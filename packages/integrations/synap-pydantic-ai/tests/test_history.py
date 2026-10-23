"""Synap's compaction reaches the model through Pydantic AI's own history seam.

These run a real ``Agent`` with a ``FunctionModel`` that records exactly what
the model is sent, so they prove the framework calls the processor on its
default path, not that a stand-in does.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from synap_pydantic_ai import (
    SynapDeps,
    install_synap_history,
    register_synap_st_system_prompt,
    register_synap_tools,
    synap_history_processor,
    synap_run,
)

BLOCK = "## Conversation so far (compacted 2026-10-22)\nThe user is planning a trip to Lisbon."


class FakeInstance:
    """Behaves like sdk.instance: listen() flips is_listening on."""

    def __init__(self):
        self.is_listening = False
        self.listen_calls = 0
        self.send_message = AsyncMock()
        self.record_tool_call = AsyncMock()
        self.record_tool_result = AsyncMock()
        self.record_thinking = AsyncMock()
        self._controller = SimpleNamespace(_transport=object())

    async def listen(self):
        self.listen_calls += 1
        self.is_listening = True

    async def stop_listening(self):
        self.is_listening = False


def make_sdk(compaction=True, recent=(("user", "flights?"), ("assistant", "TAP at 9am"))):
    ctx = None
    if compaction:
        ctx = SimpleNamespace(
            context_block=BLOCK, history_block=None, compaction_id="cmp_1",
            recent_messages=[SimpleNamespace(role=r, content=c) for r, c in recent],
        )
    get = AsyncMock(return_value=ctx if ctx is not None else SimpleNamespace(
        context_block=None, history_block=None, recent_messages=[], compaction_id=None))
    mem = SimpleNamespace(formatted_context="Prefers window seats.")
    return SimpleNamespace(
        instance=FakeInstance(),
        _initialized=True,
        fetch=AsyncMock(return_value=mem),
        conversation=SimpleNamespace(
            record_message=AsyncMock(),
            context=SimpleNamespace(get_context_for_prompt=get),
        ),
    )


def _history() -> List[ModelMessage]:
    return [
        ModelRequest(parts=[SystemPromptPart("You are a travel agent."),
                            UserPromptPart("I want to visit Portugal")]),
        ModelResponse(parts=[TextPart("Great, when?")]),
        ModelRequest(parts=[UserPromptPart("In May, for a week")]),
        ModelResponse(parts=[TextPart("Lisbon is lovely in May.")]),
    ]


class Recorder:
    """A model that answers "ok" and keeps what it was sent."""

    def __init__(self):
        self.calls: List[List[ModelMessage]] = []

    def model(self):
        async def respond(messages: List[ModelMessage], info: AgentInfo) -> ModelResponse:
            self.calls.append(list(messages))
            return ModelResponse(parts=[TextPart("ok")])

        async def stream(messages: List[ModelMessage], info: AgentInfo):
            # synap_run installs an event_stream_handler, which makes the run stream
            self.calls.append(list(messages))
            yield "ok"

        return FunctionModel(respond, stream_function=stream)

    @property
    def seen(self) -> str:
        """Everything the model was sent on its last call, as one string."""
        out = []
        for m in self.calls[-1]:
            for p in m.parts:
                out.append(f"{type(p).__name__}:{getattr(p, 'content', '')}")
        return "\n".join(out)


def _deps(sdk, conversation_id="conv-1"):
    return SynapDeps(sdk=sdk, user_id="u1", customer_id="c1", conversation_id=conversation_id)


# ---------------------------------------------------------------------------
# The seam, driven by the real framework
# ---------------------------------------------------------------------------


class TestCompactionReachesTheModel:
    @pytest.mark.asyncio
    async def test_register_synap_tools_wires_it_so_a_plain_run_gets_it(self):
        rec = Recorder()
        agent = Agent(rec.model(), deps_type=SynapDeps)
        register_synap_tools(agent)
        await agent.run("and the hotel?", deps=_deps(make_sdk()), message_history=_history())
        seen = rec.seen
        assert "The user is planning a trip to Lisbon" in seen
        assert "<synap_conversation_history>" in seen
        # the turns the compaction covers are gone; Synap's later turns are in
        assert "I want to visit Portugal" not in seen
        assert "In May, for a week" not in seen
        assert "TAP at 9am" in seen
        # the current turn and the framework's system prompt are untouched
        assert "and the hotel?" in seen
        assert "You are a travel agent." in seen

    @pytest.mark.asyncio
    async def test_messages_are_untouched_when_synap_has_no_compaction(self):
        rec = Recorder()
        agent = Agent(rec.model(), deps_type=SynapDeps)
        register_synap_tools(agent)
        await agent.run("and the hotel?", deps=_deps(make_sdk(compaction=False)),
                        message_history=_history())
        assert "I want to visit Portugal" in rec.seen
        assert "In May, for a week" in rec.seen
        assert "synap_conversation_history" not in rec.seen

    @pytest.mark.asyncio
    async def test_messages_are_untouched_when_the_sdk_fails(self):
        rec = Recorder()
        agent = Agent(rec.model(), deps_type=SynapDeps)
        register_synap_tools(agent)
        sdk = make_sdk()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(side_effect=RuntimeError("down"))
        await agent.run("and the hotel?", deps=_deps(sdk), message_history=_history())
        assert "In May, for a week" in rec.seen

    @pytest.mark.asyncio
    async def test_no_conversation_id_means_no_fetch(self):
        rec = Recorder()
        agent = Agent(rec.model(), deps_type=SynapDeps)
        register_synap_tools(agent)
        sdk = make_sdk()
        await agent.run("hi", deps=_deps(sdk, conversation_id=None), message_history=_history())
        sdk.conversation.context.get_context_for_prompt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_synap_run_installs_it_without_any_register_call(self):
        rec = Recorder()
        agent = Agent(rec.model(), deps_type=SynapDeps)
        await synap_run(agent, "and the hotel?", deps=_deps(make_sdk()), message_history=_history())
        assert "The user is planning a trip to Lisbon" in rec.seen
        assert "In May, for a week" not in rec.seen

    @pytest.mark.asyncio
    async def test_registering_twice_installs_it_once(self):
        agent = Agent(Recorder().model(), deps_type=SynapDeps)
        install_synap_history(agent)
        install_synap_history(agent)
        sdk = make_sdk()
        await agent.run("hi", deps=_deps(sdk), message_history=_history())
        assert sdk.conversation.context.get_context_for_prompt.await_count == 1

    @pytest.mark.asyncio
    async def test_a_tool_calling_run_fetches_once(self):
        calls = {"n": 0}
        seen: List[List[ModelMessage]] = []

        async def respond(messages: List[ModelMessage], info: AgentInfo) -> ModelResponse:
            from pydantic_ai.messages import ToolCallPart
            seen.append(list(messages))
            calls["n"] += 1
            if calls["n"] == 1:
                return ModelResponse(parts=[ToolCallPart("ping", {})])
            return ModelResponse(parts=[TextPart("done")])

        agent = Agent(FunctionModel(respond), deps_type=SynapDeps)

        @agent.tool_plain
        def ping() -> str:
            return "pong"

        sdk = make_sdk()
        install_synap_history(agent)
        await agent.run("go", deps=_deps(sdk), message_history=_history())
        assert calls["n"] == 2
        assert sdk.conversation.context.get_context_for_prompt.await_count == 1
        # the second request still carries the compaction, and the tool pair
        second = "".join(str(getattr(p, "content", "")) for m in seen[1] for p in m.parts)
        assert "Lisbon" in second and "pong" in second

    @pytest.mark.asyncio
    async def test_feeding_all_messages_back_replaces_the_block_never_stacks_it(self):
        rec = Recorder()
        agent = Agent(rec.model(), deps_type=SynapDeps)
        install_synap_history(agent)
        sdk = make_sdk()
        first = await agent.run("and the hotel?", deps=_deps(sdk), message_history=_history())
        await agent.run("and the car?", deps=_deps(sdk), message_history=first.all_messages())
        assert rec.seen.count("<synap_conversation_history>") == 1

    @pytest.mark.asyncio
    async def test_new_messages_is_still_only_this_turn(self):
        agent = Agent(Recorder().model(), deps_type=SynapDeps)
        install_synap_history(agent)
        result = await agent.run("and the hotel?", deps=_deps(make_sdk()), message_history=_history())
        new = result.new_messages()
        assert len(new) == 2
        assert "and the hotel?" in str(new[0])

    def test_the_processor_is_a_plain_async_function_pydantic_ai_can_inspect(self):
        proc = synap_history_processor()
        assert getattr(proc, "_synap_history_processor") is True


# ---------------------------------------------------------------------------
# Dynamic system prompts, and no double conversation context
# ---------------------------------------------------------------------------


class TestSystemPromptsFollowTheRun:
    @pytest.mark.asyncio
    async def test_the_memory_prompt_is_refreshed_on_a_run_with_history(self):
        """Without dynamic=True the second run's prompt is the first run's text."""
        rec = Recorder()
        agent = Agent(rec.model(), deps_type=SynapDeps, system_prompt="Be brief.")
        register_synap_tools(agent)
        sdk = make_sdk(compaction=False)
        first = await agent.run("hi", deps=_deps(sdk))
        assert "Prefers window seats." in rec.seen
        sdk.fetch = AsyncMock(return_value=SimpleNamespace(formatted_context="Now prefers aisle."))
        await agent.run("again", deps=_deps(sdk), message_history=first.all_messages())
        assert "Now prefers aisle." in rec.seen
        assert "Prefers window seats." not in rec.seen

    @pytest.mark.asyncio
    async def test_the_memory_prompt_does_not_ask_for_the_conversation_too(self):
        agent = Agent(Recorder().model(), deps_type=SynapDeps)
        register_synap_tools(agent)
        sdk = make_sdk(compaction=False)
        await agent.run("hi", deps=_deps(sdk))
        assert sdk.fetch.await_args.kwargs["include_conversation_context"] is False

    @pytest.mark.asyncio
    async def test_the_st_block_is_sent_when_the_framework_has_no_history(self):
        rec = Recorder()
        agent = Agent(rec.model(), deps_type=SynapDeps)
        register_synap_st_system_prompt(agent, system="Be brief.")
        sdk = make_sdk(compaction=False)
        sdk.conversation.context.get_context_for_prompt = AsyncMock(return_value=SimpleNamespace(
            available=True, formatted_context="User asked about Lisbon.",
            context_block=None, history_block=None, recent_messages=[], compaction_id=None))
        await agent.run("hi", deps=_deps(sdk))
        assert "User asked about Lisbon." in rec.seen

    @pytest.mark.asyncio
    async def test_the_st_block_is_not_sent_next_to_a_framework_history(self):
        """It holds the recent turns, which the history already carries."""
        rec = Recorder()
        agent = Agent(rec.model(), deps_type=SynapDeps)
        register_synap_st_system_prompt(agent, system="Be brief.")
        sdk = make_sdk(compaction=False)
        sdk.conversation.context.get_context_for_prompt = AsyncMock(return_value=SimpleNamespace(
            available=True, formatted_context="User asked about Lisbon.",
            context_block=None, history_block=None, recent_messages=[], compaction_id=None))
        first = await agent.run("hi", deps=_deps(sdk))
        assert "User asked about Lisbon." in rec.seen
        await agent.run("and again", deps=_deps(sdk), message_history=first.all_messages())
        assert "User asked about Lisbon." not in rec.seen
        assert "Be brief." in rec.seen


# ---------------------------------------------------------------------------
# The stream is the default path: nobody called listen()
# ---------------------------------------------------------------------------


class TestStreamOpensByDefault:
    @pytest.mark.asyncio
    async def test_synap_run_opens_the_stream_and_everything_goes_on_it(self):
        from pydantic_ai.messages import ToolCallPart

        steps = {"n": 0}

        async def respond(messages, info):
            steps["n"] += 1
            if steps["n"] == 1:
                return ModelResponse(parts=[ToolCallPart("ping", {}, tool_call_id="t1")])
            return ModelResponse(parts=[TextPart("pong received")])

        async def stream(messages, info):
            from pydantic_ai.models.function import DeltaToolCall
            steps["n"] += 1
            if steps["n"] == 1:
                yield {0: DeltaToolCall(name="ping", json_args="{}", tool_call_id="t1")}
            else:
                yield "pong received"

        agent = Agent(FunctionModel(respond, stream_function=stream), deps_type=SynapDeps)

        @agent.tool_plain
        def ping() -> str:
            return "pong"

        sdk = make_sdk(compaction=False)
        assert sdk.instance.is_listening is False
        await synap_run(agent, "ping it", deps=_deps(sdk))

        assert sdk.instance.listen_calls == 1
        sent = [c.kwargs["event_type"] for c in sdk.instance.send_message.await_args_list]
        assert sent == ["user_message", "assistant_message"]
        sdk.instance.record_tool_call.assert_awaited_once()
        sdk.instance.record_tool_result.assert_awaited_once()
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_run_with_no_conversation_id_streams_and_lets_the_server_mint_one(self):
        agent = Agent(Recorder().model(), deps_type=SynapDeps)
        sdk = make_sdk(compaction=False)
        await synap_run(agent, "hi", deps=_deps(sdk, conversation_id=None))
        assert sdk.instance.send_message.await_count == 2
        assert "conversation_id" not in sdk.instance.send_message.await_args_list[0].kwargs

    @pytest.mark.asyncio
    async def test_a_stream_that_will_not_open_falls_back_to_rest_once_per_turn(self):
        agent = Agent(Recorder().model(), deps_type=SynapDeps)
        sdk = make_sdk(compaction=False)
        sdk.instance.listen = AsyncMock(side_effect=ConnectionError("no route"))
        await synap_run(agent, "hi", deps=_deps(sdk))
        roles = [c.kwargs["role"] for c in sdk.conversation.record_message.await_args_list]
        assert roles == ["user", "assistant"]
        sdk.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_opting_out_keeps_turns_on_rest(self):
        from synap_integrations_common import set_auto_listen
        agent = Agent(Recorder().model(), deps_type=SynapDeps)
        sdk = make_sdk(compaction=False)
        set_auto_listen(sdk, False)
        await synap_run(agent, "hi", deps=_deps(sdk))
        assert sdk.instance.listen_calls == 0
        assert sdk.conversation.record_message.await_count == 2

    @pytest.mark.asyncio
    async def test_the_compaction_block_is_never_written_back_as_a_turn(self):
        from synap_pydantic_ai.stream import _report_turn
        sdk = make_sdk(compaction=False)
        await _report_turn(_deps(sdk), "user", "<synap_conversation_history>\nx\n</synap_conversation_history>")
        sdk.instance.send_message.assert_not_awaited()
        sdk.conversation.record_message.assert_not_awaited()
