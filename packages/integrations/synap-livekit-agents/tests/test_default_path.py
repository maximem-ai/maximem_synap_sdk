"""The defaults of attach_synap_recording: the gRPC stream, and Synap's compaction.

Two things happen with no opt-in from the caller:

- the first turn opens the stream (``sdk.instance.listen()``) and everything
  after rides it;
- once Synap has compacted the conversation, the model is sent Synap's
  compaction in place of the agent's earlier turns.

The second is proven against a REAL ``AgentSession`` and a fake LLM that
records the chat context it is handed, so the seam is the one the framework
actually calls, not a double of it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from livekit.agents import Agent, AgentSession, ChatContext, llm
from livekit.agents.llm import ChatChunk, ChoiceDelta, ChatMessage
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from synap_integrations_common import set_auto_listen
from synap_livekit_agents import attach_synap_recording
from synap_livekit_agents import compaction
from synap_livekit_agents.compaction import splice_synap_history

PREAMBLE_OPEN = "<synap_conversation_history>"


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _Stream(llm.LLMStream):
    async def _run(self) -> None:
        self._llm.inputs.append(
            [(i.type, getattr(i, "role", None), getattr(i, "text_content", None))
             for i in self._chat_ctx.items]
        )
        self._event_ch.send_nowait(
            ChatChunk(id="x", delta=ChoiceDelta(role="assistant", content="ok"))
        )


class FakeLLM(llm.LLM):
    """Records the chat context of every request it is sent."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs: list = []

    @property
    def model(self) -> str:
        return "fake-model"

    def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **kw):
        return _Stream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


def _sdk(compaction_block="", turns=(), *, listening=False):
    """An SDK double whose stream opens when listen() is awaited."""
    sdk = MagicMock()
    sdk.instance.is_listening = listening

    async def listen():
        sdk.instance.is_listening = True

    sdk.instance.listen = AsyncMock(side_effect=listen)
    for name in ("send_message", "record_tool_call", "record_tool_result",
                 "record_thinking", "end_session"):
        setattr(sdk.instance, name, AsyncMock())
    sdk.conversation.record_message = AsyncMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=SimpleNamespace(
            context_block=compaction_block,
            history_block="",
            recent_messages=[SimpleNamespace(role=r, content=t) for r, t in turns],
            compaction_id="c1" if compaction_block else None,
        )
    )
    return sdk


def _history_ctx() -> ChatContext:
    ctx = ChatContext()
    for role, text in (("user", "turn-1"), ("assistant", "reply-1"),
                       ("user", "turn-2"), ("assistant", "reply-2")):
        ctx.add_message(role=role, content=text)
    return ctx


class MockSession:
    def __init__(self):
        self._handlers: dict = {}

    def on(self, event, cb):
        self._handlers[event] = cb

    def fire(self, event, payload=None):
        self._handlers[event](payload)


def _texts(model_input):
    return [t for (_type, _role, t) in model_input if t]


async def _settle():
    for _ in range(8):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# (a) gRPC by default
# ---------------------------------------------------------------------------


class TestTheStreamOpensByDefault:
    async def test_the_first_turn_opens_the_stream_and_is_sent_on_it(self):
        sdk = _sdk()
        assert sdk.instance.is_listening is False
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("conversation_item_added",
                     SimpleNamespace(item=ChatMessage(role="user", content=["hello"])))
        await _settle()
        sdk.instance.listen.assert_awaited_once()
        sdk.instance.send_message.assert_awaited_once()
        assert sdk.instance.send_message.await_args.kwargs["event_type"] == "user_message"
        sdk.conversation.record_message.assert_not_awaited()

    async def test_later_turns_reuse_it(self):
        sdk = _sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        for role, text in (("user", "a"), ("assistant", "b"), ("user", "c")):
            session.fire("conversation_item_added",
                         SimpleNamespace(item=ChatMessage(role=role, content=[text])))
        await _settle()
        assert sdk.instance.listen.await_count == 1
        assert sdk.instance.send_message.await_count == 3
        sdk.conversation.record_message.assert_not_awaited()

    async def test_a_tool_call_as_the_first_event_opens_it_too(self):
        sdk = _sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        call = SimpleNamespace(call_id="k1", name="lookup", arguments='{"a": 1}')
        out = SimpleNamespace(call_id="k1", name="lookup", output="done")
        session.fire("function_tools_executed",
                     SimpleNamespace(function_calls=[call], function_call_outputs=[out]))
        await _settle()
        sdk.instance.listen.assert_awaited_once()
        sdk.instance.record_tool_call.assert_awaited_once()
        sdk.instance.record_tool_result.assert_awaited_once()

    async def test_a_real_session_streams_both_ends_of_the_turn(self):
        sdk = _sdk()
        session = AgentSession(llm=FakeLLM())
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        await session.start(Agent(instructions="be nice"))
        await session.run(user_input="hello there")
        await _settle()
        sent = [(c.kwargs["event_type"], c.kwargs["content"])
                for c in sdk.instance.send_message.await_args_list]
        assert sent == [("user_message", "hello there"), ("assistant_message", "ok")]
        sdk.instance.listen.assert_awaited_once()
        sdk.conversation.record_message.assert_not_awaited()
        await session.aclose()

    async def test_the_opt_out_keeps_it_on_rest(self, monkeypatch):
        monkeypatch.setenv("SYNAP_AUTO_LISTEN", "false")
        sdk = _sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("conversation_item_added",
                     SimpleNamespace(item=ChatMessage(role="user", content=["hello"])))
        await _settle()
        sdk.instance.listen.assert_not_awaited()
        sdk.conversation.record_message.assert_awaited_once()

    async def test_set_auto_listen_is_the_per_sdk_switch(self):
        sdk = _sdk()
        set_auto_listen(sdk, False)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("conversation_item_added",
                     SimpleNamespace(item=ChatMessage(role="assistant", content=["hi"])))
        await _settle()
        sdk.instance.listen.assert_not_awaited()
        sdk.conversation.record_message.assert_awaited_once()

    async def test_synaps_own_history_is_never_recorded_as_a_turn(self):
        sdk = _sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("conversation_item_added", SimpleNamespace(
            item=ChatMessage(role="assistant",
                             content=[f"{PREAMBLE_OPEN}\nsummary\n</synap_conversation_history>"])))
        await _settle()
        sdk.instance.send_message.assert_not_awaited()
        sdk.conversation.record_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# (b) Synap's compaction in place of the agent's earlier turns
# ---------------------------------------------------------------------------


async def _run_turn(sdk, *, agent=None, user_input="turn-4 now", chat_ctx=None):
    fake = FakeLLM()
    session = AgentSession(llm=fake)
    attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
    agent = agent or Agent(instructions="be nice", chat_ctx=chat_ctx or _history_ctx())
    await session.start(agent)
    await session.run(user_input=user_input)
    await _settle()
    return session, fake


class TestTheModelSeesSynapsCompaction:
    async def test_once_compacted_the_model_gets_the_preamble_not_the_old_turns(self):
        sdk = _sdk("SUMMARY: refund requested", [("user", "turn-3"), ("assistant", "reply-3")])
        session, fake = await _run_turn(sdk)
        (seen,) = fake.inputs
        texts = _texts(seen)
        assert texts[0] == "be nice"
        assert texts[1].startswith(PREAMBLE_OPEN) and "SUMMARY: refund requested" in texts[1]
        assert texts[2:] == ["turn-3", "reply-3", "turn-4 now"]
        for replaced in ("turn-1", "reply-1", "turn-2", "reply-2"):
            assert replaced not in texts
        await session.aclose()

    async def test_the_agents_own_history_is_not_rewritten(self):
        sdk = _sdk("SUMMARY: x", [("user", "turn-3")])
        session, _ = await _run_turn(sdk)
        kept = _texts([(i.type, None, getattr(i, "text_content", None))
                       for i in session.current_agent.chat_ctx.items])
        assert kept[:5] == ["be nice", "turn-1", "reply-1", "turn-2", "reply-2"]
        assert not any(t.startswith(PREAMBLE_OPEN) for t in kept)
        await session.aclose()

    async def test_with_no_compaction_the_model_gets_the_agents_history_untouched(self):
        sdk = _sdk("")  # Synap has not compacted yet
        session, fake = await _run_turn(sdk)
        texts = _texts(fake.inputs[0])
        assert texts == ["be nice", "turn-1", "reply-1", "turn-2", "reply-2", "turn-4 now"]
        await session.aclose()

    async def test_when_synap_fails_the_agents_history_is_untouched(self):
        sdk = _sdk("SUMMARY: x")
        sdk.conversation.context.get_context_for_prompt = AsyncMock(side_effect=RuntimeError("down"))
        session, fake = await _run_turn(sdk)
        assert _texts(fake.inputs[0])[1:5] == ["turn-1", "reply-1", "turn-2", "reply-2"]
        await session.aclose()

    async def test_a_slow_read_does_not_hold_up_the_reply_and_is_not_retried_at_once(self, monkeypatch):
        monkeypatch.setattr(compaction, "_FETCH_TIMEOUT_S", 0.05)

        async def slow(**kw):
            await asyncio.sleep(5)

        sdk = _sdk("SUMMARY: x")
        sdk.conversation.context.get_context_for_prompt = AsyncMock(side_effect=slow)
        session, fake = await _run_turn(sdk)
        assert "turn-1" in _texts(fake.inputs[0])
        await session.run(user_input="and another")
        await _settle()
        assert sdk.conversation.context.get_context_for_prompt.await_count == 1
        await session.aclose()

    async def test_a_subclass_llm_node_still_runs_on_the_spliced_context(self):
        seen = []

        class Mine(Agent):
            async def llm_node(self, chat_ctx, tools, model_settings):
                seen.append(_texts([(i.type, None, getattr(i, "text_content", None))
                                    for i in chat_ctx.items]))
                async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                    yield chunk

        sdk = _sdk("SUMMARY: x", [("user", "turn-3")])
        agent = Mine(instructions="be nice", chat_ctx=_history_ctx())
        session, fake = await _run_turn(sdk, agent=agent)
        assert seen and seen[0][1].startswith(PREAMBLE_OPEN)
        assert "turn-1" not in fake.inputs[0][1:] and "turn-1" not in _texts(fake.inputs[0])
        await session.aclose()

    async def test_the_agent_a_handoff_lands_on_is_covered_too(self):
        sdk = _sdk("SUMMARY: x", [("user", "turn-3")])
        session, fake = await _run_turn(sdk)
        second = Agent(instructions="second agent", chat_ctx=_history_ctx())
        session.update_agent(second)  # synchronous: schedules the swap
        await asyncio.sleep(0.2)
        await session.run(user_input="to the second")
        await _settle()
        texts = _texts(fake.inputs[-1])
        assert texts[0] == "second agent"
        assert texts[1].startswith(PREAMBLE_OPEN)
        assert "turn-1" not in texts
        await session.aclose()

    async def test_a_second_attach_does_not_wrap_the_agent_twice(self):
        sdk = _sdk("SUMMARY: x", [("user", "turn-3")])
        fake = FakeLLM()
        session = AgentSession(llm=fake)
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        await session.start(Agent(instructions="be nice", chat_ctx=_history_ctx()))
        await session.run(user_input="turn-4 now")
        await _settle()
        assert sdk.conversation.context.get_context_for_prompt.await_count == 1
        await session.aclose()


class TestSpliceSynapHistory:
    async def test_replaces_in_place_and_says_so(self):
        sdk = _sdk("SUMMARY: x", [("user", "turn-3")])
        ctx = _history_ctx()
        ctx.add_message(role="user", content="now")
        assert await splice_synap_history(ctx, sdk, conversation_id="c1") is True
        texts = [i.text_content for i in ctx.items]
        assert texts[0].startswith(PREAMBLE_OPEN)
        assert texts[-1] == "now"
        assert "turn-1" not in texts

    async def test_leaves_the_context_alone_when_there_is_nothing_to_splice(self):
        ctx = _history_ctx()
        before = [i.id for i in ctx.items]
        assert await splice_synap_history(ctx, _sdk(""), conversation_id="c1") is False
        assert [i.id for i in ctx.items] == before

    async def test_keeps_the_current_turns_tool_call_and_result_paired(self):
        sdk = _sdk("SUMMARY: x", [("user", "turn-3")])
        ctx = _history_ctx()
        ctx.add_message(role="user", content="now")
        ctx.items.append(llm.FunctionCall(call_id="k", name="t", arguments="{}"))
        ctx.items.append(llm.FunctionCallOutput(call_id="k", name="t", output="r", is_error=False))
        await splice_synap_history(ctx, sdk, conversation_id="c1")
        assert [i.type for i in ctx.items][-3:] == ["message", "function_call", "function_call_output"]
