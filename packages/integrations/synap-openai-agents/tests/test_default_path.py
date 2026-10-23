"""The defaults, proved on the real Agents SDK Runner with a scripted model.

Two things were missing: the stream was only used if the caller had opened it
(and the user's turn was never written at all), and the SDK's own history
decided what the model saw even after Synap had compacted the conversation.
Here a run uses `SynapRunHooks` the documented way, with no `listen()` and no
manual `report_user_turn`, and the tests look at what reached Synap and what
reached the model.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from agents import Agent, Model, ModelResponse, RunConfig, Runner, Usage, function_tool
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from pydantic import BaseModel

from synap_integrations_common import set_auto_listen
from synap_openai_agents import SynapRunHooks, report_user_turn, synap_call_model_input_filter
from synap_openai_agents.hooks import _output_text

BLOCK = "## Conversation so far (compacted 2026-10-22)\nThe user is planning a trip to Lisbon."


def reply(text: str) -> ModelResponse:
    return ModelResponse(
        output=[ResponseOutputMessage(
            id="m1", type="message", role="assistant", status="completed",
            content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
        )],
        usage=Usage(), response_id=None,
    )


class ScriptedModel(Model):
    """Answers with fixed text and keeps every input it was sent."""

    def __init__(self, text: str = "Booked.") -> None:
        self.text = text
        self.inputs: List[Any] = []
        self.instructions: List[Any] = []

    async def get_response(self, system_instructions, input, *args: Any, **kwargs: Any):
        self.inputs.append(input)
        self.instructions.append(system_instructions)
        return reply(self.text)

    def stream_response(self, *args: Any, **kwargs: Any):  # pragma: no cover - not used
        raise NotImplementedError


class StreamingModel(ScriptedModel):
    """The same scripted answer, delivered as a stream."""

    async def stream_response(self, system_instructions, input, *args: Any, **kwargs: Any):
        from openai.types.responses import Response, ResponseCompletedEvent

        self.inputs.append(input)
        response = Response(
            id="r1", created_at=0, model="m", object="response", output=reply(self.text).output,
            parallel_tool_calls=False, tool_choice="auto", tools=[])
        yield ResponseCompletedEvent(type="response.completed", response=response, sequence_number=0)


def make_sdk(*, compaction: bool = False, listening: bool = False):
    s = MagicMock()
    s.instance = MagicMock()
    s.instance.is_listening = listening

    async def listen():
        s.instance.is_listening = True

    s.instance.listen = AsyncMock(side_effect=listen)
    s.instance.send_message = AsyncMock()
    s.instance.record_tool_call = AsyncMock()
    s.instance.record_tool_result = AsyncMock()
    s.instance.record_thinking = AsyncMock()
    s.conversation = MagicMock()
    s.conversation.record_message = AsyncMock()
    if compaction:
        ctx = SimpleNamespace(
            context_block=BLOCK, history_block=None, compaction_id="cmp_1",
            recent_messages=[SimpleNamespace(role="user", content="and hotels?"),
                             SimpleNamespace(role="assistant", content="Three near the river.")],
        )
    else:
        ctx = SimpleNamespace(context_block=None, history_block=None, recent_messages=[])
    s.conversation.context.get_context_for_prompt = AsyncMock(return_value=ctx)
    return s


def sent(s):
    return [(c.kwargs["role"], c.kwargs["content"]) for c in s.instance.send_message.await_args_list]


def texts(model_input) -> List[str]:
    out = []
    for item in model_input:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content)
        out.append(content or "")
    return out


EARLIER = [
    {"role": "user", "content": "flights to Lisbon?"},
    {"role": "assistant", "content": "TAP at 9am."},
    {"role": "user", "content": "and hotels?"},
    {"role": "assistant", "content": "Three near the river."},
    {"role": "user", "content": "book the first one"},
]


class TestTheStreamIsTheDefault:
    @pytest.mark.asyncio
    async def test_a_run_with_no_listen_opens_it_and_sends_both_turns_on_it(self):
        """The user's turn is read off the model input; nobody calls report_user_turn."""
        s = make_sdk()
        hooks = SynapRunHooks(s, user_id="u1", conversation_id="c1")
        agent = Agent(name="a", instructions="x", model=ScriptedModel("Booked."))
        await Runner.run(agent, "book the first one", hooks=hooks)

        s.instance.listen.assert_awaited_once()
        assert sent(s) == [("user", "book the first one"), ("assistant", "Booked.")]
        s.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_tool_lap_writes_the_user_turn_once_and_reports_the_tool(self):
        """Two model calls in one run see the same user message."""
        s = make_sdk()
        hooks = SynapRunHooks(s, user_id="u1", conversation_id="c1")

        @function_tool
        def lookup(order: str) -> str:
            """Look an order up."""
            return "shipped"

        class ToolThenAnswer(ScriptedModel):
            async def get_response(self, system_instructions, input, *args: Any, **kwargs: Any):
                self.inputs.append(input)
                if len(self.inputs) == 1:
                    return ModelResponse(output=[ResponseFunctionToolCall(
                        type="function_call", call_id="k1", name="lookup",
                        arguments='{"order": "A1"}')], usage=Usage(), response_id=None)
                return reply("It shipped.")

        model = ToolThenAnswer()
        await Runner.run(Agent(name="a", instructions="x", model=model, tools=[lookup]),
                         "where is A1", hooks=hooks)
        assert len(model.inputs) == 2
        assert sent(s) == [("user", "where is A1"), ("assistant", "It shipped.")]
        s.instance.record_tool_call.assert_awaited_once()
        s.instance.record_tool_result.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_two_runs_write_two_user_turns_even_when_the_words_repeat(self):
        s = make_sdk()
        hooks = SynapRunHooks(s, user_id="u1", conversation_id="c1")
        agent = Agent(name="a", instructions="x", model=ScriptedModel("ok"))
        await Runner.run(agent, "yes", hooks=hooks)
        await Runner.run(agent, "yes", hooks=hooks)
        assert [t for t in sent(s) if t[0] == "user"] == [("user", "yes"), ("user", "yes")]

    @pytest.mark.asyncio
    async def test_the_reply_is_written_without_a_conversation_id(self):
        """Over REST a turn needs a conversation id; on the stream it does not,
        and the stream is the default."""
        s = make_sdk()
        hooks = SynapRunHooks(s, user_id="u1")
        await hooks.on_agent_end(MagicMock(), MagicMock(), "done")
        assert sent(s) == [("assistant", "done")]

    @pytest.mark.asyncio
    async def test_with_auto_listen_off_the_turns_go_over_rest(self):
        s = make_sdk()
        set_auto_listen(s, False)
        hooks = SynapRunHooks(s, user_id="u1", conversation_id="c1")
        agent = Agent(name="a", instructions="x", model=ScriptedModel("Booked."))
        await Runner.run(agent, "book it", hooks=hooks)
        s.instance.listen.assert_not_awaited()
        calls = s.conversation.record_message.await_args_list
        assert [(c.kwargs["role"], c.kwargs["content"]) for c in calls] == [
            ("user", "book it"), ("assistant", "Booked.")]

    @pytest.mark.asyncio
    async def test_a_manual_report_user_turn_plus_the_hook_writes_the_turn_once(self):
        """Callers who followed the old docs still call report_user_turn."""
        s = make_sdk()
        await report_user_turn(s, "book it", conversation_id="c1", user_id="u1")
        hooks = SynapRunHooks(s, user_id="u1", conversation_id="c1")
        agent = Agent(name="a", instructions="x", model=ScriptedModel("Booked."))
        await Runner.run(agent, "book it", hooks=hooks)
        assert sent(s) == [("user", "book it"), ("assistant", "Booked.")]

    @pytest.mark.asyncio
    async def test_the_compaction_block_is_never_written_back_as_a_turn(self):
        s = make_sdk()
        hooks = SynapRunHooks(s, user_id="u1", conversation_id="c1")
        await hooks.on_llm_start(
            object(), MagicMock(), None,
            [{"role": "user", "content": "<synap_conversation_history>\n" + BLOCK}])
        assert sent(s) == []


class TestRunStreamed:
    @pytest.mark.asyncio
    async def test_the_streamed_path_writes_the_turns_and_splices_too(self):
        s = make_sdk(compaction=True)
        hooks = SynapRunHooks(s, user_id="u1", conversation_id="c1")
        model = StreamingModel("Booked.")
        result = Runner.run_streamed(
            Agent(name="a", instructions="x", model=model), EARLIER, **hooks.run_kwargs())
        async for _ in result.stream_events():
            pass
        assert sent(s) == [("user", "book the first one"), ("assistant", "Booked.")]
        seen = texts(model.inputs[0])
        assert "flights to Lisbon?" not in seen
        assert any("planning a trip to Lisbon" in t for t in seen)


class TestStructuredOutput:
    def test_a_pydantic_output_is_written_as_json_not_a_repr(self):
        class Answer(BaseModel):
            text: str
            sources: List[str]

        out = _output_text(Answer(text="shipped", sources=["a", "b"]))
        assert out == '{"text":"shipped","sources":["a","b"]}'

    def test_a_dict_and_a_dataclass_are_json(self):
        import dataclasses

        @dataclasses.dataclass
        class Row:
            order: str
            eta: int

        assert _output_text({"order": "A1"}) == '{"order": "A1"}'
        assert _output_text(Row("A1", 3)) == '{"order": "A1", "eta": 3}'

    @pytest.mark.asyncio
    async def test_the_agent_end_hook_sends_the_json(self):
        class Answer(BaseModel):
            text: str

        s = make_sdk()
        await SynapRunHooks(s, user_id="u1", conversation_id="c1").on_agent_end(
            MagicMock(), MagicMock(), Answer(text="shipped"))
        assert sent(s) == [("assistant", '{"text":"shipped"}')]


class TestSynapsCompactionReplacesTheSdksHistory:
    @pytest.mark.asyncio
    async def test_the_model_sees_the_compaction_and_not_the_earlier_turns(self):
        s = make_sdk(compaction=True)
        hooks = SynapRunHooks(s, user_id="u1", conversation_id="c1")
        model = ScriptedModel()
        agent = Agent(name="a", instructions="x", model=model)
        await Runner.run(agent, EARLIER, **hooks.run_kwargs())

        seen = texts(model.inputs[0])
        assert "flights to Lisbon?" not in seen and "TAP at 9am." not in seen
        assert any("planning a trip to Lisbon" in t for t in seen)
        # Synap's turns after the compaction, then the current turn untouched.
        assert seen[-3:] == ["and hotels?", "Three near the river.", "book the first one"]

    @pytest.mark.asyncio
    async def test_the_default_run_uses_the_splice(self):
        """`hooks.run_kwargs()` is all the caller passes."""
        s = make_sdk(compaction=True)
        hooks = SynapRunHooks(s, user_id="u1", conversation_id="c1")
        cfg = hooks.run_kwargs()["run_config"]
        assert isinstance(cfg, RunConfig) and cfg.call_model_input_filter is not None
        s.conversation.context.get_context_for_prompt.assert_not_awaited()
        model = ScriptedModel()
        await Runner.run(Agent(name="a", instructions="x", model=model), EARLIER, **hooks.run_kwargs())
        s.conversation.context.get_context_for_prompt.assert_awaited()

    @pytest.mark.asyncio
    async def test_hooks_alone_leave_the_history_as_the_sdk_built_it(self):
        """Hooks cannot change the model input; this is why run_kwargs exists."""
        s = make_sdk(compaction=True)
        hooks = SynapRunHooks(s, user_id="u1", conversation_id="c1")
        model = ScriptedModel()
        await Runner.run(Agent(name="a", instructions="x", model=model), EARLIER, hooks=hooks)
        assert "flights to Lisbon?" in texts(model.inputs[0])

    @pytest.mark.asyncio
    async def test_messages_are_untouched_when_synap_has_no_compaction(self):
        s = make_sdk(compaction=False)
        hooks = SynapRunHooks(s, user_id="u1", conversation_id="c1")
        model = ScriptedModel()
        await Runner.run(Agent(name="a", instructions="x", model=model), EARLIER, **hooks.run_kwargs())
        assert texts(model.inputs[0]) == [e["content"] for e in EARLIER]

    @pytest.mark.asyncio
    async def test_messages_are_untouched_when_synap_errors(self):
        s = make_sdk(compaction=True)
        s.conversation.context.get_context_for_prompt.side_effect = RuntimeError("down")
        hooks = SynapRunHooks(s, user_id="u1", conversation_id="c1")
        model = ScriptedModel()
        await Runner.run(Agent(name="a", instructions="x", model=model), EARLIER, **hooks.run_kwargs())
        assert texts(model.inputs[0]) == [e["content"] for e in EARLIER]

    @pytest.mark.asyncio
    async def test_it_works_with_a_session_too(self):
        from agents import SQLiteSession

        s = make_sdk(compaction=True)
        hooks = SynapRunHooks(s, user_id="u1", conversation_id="c1")
        session = SQLiteSession("sess")
        await session.add_items(EARLIER[:4])
        model = ScriptedModel()
        await Runner.run(Agent(name="a", instructions="x", model=model), "book the first one",
                         session=session, **hooks.run_kwargs())
        seen = texts(model.inputs[0])
        assert "flights to Lisbon?" not in seen
        assert any("planning a trip to Lisbon" in t for t in seen)
        assert seen[-1] == "book the first one"

    @pytest.mark.asyncio
    async def test_the_callers_own_filter_still_runs_after_ours(self):
        from agents.run import ModelInputData

        s = make_sdk(compaction=True)
        hooks = SynapRunHooks(s, user_id="u1", conversation_id="c1")
        seen_by_theirs = []

        def theirs(data):
            seen_by_theirs.append(texts(data.model_data.input))
            return ModelInputData(input=data.model_data.input, instructions="theirs")

        model = ScriptedModel()
        await Runner.run(Agent(name="a", instructions="x", model=model), EARLIER,
                         **hooks.run_kwargs(call_model_input_filter=theirs))
        assert any("planning a trip" in t for t in seen_by_theirs[0])
        assert model.instructions[0] == "theirs"

    @pytest.mark.asyncio
    async def test_the_compaction_is_not_sent_twice_next_to_synap_st_instructions(self):
        from synap_openai_agents import synap_st_instructions

        s = make_sdk(compaction=True)
        ctx = s.conversation.context.get_context_for_prompt.return_value
        ctx.available = True
        hooks = SynapRunHooks(s, user_id="u1", conversation_id="c1")
        model = ScriptedModel()
        agent = Agent(name="a", model=model,
                      instructions=synap_st_instructions(s, "c1", system="Be brief."))
        await Runner.run(agent, EARLIER, **hooks.run_kwargs())
        assert "planning a trip to Lisbon" in model.instructions[0]
        assert not any("planning a trip to Lisbon" in t for t in texts(model.inputs[0]))
        assert "flights to Lisbon?" not in texts(model.inputs[0])

    @pytest.mark.asyncio
    async def test_a_tool_call_in_the_current_turn_stays_paired_with_its_result(self):
        s = make_sdk(compaction=True)
        hooks = SynapRunHooks(s, user_id="u1", conversation_id="c1")
        items = list(EARLIER) + [
            {"type": "function_call", "call_id": "k1", "name": "book", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "k1", "output": "ok"},
        ]
        from agents.run import CallModelData, ModelInputData

        data = CallModelData(model_data=ModelInputData(input=items, instructions=None),
                             agent=MagicMock(), context=None)
        out = await synap_call_model_input_filter(s, "c1")(data)
        assert out.input[-2]["type"] == "function_call" and out.input[-1]["type"] == "function_call_output"
        assert out.input[-3]["content"] == "book the first one"
