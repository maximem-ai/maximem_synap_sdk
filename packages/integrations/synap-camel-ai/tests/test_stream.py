"""Tests for the live-stream half of SynapAgentMemory.

CAMEL has no callback protocol, so ``write_records`` is the only place an
integration can see a turn — and it sees all of it: the question, the tool calls
CAMEL records on the assistant message, the tool results it records as
``FunctionCallingMessage``, the provider's reasoning, and the answer.

These pin that contract:

- stream first, REST only as a fallback, never both;
- tool arguments arrive as the parsed dict, not as the JSON string CAMEL stores;
- what is reported as a tool result survives ``json.dumps``, because the SDK
  encodes it and ``stream_events`` swallows whatever that raises;
- a tool call and its result share one ``tool_call_id``;
- a replayed record is not a second turn;
- none of it can raise into ``ChatAgent.step``.
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from camel.memories import MemoryRecord
from camel.messages import BaseMessage, FunctionCallingMessage
from camel.types import OpenAIBackendRole, RoleType
from camel.utils.tool_result import ToolResult

from synap_camel_ai import SynapAgentMemory
from synap_camel_ai import memory as memory_module
from synap_camel_ai.memory import _tool_args, _tool_result


# ── fixtures ─────────────────────────────────────────────────────────────────


def _streaming_sdk(is_listening: bool = True):
    sdk = MagicMock()
    sdk.instance.is_listening = is_listening
    sdk.instance.send_message = AsyncMock()
    sdk.instance.record_thinking = AsyncMock()
    sdk.instance.record_tool_call = AsyncMock()
    sdk.instance.record_tool_result = AsyncMock()
    sdk.memories.create = AsyncMock()
    return sdk


def _memory(sdk, **kwargs):
    kwargs.setdefault("conversation_id", "conv-1")
    return SynapAgentMemory(sdk, "alice", customer_id="acme", **kwargs)


# ── record factories, matching what ChatAgent.update_memory writes ───────────


def user_rec(text="where is my order"):
    return MemoryRecord(
        message=BaseMessage.make_user_message("user", text),
        role_at_backend=OpenAIBackendRole.USER,
    )


def asst_rec(text="it ships tomorrow", reasoning=None):
    return MemoryRecord(
        message=BaseMessage(
            role_name="assistant",
            role_type=RoleType.ASSISTANT,
            meta_dict={},
            content=text,
            reasoning_content=reasoning,
        ),
        role_at_backend=OpenAIBackendRole.ASSISTANT,
    )


def tool_call_rec(
    name="lookup_order", arguments='{"pnr": "QX41RT"}', call_id="call-1"
):
    """What ``_record_assistant_tool_calls_from_requests`` writes."""
    return MemoryRecord(
        message=BaseMessage(
            role_name="assistant",
            role_type=RoleType.ASSISTANT,
            meta_dict={
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ]
            },
            content="",
        ),
        role_at_backend=OpenAIBackendRole.ASSISTANT,
    )


def tool_result_rec(result="shipped", name="lookup_order", call_id="call-1"):
    """What ``_record_tool_calling`` writes."""
    return MemoryRecord(
        message=FunctionCallingMessage(
            role_name="assistant",
            role_type=RoleType.ASSISTANT,
            meta_dict=None,
            content="",
            func_name=name,
            result=result,
            tool_call_id=call_id,
        ),
        role_at_backend=OpenAIBackendRole.FUNCTION,
    )


# ── the conversation the events belong to ────────────────────────────────────


def test_conversation_id_defaults_to_the_document_id(mock_sdk):
    mem = SynapAgentMemory(mock_sdk, "alice")
    assert mem._conversation_id == mem._doc_id


def test_an_explicit_conversation_id_wins(mock_sdk):
    mem = SynapAgentMemory(mock_sdk, "alice", conversation_id="conv-9")
    assert mem._conversation_id == "conv-9"


def test_the_events_carry_the_conversation_and_the_scope():
    sdk = _streaming_sdk()
    _memory(sdk).write_records([user_rec()])
    kwargs = sdk.instance.send_message.await_args.kwargs
    assert kwargs["conversation_id"] == "conv-1"
    assert kwargs["user_id"] == "alice"
    assert kwargs["customer_id"] == "acme"


# ── stream first, REST only as a fallback ────────────────────────────────────


class TestStreamFirstRestFallback:
    def test_the_user_turn_goes_out_on_the_stream(self):
        sdk = _streaming_sdk()
        _memory(sdk).write_records([user_rec("where is my order")])
        kwargs = sdk.instance.send_message.await_args.kwargs
        assert kwargs["event_type"] == "user_message"
        assert kwargs["content"] == "where is my order"

    def test_the_assistant_turn_is_the_anticipation_moment(self):
        sdk = _streaming_sdk()
        _memory(sdk).write_records([asst_rec("it ships tomorrow")])
        kwargs = sdk.instance.send_message.await_args.kwargs
        assert kwargs["event_type"] == "assistant_message"
        assert kwargs["role"] == "assistant"

    def test_a_streamed_turn_is_NOT_also_ingested_over_rest(self):
        """The double-write this rule exists to prevent.

        The server persists the streamed turn and extracts from the
        conversation it builds; the transcript ingest does the same job again.
        """
        sdk = _streaming_sdk()
        mem = _memory(sdk)
        mem.write_records([user_rec()])
        mem.write_records([asst_rec()])
        sdk.memories.create.assert_not_awaited()

    def test_with_no_stream_the_transcript_is_ingested_as_before(self):
        sdk = _streaming_sdk(is_listening=False)
        mem = _memory(sdk)
        mem.write_records([user_rec()])
        mem.write_records([asst_rec()])
        sdk.memories.create.assert_awaited_once()
        sdk.instance.send_message.assert_not_awaited()

    def test_a_failed_send_falls_back_to_the_transcript(self):
        """``report_turn`` says whether it went out, so this is decidable."""
        sdk = _streaming_sdk()
        sdk.instance.send_message.side_effect = RuntimeError("stream died")
        mem = _memory(sdk)
        mem.write_records([user_rec()])
        mem.write_records([asst_rec()])
        sdk.memories.create.assert_awaited_once()

    def test_a_dropped_QUESTION_also_falls_back(self):
        """A turn is both halves. Losing the question and keeping the answer
        leaves Synap a reply to a question nobody asked."""
        sdk = _streaming_sdk()

        def only_the_question_fails(**kwargs):
            if kwargs.get("event_type") == "user_message":
                raise RuntimeError("stream died")

        sdk.instance.send_message.side_effect = only_the_question_fails
        _memory(sdk).write_records([user_rec(), asst_rec()])
        sdk.memories.create.assert_awaited_once()

    def test_with_no_stream_the_async_bridge_is_never_entered(self, monkeypatch):
        """A caller who never opened a stream behaves exactly as before.

        ``run_async`` applies ``nest_asyncio`` to whatever loop it finds, which
        patches that loop's class globally. Doing it on every write for people
        who never asked for streaming is a side effect they did not sign up
        for; ``_report`` answers no before reaching the bridge.
        """
        sdk = _streaming_sdk(is_listening=False)
        calls = []
        real = memory_module.run_async

        def counted(coro):
            calls.append(coro)
            return real(coro)

        monkeypatch.setattr(memory_module, "run_async", counted)
        _memory(sdk).write_records([user_rec()])
        assert calls == []

    def test_a_tool_lap_does_not_ingest_the_transcript_mid_turn(self):
        sdk = _streaming_sdk()
        mem = _memory(sdk)
        mem.write_records([user_rec()])
        mem.write_records([tool_call_rec()])
        mem.write_records([tool_result_rec()])
        sdk.memories.create.assert_not_awaited()


# ── tool calls ───────────────────────────────────────────────────────────────


class TestToolCalls:
    def test_a_tool_call_is_reported(self):
        sdk = _streaming_sdk()
        _memory(sdk).write_records([tool_call_rec(name="lookup_order")])
        sdk.instance.record_tool_call.assert_awaited_once()
        assert sdk.instance.record_tool_call.await_args.args[0] == "lookup_order"

    def test_the_args_are_the_parsed_dict_not_the_json_string(self):
        """⚠ CAMEL stores ``arguments`` as the string the provider sent, and
        ``report_tool_call`` keeps a dict and drops everything else. Passing the
        string through reports the call with no arguments and says nothing."""
        sdk = _streaming_sdk()
        _memory(sdk).write_records([tool_call_rec(arguments='{"pnr": "QX41RT"}')])
        assert sdk.instance.record_tool_call.await_args.args[1] == {"pnr": "QX41RT"}

    def test_every_call_in_one_message_is_reported(self):
        sdk = _streaming_sdk()
        record = tool_call_rec()
        record.message.meta_dict["tool_calls"].append(
            {
                "id": "call-2",
                "type": "function",
                "function": {"name": "check_stock", "arguments": "{}"},
            }
        )
        _memory(sdk).write_records([record])
        assert sdk.instance.record_tool_call.await_count == 2

    def test_an_assistant_message_with_no_tool_calls_reports_none(self):
        sdk = _streaming_sdk()
        _memory(sdk).write_records([asst_rec()])
        sdk.instance.record_tool_call.assert_not_awaited()


class TestToolArgsParsing:
    def test_a_json_object_becomes_a_dict(self):
        call = {"function": {"arguments": '{"pnr": "QX41RT"}'}}
        assert _tool_args(call) == {"pnr": "QX41RT"}

    def test_an_already_parsed_dict_is_left_alone(self):
        call = {"function": {"arguments": {"pnr": "QX41RT"}}}
        assert _tool_args(call) == {"pnr": "QX41RT"}

    def test_a_string_that_is_not_json_is_kept_under_one_key(self):
        call = {"function": {"arguments": "QX41RT"}}
        assert _tool_args(call) == {"arguments": "QX41RT"}

    def test_json_that_is_not_an_object_is_kept_under_one_key(self):
        call = {"function": {"arguments": "[1, 2]"}}
        assert _tool_args(call) == {"arguments": [1, 2]}

    @pytest.mark.parametrize(
        "call",
        [{}, {"function": None}, {"function": {}}, {"function": {"arguments": "  "}}],
    )
    def test_nothing_to_parse_is_no_args(self, call):
        assert _tool_args(call) is None


# ── tool results ─────────────────────────────────────────────────────────────


class TestToolResults:
    def test_a_tool_result_is_reported(self):
        sdk = _streaming_sdk()
        _memory(sdk).write_records([tool_result_rec(result="shipped")])
        sdk.instance.record_tool_result.assert_awaited_once()
        assert sdk.instance.record_tool_result.await_args.args[0] == "shipped"

    def test_the_call_and_its_result_share_an_id(self):
        """Without this the anticipation agent sees a call and a result and
        cannot tell they are the same invocation."""
        sdk = _streaming_sdk()
        mem = _memory(sdk)
        mem.write_records([tool_call_rec(call_id="call-7")])
        mem.write_records([tool_result_rec(call_id="call-7")])
        assert (
            sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
            == sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"]
            == "call-7"
        )

    def test_the_tool_name_travels_with_the_result(self):
        sdk = _streaming_sdk()
        _memory(sdk).write_records([tool_result_rec(name="lookup_order")])
        assert (
            sdk.instance.record_tool_result.await_args.kwargs["tool_name"]
            == "lookup_order"
        )

    def test_a_tool_result_is_not_a_turn(self):
        sdk = _streaming_sdk()
        _memory(sdk).write_records([tool_result_rec()])
        sdk.instance.send_message.assert_not_awaited()


class TestTheToolResultSurvivesSerialisation:
    """The SDK does ``json.dumps`` on a tool result and ``stream_events``
    swallows what that raises, so an un-encodable result is not an error: it is
    a result the anticipation agent never sees."""

    def test_a_raw_toolresult_would_not_have_survived(self):
        with pytest.raises(TypeError):
            json.dumps(ToolResult(text="a chart", images=["data:image/png;base64,x"]))

    def test_a_toolresult_is_reduced_to_its_text(self):
        assert _tool_result(ToolResult(text="a chart")) == "a chart"

    def test_everything_reported_is_json_encodable(self):
        for result in (
            ToolResult(text="a chart", images=["data:image/png;base64,x"]),
            {"status": "shipped"},
            ["a", "b"],
            "plain string",
            object(),
            None,
        ):
            json.dumps(_tool_result(result))

    def test_a_plain_result_is_left_alone(self):
        assert _tool_result({"status": "shipped"}) == {"status": "shipped"}

    def test_an_unencodable_result_reaches_the_stream_as_its_text(self):
        sdk = _streaming_sdk()
        _memory(sdk).write_records(
            [tool_result_rec(result=ToolResult(text="a chart"))]
        )
        assert sdk.instance.record_tool_result.await_args.args[0] == "a chart"


# ── reasoning ────────────────────────────────────────────────────────────────


class TestReasoning:
    def test_reasoning_content_is_reported(self):
        sdk = _streaming_sdk()
        _memory(sdk).write_records(
            [asst_rec("it ships tomorrow", reasoning="the PNR is a booking")]
        )
        sdk.instance.record_thinking.assert_awaited_once()
        assert (
            sdk.instance.record_thinking.await_args.args[0] == "the PNR is a booking"
        )

    def test_it_goes_out_before_the_answer(self):
        sdk = _streaming_sdk()
        order = []
        sdk.instance.record_thinking.side_effect = lambda *a, **k: order.append(
            "thinking"
        )
        sdk.instance.send_message.side_effect = lambda *a, **k: order.append("turn")
        _memory(sdk).write_records([asst_rec("answer", reasoning="a thought")])
        assert order == ["thinking", "turn"]

    def test_a_turn_with_no_reasoning_reports_none(self):
        sdk = _streaming_sdk()
        _memory(sdk).write_records([asst_rec("it ships tomorrow")])
        sdk.instance.record_thinking.assert_not_awaited()

    def test_blank_reasoning_reports_none(self):
        sdk = _streaming_sdk()
        _memory(sdk).write_records([asst_rec("answer", reasoning="   ")])
        sdk.instance.record_thinking.assert_not_awaited()


# ── a replayed record is not a second turn ───────────────────────────────────


class TestARecordIsReportedOnce:
    def test_replaying_the_same_record_reports_it_once(self):
        """``ChatAgent.load_memory`` writes another memory's records into this
        one. Without the guard every historical turn is reported again."""
        sdk = _streaming_sdk()
        mem = _memory(sdk)
        record = user_rec()
        mem.write_records([record])
        mem.write_records([record])
        assert sdk.instance.send_message.await_count == 1

    def test_the_same_words_in_two_turns_are_two_turns(self):
        sdk = _streaming_sdk()
        mem = _memory(sdk)
        mem.write_records([user_rec("hello")])
        mem.write_records([user_rec("hello")])
        assert sdk.instance.send_message.await_count == 2

    def test_two_memories_do_not_silence_each_other(self):
        """The guard is per instance, not per class."""
        sdk = _streaming_sdk()
        record = user_rec()
        _memory(sdk).write_records([record])
        _memory(sdk).write_records([record])
        assert sdk.instance.send_message.await_count == 2


# ── nothing here breaks the agent ────────────────────────────────────────────


class TestNoneOfItBreaksTheStep:
    """``write_records`` runs inside ``ChatAgent.step``."""

    def test_a_bridge_that_cannot_run_the_coroutine_is_still_silent(
        self, monkeypatch, caplog
    ):
        """``run_async`` can fail on its own, on a loop nest_asyncio cannot patch.

        That is not a ``SynapIntegrationError``, so it travelled straight out of
        ``write_records`` into ``ChatAgent.step`` and ended the run — the one
        thing ``on_error="fallback"`` exists to prevent.
        """
        sdk = _streaming_sdk()

        def boom(coro):
            coro.close()
            raise RuntimeError("no loop to run this on")

        monkeypatch.setattr(memory_module, "run_async", boom)
        _memory(sdk).write_records([user_rec(), asst_rec()])

    def test_a_bridge_failure_still_raises_when_strict(self, monkeypatch):
        sdk = _streaming_sdk()

        def boom(coro):
            coro.close()
            raise RuntimeError("no loop to run this on")

        monkeypatch.setattr(memory_module, "run_async", boom)
        mem = _memory(sdk, on_error="raise")
        with pytest.raises(RuntimeError, match="no loop"):
            mem.write_records([user_rec(), asst_rec()])

    def test_the_transcript_is_still_attempted_when_the_stream_bridge_fails(
        self, monkeypatch
    ):
        sdk = _streaming_sdk()
        calls = []

        def boom(coro):
            calls.append(coro)
            coro.close()
            raise RuntimeError("no loop to run this on")

        monkeypatch.setattr(memory_module, "run_async", boom)
        mem = _memory(sdk)
        mem.write_records([user_rec(), asst_rec()])
        # The stream never took the write, so the REST path must still try:
        # one coroutine for the report, one for the transcript ingest.
        assert len(calls) == 2

    def test_a_stream_that_raises_does_not_raise_out(self):
        sdk = _streaming_sdk()
        sdk.instance.send_message.side_effect = RuntimeError("stream died")
        sdk.instance.record_tool_call.side_effect = RuntimeError("stream died")
        sdk.instance.record_tool_result.side_effect = RuntimeError("stream died")
        sdk.instance.record_thinking.side_effect = RuntimeError("stream died")
        mem = _memory(sdk)
        mem.write_records([user_rec(), tool_call_rec()])
        mem.write_records([tool_result_rec(), asst_rec(reasoning="a thought")])

    def test_an_sdk_with_no_instance_namespace_does_not_raise(self, mock_sdk):
        _memory(mock_sdk).write_records([user_rec(), asst_rec()])

    def test_local_history_is_kept_even_when_the_stream_explodes(self):
        sdk = _streaming_sdk()
        sdk.instance.send_message.side_effect = RuntimeError("stream died")
        mem = _memory(sdk)
        mem.write_records([user_rec("where is my order"), asst_rec("tomorrow")])
        contents = [r.memory_record.message.content for r in mem.retrieve()]
        assert "where is my order" in contents
        assert "tomorrow" in contents


# ── the same thing, driven by a real ChatAgent ───────────────────────────────


class TestARealChatAgentReportsTheWholeTurn:
    """The records above are hand-built. These come from CAMEL's own recorders.

    Every bug this module's comments describe came from assuming a framework's
    shape instead of running it, so the shapes are taken from
    ``ChatAgent.update_memory``, ``_record_assistant_tool_calls_from_requests``
    and ``_record_tool_calling`` rather than from a fixture agreeing with the
    code it tests.
    """

    @pytest.fixture
    def agent_and_sdk(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from camel.agents import ChatAgent

        sdk = _streaming_sdk()
        mem = SynapAgentMemory(sdk, "alice", conversation_id="conv-1")
        agent = ChatAgent(system_message="You are helpful", memory=mem)

        # `_types` is where ChatAgent itself imports ToolCallRequest from.
        from camel.agents._types import ToolCallRequest

        agent.update_memory(
            BaseMessage.make_user_message("user", "where is my order"),
            OpenAIBackendRole.USER,
        )
        agent._record_assistant_tool_calls_from_requests(
            [
                ToolCallRequest(
                    tool_name="lookup_order",
                    args={"pnr": "QX41RT"},
                    tool_call_id="call-1",
                )
            ]
        )
        agent._record_tool_calling(
            "lookup_order", {"pnr": "QX41RT"}, {"status": "shipped"}, "call-1"
        )
        agent.record_message(
            BaseMessage(
                role_name="assistant",
                role_type=agent.role_type,
                meta_dict={},
                content="it ships tomorrow",
                reasoning_content="the PNR is a booking",
            )
        )
        return agent, sdk

    def test_both_turns_are_reported(self, agent_and_sdk):
        _agent, sdk = agent_and_sdk
        assert [
            (call.kwargs["event_type"], call.kwargs["content"])
            for call in sdk.instance.send_message.await_args_list
        ] == [
            ("user_message", "where is my order"),
            ("assistant_message", "it ships tomorrow"),
        ]

    def test_the_tool_call_carries_the_parsed_args(self, agent_and_sdk):
        _agent, sdk = agent_and_sdk
        call = sdk.instance.record_tool_call.await_args
        assert call.args[0] == "lookup_order"
        assert call.args[1] == {"pnr": "QX41RT"}

    def test_the_tool_result_pairs_with_its_call(self, agent_and_sdk):
        _agent, sdk = agent_and_sdk
        assert (
            sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
            == sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"]
            == "call-1"
        )
        assert sdk.instance.record_tool_result.await_args.args[0] == {
            "status": "shipped"
        }

    def test_the_reasoning_is_reported(self, agent_and_sdk):
        _agent, sdk = agent_and_sdk
        assert [
            call.args[0] for call in sdk.instance.record_thinking.await_args_list
        ] == ["the PNR is a booking"]

    def test_and_nothing_is_written_twice_over_rest(self, agent_and_sdk):
        _agent, sdk = agent_and_sdk
        sdk.memories.create.assert_not_awaited()
