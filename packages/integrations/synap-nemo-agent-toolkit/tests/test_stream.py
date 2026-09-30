"""Tests for synap_nemo_agent_toolkit.stream — the NAT telemetry exporter
that reports a whole run onto Synap's live gRPC stream.

Contract under test:

- the turn is the workflow run (``WORKFLOW_START`` / ``WORKFLOW_END``), not
  the model call, so nothing is reported twice;
- a tool call and its result carry the same ``tool_call_id``;
- a tool result survives ``json.dumps`` — the SDK serialises it and
  ``stream_events._send`` swallows the failure, so a non-serialisable one is
  dropped in silence;
- tool args are the parsed dict, not the framework's string rendering of one;
- reasoning is every LLM output except the last, so the answer is never
  reported as both a thought and the assistant turn;
- events reach Synap in the order they happened;
- stream only: nothing is ever written over REST, because the server persists
  the turn from the stream itself;
- nothing here can raise into the agent's run.
"""

from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# ⚠ ``test_short_term.py`` installs stub ``nat`` modules into ``sys.modules``
# and never takes them out again. Pytest imports test modules in filename
# order, so whether this module binds the real toolkit or those stubs would
# otherwise depend on what this file is called. Drop any ``nat`` entry with no
# ``__file__`` — a hand-built stub has none, the real package always does.
for _name in [n for n in list(sys.modules) if n == "nat" or n.startswith("nat.")]:
    if getattr(sys.modules[_name], "__file__", None) is None:
        del sys.modules[_name]

from nat.data_models.intermediate_step import (  # noqa: E402
    IntermediateStep,
    IntermediateStepPayload,
    IntermediateStepType,
    StreamEventData,
    TraceMetadata,
)
from nat.data_models.invocation_node import InvocationNode  # noqa: E402

from synap_nemo_agent_toolkit.stream import (  # noqa: E402
    SynapStreamExporter,
    SynapStreamExporterConfig,
    _assistant_text,
    _conversation_id_from,
    _json_safe,
    _tool_args,
    _unwrap_result,
    _user_text,
)

_ANCESTRY = InvocationNode(function_id="fn-1", function_name="agent")


def _step(event_type, *, uuid="step-1", name=None, data=None, metadata=None):
    """One NAT intermediate step, built the way the runtime builds them."""
    return IntermediateStep(
        parent_id="root",
        function_ancestry=_ANCESTRY,
        payload=IntermediateStepPayload(
            UUID=uuid,
            event_type=event_type,
            name=name,
            data=data,
            metadata=metadata,
        ),
    )


def _streaming_sdk(is_listening: bool = True):
    sdk = MagicMock()
    sdk.instance.is_listening = is_listening
    sdk.instance.send_message = AsyncMock()
    sdk.instance.record_thinking = AsyncMock()
    sdk.instance.record_tool_call = AsyncMock()
    sdk.instance.record_tool_result = AsyncMock()
    sdk.conversation.record_message = AsyncMock()
    return sdk


def _exporter(sdk, **kwargs):
    kwargs.setdefault("conversation_id", "conv-1")
    kwargs.setdefault("user_id", "user-1")
    kwargs.setdefault("customer_id", "cust-1")
    return SynapStreamExporter(sdk=sdk, **kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_requires_non_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapStreamExporter(sdk=None)  # type: ignore[arg-type]

    def test_config_defaults(self):
        cfg = SynapStreamExporterConfig()
        assert cfg.conversation_id == ""
        assert cfg.user_id == ""
        assert cfg.customer_id == ""
        assert cfg.drain_timeout_s == 5.0

    def test_config_yaml_name_is_synap_stream(self):
        assert SynapStreamExporterConfig().type == "synap_stream"


# ---------------------------------------------------------------------------
# Reading a NAT event
# ---------------------------------------------------------------------------


class TestJsonSafe:
    def test_a_string_passes_through_unquoted(self):
        assert _json_safe("shipped") == "shipped"

    def test_a_serialisable_value_is_left_alone(self):
        assert _json_safe({"status": "shipped"}) == {"status": "shipped"}

    def test_an_unserialisable_value_becomes_its_string_form(self):
        class Opaque:
            def __repr__(self):
                return "<Opaque>"

        assert _json_safe(Opaque()) == "<Opaque>"


class TestTheToolResultSurvivesSerialisation:
    """The SDK does json.dumps on the result and stream_events swallows the
    failure, so an unserialisable result is never reported and nothing fails."""

    def test_a_framework_envelope_is_unwrapped_to_its_content(self):
        message = MagicMock()
        message.content = {"status": "shipped"}
        assert _unwrap_result(message) == {"status": "shipped"}

    def test_the_unwrapped_result_is_json_serialisable(self):
        class ToolMessage:
            content = {"status": "shipped"}

        json.dumps(_unwrap_result(ToolMessage()))  # must not raise

    def test_a_plain_result_is_left_alone(self):
        assert _unwrap_result({"status": "shipped"}) == {"status": "shipped"}

    def test_an_unserialisable_result_still_reaches_synap_as_text(self):
        class Row:
            content = None

            def __repr__(self):
                return "Row(order=7)"

        assert _unwrap_result(Row()) == "Row(order=7)"


class TestUserText:
    def test_a_bare_string_input(self):
        assert _user_text("where is my order") == "where is my order"

    def test_a_request_model_with_an_input_message(self):
        request = MagicMock()
        request.model_dump.return_value = {"input_message": "where is my order"}
        assert _user_text(request) == "where is my order"

    def test_the_last_user_message_of_a_chat_request(self):
        request = MagicMock()
        request.model_dump.return_value = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "where is my order"},
            ]
        }
        assert _user_text(request) == "where is my order"

    def test_nothing_gives_an_empty_string(self):
        assert _user_text(None) == ""


class TestAssistantText:
    def test_a_plain_answer(self):
        assert _assistant_text("it ships tomorrow") == "it ships tomorrow"

    def test_a_streamed_answer_is_put_back_together(self):
        """`astream` reports a list of chunks, not the joined string."""
        assert _assistant_text(["it ", "ships ", "tomorrow"]) == "it ships tomorrow"

    def test_nothing_gives_an_empty_string(self):
        assert _assistant_text(None) == ""


class TestToolArgsAreTheParsedDict:
    """NAT's wrappers put the framework's STRING rendering of the arguments on
    `data.input` and the parsed dict on `metadata.tool_inputs`. Sending the
    string gives the anticipation agent a repr to reason over."""

    def test_the_parsed_inputs_win_over_the_string(self):
        step = _step(
            IntermediateStepType.TOOL_START,
            name="lookup_order",
            data=StreamEventData(input="{'pnr': 'QX41RT'}"),
            metadata=TraceMetadata(tool_inputs={"pnr": "QX41RT"}),
        )
        assert _tool_args(step) == {"pnr": "QX41RT"}

    def test_a_json_string_input_is_decoded(self):
        step = _step(
            IntermediateStepType.TOOL_START,
            data=StreamEventData(input='{"pnr": "QX41RT"}'),
        )
        assert _tool_args(step) == {"pnr": "QX41RT"}

    def test_an_opaque_string_falls_back_to_one_named_field(self):
        step = _step(
            IntermediateStepType.TOOL_START,
            data=StreamEventData(input="QX41RT"),
        )
        assert _tool_args(step) == {"input": "QX41RT"}

    def test_the_args_are_json_serialisable(self):
        class Opaque:
            def __repr__(self):
                return "<Opaque>"

        step = _step(
            IntermediateStepType.TOOL_START,
            metadata=TraceMetadata(tool_inputs={"row": Opaque()}),
        )
        json.dumps(_tool_args(step))  # must not raise

    def test_no_input_at_all_is_an_empty_dict(self):
        assert _tool_args(_step(IntermediateStepType.TOOL_START)) == {}


class TestConversationId:
    def test_it_is_read_off_the_workflow_event(self):
        step = _step(
            IntermediateStepType.WORKFLOW_START,
            metadata=TraceMetadata(provided_metadata={"conversation_id": "conv-nat"}),
        )
        assert _conversation_id_from(step) == "conv-nat"

    def test_other_events_do_not_carry_one(self):
        assert _conversation_id_from(_step(IntermediateStepType.TOOL_START)) == ""


# ---------------------------------------------------------------------------
# The mapping: one NAT event in, one Synap event out
# ---------------------------------------------------------------------------


class TestTheTurnIsTheWorkflowRun:
    @pytest.mark.asyncio
    async def test_workflow_start_is_the_user_turn(self):
        sdk = _streaming_sdk()
        await _exporter(sdk)._report(
            _step(
                IntermediateStepType.WORKFLOW_START,
                data=StreamEventData(input="where is my order"),
            )
        )
        kwargs = sdk.instance.send_message.await_args.kwargs
        assert kwargs["event_type"] == "user_message"
        assert kwargs["content"] == "where is my order"

    @pytest.mark.asyncio
    async def test_workflow_end_is_the_assistant_turn(self):
        """The event the anticipation agent actually acts on."""
        sdk = _streaming_sdk()
        await _exporter(sdk)._report(
            _step(
                IntermediateStepType.WORKFLOW_END,
                data=StreamEventData(output="it ships tomorrow"),
            )
        )
        kwargs = sdk.instance.send_message.await_args.kwargs
        assert kwargs["event_type"] == "assistant_message"
        assert kwargs["role"] == "assistant"
        assert kwargs["content"] == "it ships tomorrow"

    @pytest.mark.asyncio
    async def test_an_empty_turn_is_not_reported(self):
        sdk = _streaming_sdk()
        await _exporter(sdk)._report(
            _step(IntermediateStepType.WORKFLOW_START, data=StreamEventData(input=None))
        )
        sdk.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_llm_call_is_not_a_turn(self):
        """LLM_START fires once per lap of a tool loop. Reading the user turn
        off it reports the same question once per tool."""
        sdk = _streaming_sdk()
        await _exporter(sdk)._report(
            _step(
                IntermediateStepType.LLM_START,
                data=StreamEventData(input="where is my order"),
            )
        )
        sdk.instance.send_message.assert_not_awaited()


class TestToolCallsAndResults:
    @pytest.mark.asyncio
    async def test_a_tool_call_is_reported(self):
        sdk = _streaming_sdk()
        await _exporter(sdk)._report(
            _step(
                IntermediateStepType.TOOL_START,
                uuid="tool-run-9",
                name="lookup_order",
                metadata=TraceMetadata(tool_inputs={"pnr": "QX41RT"}),
            )
        )
        kwargs = sdk.instance.record_tool_call.await_args.kwargs
        assert sdk.instance.record_tool_call.await_args.args[0] == "lookup_order"
        assert kwargs["tool_call_id"] == "tool-run-9"

    @pytest.mark.asyncio
    async def test_a_tool_result_is_reported(self):
        sdk = _streaming_sdk()
        await _exporter(sdk)._report(
            _step(
                IntermediateStepType.TOOL_END,
                uuid="tool-run-9",
                name="lookup_order",
                data=StreamEventData(output={"status": "shipped"}),
            )
        )
        sdk.instance.record_tool_result.assert_awaited_once()
        assert (
            sdk.instance.record_tool_result.await_args.args[0] == {"status": "shipped"}
        )

    @pytest.mark.asyncio
    async def test_the_call_and_its_result_share_an_id(self):
        """NAT gives the START and the END of one step the same UUID. Without
        using it the anticipation agent cannot tell they are one invocation."""
        sdk = _streaming_sdk()
        exporter = _exporter(sdk)
        await exporter._report(
            _step(IntermediateStepType.TOOL_START, uuid="tool-run-9", name="t")
        )
        await exporter._report(
            _step(
                IntermediateStepType.TOOL_END,
                uuid="tool-run-9",
                name="t",
                data=StreamEventData(output="ok"),
            )
        )
        assert (
            sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
            == sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"]
        )


class TestReasoningIsEveryLlmOutputExceptTheLast:
    @pytest.mark.asyncio
    async def test_one_llm_output_on_its_own_reports_nothing_yet(self):
        sdk = _streaming_sdk()
        await _exporter(sdk)._report(
            _step(
                IntermediateStepType.LLM_END,
                data=StreamEventData(output="I should look up the booking"),
            )
        )
        sdk.instance.record_thinking.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_it_is_released_when_the_run_calls_a_tool(self):
        sdk = _streaming_sdk()
        exporter = _exporter(sdk)
        await exporter._report(
            _step(
                IntermediateStepType.LLM_END,
                data=StreamEventData(output="I should look up the booking"),
            )
        )
        await exporter._report(
            _step(IntermediateStepType.TOOL_START, uuid="t1", name="lookup_order")
        )
        sdk.instance.record_thinking.assert_awaited_once()
        assert (
            sdk.instance.record_thinking.await_args.args[0]
            == "I should look up the booking"
        )

    @pytest.mark.asyncio
    async def test_the_final_output_is_never_reported_as_a_thought(self):
        """It is about to go out as the assistant turn; saying it twice is
        exactly the duplication this exporter exists to avoid."""
        sdk = _streaming_sdk()
        exporter = _exporter(sdk)
        await exporter._report(
            _step(
                IntermediateStepType.LLM_END,
                data=StreamEventData(output="it ships tomorrow"),
            )
        )
        await exporter._report(
            _step(
                IntermediateStepType.WORKFLOW_END,
                data=StreamEventData(output="it ships tomorrow"),
            )
        )
        sdk.instance.record_thinking.assert_not_awaited()
        assert (
            sdk.instance.send_message.await_args.kwargs["event_type"]
            == "assistant_message"
        )

    @pytest.mark.asyncio
    async def test_a_second_llm_call_releases_the_first(self):
        sdk = _streaming_sdk()
        exporter = _exporter(sdk)
        await exporter._report(
            _step(IntermediateStepType.LLM_END, data=StreamEventData(output="lap one"))
        )
        await exporter._report(
            _step(IntermediateStepType.LLM_END, data=StreamEventData(output="lap two"))
        )
        sdk.instance.record_thinking.assert_awaited_once()
        assert sdk.instance.record_thinking.await_args.args[0] == "lap one"

    @pytest.mark.asyncio
    async def test_an_empty_llm_output_is_not_held(self):
        sdk = _streaming_sdk()
        exporter = _exporter(sdk)
        await exporter._report(
            _step(IntermediateStepType.LLM_END, data=StreamEventData(output="   "))
        )
        await exporter._report(_step(IntermediateStepType.TOOL_START, name="t"))
        sdk.instance.record_thinking.assert_not_awaited()


# ---------------------------------------------------------------------------
# Stream only, never REST
# ---------------------------------------------------------------------------


class TestStreamOnly:
    @pytest.mark.asyncio
    async def test_a_turn_goes_out_on_the_stream_when_one_is_open(self):
        sdk = _streaming_sdk(True)
        await _exporter(sdk)._report(
            _step(
                IntermediateStepType.WORKFLOW_START,
                data=StreamEventData(input="where is my order"),
            )
        )
        sdk.instance.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_and_is_NOT_also_written_over_rest(self):
        """The server persists user_message/assistant_message from the stream
        itself, so a record_message on top writes and extracts it twice."""
        sdk = _streaming_sdk(True)
        await _exporter(sdk)._report(
            _step(
                IntermediateStepType.WORKFLOW_START,
                data=StreamEventData(input="where is my order"),
            )
        )
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_with_no_stream_nothing_is_sent_anywhere(self):
        sdk = _streaming_sdk(False)
        await _exporter(sdk)._report(
            _step(
                IntermediateStepType.WORKFLOW_END,
                data=StreamEventData(output="it ships tomorrow"),
            )
        )
        sdk.instance.send_message.assert_not_awaited()
        sdk.conversation.record_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


class TestScope:
    @pytest.mark.asyncio
    async def test_nats_own_conversation_id_wins(self):
        sdk = _streaming_sdk()
        await _exporter(sdk)._report(
            _step(
                IntermediateStepType.WORKFLOW_START,
                data=StreamEventData(input="hello"),
                metadata=TraceMetadata(
                    provided_metadata={"conversation_id": "conv-from-nat"}
                ),
            )
        )
        assert (
            sdk.instance.send_message.await_args.kwargs["conversation_id"]
            == "conv-from-nat"
        )

    @pytest.mark.asyncio
    async def test_the_configured_id_is_the_fallback(self):
        sdk = _streaming_sdk()
        await _exporter(sdk)._report(
            _step(
                IntermediateStepType.WORKFLOW_START,
                data=StreamEventData(input="hello"),
            )
        )
        kwargs = sdk.instance.send_message.await_args.kwargs
        assert kwargs["conversation_id"] == "conv-1"
        assert kwargs["user_id"] == "user-1"
        assert kwargs["customer_id"] == "cust-1"

    @pytest.mark.asyncio
    async def test_an_unset_customer_id_is_dropped_not_sent_empty(self):
        """A B2C instance refuses a customer_id, so an empty one must not go."""
        sdk = _streaming_sdk()
        await _exporter(sdk, customer_id="")._report(
            _step(
                IntermediateStepType.WORKFLOW_START,
                data=StreamEventData(input="hello"),
            )
        )
        assert "customer_id" not in sdk.instance.send_message.await_args.kwargs


# ---------------------------------------------------------------------------
# Queue, order, and shutdown
# ---------------------------------------------------------------------------


async def _run_through_the_queue(exporter, steps):
    """Push steps the way NAT does — synchronously, through export() — and
    leave the context immediately.

    Deliberately no ``await`` after the pushes: the only thing that gets these
    events out is the drain in ``stop()``, which is exactly the path a real
    workflow takes when it emits ``WORKFLOW_END`` and tears down.
    """
    async with exporter.start():
        for step in steps:
            exporter.export(step)


class TestTheQueue:
    @pytest.mark.asyncio
    async def test_events_reach_synap_in_the_order_they_happened(self):
        """A task per event lets them interleave at the first await, and the
        user turn arriving after the tool call it caused is nonsense."""
        sdk = _streaming_sdk()
        exporter = _exporter(sdk)
        await _run_through_the_queue(
            exporter,
            [
                _step(
                    IntermediateStepType.WORKFLOW_START,
                    data=StreamEventData(input="where is my order"),
                ),
                _step(IntermediateStepType.TOOL_START, uuid="t1", name="lookup"),
                _step(
                    IntermediateStepType.TOOL_END,
                    uuid="t1",
                    name="lookup",
                    data=StreamEventData(output="shipped"),
                ),
                _step(
                    IntermediateStepType.WORKFLOW_END,
                    data=StreamEventData(output="it ships tomorrow"),
                ),
            ],
        )
        order = [c.kwargs["event_type"] for c in sdk.instance.send_message.await_args_list]
        assert order == ["user_message", "assistant_message"]
        sdk.instance.record_tool_call.assert_awaited_once()
        sdk.instance.record_tool_result.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_slow_report_is_not_overtaken_by_the_next_one(self):
        """One worker, one queue. A task per event would let the tool call
        land before the user turn that caused it the moment the turn's send
        hits an await, which on a real network it always does."""
        seen: list[str] = []

        async def _slow_turn(*args, **kwargs):
            await asyncio.sleep(0.02)
            seen.append("turn")

        async def _quick_tool(*args, **kwargs):
            seen.append("tool")

        sdk = _streaming_sdk()
        sdk.instance.send_message = AsyncMock(side_effect=_slow_turn)
        sdk.instance.record_tool_call = AsyncMock(side_effect=_quick_tool)
        await _run_through_the_queue(
            _exporter(sdk),
            [
                _step(
                    IntermediateStepType.WORKFLOW_START,
                    data=StreamEventData(input="where is my order"),
                ),
                _step(IntermediateStepType.TOOL_START, uuid="t1", name="lookup"),
            ],
        )
        assert seen == ["turn", "tool"]

    @pytest.mark.asyncio
    async def test_the_assistant_turn_queued_last_still_goes_out(self):
        """It is queued immediately before teardown, so without a drain on
        stop() the one event anticipation needs is the one that is lost.

        The send takes a moment on purpose: a report that only finishes
        because the test happened to yield once proves nothing about a real
        network round trip.
        """
        delivered: list[str] = []

        async def _slow_send(*args, **kwargs):
            await asyncio.sleep(0.02)
            delivered.append(kwargs["event_type"])

        sdk = _streaming_sdk()
        sdk.instance.send_message = AsyncMock(side_effect=_slow_send)
        await _run_through_the_queue(
            _exporter(sdk),
            [
                _step(
                    IntermediateStepType.WORKFLOW_END,
                    data=StreamEventData(output="it ships tomorrow"),
                )
            ],
        )
        assert delivered == ["assistant_message"]

    @pytest.mark.asyncio
    async def test_an_event_before_start_is_dropped_and_does_not_raise(self):
        exporter = _exporter(_streaming_sdk())
        exporter.export(_step(IntermediateStepType.WORKFLOW_START))
        assert exporter._events.qsize() == 0

    @pytest.mark.asyncio
    async def test_an_event_after_shutdown_is_dropped(self):
        """The worker is gone by then, so anything queued would sit in memory
        for the life of the process and never be reported."""
        sdk = _streaming_sdk()
        exporter = _exporter(sdk)
        async with exporter.start():
            pass
        exporter.export(
            _step(
                IntermediateStepType.WORKFLOW_END,
                data=StreamEventData(output="too late"),
            )
        )
        assert exporter._events.qsize() == 0
        sdk.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_something_that_is_not_a_nat_event_is_ignored(self):
        sdk = _streaming_sdk()
        exporter = _exporter(sdk)
        async with exporter.start():
            exporter.export("not an IntermediateStep")  # type: ignore[arg-type]
        sdk.instance.send_message.assert_not_awaited()


class TestNothingBreaksTheRun:
    @pytest.mark.asyncio
    async def test_an_sdk_that_raises_does_not_escape_the_drain(self):
        sdk = _streaming_sdk()
        sdk.instance.send_message = AsyncMock(side_effect=RuntimeError("boom"))
        exporter = _exporter(sdk)
        await _run_through_the_queue(
            exporter,
            [
                _step(
                    IntermediateStepType.WORKFLOW_END,
                    data=StreamEventData(output="it ships tomorrow"),
                )
            ],
        )

    @pytest.mark.asyncio
    async def test_an_sdk_with_no_instance_namespace_does_not_raise(self):
        """An older SDK, or a test double. Every report must no-op."""
        sdk = MagicMock(spec=[])
        exporter = _exporter(sdk)
        await exporter._report(
            _step(
                IntermediateStepType.WORKFLOW_START,
                data=StreamEventData(input="hello"),
            )
        )
        await exporter._report(
            _step(IntermediateStepType.TOOL_START, uuid="t", name="t")
        )
        await exporter._report(
            _step(
                IntermediateStepType.TOOL_END,
                uuid="t",
                name="t",
                data=StreamEventData(output="x"),
            )
        )

    @pytest.mark.asyncio
    async def test_a_malformed_event_does_not_stop_the_ones_after_it(self):
        sdk = _streaming_sdk()
        exporter = _exporter(sdk)
        async with exporter.start():
            exporter._events.put_nowait(object())  # not an IntermediateStep
            exporter.export(
                _step(
                    IntermediateStepType.WORKFLOW_END,
                    data=StreamEventData(output="it ships tomorrow"),
                )
            )
        sdk.instance.send_message.assert_awaited_once()


class TestIsolationBetweenConcurrentRuns:
    """ExporterManager hands each run a `copy.copy` of the exporter, which is
    shallow. State that is not an IsolatedAttribute would be one queue and one
    reasoning buffer shared by every conversation in flight."""

    def test_each_run_gets_its_own_queue_and_reasoning_buffer(self):
        from nat.builder.context import ContextState

        original = _exporter(_streaming_sdk())
        original._pending_reasoning.append("lap one")
        original._events.put_nowait(_step(IntermediateStepType.TOOL_START))

        isolated = original.create_isolated_instance(ContextState.get())

        assert isolated._pending_reasoning == []
        assert isolated._events.qsize() == 0
        assert isolated.sdk is original.sdk
