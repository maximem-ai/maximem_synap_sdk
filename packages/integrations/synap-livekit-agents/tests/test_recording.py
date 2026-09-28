"""Tests for synap_livekit_agents.recording — attach_synap_recording.

Documented error-handling contract (from recording.py docstring):
- The callback is synchronous (LiveKit EventEmitter contract) but dispatches
  async writes via asyncio.create_task (if a loop is running) or asyncio.run
  (if called from a sync context).
- Callbacks NEVER raise — SDK write failures are caught and logged at ERROR.
- Roles other than "user" / "assistant" are silently ignored.
- Events without an item, items without text, or items with falsy text are
  silently ignored.
- conversation_id is returned so callers can stitch downstream reads.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from synap_livekit_agents.recording import attach_synap_recording


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class MockSession:
    """Minimal LiveKit-style EventEmitter double."""

    def __init__(self):
        self._handlers: dict = {}

    def on(self, event: str, cb) -> None:
        self._handlers[event] = cb

    def fire(self, event: str, payload) -> None:
        self._handlers[event](payload)


_UNSET = object()


def _make_item(role: str = "user", text_content=_UNSET):
    item = MagicMock()
    item.role = role
    item.text_content = "Hello world" if text_content is _UNSET else text_content
    return item


def _make_event(item=None):
    event = MagicMock()
    event.item = item
    return event


def _make_sdk():
    sdk = MagicMock()
    sdk.conversation = MagicMock()
    sdk.conversation.record_message = AsyncMock(return_value={"message_id": "msg-1"})
    return sdk


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_requires_non_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            attach_synap_recording(MockSession(), None, user_id="u1")  # type: ignore[arg-type]

    def test_requires_non_empty_user_id(self):
        with pytest.raises(ValueError, match="non-empty user_id"):
            attach_synap_recording(MockSession(), _make_sdk(), user_id="")

    def test_requires_session_with_on_method(self):
        with pytest.raises(ValueError, match=r"\.on\(event"):
            attach_synap_recording(None, _make_sdk(), user_id="u1")  # type: ignore[arg-type]

    def test_requires_session_object_with_on_attribute(self):
        class BadSession:
            pass

        with pytest.raises(ValueError, match=r"\.on\(event"):
            attach_synap_recording(BadSession(), _make_sdk(), user_id="u1")


# ---------------------------------------------------------------------------
# Return value: conversation_id
# ---------------------------------------------------------------------------


class TestConversationId:
    def test_returns_explicit_conversation_id(self):
        sdk = _make_sdk()
        conv_id = attach_synap_recording(
            MockSession(), sdk, user_id="u1", conversation_id="conv-explicit"
        )
        assert conv_id == "conv-explicit"

    def test_auto_generates_conv_id_when_absent(self):
        sdk = _make_sdk()
        conv_id = attach_synap_recording(MockSession(), sdk, user_id="u1")
        assert conv_id.startswith("livekit-")
        # livekit- + 12 hex chars
        suffix = conv_id[len("livekit-"):]
        assert len(suffix) == 12
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_two_calls_get_different_auto_ids(self):
        sdk = _make_sdk()
        id1 = attach_synap_recording(MockSession(), sdk, user_id="u1")
        id2 = attach_synap_recording(MockSession(), sdk, user_id="u1")
        assert id1 != id2


# ---------------------------------------------------------------------------
# Event wiring — listener is registered on the session
# ---------------------------------------------------------------------------


class TestEventWiring:
    def test_registers_conversation_item_added_listener(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1")
        assert "conversation_item_added" in session._handlers


# ---------------------------------------------------------------------------
# Guard clauses — events / items that must be silently ignored
# ---------------------------------------------------------------------------


class TestGuardClauses:
    def test_ignores_event_with_no_item(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1")
        session.fire("conversation_item_added", _make_event(item=None))
        assert sdk.conversation.record_message.await_count == 0

    def test_ignores_system_role(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1")
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="system", text_content="sys")),
        )
        assert sdk.conversation.record_message.await_count == 0

    def test_ignores_developer_role(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1")
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="developer", text_content="dev")),
        )
        assert sdk.conversation.record_message.await_count == 0

    def test_ignores_item_with_no_role_attr(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1")
        item = MagicMock(spec=[])  # no role attr
        session.fire("conversation_item_added", _make_event(item))
        assert sdk.conversation.record_message.await_count == 0

    def test_ignores_empty_text_content(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1")
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="user", text_content="")),
        )
        assert sdk.conversation.record_message.await_count == 0

    def test_ignores_none_text_content(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1")
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="user", text_content=None)),
        )
        assert sdk.conversation.record_message.await_count == 0


# ---------------------------------------------------------------------------
# Happy paths — sync context (no running loop → asyncio.run)
# ---------------------------------------------------------------------------


class TestHappyPathSync:
    def test_user_turn_triggers_record_message(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="user", text_content="Hello")),
        )
        assert sdk.conversation.record_message.await_count == 1

    def test_assistant_turn_triggers_record_message(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="assistant", text_content="Hi there")),
        )
        assert sdk.conversation.record_message.await_count == 1

    def test_record_message_called_with_correct_role_user(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="user", text_content="Hello")),
        )
        kw = sdk.conversation.record_message.call_args.kwargs
        assert kw["role"] == "user"

    def test_record_message_called_with_correct_role_assistant(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="assistant", text_content="AI reply")),
        )
        kw = sdk.conversation.record_message.call_args.kwargs
        assert kw["role"] == "assistant"

    def test_record_message_called_with_content(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="user", text_content="My content")),
        )
        kw = sdk.conversation.record_message.call_args.kwargs
        assert kw["content"] == "My content"

    def test_record_message_called_with_user_id(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="user-xyz", conversation_id="c1")
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="user", text_content="hi")),
        )
        kw = sdk.conversation.record_message.call_args.kwargs
        assert kw["user_id"] == "user-xyz"

    def test_record_message_called_with_customer_id(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(
            session, sdk, user_id="u1", customer_id="cust-42", conversation_id="c1"
        )
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="user", text_content="hi")),
        )
        kw = sdk.conversation.record_message.call_args.kwargs
        assert kw["customer_id"] == "cust-42"

    def test_record_message_called_with_conversation_id(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(
            session, sdk, user_id="u1", conversation_id="conv-explicit"
        )
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="user", text_content="hi")),
        )
        kw = sdk.conversation.record_message.call_args.kwargs
        assert kw["conversation_id"] == "conv-explicit"

    def test_callable_text_content_is_invoked(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        item = _make_item(role="user", text_content=lambda: "Text from callable")
        session.fire("conversation_item_added", _make_event(item))
        kw = sdk.conversation.record_message.call_args.kwargs
        assert kw["content"] == "Text from callable"

    def test_two_events_result_in_two_sdk_calls(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="user", text_content="msg1")),
        )
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="assistant", text_content="msg2")),
        )
        assert sdk.conversation.record_message.await_count == 2


# ---------------------------------------------------------------------------
# Happy paths — async context (running loop → create_task)
# ---------------------------------------------------------------------------


class TestHappyPathAsync:
    @pytest.mark.asyncio
    async def test_user_turn_recorded_in_async_context(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="user", text_content="async hello")),
        )
        await asyncio.sleep(0)  # yield to let the task complete
        assert sdk.conversation.record_message.await_count == 1

    @pytest.mark.asyncio
    async def test_assistant_turn_recorded_in_async_context(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="assistant", text_content="async reply")),
        )
        await asyncio.sleep(0)
        assert sdk.conversation.record_message.await_count == 1

    @pytest.mark.asyncio
    async def test_correct_kwargs_in_async_context(self):
        sdk = _make_sdk()
        session = MockSession()
        attach_synap_recording(
            session, sdk, user_id="u42", customer_id="c99", conversation_id="conv-99"
        )
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="user", text_content="async content")),
        )
        await asyncio.sleep(0)
        kw = sdk.conversation.record_message.call_args.kwargs
        assert kw["user_id"] == "u42"
        assert kw["customer_id"] == "c99"
        assert kw["conversation_id"] == "conv-99"
        assert kw["content"] == "async content"
        assert kw["role"] == "user"


# ---------------------------------------------------------------------------
# Failure / degradation paths — callback must NEVER raise
# ---------------------------------------------------------------------------


class TestFailureDegradation:
    def test_sdk_failure_does_not_propagate_from_sync_callback(self):
        """A Synap write outage must NOT tear down the synchronous event fire."""
        sdk = MagicMock()
        sdk.conversation = MagicMock()
        sdk.conversation.record_message = AsyncMock(
            side_effect=RuntimeError("sdk boom")
        )
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")

        # Must not raise
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="user", text_content="hello")),
        )

    def test_sdk_failure_logs_error_in_sync_context(self, caplog):
        sdk = MagicMock()
        sdk.conversation = MagicMock()
        sdk.conversation.record_message = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        with caplog.at_level(logging.ERROR):
            session.fire(
                "conversation_item_added",
                _make_event(_make_item(role="user", text_content="hello")),
            )
        assert any(
            "record_message" in r.message or "record_turn" in r.message
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_sdk_failure_does_not_propagate_from_async_context(self):
        """A Synap write outage must NOT propagate from the async task path."""
        sdk = MagicMock()
        sdk.conversation = MagicMock()
        sdk.conversation.record_message = AsyncMock(
            side_effect=RuntimeError("sdk boom")
        )
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        # Fire from inside async context — uses create_task path
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="user", text_content="hello")),
        )
        # Awaiting sleep gives the task a chance to run and fail
        await asyncio.sleep(0)
        # No unhandled exception propagated — test passes if we reach here

    @pytest.mark.asyncio
    async def test_failing_sdk_fixture_swallowed(self, failing_sdk):
        """Shared failing_sdk: all calls raise — recording must swallow and log."""
        session = MockSession()
        attach_synap_recording(session, failing_sdk, user_id="u1", conversation_id="c1")
        # Must not raise
        session.fire(
            "conversation_item_added",
            _make_event(_make_item(role="user", text_content="hi")),
        )
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_surface_exports():
    import synap_livekit_agents
    assert hasattr(synap_livekit_agents, "attach_synap_recording")
    assert "attach_synap_recording" in synap_livekit_agents.__all__


# ---------------------------------------------------------------------------
# The live stream
#
# This integration recorded the user turn and the assistant turn over REST and
# stopped there, so anticipation saw a conversation with questions and answers
# and nothing in between. On a VOICE call that gap is the worst place to be
# blind: the seconds between a caller's question and the answer are dead air
# they sit through, and the tool call the agent just dispatched is the clearest
# statement of what it is about to need.
#
# The rule these pin is "stream first, REST only as a fallback, never both".
# The server persists `user_message` and `assistant_message` from the stream
# itself, so a `record_message` on top of a delivered stream event writes the
# same turn twice and extracts it twice.
#
# These use the REAL LiveKit event and content models, not doubles. Every bug
# worth catching here is a wrong assumption about their shape.
# ---------------------------------------------------------------------------

import json  # noqa: E402

from livekit.agents.llm import (  # noqa: E402
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
)
from livekit.agents.voice.events import (  # noqa: E402
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
    ToolCallEnded,
    ToolCallStarted,
    ToolExecutionUpdatedEvent,
)


def _streaming_sdk(is_listening: bool = True):
    sdk = MagicMock()
    sdk.instance.is_listening = is_listening
    sdk.instance.send_message = AsyncMock()
    sdk.instance.record_tool_call = AsyncMock()
    sdk.instance.record_tool_result = AsyncMock()
    sdk.instance.record_thinking = AsyncMock()
    sdk.instance.end_session = AsyncMock()
    sdk.conversation.record_message = AsyncMock()
    return sdk


def _turn(role: str, text: str) -> ConversationItemAddedEvent:
    return ConversationItemAddedEvent(item=ChatMessage(role=role, content=[text]))


def _started(call_id: str, name: str = "lookup_order", arguments: str = "{}"):
    return ToolExecutionUpdatedEvent(
        update=ToolCallStarted(
            function_call=FunctionCall(
                call_id=call_id, name=name, arguments=arguments
            )
        )
    )


def _executed(call_id: str, name: str = "lookup_order",
              arguments: str = "{}", output: str = "shipped"):
    return FunctionToolsExecutedEvent(
        function_calls=[
            FunctionCall(call_id=call_id, name=name, arguments=arguments)
        ],
        function_call_outputs=[
            FunctionCallOutput(
                call_id=call_id, name=name, output=output, is_error=False
            )
        ],
    )


async def _settle():
    """Let the dispatched report tasks run to completion."""
    for _ in range(6):
        await asyncio.sleep(0)


class TestStreamFirstRestFallback:
    @pytest.mark.asyncio
    async def test_a_turn_goes_out_on_the_stream_when_one_is_open(self):
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("conversation_item_added", _turn("user", "where is my order"))
        await _settle()
        sdk.instance.send_message.assert_awaited_once()
        assert sdk.instance.send_message.await_args.kwargs[
            "event_type"] == "user_message"

    @pytest.mark.asyncio
    async def test_and_is_NOT_also_written_over_rest(self):
        """The double-write this rule exists to prevent."""
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("conversation_item_added", _turn("user", "where is my order"))
        await _settle()
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_with_no_stream_it_falls_back_to_rest(self):
        sdk = _streaming_sdk(False)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("conversation_item_added", _turn("assistant", "it ships tomorrow"))
        await _settle()
        sdk.conversation.record_message.assert_awaited_once()
        sdk.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_assistant_turn_is_the_anticipation_moment(self):
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("conversation_item_added", _turn("assistant", "it ships tomorrow"))
        await _settle()
        kw = sdk.instance.send_message.await_args.kwargs
        assert kw["event_type"] == "assistant_message"
        assert kw["role"] == "assistant"


class TestTheDeadAirBetweenQuestionAndAnswer:
    """A tool call on a phone call is silence the caller sits through. It is
    also the event that says what the agent is about to need, so it has to go
    out when the tool is DISPATCHED, not when the whole batch has finished."""

    @pytest.mark.asyncio
    async def test_a_tool_call_is_reported_at_dispatch(self):
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("tool_execution_updated", _started("call_1"))
        await _settle()
        sdk.instance.record_tool_call.assert_awaited_once()
        assert sdk.instance.record_tool_call.await_args.args[0] == "lookup_order"

    @pytest.mark.asyncio
    async def test_a_tool_result_is_reported(self):
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("function_tools_executed", _executed("call_1"))
        await _settle()
        sdk.instance.record_tool_result.assert_awaited_once()
        assert sdk.instance.record_tool_result.await_args.args[0] == "shipped"

    @pytest.mark.asyncio
    async def test_the_call_and_its_result_share_a_call_id(self):
        """Without this the anticipation agent sees a call and a result and
        cannot tell they are the same tool invocation."""
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("tool_execution_updated", _started("call_7"))
        session.fire("function_tools_executed", _executed("call_7"))
        await _settle()
        assert (sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
                == sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"]
                == "call_7")

    @pytest.mark.asyncio
    async def test_the_call_is_reported_once_though_both_events_carry_it(self):
        """`function_tools_executed` repeats every call it ran. Reporting it
        again would tell anticipation the agent called the tool twice."""
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("tool_execution_updated", _started("call_1"))
        session.fire("function_tools_executed", _executed("call_1"))
        await _settle()
        assert sdk.instance.record_tool_call.await_count == 1

    @pytest.mark.asyncio
    async def test_a_call_only_seen_in_the_batch_event_is_still_reported(self):
        """`tool_execution_updated` is newer than `livekit-agents>=1.0`. On a
        version that never emits it the batch event must carry both halves."""
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("function_tools_executed", _executed("call_1"))
        await _settle()
        assert sdk.instance.record_tool_call.await_count == 1
        assert sdk.instance.record_tool_result.await_count == 1

    @pytest.mark.asyncio
    async def test_a_result_is_not_reported_twice(self):
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("function_tools_executed", _executed("call_1"))
        session.fire("function_tools_executed", _executed("call_1"))
        await _settle()
        assert sdk.instance.record_tool_result.await_count == 1

    @pytest.mark.asyncio
    async def test_a_terminal_update_is_not_mistaken_for_a_new_call(self):
        """`ToolCallEnded.message` is the text voiced to the caller, not what
        the tool returned, and it arrives on the same event as the dispatch."""
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire(
            "tool_execution_updated",
            ToolExecutionUpdatedEvent(
                update=ToolCallEnded(
                    id="call_1", call_id="call_1", message="done", status="done"
                )
            ),
        )
        await _settle()
        sdk.instance.record_tool_call.assert_not_awaited()
        sdk.instance.record_tool_result.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_call_is_reported_before_its_result(self):
        """Two events, two tasks. Unordered, a slow call report and a fast
        result report arrive back to front and the exchange reads inverted."""
        order = []

        async def slow_call(*a, **k):
            await asyncio.sleep(0.02)
            order.append("call")

        async def fast_result(*a, **k):
            order.append("result")

        sdk = _streaming_sdk(True)
        sdk.instance.record_tool_call = AsyncMock(side_effect=slow_call)
        sdk.instance.record_tool_result = AsyncMock(side_effect=fast_result)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("tool_execution_updated", _started("call_1"))
        session.fire("function_tools_executed", _executed("call_1"))
        await asyncio.sleep(0.1)
        assert order == ["call", "result"]


class TestToolArgsAreTheParsedDict:
    """`FunctionCall.arguments` is a STRING: the raw JSON the model emitted.
    `report_tool_call` only forwards a dict, so passing the string through
    would have sent no arguments at all."""

    @pytest.mark.asyncio
    async def test_the_json_livekit_carries_as_text_is_parsed(self):
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire(
            "tool_execution_updated",
            _started("call_1", arguments='{"pnr": "QX41RT"}'),
        )
        await _settle()
        assert sdk.instance.record_tool_call.await_args.args[1] == {"pnr": "QX41RT"}

    @pytest.mark.asyncio
    async def test_the_parsed_args_are_json_serialisable(self):
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire(
            "tool_execution_updated",
            _started("call_1", arguments='{"pnr": "QX41RT"}'),
        )
        await _settle()
        json.dumps(sdk.instance.record_tool_call.await_args.args[1])

    @pytest.mark.asyncio
    async def test_arguments_that_are_not_json_fall_back_to_the_raw_text(self):
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("tool_execution_updated", _started("call_1", arguments="not json"))
        await _settle()
        assert sdk.instance.record_tool_call.await_args.args[1] == {"input": "not json"}

    @pytest.mark.asyncio
    async def test_a_tool_taking_no_arguments_reports_an_empty_dict(self):
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("tool_execution_updated", _started("call_1", arguments=""))
        await _settle()
        assert sdk.instance.record_tool_call.await_args.args[1] == {}


class TestTheToolResultSurvivesSerialisation:
    """The SDK does `json.dumps` on the result and `stream_events._send`
    swallows every exception by design, so a result it cannot serialise is
    not an error anybody sees — it is a result that silently never arrives."""

    @pytest.mark.asyncio
    async def test_the_reported_result_is_json_serialisable(self):
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("function_tools_executed", _executed("call_1", output="shipped"))
        await _settle()
        json.dumps(sdk.instance.record_tool_result.await_args.args[0])

    @pytest.mark.asyncio
    async def test_a_result_json_cannot_touch_is_stringified_not_dropped(self):
        class NotSerialisable:
            def __str__(self):
                return "opaque-result"

        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        event = _executed("call_1")
        # LiveKit types `output` as str; a provider adapter can still put an
        # object there, and the annotation is not a runtime guarantee.
        object.__setattr__(
            event.function_call_outputs[0], "output", NotSerialisable()
        )
        session.fire("function_tools_executed", event)
        await _settle()
        sent = sdk.instance.record_tool_result.await_args.args[0]
        json.dumps(sent)
        assert sent == "opaque-result"


class TestATurnIsReportedOncePerTurn:
    @pytest.mark.asyncio
    async def test_the_same_item_is_not_reported_twice(self):
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        event = _turn("user", "yes")
        session.fire("conversation_item_added", event)
        session.fire("conversation_item_added", event)
        await _settle()
        assert sdk.instance.send_message.await_count == 1

    @pytest.mark.asyncio
    async def test_the_same_words_in_two_items_are_two_turns(self):
        """A caller saying "yes" twice is two turns. Keying on the words
        alone would swallow the second."""
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("conversation_item_added", _turn("user", "yes"))
        session.fire("conversation_item_added", _turn("user", "yes"))
        await _settle()
        assert sdk.instance.send_message.await_count == 2

    @pytest.mark.asyncio
    async def test_two_attachments_do_not_silence_each_other(self):
        sdk = _streaming_sdk(True)
        event = _turn("user", "hello")
        for _ in range(2):
            session = MockSession()
            attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
            session.fire("conversation_item_added", event)
        await _settle()
        assert sdk.instance.send_message.await_count == 2

    @pytest.mark.asyncio
    async def test_an_assistant_item_carrying_only_a_tool_call_is_not_a_turn(self):
        sdk = _streaming_sdk(True)
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire(
            "conversation_item_added",
            ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=[])),
        )
        await _settle()
        sdk.instance.send_message.assert_not_awaited()


class TestTheEndOfTheCall:
    @pytest.mark.asyncio
    async def test_close_ends_the_synap_session(self):
        sdk = _streaming_sdk(True)
        session = MockSession()
        conv = attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("close", CloseEvent(reason=CloseReason.USER_INITIATED))
        await _settle()
        sdk.instance.end_session.assert_awaited_once_with(conv)


class TestNoneOfItTearsDownTheCall:
    """LiveKit invokes listeners synchronously and a raising one is logged,
    not survived gracefully by the caller. None of these may raise."""

    def test_all_four_listeners_are_registered(self):
        session = MockSession()
        attach_synap_recording(session, _make_sdk(), user_id="u1")
        for event in (
            "conversation_item_added",
            "tool_execution_updated",
            "function_tools_executed",
            "close",
        ):
            assert event in session._handlers

    def test_a_real_livekit_emitter_accepts_every_listener(self):
        """`EventEmitter.on` REFUSES an async callback outright (ValueError).
        Every listener here has to be synchronous."""
        from livekit.rtc import EventEmitter

        attach_synap_recording(EventEmitter(), _make_sdk(), user_id="u1")

    @pytest.mark.asyncio
    async def test_a_dead_stream_does_not_raise_from_the_tool_hook(self, caplog):
        """Every report goes through `stream_events`, which swallows by
        design. Reaching for `sdk.instance` directly would put a raise back
        in the middle of a live call."""
        sdk = _streaming_sdk(True)
        sdk.instance.record_tool_call = AsyncMock(side_effect=RuntimeError("boom"))
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        with caplog.at_level(logging.ERROR):
            session.fire("tool_execution_updated", _started("call_1"))
            await _settle()
        assert caplog.records == []

    @pytest.mark.asyncio
    async def test_one_failed_report_does_not_swallow_the_next(self):
        """Reports are chained so they land in order. A chain that propagates
        the previous link's failure would let one bad write silence every
        turn after it for the rest of the call."""
        sdk = _streaming_sdk(False)
        sdk.conversation.record_message = AsyncMock(
            side_effect=[RuntimeError("boom"), None]
        )
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("conversation_item_added", _turn("user", "one"))
        session.fire("conversation_item_added", _turn("assistant", "two"))
        await _settle()
        assert sdk.conversation.record_message.await_count == 2

    @pytest.mark.asyncio
    async def test_an_sdk_with_no_instance_namespace_does_not_raise(self):
        """An older SDK, or a test double. Every stream call must no-op and
        the turn must still reach REST."""
        sdk = MagicMock(spec=["conversation"])
        sdk.conversation.record_message = AsyncMock()
        session = MockSession()
        attach_synap_recording(session, sdk, user_id="u1", conversation_id="c1")
        session.fire("function_tools_executed", _executed("call_1"))
        session.fire("conversation_item_added", _turn("user", "hi"))
        await _settle()
        sdk.conversation.record_message.assert_awaited_once()
