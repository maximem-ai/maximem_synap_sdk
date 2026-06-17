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
