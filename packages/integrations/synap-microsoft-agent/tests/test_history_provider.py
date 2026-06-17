"""Tests for SynapHistoryProvider — MAF HistoryProvider backed by Synap.

Documented contracts (from history_provider.py docstring):
- get_messages: best-effort — swallows SDK errors, returns [] + ERROR log.
  A history read outage must not abort the agent turn.
- save_messages: strict — SDK errors are surfaced as SynapIntegrationError.
  Silent drops would hide ingestion problems from the caller.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_framework import Message
from synap_microsoft_agent.history_provider import SynapHistoryProvider
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_sdk(
    *,
    get_context_result=None,
    get_context_side_effect=None,
    record_message_side_effect=None,
) -> MagicMock:
    sdk = MagicMock()
    sdk.conversation = MagicMock()

    if get_context_side_effect is not None:
        sdk.conversation.context = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=get_context_side_effect
        )
    else:
        ctx_response = get_context_result or MagicMock(recent_messages=[])
        sdk.conversation.context = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            return_value=ctx_response
        )

    if record_message_side_effect is not None:
        sdk.conversation.record_message = AsyncMock(side_effect=record_message_side_effect)
    else:
        sdk.conversation.record_message = AsyncMock(return_value={"message_id": "m1"})

    return sdk


def _make_recent_messages(*role_content_pairs: tuple[str, str]) -> MagicMock:
    """Build a mock context response with given (role, content) tuples as recent_messages."""
    msgs = []
    for role, content in role_content_pairs:
        m = MagicMock()
        m.role = role
        m.content = content
        msgs.append(m)
    response = MagicMock()
    response.recent_messages = msgs
    return response


def _make_message(role: str, text: str) -> Message:
    return Message(role=role, contents=[text])


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_requires_non_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapHistoryProvider(None, user_id="alice")  # type: ignore[arg-type]

    def test_requires_non_empty_user_id(self):
        sdk = _make_sdk()
        with pytest.raises(ValueError, match="non-empty user_id"):
            SynapHistoryProvider(sdk, user_id="")

    def test_defaults_stored_on_instance(self):
        sdk = _make_sdk()
        provider = SynapHistoryProvider(sdk, user_id="alice")
        assert provider.user_id == "alice"
        assert provider.customer_id == ""
        assert provider.conversation_id is None
        assert provider.source_id == SynapHistoryProvider.DEFAULT_SOURCE_ID

    def test_custom_params_stored(self):
        sdk = _make_sdk()
        provider = SynapHistoryProvider(
            sdk,
            user_id="bob",
            customer_id="acme",
            conversation_id="conv-1",
            source_id="my_synap_history",
            load_messages=False,
            store_inputs=False,
            store_outputs=False,
        )
        assert provider.user_id == "bob"
        assert provider.customer_id == "acme"
        assert provider.conversation_id == "conv-1"
        assert provider.source_id == "my_synap_history"
        assert provider.load_messages is False
        assert provider.store_inputs is False
        assert provider.store_outputs is False

    def test_default_source_id_is_synap_history(self):
        sdk = _make_sdk()
        provider = SynapHistoryProvider(sdk, user_id="alice")
        assert provider.source_id == "synap_history"


# ---------------------------------------------------------------------------
# get_messages — happy paths
# ---------------------------------------------------------------------------


class TestGetMessagesHappyPath:
    @pytest.mark.asyncio
    async def test_returns_user_and_assistant_messages(self):
        ctx_response = _make_recent_messages(("user", "Hello"), ("assistant", "Hi!"))
        sdk = _make_sdk(get_context_result=ctx_response)
        provider = SynapHistoryProvider(sdk, user_id="alice", conversation_id="conv-1")

        msgs = await provider.get_messages("conv-1")

        assert len(msgs) == 2
        assert isinstance(msgs[0], Message)
        assert msgs[0].role == "user"
        assert msgs[0].text == "Hello"
        assert msgs[1].role == "assistant"
        assert msgs[1].text == "Hi!"

    @pytest.mark.asyncio
    async def test_calls_sdk_with_correct_conversation_id(self):
        ctx_response = _make_recent_messages()
        sdk = _make_sdk(get_context_result=ctx_response)
        provider = SynapHistoryProvider(sdk, user_id="alice", conversation_id="conv-explicit")

        await provider.get_messages("sess-fallback")

        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv-explicit"
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_session_id(self):
        ctx_response = _make_recent_messages()
        sdk = _make_sdk(get_context_result=ctx_response)
        provider = SynapHistoryProvider(sdk, user_id="alice")

        await provider.get_messages("sess-from-arg")

        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="sess-from-arg"
        )

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_conversation_id(self):
        sdk = _make_sdk()
        provider = SynapHistoryProvider(sdk, user_id="alice")

        msgs = await provider.get_messages(None)

        assert msgs == []
        sdk.conversation.context.get_context_for_prompt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_recent_messages_returns_empty_list(self):
        ctx_response = _make_recent_messages()  # no messages
        sdk = _make_sdk(get_context_result=ctx_response)
        provider = SynapHistoryProvider(sdk, user_id="alice", conversation_id="conv-1")

        msgs = await provider.get_messages("conv-1")

        assert msgs == []

    @pytest.mark.asyncio
    async def test_skips_messages_with_missing_role(self):
        response = MagicMock()
        m = MagicMock()
        m.role = None
        m.content = "some content"
        response.recent_messages = [m]
        sdk = _make_sdk(get_context_result=response)
        provider = SynapHistoryProvider(sdk, user_id="alice", conversation_id="conv-1")

        msgs = await provider.get_messages("conv-1")

        assert msgs == []

    @pytest.mark.asyncio
    async def test_skips_messages_with_missing_content(self):
        response = MagicMock()
        m = MagicMock()
        m.role = "user"
        m.content = None
        response.recent_messages = [m]
        sdk = _make_sdk(get_context_result=response)
        provider = SynapHistoryProvider(sdk, user_id="alice", conversation_id="conv-1")

        msgs = await provider.get_messages("conv-1")

        assert msgs == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_recent_messages_is_none(self):
        response = MagicMock()
        response.recent_messages = None
        sdk = _make_sdk(get_context_result=response)
        provider = SynapHistoryProvider(sdk, user_id="alice", conversation_id="conv-1")

        msgs = await provider.get_messages("conv-1")

        assert msgs == []


# ---------------------------------------------------------------------------
# get_messages — failure path (documented: swallow + empty list)
# ---------------------------------------------------------------------------


class TestGetMessagesFailurePath:
    @pytest.mark.asyncio
    async def test_sdk_error_returns_empty_list(self):
        sdk = _make_sdk(get_context_side_effect=RuntimeError("sdk boom"))
        provider = SynapHistoryProvider(sdk, user_id="alice", conversation_id="conv-1")

        msgs = await provider.get_messages("conv-1")

        assert msgs == []

    @pytest.mark.asyncio
    async def test_sdk_error_does_not_raise(self):
        sdk = _make_sdk(get_context_side_effect=Exception("outage"))
        provider = SynapHistoryProvider(sdk, user_id="alice", conversation_id="conv-1")

        # Must not raise
        await provider.get_messages("conv-1")

    @pytest.mark.asyncio
    async def test_sdk_error_is_logged_at_error_level(self, caplog):
        sdk = _make_sdk(get_context_side_effect=RuntimeError("outage"))
        provider = SynapHistoryProvider(sdk, user_id="alice", conversation_id="conv-1")

        with caplog.at_level(logging.ERROR, logger="synap_microsoft_agent.history_provider"):
            await provider.get_messages("conv-1")

        assert len(caplog.records) >= 1
        assert any("conv-1" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# save_messages — happy paths
# ---------------------------------------------------------------------------


class TestSaveMessagesHappyPath:
    @pytest.mark.asyncio
    async def test_records_each_message_to_sdk(self):
        sdk = _make_sdk()
        provider = SynapHistoryProvider(
            sdk, user_id="alice", customer_id="acme", conversation_id="conv-1"
        )
        msgs = [
            _make_message("user", "Hello"),
            _make_message("assistant", "Hi there"),
        ]

        await provider.save_messages("conv-1", msgs)

        assert sdk.conversation.record_message.await_count == 2

    @pytest.mark.asyncio
    async def test_passes_correct_ids_on_each_call(self):
        sdk = _make_sdk()
        provider = SynapHistoryProvider(
            sdk, user_id="alice", customer_id="acme", conversation_id="conv-1"
        )
        msgs = [_make_message("user", "Test message")]

        await provider.save_messages("sess-arg", msgs)

        kw = sdk.conversation.record_message.call_args.kwargs
        assert kw["conversation_id"] == "conv-1"
        assert kw["user_id"] == "alice"
        assert kw["customer_id"] == "acme"
        assert kw["role"] == "user"
        assert kw["content"] == "Test message"

    @pytest.mark.asyncio
    async def test_falls_back_to_session_id_arg_when_no_conversation_id(self):
        sdk = _make_sdk()
        provider = SynapHistoryProvider(sdk, user_id="alice")
        msgs = [_make_message("user", "Hello")]

        await provider.save_messages("sess-from-arg", msgs)

        kw = sdk.conversation.record_message.call_args.kwargs
        assert kw["conversation_id"] == "sess-from-arg"

    @pytest.mark.asyncio
    async def test_empty_messages_sequence_is_noop(self):
        sdk = _make_sdk()
        provider = SynapHistoryProvider(sdk, user_id="alice", conversation_id="conv-1")

        await provider.save_messages("conv-1", [])

        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_record_when_no_conversation_id_available(self):
        sdk = _make_sdk()
        provider = SynapHistoryProvider(sdk, user_id="alice")
        msgs = [_make_message("user", "Hello")]

        await provider.save_messages(None, msgs)

        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_messages_with_unsupported_role(self):
        """Messages with role='tool' must be silently skipped."""
        sdk = _make_sdk()
        provider = SynapHistoryProvider(sdk, user_id="alice", conversation_id="conv-1")
        tool_msg = _make_message("tool", "Tool result")

        await provider.save_messages("conv-1", [tool_msg])

        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_messages_with_blank_text(self):
        """Messages whose text is whitespace-only must be silently skipped."""
        sdk = _make_sdk()
        provider = SynapHistoryProvider(sdk, user_id="alice", conversation_id="conv-1")
        blank_msg = MagicMock()
        blank_msg.role = "user"
        blank_msg.text = "   "

        await provider.save_messages("conv-1", [blank_msg])

        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_role_enum_values_unwrapped(self):
        """Message.role values must be converted to plain strings before forwarding."""
        sdk = _make_sdk()
        provider = SynapHistoryProvider(sdk, user_id="alice", conversation_id="conv-1")
        msg = _make_message("user", "Hello")

        await provider.save_messages("conv-1", [msg])

        kw = sdk.conversation.record_message.call_args.kwargs
        assert isinstance(kw["role"], str)
        assert kw["role"] == "user"


# ---------------------------------------------------------------------------
# save_messages — failure path (documented: raise SynapIntegrationError)
# ---------------------------------------------------------------------------


class TestSaveMessagesFailurePath:
    @pytest.mark.asyncio
    async def test_sdk_error_raises_synap_integration_error(self):
        """save_messages MUST raise SynapIntegrationError when SDK fails.

        The docstring is explicit: save_messages surfaces SDK errors as
        SynapIntegrationError. Silent drops would hide ingestion problems.
        """
        sdk = _make_sdk(record_message_side_effect=RuntimeError("sdk boom"))
        provider = SynapHistoryProvider(sdk, user_id="alice", conversation_id="conv-1")
        msgs = [_make_message("user", "Hello")]

        with pytest.raises(SynapIntegrationError):
            await provider.save_messages("conv-1", msgs)

    @pytest.mark.asyncio
    async def test_sdk_error_preserves_original_cause(self):
        """SynapIntegrationError must chain the original SDK exception as __cause__."""
        original = RuntimeError("original sdk error")
        sdk = _make_sdk(record_message_side_effect=original)
        provider = SynapHistoryProvider(sdk, user_id="alice", conversation_id="conv-1")
        msgs = [_make_message("user", "Hello")]

        with pytest.raises(SynapIntegrationError) as exc_info:
            await provider.save_messages("conv-1", msgs)

        assert exc_info.value.__cause__ is original

    @pytest.mark.asyncio
    async def test_already_synap_integration_error_is_re_raised(self):
        """A SynapIntegrationError raised by the SDK should surface unchanged."""
        sdk_error = SynapIntegrationError(
            "microsoft_agent.save_messages", "direct error from sdk"
        )
        sdk = _make_sdk(record_message_side_effect=sdk_error)
        provider = SynapHistoryProvider(sdk, user_id="alice", conversation_id="conv-1")
        msgs = [_make_message("user", "Hello")]

        with pytest.raises(SynapIntegrationError):
            await provider.save_messages("conv-1", msgs)


# ---------------------------------------------------------------------------
# Integration with shared harness fixtures
# ---------------------------------------------------------------------------


class TestWithSharedHarness:
    @pytest.mark.asyncio
    async def test_get_messages_with_mock_sdk(self, mock_sdk):
        """mock_sdk returns a ContextForPromptResponse — get_messages must complete without error."""
        provider = SynapHistoryProvider(
            mock_sdk, user_id="alice", conversation_id="conv-1"
        )
        # mock_sdk returns a ContextForPromptResponse which may or may not have recent_messages
        # The important thing is get_messages does not raise and returns a list
        msgs = await provider.get_messages("conv-1")
        assert isinstance(msgs, list)

    @pytest.mark.asyncio
    async def test_get_messages_with_failing_sdk_returns_empty(self, failing_sdk):
        provider = SynapHistoryProvider(
            failing_sdk, user_id="alice", conversation_id="conv-1"
        )
        msgs = await provider.get_messages("conv-1")
        assert msgs == []

    @pytest.mark.asyncio
    async def test_save_messages_with_mock_sdk(self, mock_sdk):
        provider = SynapHistoryProvider(
            mock_sdk, user_id="alice", customer_id="acme", conversation_id="conv-1"
        )
        msgs = [_make_message("user", "Hello"), _make_message("assistant", "Hi")]

        await provider.save_messages("conv-1", msgs)

        assert mock_sdk.conversation.record_message.await_count == 2

    @pytest.mark.asyncio
    async def test_save_messages_with_failing_sdk_raises(self, failing_sdk):
        provider = SynapHistoryProvider(
            failing_sdk, user_id="alice", conversation_id="conv-1"
        )
        msgs = [_make_message("user", "Hello")]

        with pytest.raises(SynapIntegrationError):
            await provider.save_messages("conv-1", msgs)
