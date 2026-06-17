"""Tests for SynapContextProvider — MAF ContextProvider backed by Synap.

Documented contracts (from context_provider.py docstring):
- before_run: Fetches Synap context via sdk.fetch(); extends instructions via
  context.extend_instructions(source_id, ...) when formatted_context is non-empty.
  Read failures degrade gracefully — logged + skipped, never raised.
- after_run: Records each input + response message via sdk.conversation.record_message().
  Write failures are logged but never re-raised (MAF "context providers must not crash" contract).
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from agent_framework import Message, SessionContext
from synap_microsoft_agent.context_provider import SynapContextProvider
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_sdk(formatted: str | None = "User is a power user.", response_ok: bool = True) -> MagicMock:
    """Build a minimal SDK mock for context-provider tests."""
    sdk = MagicMock()

    fetch_response = MagicMock()
    fetch_response.formatted_context = formatted
    sdk.fetch = AsyncMock(return_value=fetch_response)

    if response_ok:
        sdk.conversation = MagicMock()
        sdk.conversation.record_message = AsyncMock(return_value={"message_id": "m1"})
    else:
        sdk.conversation = MagicMock()
        sdk.conversation.record_message = AsyncMock(side_effect=RuntimeError("sdk boom"))
    return sdk


def _make_fetch_failing_sdk() -> MagicMock:
    sdk = MagicMock()
    sdk.fetch = AsyncMock(side_effect=RuntimeError("sdk boom"))
    sdk.conversation = MagicMock()
    sdk.conversation.record_message = AsyncMock(return_value={"message_id": "m1"})
    return sdk


def _make_context(
    *,
    session_id: str | None = "conv-test-1",
    input_messages: list[Message] | None = None,
    response_messages: list[Message] | None = None,
) -> SessionContext:
    """Build a real SessionContext with optional input and response messages.

    Passing ``input_messages=None`` (default) produces a single user message.
    Passing ``input_messages=[]`` explicitly gives an empty input list.
    """
    if input_messages is None:
        inputs: list[Message] = [Message(role="user", contents=["Hello"])]
    else:
        inputs = input_messages
    ctx = SessionContext(session_id=session_id, input_messages=inputs)
    if response_messages is not None:
        from agent_framework import AgentResponse
        ctx._response = AgentResponse(messages=response_messages)
    return ctx


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_requires_non_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapContextProvider(None, user_id="alice")  # type: ignore[arg-type]

    def test_requires_non_empty_user_id(self):
        sdk = _make_sdk()
        with pytest.raises(ValueError, match="non-empty user_id"):
            SynapContextProvider(sdk, user_id="")

    def test_defaults_stored_on_instance(self):
        sdk = _make_sdk()
        provider = SynapContextProvider(sdk, user_id="alice")
        assert provider.user_id == "alice"
        assert provider.customer_id == ""
        assert provider.conversation_id is None
        assert provider.mode == "accurate"
        assert provider.max_results == 20
        assert provider.source_id == SynapContextProvider.DEFAULT_SOURCE_ID
        assert provider.include_scope_labels is False

    def test_custom_params_stored(self):
        sdk = _make_sdk()
        provider = SynapContextProvider(
            sdk,
            user_id="bob",
            customer_id="acme",
            conversation_id="conv-1",
            source_id="my_synap",
            mode="fast",
            max_results=5,
            context_prompt="## Custom Prompt",
            include_scope_labels=True,
        )
        assert provider.customer_id == "acme"
        assert provider.conversation_id == "conv-1"
        assert provider.source_id == "my_synap"
        assert provider.mode == "fast"
        assert provider.max_results == 5
        assert provider.context_prompt == "## Custom Prompt"
        assert provider.include_scope_labels is True

    def test_default_context_prompt_not_empty(self):
        sdk = _make_sdk()
        provider = SynapContextProvider(sdk, user_id="alice")
        assert provider.context_prompt
        assert len(provider.context_prompt) > 0


# ---------------------------------------------------------------------------
# before_run — happy paths
# ---------------------------------------------------------------------------


class TestBeforeRunHappyPath:
    @pytest.mark.asyncio
    async def test_extends_instructions_when_context_available(self):
        sdk = _make_sdk(formatted="User is a premium customer.")
        provider = SynapContextProvider(sdk, user_id="alice", customer_id="acme")
        ctx = _make_context()

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        assert len(ctx.instructions) == 1
        instruction_text = ctx.instructions[0]
        assert "User is a premium customer." in instruction_text

    @pytest.mark.asyncio
    async def test_instructions_include_context_prompt_preamble(self):
        sdk = _make_sdk(formatted="User likes Python.")
        provider = SynapContextProvider(sdk, user_id="alice")
        ctx = _make_context()

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        text = ctx.instructions[0]
        assert provider.context_prompt in text
        assert "User likes Python." in text

    @pytest.mark.asyncio
    async def test_calls_sdk_fetch_with_correct_args(self):
        sdk = _make_sdk()
        provider = SynapContextProvider(
            sdk,
            user_id="alice",
            customer_id="acme",
            conversation_id="conv-explicit",
            mode="fast",
            max_results=10,
            include_scope_labels=True,
        )
        ctx = _make_context(session_id="sess-xyz")

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        sdk.fetch.assert_awaited_once_with(
            conversation_id="conv-explicit",
            user_id="alice",
            customer_id="acme",
            search_query=["Hello"],
            max_results=10,
            mode="fast",
            include_conversation_context=False,
            include_scope_labels=True,
        )

    @pytest.mark.asyncio
    async def test_uses_session_id_when_no_conversation_id_set(self):
        sdk = _make_sdk()
        provider = SynapContextProvider(sdk, user_id="alice")
        ctx = _make_context(session_id="sess-from-ctx")

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        call_kwargs = sdk.fetch.call_args.kwargs
        assert call_kwargs["conversation_id"] == "sess-from-ctx"

    @pytest.mark.asyncio
    async def test_empty_customer_id_passed_as_none_to_sdk(self):
        """customer_id='' should be forwarded as None to the SDK fetch call."""
        sdk = _make_sdk()
        provider = SynapContextProvider(sdk, user_id="alice", customer_id="")
        ctx = _make_context()

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        call_kwargs = sdk.fetch.call_args.kwargs
        assert call_kwargs["customer_id"] is None

    @pytest.mark.asyncio
    async def test_no_extension_when_formatted_context_empty(self):
        sdk = _make_sdk(formatted="")
        provider = SynapContextProvider(sdk, user_id="alice")
        ctx = _make_context()

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        assert ctx.instructions == []

    @pytest.mark.asyncio
    async def test_no_extension_when_formatted_context_whitespace_only(self):
        sdk = _make_sdk(formatted="   ")
        provider = SynapContextProvider(sdk, user_id="alice")
        ctx = _make_context()

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        assert ctx.instructions == []

    @pytest.mark.asyncio
    async def test_no_extension_when_formatted_context_none(self):
        sdk = _make_sdk(formatted=None)
        provider = SynapContextProvider(sdk, user_id="alice")
        ctx = _make_context()

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        assert ctx.instructions == []

    @pytest.mark.asyncio
    async def test_no_fetch_when_input_messages_empty(self):
        sdk = _make_sdk()
        provider = SynapContextProvider(sdk, user_id="alice")
        ctx = _make_context(input_messages=[])

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        sdk.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_fetch_when_input_messages_text_blank(self):
        """Messages with whitespace-only text should produce an empty query → no fetch."""
        blank_msg = MagicMock()
        blank_msg.text = "   "
        sdk = _make_sdk()
        provider = SynapContextProvider(sdk, user_id="alice")
        ctx = _make_context(input_messages=[blank_msg])

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        sdk.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multiple_input_messages_concatenated(self):
        """All input message texts must be joined and passed as a single search_query item."""
        m1 = Message(role="user", contents=["First turn."])
        m2 = Message(role="user", contents=["Second turn."])
        sdk = _make_sdk()
        provider = SynapContextProvider(sdk, user_id="alice")
        ctx = _make_context(input_messages=[m1, m2])

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        call_kwargs = sdk.fetch.call_args.kwargs
        query = call_kwargs["search_query"]
        assert isinstance(query, list)
        assert len(query) == 1
        assert "First turn." in query[0]
        assert "Second turn." in query[0]


# ---------------------------------------------------------------------------
# before_run — failure path (documented: degrade gracefully, never raise)
# ---------------------------------------------------------------------------


class TestBeforeRunFailurePath:
    @pytest.mark.asyncio
    async def test_sdk_fetch_error_does_not_raise(self):
        sdk = _make_fetch_failing_sdk()
        provider = SynapContextProvider(sdk, user_id="alice")
        ctx = _make_context()

        # Must not raise — degrades gracefully
        await provider.before_run(agent=None, session=None, context=ctx, state={})

    @pytest.mark.asyncio
    async def test_sdk_fetch_error_no_instructions_extended(self):
        sdk = _make_fetch_failing_sdk()
        provider = SynapContextProvider(sdk, user_id="alice")
        ctx = _make_context()

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        assert ctx.instructions == []

    @pytest.mark.asyncio
    async def test_sdk_fetch_error_is_logged(self, caplog):
        sdk = _make_fetch_failing_sdk()
        provider = SynapContextProvider(sdk, user_id="alice")
        ctx = _make_context()

        with caplog.at_level(logging.ERROR, logger="synap_microsoft_agent.context_provider"):
            await provider.before_run(agent=None, session=None, context=ctx, state={})

        assert any("before_run" in r.message or "sdk" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# after_run — happy paths
# ---------------------------------------------------------------------------


class TestAfterRunHappyPath:
    @pytest.mark.asyncio
    async def test_records_input_messages_to_sdk(self):
        sdk = _make_sdk()
        provider = SynapContextProvider(
            sdk, user_id="alice", customer_id="acme", conversation_id="conv-1"
        )
        user_msg = Message(role="user", contents=["How can I upgrade?"])
        ctx = _make_context(session_id="conv-1", input_messages=[user_msg], response_messages=[])

        await provider.after_run(agent=None, session=None, context=ctx, state={})

        sdk.conversation.record_message.assert_awaited_once_with(
            conversation_id="conv-1",
            role="user",
            content="How can I upgrade?",
            user_id="alice",
            customer_id="acme",
        )

    @pytest.mark.asyncio
    async def test_records_response_messages_to_sdk(self):
        sdk = _make_sdk()
        provider = SynapContextProvider(
            sdk, user_id="alice", customer_id="acme", conversation_id="conv-1"
        )
        assistant_msg = Message(role="assistant", contents=["Here is how to upgrade."])
        ctx = _make_context(session_id="conv-1", input_messages=[], response_messages=[assistant_msg])

        await provider.after_run(agent=None, session=None, context=ctx, state={})

        sdk.conversation.record_message.assert_awaited_once_with(
            conversation_id="conv-1",
            role="assistant",
            content="Here is how to upgrade.",
            user_id="alice",
            customer_id="acme",
        )

    @pytest.mark.asyncio
    async def test_records_both_input_and_response_messages(self):
        sdk = _make_sdk()
        provider = SynapContextProvider(
            sdk, user_id="alice", customer_id="acme", conversation_id="conv-1"
        )
        user_msg = Message(role="user", contents=["Question?"])
        assistant_msg = Message(role="assistant", contents=["Answer."])
        ctx = _make_context(
            session_id="conv-1",
            input_messages=[user_msg],
            response_messages=[assistant_msg],
        )

        await provider.after_run(agent=None, session=None, context=ctx, state={})

        assert sdk.conversation.record_message.await_count == 2
        roles_recorded = [
            c.kwargs["role"]
            for c in sdk.conversation.record_message.call_args_list
        ]
        assert "user" in roles_recorded
        assert "assistant" in roles_recorded

    @pytest.mark.asyncio
    async def test_uses_session_id_when_no_explicit_conversation_id(self):
        """after_run must fall back to context.session_id if conversation_id not set."""
        sdk = _make_sdk()
        provider = SynapContextProvider(sdk, user_id="alice")
        user_msg = Message(role="user", contents=["Hi"])
        ctx = _make_context(session_id="sess-fallback", input_messages=[user_msg])

        await provider.after_run(agent=None, session=None, context=ctx, state={})

        call_kwargs = sdk.conversation.record_message.call_args.kwargs
        assert call_kwargs["conversation_id"] == "sess-fallback"

    @pytest.mark.asyncio
    async def test_no_record_when_no_conversation_id_available(self):
        """If conversation_id is not set and context has no session_id → no SDK call."""
        sdk = _make_sdk()
        provider = SynapContextProvider(sdk, user_id="alice")
        user_msg = Message(role="user", contents=["Hello"])
        ctx = _make_context(session_id=None, input_messages=[user_msg])

        await provider.after_run(agent=None, session=None, context=ctx, state={})

        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_messages_with_unsupported_role(self):
        """Messages with role='tool' must be silently skipped."""
        sdk = _make_sdk()
        provider = SynapContextProvider(
            sdk, user_id="alice", conversation_id="conv-1"
        )
        tool_msg = Message(role="tool", contents=["Tool result"])
        ctx = _make_context(session_id="conv-1", input_messages=[tool_msg])

        await provider.after_run(agent=None, session=None, context=ctx, state={})

        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_messages_with_blank_text(self):
        """Messages with only whitespace text must not be recorded."""
        sdk = _make_sdk()
        provider = SynapContextProvider(
            sdk, user_id="alice", conversation_id="conv-1"
        )
        blank_msg = MagicMock()
        blank_msg.role = "user"
        blank_msg.text = "   "
        ctx = _make_context(session_id="conv-1", input_messages=[blank_msg])

        await provider.after_run(agent=None, session=None, context=ctx, state={})

        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_record_when_input_and_response_empty(self):
        sdk = _make_sdk()
        provider = SynapContextProvider(
            sdk, user_id="alice", conversation_id="conv-1"
        )
        ctx = _make_context(session_id="conv-1", input_messages=[], response_messages=[])

        await provider.after_run(agent=None, session=None, context=ctx, state={})

        sdk.conversation.record_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# after_run — failure path (documented: never re-raise)
# ---------------------------------------------------------------------------


class TestAfterRunFailurePath:
    @pytest.mark.asyncio
    async def test_record_message_error_does_not_raise(self):
        sdk = _make_sdk(response_ok=False)
        provider = SynapContextProvider(
            sdk, user_id="alice", conversation_id="conv-1"
        )
        user_msg = Message(role="user", contents=["Hi"])
        ctx = _make_context(session_id="conv-1", input_messages=[user_msg])

        # Must not raise — MAF contract: context providers must not crash the agent
        await provider.after_run(agent=None, session=None, context=ctx, state={})

    @pytest.mark.asyncio
    async def test_record_message_error_is_logged(self, caplog):
        sdk = _make_sdk(response_ok=False)
        provider = SynapContextProvider(
            sdk, user_id="alice", conversation_id="conv-1"
        )
        user_msg = Message(role="user", contents=["Hi"])
        ctx = _make_context(session_id="conv-1", input_messages=[user_msg])

        with caplog.at_level(logging.ERROR, logger="synap_microsoft_agent.context_provider"):
            await provider.after_run(agent=None, session=None, context=ctx, state={})

        assert any(
            "after_run" in r.message or "record_message" in r.message
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_record_message_failure_continues_remaining_messages(self):
        """Even if one record_message call fails, the subsequent messages must still be attempted."""
        sdk = MagicMock()
        # Fails on first call, succeeds on second
        sdk.conversation = MagicMock()
        sdk.conversation.record_message = AsyncMock(
            side_effect=[RuntimeError("fail on first"), {"message_id": "m2"}]
        )
        provider = SynapContextProvider(
            sdk, user_id="alice", conversation_id="conv-1"
        )
        msg1 = Message(role="user", contents=["First"])
        msg2 = Message(role="user", contents=["Second"])
        ctx = _make_context(session_id="conv-1", input_messages=[msg1, msg2])

        await provider.after_run(agent=None, session=None, context=ctx, state={})

        # Both calls must have been attempted
        assert sdk.conversation.record_message.await_count == 2


# ---------------------------------------------------------------------------
# _resolve_conversation_id
# ---------------------------------------------------------------------------


class TestResolveConversationId:
    def test_explicit_conversation_id_takes_precedence(self):
        sdk = _make_sdk()
        provider = SynapContextProvider(sdk, user_id="alice", conversation_id="explicit-conv")
        ctx = MagicMock()
        ctx.session_id = "session-conv"
        assert provider._resolve_conversation_id(ctx) == "explicit-conv"

    def test_falls_back_to_session_id(self):
        sdk = _make_sdk()
        provider = SynapContextProvider(sdk, user_id="alice")
        ctx = MagicMock()
        ctx.session_id = "session-from-ctx"
        assert provider._resolve_conversation_id(ctx) == "session-from-ctx"

    def test_returns_none_when_no_id_available(self):
        sdk = _make_sdk()
        provider = SynapContextProvider(sdk, user_id="alice")
        ctx = MagicMock()
        ctx.session_id = None
        assert provider._resolve_conversation_id(ctx) is None


# ---------------------------------------------------------------------------
# _concat_text
# ---------------------------------------------------------------------------


class TestConcatText:
    def test_empty_messages_returns_empty_string(self):
        assert SynapContextProvider._concat_text([]) == ""

    def test_none_messages_returns_empty_string(self):
        assert SynapContextProvider._concat_text(None) == ""

    def test_single_message_text(self):
        msg = MagicMock()
        msg.text = "hello"
        assert SynapContextProvider._concat_text([msg]) == "hello"

    def test_multiple_messages_joined_by_newline(self):
        m1, m2 = MagicMock(), MagicMock()
        m1.text = "first"
        m2.text = "second"
        result = SynapContextProvider._concat_text([m1, m2])
        assert result == "first\nsecond"

    def test_messages_without_text_attribute_skipped(self):
        msg = MagicMock(spec=[])  # no .text
        assert SynapContextProvider._concat_text([msg]) == ""

    def test_blank_text_messages_skipped(self):
        msg = MagicMock()
        msg.text = "   "
        assert SynapContextProvider._concat_text([msg]) == ""


# ---------------------------------------------------------------------------
# _message_to_role_text
# ---------------------------------------------------------------------------


class TestMessageToRoleText:
    def test_user_role_returns_correctly(self):
        msg = MagicMock()
        msg.role = "user"
        msg.text = "Hello"
        role, text = SynapContextProvider._message_to_role_text(msg)
        assert role == "user"
        assert text == "Hello"

    def test_assistant_role_returns_correctly(self):
        msg = MagicMock()
        msg.role = "assistant"
        msg.text = "Reply"
        role, text = SynapContextProvider._message_to_role_text(msg)
        assert role == "assistant"
        assert text == "Reply"

    def test_system_role_returns_correctly(self):
        msg = MagicMock()
        msg.role = "system"
        msg.text = "You are helpful"
        role, text = SynapContextProvider._message_to_role_text(msg)
        assert role == "system"
        assert text == "You are helpful"

    def test_tool_role_rejected(self):
        msg = MagicMock()
        msg.role = "tool"
        msg.text = "Tool result"
        role, text = SynapContextProvider._message_to_role_text(msg)
        assert role is None
        assert text == ""

    def test_unknown_role_rejected(self):
        msg = MagicMock()
        msg.role = "unknown"
        msg.text = "Something"
        role, text = SynapContextProvider._message_to_role_text(msg)
        assert role is None

    def test_no_role_attribute_returns_none(self):
        msg = MagicMock(spec=[])
        role, text = SynapContextProvider._message_to_role_text(msg)
        assert role is None

    def test_blank_text_returns_none_role(self):
        msg = MagicMock()
        msg.role = "user"
        msg.text = "   "
        role, text = SynapContextProvider._message_to_role_text(msg)
        assert role is None

    def test_role_with_value_attribute_is_unwrapped(self):
        """Enum-style role objects with .value must be unwrapped to the string."""
        msg = MagicMock()
        role_enum = MagicMock()
        role_enum.value = "user"
        msg.role = role_enum
        msg.text = "Hi"
        role, text = SynapContextProvider._message_to_role_text(msg)
        assert role == "user"
        assert text == "Hi"


# ---------------------------------------------------------------------------
# Integration with shared harness fixtures (mock_sdk / failing_sdk)
# ---------------------------------------------------------------------------


class TestWithSharedHarness:
    @pytest.mark.asyncio
    async def test_before_run_with_mock_sdk(self, mock_sdk):
        """mock_sdk returns a real ContextForPromptResponse with formatted_context."""
        provider = SynapContextProvider(
            mock_sdk, user_id="alice", customer_id="acme", conversation_id="conv-1"
        )
        ctx = _make_context(session_id="conv-1")

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        # mock_sdk.fetch returns make_unified_response() which has a non-empty formatted_context
        assert len(ctx.instructions) == 1
        assert "## User Context" in ctx.instructions[0] or ctx.instructions[0]  # any non-empty text

    @pytest.mark.asyncio
    async def test_before_run_with_failing_sdk_does_not_raise(self, failing_sdk):
        """failing_sdk.fetch raises RuntimeError — before_run must swallow it."""
        provider = SynapContextProvider(
            failing_sdk, user_id="alice", conversation_id="conv-1"
        )
        ctx = _make_context(session_id="conv-1")

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        assert ctx.instructions == []

    @pytest.mark.asyncio
    async def test_after_run_with_failing_sdk_does_not_raise(self, failing_sdk):
        """failing_sdk.conversation.record_message raises — after_run must swallow it."""
        provider = SynapContextProvider(
            failing_sdk, user_id="alice", conversation_id="conv-1"
        )
        user_msg = Message(role="user", contents=["Hi"])
        ctx = _make_context(session_id="conv-1", input_messages=[user_msg])

        await provider.after_run(agent=None, session=None, context=ctx, state={})
