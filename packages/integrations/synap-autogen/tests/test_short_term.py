"""Bedrock-quality tests for synap_autogen.short_term — SynapShortTermChatContext.

Covers:
- Construction validation (sdk, conversation_id, style, on_error).
- Happy path: ST prepended as SystemMessage, preamble tags, style forwarded.
- Empty / unavailable ST: no prepend, inner messages returned unchanged.
- SDK failure paths: fallback (silent) and raise (SynapIntegrationError).
- Delegation: add_message / clear forwarded to inner context.
- State persistence: save_state / load_state round-trip.
- Default inner context: UnboundedChatCompletionContext created automatically.
- Public surface / module exports.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage, SystemMessage, UserMessage

from synap_autogen.short_term import (
    SynapShortTermChatContext,
    _DEFAULT_CLOSE,
    _DEFAULT_OPEN,
    _SUPPORTED_STYLES,
)
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(formatted: str | None, available: bool):
    resp = MagicMock()
    resp.available = available
    resp.formatted_context = formatted
    return resp


def _fake_sdk(
    formatted: str | None = "User asked about pricing twice.",
    available: bool = True,
):
    sdk = MagicMock()
    sdk.conversation = MagicMock()
    sdk.conversation.context = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=_make_response(formatted, available)
    )
    return sdk


async def _seed_inner(ctx: SynapShortTermChatContext, n: int = 2) -> None:
    await ctx.add_message(UserMessage(content="hi", source="user"))
    if n >= 2:
        await ctx.add_message(AssistantMessage(content="hello!", source="assistant"))


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_package_exports_class(self):
        import synap_autogen
        assert hasattr(synap_autogen, "SynapShortTermChatContext")
        assert "SynapShortTermChatContext" in synap_autogen.__all__

    def test_module_all_exports(self):
        from synap_autogen import short_term
        assert "SynapShortTermChatContext" in short_term.__all__


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapShortTermChatContext(None, "conv_abc")  # type: ignore[arg-type]

    def test_requires_non_empty_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            SynapShortTermChatContext(_fake_sdk(), "")

    def test_requires_non_whitespace_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            SynapShortTermChatContext(_fake_sdk(), "   ")

    def test_rejects_unknown_style(self):
        with pytest.raises(ValueError, match="unsupported style"):
            SynapShortTermChatContext(_fake_sdk(), "conv_abc", style="bogus")

    def test_rejects_invalid_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            SynapShortTermChatContext(  # type: ignore[arg-type]
                _fake_sdk(), "conv_abc", on_error="ignore"
            )

    @pytest.mark.parametrize("style", _SUPPORTED_STYLES)
    def test_accepts_all_documented_styles(self, style):
        """Every documented style must be accepted without error."""
        ctx = SynapShortTermChatContext(_fake_sdk(), "conv_abc", style=style)
        assert ctx is not None

    def test_accepts_fallback_on_error(self):
        ctx = SynapShortTermChatContext(_fake_sdk(), "conv_abc", on_error="fallback")
        assert ctx is not None

    def test_accepts_raise_on_error(self):
        ctx = SynapShortTermChatContext(_fake_sdk(), "conv_abc", on_error="raise")
        assert ctx is not None

    def test_accepts_none_preamble_both(self):
        ctx = SynapShortTermChatContext(
            _fake_sdk(), "conv_abc", preamble_open=None, preamble_close=None
        )
        assert ctx is not None

    def test_creates_default_inner_context_when_none_passed(self):
        ctx = SynapShortTermChatContext(_fake_sdk(), "conv_abc")
        assert isinstance(ctx._inner, UnboundedChatCompletionContext)

    def test_uses_provided_inner_context(self):
        inner = UnboundedChatCompletionContext()
        ctx = SynapShortTermChatContext(_fake_sdk(), "conv_abc", inner=inner)
        assert ctx._inner is inner


# ---------------------------------------------------------------------------
# get_messages — happy path
# ---------------------------------------------------------------------------


class TestGetMessagesHappyPath:
    @pytest.mark.asyncio
    async def test_prepends_system_message_before_inner_messages(self):
        sdk = _fake_sdk(formatted="User is VIP.")
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        await _seed_inner(ctx)
        messages = await ctx.get_messages()
        assert len(messages) == 3
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], UserMessage)
        assert isinstance(messages[2], AssistantMessage)

    @pytest.mark.asyncio
    async def test_st_content_embedded_in_system_message(self):
        sdk = _fake_sdk(formatted="User is VIP.")
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        messages = await ctx.get_messages()
        assert "User is VIP." in messages[0].content

    @pytest.mark.asyncio
    async def test_default_preamble_open_tag_present(self):
        sdk = _fake_sdk(formatted="some context")
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        messages = await ctx.get_messages()
        assert _DEFAULT_OPEN in messages[0].content

    @pytest.mark.asyncio
    async def test_default_preamble_close_tag_present(self):
        sdk = _fake_sdk(formatted="some context")
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        messages = await ctx.get_messages()
        assert _DEFAULT_CLOSE in messages[0].content

    @pytest.mark.asyncio
    async def test_open_tag_before_close_tag(self):
        sdk = _fake_sdk(formatted="content")
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        messages = await ctx.get_messages()
        content = messages[0].content
        assert content.index(_DEFAULT_OPEN) < content.index(_DEFAULT_CLOSE)

    @pytest.mark.asyncio
    async def test_custom_preamble_tags_used(self):
        sdk = _fake_sdk(formatted="custom wrapped")
        ctx = SynapShortTermChatContext(
            sdk, "conv_abc", preamble_open="[BEGIN]", preamble_close="[END]"
        )
        messages = await ctx.get_messages()
        assert "[BEGIN]" in messages[0].content
        assert "[END]" in messages[0].content
        assert _DEFAULT_OPEN not in messages[0].content

    @pytest.mark.asyncio
    async def test_none_preamble_returns_raw_st_content(self):
        """Both preamble tags None → content is the raw ST string."""
        sdk = _fake_sdk(formatted="raw st")
        ctx = SynapShortTermChatContext(
            sdk, "conv_abc", preamble_open=None, preamble_close=None
        )
        messages = await ctx.get_messages()
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "raw st"

    @pytest.mark.asyncio
    async def test_forwards_conversation_id_to_sdk(self):
        sdk = _fake_sdk()
        ctx = SynapShortTermChatContext(sdk, "conv_xyz")
        await ctx.get_messages()
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_xyz",
            style="narrative",
        )

    @pytest.mark.asyncio
    async def test_forwards_style_to_sdk(self):
        sdk = _fake_sdk()
        ctx = SynapShortTermChatContext(sdk, "conv_abc", style="bullet_points")
        await ctx.get_messages()
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="bullet_points",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("style", _SUPPORTED_STYLES)
    async def test_each_style_forwarded_correctly(self, style):
        sdk = _fake_sdk()
        ctx = SynapShortTermChatContext(sdk, "conv_abc", style=style)
        await ctx.get_messages()
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style=style,
        )

    @pytest.mark.asyncio
    async def test_sdk_called_exactly_once_per_get_messages(self):
        sdk = _fake_sdk()
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        await ctx.get_messages()
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_list_not_generator(self):
        sdk = _fake_sdk()
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        result = await ctx.get_messages()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_mock_sdk_fixture_happy_path(self, mock_sdk):
        """Shared mock_sdk fixture wired with ContextForPromptResponse(available=True)."""
        ctx = SynapShortTermChatContext(mock_sdk, "conv_abc")
        await ctx.add_message(UserMessage(content="hello", source="user"))
        messages = await ctx.get_messages()
        # Should prepend a system message because mock_sdk returns available=True
        assert any(isinstance(m, SystemMessage) for m in messages)


# ---------------------------------------------------------------------------
# get_messages — empty / unavailable ST (no-op contract)
# ---------------------------------------------------------------------------


class TestGetMessagesEmptyST:
    @pytest.mark.asyncio
    async def test_unavailable_returns_inner_messages_only(self):
        sdk = _fake_sdk(formatted=None, available=False)
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        await _seed_inner(ctx)
        messages = await ctx.get_messages()
        assert len(messages) == 2
        assert all(not isinstance(m, SystemMessage) for m in messages)

    @pytest.mark.asyncio
    async def test_available_but_empty_string_no_prepend(self):
        sdk = _fake_sdk(formatted="", available=True)
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        await _seed_inner(ctx)
        messages = await ctx.get_messages()
        assert len(messages) == 2
        assert all(not isinstance(m, SystemMessage) for m in messages)

    @pytest.mark.asyncio
    async def test_available_but_whitespace_only_no_prepend(self):
        """Whitespace-only formatted_context must not produce a prepended message."""
        sdk = _fake_sdk(formatted="   ", available=True)
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        await _seed_inner(ctx)
        messages = await ctx.get_messages()
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_empty_inner_with_unavailable_st_returns_empty_list(self):
        sdk = _fake_sdk(formatted=None, available=False)
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        messages = await ctx.get_messages()
        assert messages == []


# ---------------------------------------------------------------------------
# get_messages — error handling
# ---------------------------------------------------------------------------


class TestGetMessagesErrorHandling:
    @pytest.mark.asyncio
    async def test_fallback_mode_swallows_sdk_error_returns_inner(self):
        sdk = MagicMock()
        sdk.conversation = MagicMock()
        sdk.conversation.context = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("sdk boom")
        )
        ctx = SynapShortTermChatContext(sdk, "conv_abc", on_error="fallback")
        await _seed_inner(ctx)
        messages = await ctx.get_messages()
        # Must return inner messages unchanged — not crash
        assert len(messages) == 2
        assert all(not isinstance(m, SystemMessage) for m in messages)

    @pytest.mark.asyncio
    async def test_fallback_mode_never_crashes_agent(self):
        """on_error='fallback' must never raise — the agent must keep running."""
        sdk = MagicMock()
        sdk.conversation = MagicMock()
        sdk.conversation.context = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=Exception("any error")
        )
        ctx = SynapShortTermChatContext(sdk, "conv_abc")  # default: fallback
        # Must not raise
        messages = await ctx.get_messages()
        assert isinstance(messages, list)

    @pytest.mark.asyncio
    async def test_raise_mode_propagates_synap_integration_error(self):
        sdk = MagicMock()
        sdk.conversation = MagicMock()
        sdk.conversation.context = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("sdk boom")
        )
        ctx = SynapShortTermChatContext(sdk, "conv_abc", on_error="raise")
        with pytest.raises(SynapIntegrationError):
            await ctx.get_messages()

    @pytest.mark.asyncio
    async def test_raise_mode_does_not_leak_raw_runtime_error(self):
        """The raised exception must be SynapIntegrationError, not RuntimeError."""
        sdk = MagicMock()
        sdk.conversation = MagicMock()
        sdk.conversation.context = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("raw error leak check")
        )
        ctx = SynapShortTermChatContext(sdk, "conv_abc", on_error="raise")
        raised_type = None
        try:
            await ctx.get_messages()
        except Exception as exc:
            raised_type = type(exc)
        assert raised_type is SynapIntegrationError

    @pytest.mark.asyncio
    async def test_failing_sdk_fixture_fallback_returns_inner(self, failing_sdk):
        """Shared failing_sdk fixture with on_error='fallback' returns inner messages."""
        ctx = SynapShortTermChatContext(failing_sdk, "conv_abc", on_error="fallback")
        await ctx.add_message(UserMessage(content="query", source="user"))
        messages = await ctx.get_messages()
        assert len(messages) == 1
        assert isinstance(messages[0], UserMessage)

    @pytest.mark.asyncio
    async def test_failing_sdk_fixture_raise_propagates(self, failing_sdk):
        """Shared failing_sdk fixture with on_error='raise' raises SynapIntegrationError."""
        ctx = SynapShortTermChatContext(failing_sdk, "conv_abc", on_error="raise")
        with pytest.raises(SynapIntegrationError):
            await ctx.get_messages()


# ---------------------------------------------------------------------------
# add_message / clear — delegation
# ---------------------------------------------------------------------------


class TestInnerDelegation:
    @pytest.mark.asyncio
    async def test_add_message_stored_in_inner(self):
        sdk = _fake_sdk()
        inner = UnboundedChatCompletionContext()
        ctx = SynapShortTermChatContext(sdk, "conv_abc", inner=inner)
        await ctx.add_message(UserMessage(content="hello", source="user"))
        inner_msgs = await inner.get_messages()
        assert len(inner_msgs) == 1
        assert inner_msgs[0].content == "hello"

    @pytest.mark.asyncio
    async def test_clear_empties_inner(self):
        sdk = _fake_sdk()
        inner = UnboundedChatCompletionContext()
        ctx = SynapShortTermChatContext(sdk, "conv_abc", inner=inner)
        await ctx.add_message(UserMessage(content="hi", source="user"))
        await ctx.clear()
        inner_msgs = await inner.get_messages()
        assert inner_msgs == []

    @pytest.mark.asyncio
    async def test_multiple_messages_preserved_in_order(self):
        sdk = _fake_sdk(formatted=None, available=False)  # no ST prepend
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        await ctx.add_message(UserMessage(content="msg1", source="user"))
        await ctx.add_message(AssistantMessage(content="msg2", source="assistant"))
        await ctx.add_message(UserMessage(content="msg3", source="user"))
        messages = await ctx.get_messages()
        assert len(messages) == 3
        assert messages[0].content == "msg1"
        assert messages[1].content == "msg2"
        assert messages[2].content == "msg3"


# ---------------------------------------------------------------------------
# save_state / load_state — round-trip
# ---------------------------------------------------------------------------


class TestStatePersistence:
    @pytest.mark.asyncio
    async def test_save_state_has_inner_key(self):
        sdk = _fake_sdk()
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        state = await ctx.save_state()
        assert "inner" in state

    @pytest.mark.asyncio
    async def test_load_state_restores_messages(self):
        sdk = _fake_sdk()
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        await ctx.add_message(UserMessage(content="persisted", source="user"))
        state = await ctx.save_state()

        ctx2 = SynapShortTermChatContext(sdk, "conv_abc")
        await ctx2.load_state(state)
        msgs = await ctx2._inner.get_messages()
        assert len(msgs) == 1
        assert msgs[0].content == "persisted"

    @pytest.mark.asyncio
    async def test_load_state_with_empty_dict_does_not_raise(self):
        sdk = _fake_sdk()
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        # should be graceful when the inner key is absent
        await ctx.load_state({})  # must not raise

    @pytest.mark.asyncio
    async def test_round_trip_preserves_message_count(self):
        sdk = _fake_sdk()
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        await ctx.add_message(UserMessage(content="a", source="user"))
        await ctx.add_message(AssistantMessage(content="b", source="assistant"))
        state = await ctx.save_state()

        ctx2 = SynapShortTermChatContext(sdk, "conv_abc")
        await ctx2.load_state(state)
        msgs = await ctx2._inner.get_messages()
        assert len(msgs) == 2
