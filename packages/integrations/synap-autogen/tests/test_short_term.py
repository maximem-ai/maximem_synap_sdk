"""Tests for synap_autogen.short_term — SynapShortTermChatContext."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage, SystemMessage, UserMessage

from synap_autogen.short_term import SynapShortTermChatContext
from synap_integrations_common import SynapIntegrationError


def _make_response(formatted: str | None, available: bool):
    resp = MagicMock()
    resp.available = available
    resp.formatted_context = formatted
    return resp


def _fake_sdk(formatted: str | None = "User asked about pricing twice.", available: bool = True):
    sdk = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=_make_response(formatted, available)
    )
    return sdk


async def _seed_inner(ctx):
    await ctx.add_message(UserMessage(content="hi", source="user"))
    await ctx.add_message(AssistantMessage(content="hello!", source="assistant"))


class TestValidation:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapShortTermChatContext(None, "conv_abc")  # type: ignore[arg-type]

    def test_requires_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            SynapShortTermChatContext(_fake_sdk(), "")

    def test_rejects_unknown_style(self):
        with pytest.raises(ValueError, match="unsupported style"):
            SynapShortTermChatContext(_fake_sdk(), "conv_abc", style="bogus")

    def test_rejects_invalid_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            SynapShortTermChatContext(  # type: ignore[arg-type]
                _fake_sdk(), "conv_abc", on_error="ignore"
            )


class TestGetMessages:
    @pytest.mark.asyncio
    async def test_prepends_st_system_message(self):
        sdk = _fake_sdk(formatted="User is VIP.")
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        await _seed_inner(ctx)
        messages = await ctx.get_messages()
        assert len(messages) == 3
        assert isinstance(messages[0], SystemMessage)
        assert "<synap_short_term_context>" in messages[0].content
        assert "User is VIP." in messages[0].content
        # Inner messages preserved in order
        assert isinstance(messages[1], UserMessage)
        assert isinstance(messages[2], AssistantMessage)

    @pytest.mark.asyncio
    async def test_passes_conv_and_style(self):
        sdk = _fake_sdk()
        ctx = SynapShortTermChatContext(sdk, "conv_abc", style="bullet_points")
        await ctx.get_messages()
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="bullet_points",
        )

    @pytest.mark.asyncio
    async def test_no_preamble_means_raw(self):
        sdk = _fake_sdk(formatted="raw st")
        ctx = SynapShortTermChatContext(
            sdk, "conv_abc",
            preamble_open=None, preamble_close=None,
        )
        messages = await ctx.get_messages()
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "raw st"


class TestEmptyST:
    @pytest.mark.asyncio
    async def test_unavailable_returns_inner_messages_only(self):
        sdk = _fake_sdk(formatted=None, available=False)
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        await _seed_inner(ctx)
        messages = await ctx.get_messages()
        # No SystemMessage prepended
        assert all(not isinstance(m, SystemMessage) for m in messages)
        assert len(messages) == 2


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_fallback_returns_inner_messages(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        ctx = SynapShortTermChatContext(sdk, "conv_abc")
        await _seed_inner(ctx)
        messages = await ctx.get_messages()
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_raise_propagates(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        ctx = SynapShortTermChatContext(sdk, "conv_abc", on_error="raise")
        await _seed_inner(ctx)
        with pytest.raises(SynapIntegrationError):
            await ctx.get_messages()


class TestInnerDelegation:
    @pytest.mark.asyncio
    async def test_add_and_clear_delegate_to_inner(self):
        sdk = _fake_sdk()
        inner = UnboundedChatCompletionContext()
        ctx = SynapShortTermChatContext(sdk, "conv_abc", inner=inner)
        await ctx.add_message(UserMessage(content="hi", source="user"))
        # Inner should reflect the add
        inner_msgs = await inner.get_messages()
        assert len(inner_msgs) == 1
        await ctx.clear()
        inner_msgs = await inner.get_messages()
        assert inner_msgs == []


def test_public_surface_exports():
    import synap_autogen
    assert hasattr(synap_autogen, "SynapShortTermChatContext")
    assert "SynapShortTermChatContext" in synap_autogen.__all__
