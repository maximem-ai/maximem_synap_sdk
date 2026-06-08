"""Tests for synap_llamaindex.short_term — synap_st_chat_message."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from synap_llamaindex.short_term import synap_st_chat_message
from synap_integrations_common import SynapIntegrationError


def _make_response(formatted: str | None, available: bool):
    resp = MagicMock()
    resp.available = available
    resp.formatted_context = formatted
    return resp


def _fake_sdk(formatted: str | None = "User likes brevity.", available: bool = True):
    sdk = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=_make_response(formatted, available)
    )
    return sdk


class TestValidation:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            synap_st_chat_message(None, "conv_abc")  # type: ignore[arg-type]

    def test_requires_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            synap_st_chat_message(_fake_sdk(), "")

    def test_rejects_unknown_style(self):
        with pytest.raises(ValueError, match="unsupported style"):
            synap_st_chat_message(_fake_sdk(), "conv_abc", style="bogus")

    def test_rejects_invalid_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            synap_st_chat_message(_fake_sdk(), "conv_abc", on_error="ignore")  # type: ignore[arg-type]


class TestCacheHit:
    @pytest.mark.asyncio
    async def test_returns_system_message_with_combined_content(self):
        sdk = _fake_sdk(formatted="User is VIP.")
        factory = synap_st_chat_message(sdk, "conv_abc", system="You are helpful.")
        msg = await factory()
        assert isinstance(msg, ChatMessage)
        assert msg.role == MessageRole.SYSTEM
        content = str(msg.content)
        assert "<synap_short_term_context>" in content
        assert "User is VIP." in content
        assert "</synap_short_term_context>" in content
        assert "You are helpful." in content
        assert content.index("User is VIP") < content.index("You are helpful")

    @pytest.mark.asyncio
    async def test_passes_conv_and_style(self):
        sdk = _fake_sdk()
        factory = synap_st_chat_message(sdk, "conv_abc", style="structured")
        await factory()
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="structured",
        )

    @pytest.mark.asyncio
    async def test_custom_preamble(self):
        sdk = _fake_sdk(formatted="X")
        factory = synap_st_chat_message(
            sdk, "conv_abc", system="sys",
            preamble_open="[B]", preamble_close="[E]",
        )
        msg = await factory()
        assert "[B]" in str(msg.content) and "[E]" in str(msg.content)


class TestEmptyST:
    @pytest.mark.asyncio
    async def test_unavailable_keeps_user_system(self):
        sdk = _fake_sdk(formatted=None, available=False)
        factory = synap_st_chat_message(sdk, "conv_abc", system="Be precise.")
        msg = await factory()
        assert isinstance(msg, ChatMessage)
        assert str(msg.content) == "Be precise."

    @pytest.mark.asyncio
    async def test_both_empty_returns_none(self):
        sdk = _fake_sdk(formatted=None, available=False)
        factory = synap_st_chat_message(sdk, "conv_abc", system="")
        msg = await factory()
        assert msg is None


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_fallback_returns_user_system_only(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        factory = synap_st_chat_message(sdk, "conv_abc", system="Stay calm.")
        msg = await factory()
        assert isinstance(msg, ChatMessage)
        assert str(msg.content) == "Stay calm."

    @pytest.mark.asyncio
    async def test_raise_propagates(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        factory = synap_st_chat_message(sdk, "conv_abc", system="X", on_error="raise")
        with pytest.raises(SynapIntegrationError):
            await factory()


def test_public_surface_exports():
    import synap_llamaindex
    assert hasattr(synap_llamaindex, "synap_st_chat_message")
    assert "synap_st_chat_message" in synap_llamaindex.__all__
