"""Tests for synap_llamaindex.short_term — synap_st_chat_message.

Covers construction validation, cache-hit behaviour, empty-ST safety,
error policy (fallback vs raise), factory call patterns, and public-surface
exports.

Quality contract mirrors the LangGraph adapter (see short_term.py docstring).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from synap_llamaindex.short_term import synap_st_chat_message
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_surface_exports():
    import synap_llamaindex
    assert hasattr(synap_llamaindex, "synap_st_chat_message")
    assert "synap_st_chat_message" in synap_llamaindex.__all__


def test_factory_function_is_callable():
    """synap_st_chat_message is a regular function (not a class/coroutine)."""
    import inspect
    assert callable(synap_st_chat_message)
    assert not inspect.iscoroutinefunction(synap_st_chat_message)


def test_factory_returns_async_callable():
    """synap_st_chat_message returns an async callable."""
    import inspect
    sdk = _fake_sdk()
    factory = synap_st_chat_message(sdk, "conv_abc")
    assert callable(factory)
    assert inspect.iscoroutinefunction(factory)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            synap_st_chat_message(None, "conv_abc")  # type: ignore[arg-type]

    def test_requires_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            synap_st_chat_message(_fake_sdk(), "")

    def test_requires_non_whitespace_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            synap_st_chat_message(_fake_sdk(), "   ")

    def test_rejects_unknown_style(self):
        with pytest.raises(ValueError, match="unsupported style"):
            synap_st_chat_message(_fake_sdk(), "conv_abc", style="bogus")

    def test_rejects_invalid_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            synap_st_chat_message(_fake_sdk(), "conv_abc", on_error="ignore")  # type: ignore[arg-type]

    def test_all_supported_styles_accepted(self):
        for style in ("structured", "narrative", "bullet_points"):
            factory = synap_st_chat_message(_fake_sdk(), "conv_abc", style=style)
            assert callable(factory)

    def test_both_on_error_values_accepted(self):
        for on_error in ("fallback", "raise"):
            factory = synap_st_chat_message(_fake_sdk(), "conv_abc", on_error=on_error)
            assert callable(factory)


# ---------------------------------------------------------------------------
# Cache-hit / happy-path behaviour
# ---------------------------------------------------------------------------


class TestCacheHit:
    @pytest.mark.asyncio
    async def test_returns_system_message_with_combined_content(self):
        """Result is a ChatMessage with role SYSTEM containing ST + user system."""
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
        # ST block must come first
        assert content.index("User is VIP") < content.index("You are helpful")

    @pytest.mark.asyncio
    async def test_passes_conv_id_and_style_to_sdk(self):
        """SDK gets the exact conversation_id and style that were configured."""
        sdk = _fake_sdk()
        factory = synap_st_chat_message(sdk, "conv_abc", style="structured")
        await factory()
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="structured",
        )

    @pytest.mark.asyncio
    async def test_passes_narrative_style_by_default(self):
        """Default style is 'narrative'."""
        sdk = _fake_sdk()
        factory = synap_st_chat_message(sdk, "conv_abc")
        await factory()
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="narrative",
        )

    @pytest.mark.asyncio
    async def test_custom_preamble_tags(self):
        """Custom preamble_open/close are used instead of defaults."""
        sdk = _fake_sdk(formatted="X")
        factory = synap_st_chat_message(
            sdk, "conv_abc", system="sys",
            preamble_open="[B]", preamble_close="[E]",
        )
        msg = await factory()
        assert "[B]" in str(msg.content)
        assert "[E]" in str(msg.content)
        assert "<synap_short_term_context>" not in str(msg.content)

    @pytest.mark.asyncio
    async def test_no_preamble_raw_concat(self):
        """preamble_open=None + preamble_close=None → raw ST + user system."""
        sdk = _fake_sdk(formatted="raw st")
        factory = synap_st_chat_message(
            sdk, "conv_abc", system="user sys",
            preamble_open=None, preamble_close=None,
        )
        msg = await factory()
        assert str(msg.content) == "raw st\n\nuser sys"

    @pytest.mark.asyncio
    async def test_system_only_no_st(self):
        """When ST block is empty but user system is set, return system-only message."""
        sdk = _fake_sdk(formatted="", available=True)
        factory = synap_st_chat_message(sdk, "conv_abc", system="Only user system.")
        msg = await factory()
        assert isinstance(msg, ChatMessage)
        assert str(msg.content) == "Only user system."

    @pytest.mark.asyncio
    async def test_factory_is_reusable(self):
        """Calling the returned factory twice should work (re-invocable)."""
        sdk = _fake_sdk(formatted="ST content.")
        factory = synap_st_chat_message(sdk, "conv_abc", system="sys")
        msg1 = await factory()
        msg2 = await factory()
        assert str(msg1.content) == str(msg2.content)
        assert sdk.conversation.context.get_context_for_prompt.await_count == 2


# ---------------------------------------------------------------------------
# Empty ST safety
# ---------------------------------------------------------------------------


class TestEmptyST:
    @pytest.mark.asyncio
    async def test_unavailable_keeps_user_system(self):
        """available=False → ST block empty, user system returned as-is."""
        sdk = _fake_sdk(formatted=None, available=False)
        factory = synap_st_chat_message(sdk, "conv_abc", system="Be precise.")
        msg = await factory()
        assert isinstance(msg, ChatMessage)
        assert str(msg.content) == "Be precise."

    @pytest.mark.asyncio
    async def test_both_empty_returns_none(self):
        """Both ST block and user system empty → returns None (no blank message)."""
        sdk = _fake_sdk(formatted=None, available=False)
        factory = synap_st_chat_message(sdk, "conv_abc", system="")
        msg = await factory()
        assert msg is None

    @pytest.mark.asyncio
    async def test_st_only_no_user_system(self):
        """Only ST block, no user system → message contains only ST block."""
        sdk = _fake_sdk(formatted="Only ST here.")
        factory = synap_st_chat_message(sdk, "conv_abc", system="")
        msg = await factory()
        assert isinstance(msg, ChatMessage)
        content = str(msg.content)
        assert "Only ST here." in content
        # No blank trailing separator
        assert content == content.strip()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_fallback_returns_user_system_on_sdk_failure(self):
        """on_error='fallback' (default) returns user system text when SDK fails."""
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        factory = synap_st_chat_message(sdk, "conv_abc", system="Stay calm.")
        msg = await factory()
        assert isinstance(msg, ChatMessage)
        assert str(msg.content) == "Stay calm."

    @pytest.mark.asyncio
    async def test_fallback_returns_none_when_both_empty_on_failure(self):
        """Fallback with empty user system and SDK failure → None."""
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        factory = synap_st_chat_message(sdk, "conv_abc", system="")
        msg = await factory()
        assert msg is None

    @pytest.mark.asyncio
    async def test_raise_propagates_synap_integration_error(self):
        """on_error='raise' propagates SynapIntegrationError when SDK fails."""
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        factory = synap_st_chat_message(sdk, "conv_abc", system="X", on_error="raise")
        with pytest.raises(SynapIntegrationError):
            await factory()

    @pytest.mark.asyncio
    async def test_raise_does_not_swallow_error(self):
        """on_error='raise' must propagate; system text must NOT be returned."""
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        factory = synap_st_chat_message(sdk, "conv_abc", system="visible?", on_error="raise")
        with pytest.raises(SynapIntegrationError):
            await factory()

    @pytest.mark.asyncio
    async def test_using_failing_sdk_fixture_with_fallback(self, failing_sdk):
        """failing_sdk fixture verifies fallback works when all SDK calls fail."""
        factory = synap_st_chat_message(failing_sdk, "conv_abc", system="Fallback text.")
        msg = await factory()
        assert isinstance(msg, ChatMessage)
        assert str(msg.content) == "Fallback text."

    @pytest.mark.asyncio
    async def test_using_failing_sdk_fixture_with_raise(self, failing_sdk):
        """failing_sdk fixture verifies raise mode surfaces errors."""
        factory = synap_st_chat_message(failing_sdk, "conv_abc", system="X", on_error="raise")
        with pytest.raises(SynapIntegrationError):
            await factory()
