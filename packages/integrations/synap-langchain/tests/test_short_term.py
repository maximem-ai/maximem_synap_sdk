"""Tests for synap_langchain.short_term — synap_st_runnable + synap_st_message.

Covers construction validation, cache-hit behaviour, empty-ST safety,
error policy (fallback vs raise), Runnable sync/async invocation, and
public-surface exports.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable

from synap_langchain.short_term import synap_st_message, synap_st_runnable
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_response(formatted: str | None, available: bool):
    resp = MagicMock()
    resp.available = available
    resp.formatted_context = formatted
    return resp


def _fake_sdk(formatted: str | None = "User prefers concise replies.", available: bool = True):
    sdk = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=_make_response(formatted, available)
    )
    return sdk


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestRunnableValidation:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            synap_st_runnable(None, "conv_abc")  # type: ignore[arg-type]

    def test_requires_conversation_id(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            synap_st_runnable(sdk, "")

    def test_rejects_unknown_style(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="unsupported style"):
            synap_st_runnable(sdk, "conv_abc", style="poetic")

    def test_rejects_invalid_on_error(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="on_error"):
            synap_st_runnable(sdk, "conv_abc", on_error="ignore")  # type: ignore[arg-type]


class TestMessageValidation:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            synap_st_message(None, "conv_abc")  # type: ignore[arg-type]

    def test_requires_conversation_id(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            synap_st_message(sdk, "  ")


# ---------------------------------------------------------------------------
# Runnable behaviour
# ---------------------------------------------------------------------------


class TestRunnable:
    def test_returns_runnable_instance(self):
        sdk = _fake_sdk()
        r = synap_st_runnable(sdk, "conv_abc")
        assert isinstance(r, Runnable)

    @pytest.mark.asyncio
    async def test_ainvoke_returns_formatted_string(self):
        sdk = _fake_sdk(formatted="Likes dark mode.")
        r = synap_st_runnable(sdk, "conv_abc")
        result = await r.ainvoke({"messages": []})
        assert result == "Likes dark mode."
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="narrative",
        )

    @pytest.mark.asyncio
    async def test_ainvoke_empty_when_unavailable(self):
        sdk = _fake_sdk(formatted=None, available=False)
        r = synap_st_runnable(sdk, "conv_abc")
        result = await r.ainvoke({})
        assert result == ""

    @pytest.mark.asyncio
    async def test_fallback_returns_empty_on_failure(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        r = synap_st_runnable(sdk, "conv_abc")
        assert await r.ainvoke({}) == ""

    @pytest.mark.asyncio
    async def test_raise_propagates_synap_integration_error(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        r = synap_st_runnable(sdk, "conv_abc", on_error="raise")
        with pytest.raises(SynapIntegrationError):
            await r.ainvoke({})

    @pytest.mark.asyncio
    async def test_passes_style_through(self):
        sdk = _fake_sdk()
        r = synap_st_runnable(sdk, "conv_abc", style="structured")
        await r.ainvoke({})
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="structured",
        )


# ---------------------------------------------------------------------------
# SystemMessage factory behaviour
# ---------------------------------------------------------------------------


class TestSystemMessageFactory:
    @pytest.mark.asyncio
    async def test_returns_system_message_with_combined_content(self):
        sdk = _fake_sdk(formatted="User context here.")
        factory = synap_st_message(sdk, "conv_abc", system="You are helpful.")
        msg = await factory()
        assert isinstance(msg, SystemMessage)
        content = msg.content
        assert "<synap_short_term_context>" in content
        assert "User context here." in content
        assert "</synap_short_term_context>" in content
        assert "You are helpful." in content
        # ST first, user system second
        assert content.index("User context here") < content.index("You are helpful")

    @pytest.mark.asyncio
    async def test_custom_preamble(self):
        sdk = _fake_sdk(formatted="X")
        factory = synap_st_message(
            sdk,
            "conv_abc",
            system="sys",
            preamble_open="[BEGIN]",
            preamble_close="[END]",
        )
        msg = await factory()
        assert "[BEGIN]" in msg.content
        assert "[END]" in msg.content

    @pytest.mark.asyncio
    async def test_no_preamble_raw_concat(self):
        sdk = _fake_sdk(formatted="raw st")
        factory = synap_st_message(
            sdk,
            "conv_abc",
            system="user sys",
            preamble_open=None,
            preamble_close=None,
        )
        msg = await factory()
        assert msg.content == "raw st\n\nuser sys"

    @pytest.mark.asyncio
    async def test_empty_st_keeps_user_system(self):
        sdk = _fake_sdk(formatted=None, available=False)
        factory = synap_st_message(sdk, "conv_abc", system="You are friendly.")
        msg = await factory()
        assert isinstance(msg, SystemMessage)
        assert msg.content == "You are friendly."

    @pytest.mark.asyncio
    async def test_both_empty_returns_none(self):
        sdk = _fake_sdk(formatted=None, available=False)
        factory = synap_st_message(sdk, "conv_abc", system="")
        msg = await factory()
        assert msg is None

    @pytest.mark.asyncio
    async def test_fallback_returns_user_system_only(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        factory = synap_st_message(sdk, "conv_abc", system="Stay calm.")
        msg = await factory()
        assert isinstance(msg, SystemMessage)
        assert msg.content == "Stay calm."

    @pytest.mark.asyncio
    async def test_raise_propagates(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        factory = synap_st_message(sdk, "conv_abc", system="X", on_error="raise")
        with pytest.raises(SynapIntegrationError):
            await factory()


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_surface_exports():
    import synap_langchain

    assert hasattr(synap_langchain, "synap_st_runnable")
    assert hasattr(synap_langchain, "synap_st_message")
    assert "synap_st_runnable" in synap_langchain.__all__
    assert "synap_st_message" in synap_langchain.__all__
