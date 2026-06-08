"""Tests for synap_google_adk.short_term — synap_st_instruction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_google_adk.short_term import synap_st_instruction
from synap_integrations_common import SynapIntegrationError


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


class TestValidation:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            synap_st_instruction(None, "conv_abc")  # type: ignore[arg-type]

    def test_requires_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            synap_st_instruction(_fake_sdk(), "")

    def test_rejects_unknown_style(self):
        with pytest.raises(ValueError, match="unsupported style"):
            synap_st_instruction(_fake_sdk(), "conv_abc", style="bogus")

    def test_rejects_invalid_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            synap_st_instruction(_fake_sdk(), "conv_abc", on_error="ignore")  # type: ignore[arg-type]


class TestCacheHit:
    @pytest.mark.asyncio
    async def test_returns_combined_instruction(self):
        sdk = _fake_sdk(formatted="User is a VIP.")
        instr = synap_st_instruction(sdk, "conv_abc", instruction="You are polite.")
        out = await instr(MagicMock())
        assert "<synap_short_term_context>" in out
        assert "User is a VIP." in out
        assert "</synap_short_term_context>" in out
        assert "You are polite." in out
        assert out.index("User is a VIP") < out.index("You are polite")

    @pytest.mark.asyncio
    async def test_passes_conv_and_style(self):
        sdk = _fake_sdk()
        instr = synap_st_instruction(sdk, "conv_abc", style="structured", instruction="X")
        await instr(MagicMock())
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="structured",
        )

    @pytest.mark.asyncio
    async def test_custom_preamble(self):
        sdk = _fake_sdk(formatted="X")
        instr = synap_st_instruction(
            sdk, "conv_abc", instruction="sys",
            preamble_open="[B]", preamble_close="[E]",
        )
        out = await instr(MagicMock())
        assert "[B]" in out and "[E]" in out


class TestEmptyST:
    @pytest.mark.asyncio
    async def test_unavailable_keeps_user_instruction(self):
        sdk = _fake_sdk(formatted=None, available=False)
        instr = synap_st_instruction(sdk, "conv_abc", instruction="Be precise.")
        out = await instr(MagicMock())
        assert out == "Be precise."

    @pytest.mark.asyncio
    async def test_both_empty_returns_empty(self):
        sdk = _fake_sdk(formatted=None, available=False)
        instr = synap_st_instruction(sdk, "conv_abc", instruction="")
        out = await instr(MagicMock())
        assert out == ""


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_fallback_returns_user_instruction_only(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        instr = synap_st_instruction(sdk, "conv_abc", instruction="Stay calm.")
        out = await instr(MagicMock())
        assert out == "Stay calm."

    @pytest.mark.asyncio
    async def test_raise_propagates(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        instr = synap_st_instruction(sdk, "conv_abc", instruction="X", on_error="raise")
        with pytest.raises(SynapIntegrationError):
            await instr(MagicMock())


def test_public_surface_exports():
    import synap_google_adk
    assert hasattr(synap_google_adk, "synap_st_instruction")
    assert "synap_st_instruction" in synap_google_adk.__all__
