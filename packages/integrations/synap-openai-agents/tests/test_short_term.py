"""Tests for synap_openai_agents.short_term — synap_st_instructions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_openai_agents.short_term import synap_st_instructions
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
            synap_st_instructions(None, "conv_abc")  # type: ignore[arg-type]

    def test_requires_conversation_id(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            synap_st_instructions(sdk, "")

    def test_rejects_unknown_style(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="unsupported style"):
            synap_st_instructions(sdk, "conv_abc", style="bogus")

    def test_rejects_invalid_on_error(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="on_error"):
            synap_st_instructions(sdk, "conv_abc", on_error="ignore")  # type: ignore[arg-type]


class TestCacheHit:
    @pytest.mark.asyncio
    async def test_returns_combined_instructions(self):
        sdk = _fake_sdk(formatted="User likes Markdown.")
        instr = synap_st_instructions(sdk, "conv_abc", system="You are a coding agent.")
        result = await instr(MagicMock(), MagicMock())
        assert "<synap_short_term_context>" in result
        assert "User likes Markdown." in result
        assert "</synap_short_term_context>" in result
        assert "You are a coding agent." in result
        assert result.index("User likes Markdown") < result.index("coding agent")

    @pytest.mark.asyncio
    async def test_passes_conv_and_style(self):
        sdk = _fake_sdk()
        instr = synap_st_instructions(sdk, "conv_abc", style="bullet_points", system="X")
        await instr(MagicMock(), MagicMock())
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="bullet_points",
        )

    @pytest.mark.asyncio
    async def test_custom_preamble(self):
        sdk = _fake_sdk(formatted="X")
        instr = synap_st_instructions(
            sdk, "conv_abc", system="sys",
            preamble_open="[BEGIN]", preamble_close="[END]",
        )
        out = await instr(MagicMock(), MagicMock())
        assert "[BEGIN]" in out and "[END]" in out


class TestEmptyST:
    @pytest.mark.asyncio
    async def test_unavailable_keeps_user_system(self):
        sdk = _fake_sdk(formatted=None, available=False)
        instr = synap_st_instructions(sdk, "conv_abc", system="You are friendly.")
        out = await instr(MagicMock(), MagicMock())
        assert out == "You are friendly."

    @pytest.mark.asyncio
    async def test_both_empty_returns_empty_string(self):
        sdk = _fake_sdk(formatted=None, available=False)
        instr = synap_st_instructions(sdk, "conv_abc", system="")
        out = await instr(MagicMock(), MagicMock())
        assert out == ""


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_fallback_returns_user_system_only(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        instr = synap_st_instructions(sdk, "conv_abc", system="Stay calm.")
        out = await instr(MagicMock(), MagicMock())
        assert out == "Stay calm."

    @pytest.mark.asyncio
    async def test_raise_propagates(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        instr = synap_st_instructions(sdk, "conv_abc", system="X", on_error="raise")
        with pytest.raises(SynapIntegrationError):
            await instr(MagicMock(), MagicMock())


def test_public_surface_exports():
    import synap_openai_agents
    assert hasattr(synap_openai_agents, "synap_st_instructions")
    assert "synap_st_instructions" in synap_openai_agents.__all__
