"""Tests for synap_agno.short_term — synap_st_instructions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_agno.short_term import synap_st_instructions
from synap_integrations_common import SynapIntegrationError


def _make_response(formatted: str | None, available: bool):
    resp = MagicMock()
    resp.available = available
    resp.formatted_context = formatted
    return resp


def _fake_sdk(formatted: str | None = "User prefers Markdown.", available: bool = True):
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
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            synap_st_instructions(_fake_sdk(), "")

    def test_rejects_unknown_style(self):
        with pytest.raises(ValueError, match="unsupported style"):
            synap_st_instructions(_fake_sdk(), "conv_abc", style="bogus")

    def test_rejects_invalid_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            synap_st_instructions(_fake_sdk(), "conv_abc", on_error="ignore")  # type: ignore[arg-type]


class TestCacheHit:
    @pytest.mark.asyncio
    async def test_returns_combined_instructions(self):
        sdk = _fake_sdk(formatted="User is VIP.")
        cb = synap_st_instructions(sdk, "conv_abc", instructions="You are helpful.")
        out = await cb()
        assert "<synap_short_term_context>" in out
        assert "User is VIP." in out
        assert "</synap_short_term_context>" in out
        assert "You are helpful." in out

    @pytest.mark.asyncio
    async def test_passes_conv_and_style(self):
        sdk = _fake_sdk()
        cb = synap_st_instructions(sdk, "conv_abc", style="structured")
        await cb()
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="structured",
        )

    @pytest.mark.asyncio
    async def test_accepts_agno_signature_kwargs(self):
        """Agno introspects signature and passes optional kwargs."""
        sdk = _fake_sdk()
        cb = synap_st_instructions(sdk, "conv_abc", instructions="X")
        # Agno may pass agent / team / session_state / run_context — must not crash
        out = await cb(agent=MagicMock(), team=None, session_state={}, run_context=None)
        assert "X" in out


class TestEmptyST:
    @pytest.mark.asyncio
    async def test_unavailable_keeps_user_instructions(self):
        sdk = _fake_sdk(formatted=None, available=False)
        cb = synap_st_instructions(sdk, "conv_abc", instructions="Be precise.")
        out = await cb()
        assert out == "Be precise."

    @pytest.mark.asyncio
    async def test_both_empty_returns_empty(self):
        sdk = _fake_sdk(formatted=None, available=False)
        cb = synap_st_instructions(sdk, "conv_abc", instructions="")
        assert (await cb()) == ""


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_fallback_returns_user_instructions_only(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        cb = synap_st_instructions(sdk, "conv_abc", instructions="Stay calm.")
        assert (await cb()) == "Stay calm."

    @pytest.mark.asyncio
    async def test_raise_propagates(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        cb = synap_st_instructions(sdk, "conv_abc", instructions="X", on_error="raise")
        with pytest.raises(SynapIntegrationError):
            await cb()


def test_public_surface_exports():
    import synap_agno
    assert hasattr(synap_agno, "synap_st_instructions")
    assert "synap_st_instructions" in synap_agno.__all__
