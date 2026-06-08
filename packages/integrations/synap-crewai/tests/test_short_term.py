"""Tests for synap_crewai.short_term — build_synap_st_backstory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_crewai.short_term import build_synap_st_backstory
from synap_integrations_common import SynapIntegrationError


def _make_response(formatted: str | None, available: bool):
    resp = MagicMock()
    resp.available = available
    resp.formatted_context = formatted
    return resp


def _fake_sdk(formatted: str | None = "User: VIP customer.", available: bool = True):
    sdk = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=_make_response(formatted, available)
    )
    return sdk


class TestValidation:
    @pytest.mark.asyncio
    async def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            await build_synap_st_backstory(None, "conv_abc")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_requires_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            await build_synap_st_backstory(_fake_sdk(), "")

    @pytest.mark.asyncio
    async def test_rejects_unknown_style(self):
        with pytest.raises(ValueError, match="unsupported style"):
            await build_synap_st_backstory(_fake_sdk(), "conv_abc", style="bogus")

    @pytest.mark.asyncio
    async def test_rejects_invalid_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            await build_synap_st_backstory(
                _fake_sdk(), "conv_abc", on_error="ignore"  # type: ignore[arg-type]
            )


class TestCacheHit:
    @pytest.mark.asyncio
    async def test_returns_combined_backstory(self):
        sdk = _fake_sdk(formatted="User is on the Pro plan.")
        out = await build_synap_st_backstory(
            sdk, "conv_abc",
            base_backstory="You are a customer-support specialist.",
        )
        assert "<synap_short_term_context>" in out
        assert "User is on the Pro plan." in out
        assert "</synap_short_term_context>" in out
        assert "customer-support specialist" in out
        assert out.index("User is on the Pro plan") < out.index("customer-support")

    @pytest.mark.asyncio
    async def test_passes_conv_and_style(self):
        sdk = _fake_sdk()
        await build_synap_st_backstory(sdk, "conv_abc", style="bullet_points")
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="bullet_points",
        )

    @pytest.mark.asyncio
    async def test_custom_preamble(self):
        sdk = _fake_sdk(formatted="X")
        out = await build_synap_st_backstory(
            sdk, "conv_abc",
            base_backstory="sys",
            preamble_open="[B]", preamble_close="[E]",
        )
        assert "[B]" in out and "[E]" in out


class TestEmptyST:
    @pytest.mark.asyncio
    async def test_unavailable_keeps_base_backstory(self):
        sdk = _fake_sdk(formatted=None, available=False)
        out = await build_synap_st_backstory(
            sdk, "conv_abc", base_backstory="Be polite."
        )
        assert out == "Be polite."

    @pytest.mark.asyncio
    async def test_both_empty_returns_empty_string(self):
        sdk = _fake_sdk(formatted=None, available=False)
        out = await build_synap_st_backstory(sdk, "conv_abc", base_backstory="")
        assert out == ""


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_fallback_returns_base_backstory_only(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        out = await build_synap_st_backstory(
            sdk, "conv_abc", base_backstory="Stay calm."
        )
        assert out == "Stay calm."

    @pytest.mark.asyncio
    async def test_raise_propagates(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        with pytest.raises(SynapIntegrationError):
            await build_synap_st_backstory(
                sdk, "conv_abc", base_backstory="X", on_error="raise"
            )


def test_public_surface_exports():
    import synap_crewai
    assert hasattr(synap_crewai, "build_synap_st_backstory")
    assert "build_synap_st_backstory" in synap_crewai.__all__
