"""Tests for synap_crewai.short_term — build_synap_st_backstory.

Covers:
- Construction validation (sdk, conversation_id, style, on_error)
- Happy-path: ST available → combined backstory with preamble tags
- SDK passthrough: correct args forwarded
- Custom preamble tags (including no-preamble mode)
- Empty / unavailable ST: no-op, returns base_backstory
- All three supported styles
- Error policy: fallback returns base_backstory; raise propagates SynapIntegrationError
- Public surface: __all__ exports
- failing_sdk: SDK error degrades gracefully (fallback) or raises (raise)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_crewai.short_term import build_synap_st_backstory
from synap_integrations_common import SynapIntegrationError
from synap_integrations_common.testing import failing_sdk  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Validation — invalid arguments must raise ValueError immediately
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.asyncio
    async def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            await build_synap_st_backstory(None, "conv_abc")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_requires_non_empty_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            await build_synap_st_backstory(_fake_sdk(), "")

    @pytest.mark.asyncio
    async def test_requires_non_whitespace_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            await build_synap_st_backstory(_fake_sdk(), "   ")

    @pytest.mark.asyncio
    async def test_rejects_unknown_style(self):
        with pytest.raises(ValueError, match="unsupported style"):
            await build_synap_st_backstory(_fake_sdk(), "conv_abc", style="poetic")

    @pytest.mark.asyncio
    async def test_rejects_invalid_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            await build_synap_st_backstory(
                _fake_sdk(), "conv_abc", on_error="ignore"  # type: ignore[arg-type]
            )

    @pytest.mark.asyncio
    async def test_validation_fires_before_sdk_call(self):
        """Validation must short-circuit — SDK must not be called when args are invalid."""
        sdk = _fake_sdk()
        with pytest.raises(ValueError):
            await build_synap_st_backstory(sdk, "", style="structured")
        sdk.conversation.context.get_context_for_prompt.assert_not_awaited()


# ---------------------------------------------------------------------------
# Happy path — ST available
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_returns_combined_backstory_with_preamble(self):
        sdk = _fake_sdk(formatted="User is on the Pro plan.")
        out = await build_synap_st_backstory(
            sdk, "conv_abc",
            base_backstory="You are a customer-support specialist.",
        )
        assert "<synap_short_term_context>" in out
        assert "User is on the Pro plan." in out
        assert "</synap_short_term_context>" in out
        assert "customer-support specialist" in out

    @pytest.mark.asyncio
    async def test_st_block_precedes_base_backstory(self):
        """ST content MUST appear before the base backstory."""
        sdk = _fake_sdk(formatted="Recent context")
        out = await build_synap_st_backstory(
            sdk, "conv_abc", base_backstory="You are helpful."
        )
        assert out.index("Recent context") < out.index("You are helpful")

    @pytest.mark.asyncio
    async def test_returns_string_type(self):
        sdk = _fake_sdk(formatted="Context here.")
        result = await build_synap_st_backstory(sdk, "conv_abc")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_returns_non_empty_with_st_available(self):
        sdk = _fake_sdk(formatted="Some context.")
        result = await build_synap_st_backstory(sdk, "conv_abc")
        assert result.strip() != ""

    @pytest.mark.asyncio
    async def test_passes_conversation_id_to_sdk(self):
        sdk = _fake_sdk()
        await build_synap_st_backstory(sdk, "conv_xyz123")
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_xyz123",
            style="narrative",
        )

    @pytest.mark.asyncio
    async def test_passes_style_to_sdk(self):
        sdk = _fake_sdk()
        await build_synap_st_backstory(sdk, "conv_abc", style="bullet_points")
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="bullet_points",
        )

    @pytest.mark.asyncio
    async def test_all_three_styles_accepted(self):
        for style in ("narrative", "structured", "bullet_points"):
            sdk = _fake_sdk()
            result = await build_synap_st_backstory(sdk, "conv_abc", style=style)
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_default_style_is_narrative(self):
        sdk = _fake_sdk()
        await build_synap_st_backstory(sdk, "conv_abc")
        call_kwargs = sdk.conversation.context.get_context_for_prompt.call_args.kwargs
        assert call_kwargs["style"] == "narrative"


# ---------------------------------------------------------------------------
# Preamble customisation
# ---------------------------------------------------------------------------


class TestPreamble:
    @pytest.mark.asyncio
    async def test_default_preamble_tags_present(self):
        sdk = _fake_sdk(formatted="context text")
        out = await build_synap_st_backstory(sdk, "conv_abc")
        assert "<synap_short_term_context>" in out
        assert "</synap_short_term_context>" in out

    @pytest.mark.asyncio
    async def test_custom_preamble_open_close(self):
        sdk = _fake_sdk(formatted="X")
        out = await build_synap_st_backstory(
            sdk, "conv_abc",
            base_backstory="sys",
            preamble_open="[B]",
            preamble_close="[E]",
        )
        assert "[B]" in out
        assert "[E]" in out
        assert "<synap_short_term_context>" not in out

    @pytest.mark.asyncio
    async def test_no_preamble_raw_concat(self):
        sdk = _fake_sdk(formatted="raw st")
        out = await build_synap_st_backstory(
            sdk, "conv_abc",
            base_backstory="user sys",
            preamble_open=None,
            preamble_close=None,
        )
        # Both parts present, no XML tags
        assert "raw st" in out
        assert "user sys" in out
        assert "<synap_short_term_context>" not in out

    @pytest.mark.asyncio
    async def test_no_preamble_ordering(self):
        """Even without tags, ST must appear before base_backstory."""
        sdk = _fake_sdk(formatted="st content")
        out = await build_synap_st_backstory(
            sdk, "conv_abc",
            base_backstory="base content",
            preamble_open=None,
            preamble_close=None,
        )
        assert out.index("st content") < out.index("base content")


# ---------------------------------------------------------------------------
# Empty / unavailable ST — must be a no-op on base_backstory
# ---------------------------------------------------------------------------


class TestEmptyST:
    @pytest.mark.asyncio
    async def test_unavailable_returns_base_backstory_unchanged(self):
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

    @pytest.mark.asyncio
    async def test_empty_formatted_context_is_noop(self):
        """An available response with empty formatted_context == no ST block."""
        sdk = _fake_sdk(formatted="   ", available=True)
        out = await build_synap_st_backstory(
            sdk, "conv_abc", base_backstory="Base."
        )
        # Empty ST → only base backstory
        assert out == "Base."
        assert "<synap_short_term_context>" not in out

    @pytest.mark.asyncio
    async def test_st_not_inserted_when_empty_even_with_preamble(self):
        sdk = _fake_sdk(formatted=None, available=False)
        out = await build_synap_st_backstory(
            sdk, "conv_abc",
            base_backstory="Keep base.",
            preamble_open="<ctx>",
            preamble_close="</ctx>",
        )
        assert "<ctx>" not in out
        assert "Keep base." in out

    @pytest.mark.asyncio
    async def test_st_only_no_base_returns_just_st(self):
        sdk = _fake_sdk(formatted="ST context.")
        out = await build_synap_st_backstory(sdk, "conv_abc", base_backstory="")
        assert "ST context." in out
        assert "base_backstory" not in out  # literal string not present


# ---------------------------------------------------------------------------
# Error handling — fallback and raise modes
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_fallback_returns_base_backstory_only_on_runtime_error(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        out = await build_synap_st_backstory(
            sdk, "conv_abc", base_backstory="Stay calm."
        )
        assert out == "Stay calm."

    @pytest.mark.asyncio
    async def test_fallback_returns_empty_string_when_no_base_backstory(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        out = await build_synap_st_backstory(sdk, "conv_abc", base_backstory="")
        assert out == ""

    @pytest.mark.asyncio
    async def test_raise_propagates_synap_integration_error(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        with pytest.raises(SynapIntegrationError):
            await build_synap_st_backstory(
                sdk, "conv_abc", base_backstory="X", on_error="raise"
            )

    @pytest.mark.asyncio
    async def test_raise_mode_does_not_return(self):
        """Ensure on_error='raise' never silently swallows exceptions."""
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("network timeout")
        )
        raised = False
        try:
            await build_synap_st_backstory(
                sdk, "conv_abc", on_error="raise"
            )
        except SynapIntegrationError:
            raised = True
        assert raised, "Expected SynapIntegrationError to be raised"

    @pytest.mark.asyncio
    async def test_fallback_is_default_on_error_mode(self):
        """Default on_error must be 'fallback' — no exception raised."""
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("sdk down")
        )
        # Must NOT raise
        result = await build_synap_st_backstory(
            sdk, "conv_abc", base_backstory="Fallback backstory."
        )
        assert result == "Fallback backstory."

    @pytest.mark.asyncio
    async def test_failing_sdk_fallback_returns_base(self, failing_sdk):
        """failing_sdk fixture: SDK raises RuntimeError — fallback returns base."""
        out = await build_synap_st_backstory(
            failing_sdk, "conv_abc", base_backstory="My backstory."
        )
        assert out == "My backstory."

    @pytest.mark.asyncio
    async def test_failing_sdk_raise_propagates(self, failing_sdk):
        """failing_sdk fixture: SDK raises RuntimeError — raise mode propagates."""
        with pytest.raises(SynapIntegrationError):
            await build_synap_st_backstory(
                failing_sdk, "conv_abc", on_error="raise"
            )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_surface_exports_build_synap_st_backstory():
    import synap_crewai
    assert hasattr(synap_crewai, "build_synap_st_backstory")
    assert "build_synap_st_backstory" in synap_crewai.__all__


def test_public_surface_exports_storage_backend():
    import synap_crewai
    assert hasattr(synap_crewai, "SynapStorageBackend")
    assert "SynapStorageBackend" in synap_crewai.__all__


def test_build_synap_st_backstory_is_coroutine():
    """build_synap_st_backstory must be an async function."""
    import asyncio
    sdk = _fake_sdk()
    result = build_synap_st_backstory(sdk, "conv_abc")
    assert asyncio.iscoroutine(result)
    # Clean up the unawaited coroutine
    result.close()
