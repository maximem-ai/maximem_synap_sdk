"""Tests for synap_haystack.short_term — SynapShortTermContext.

Quality contract:
- conversation_id required + explicit at construction.
- on_error="fallback" (default): SDK failure emits bare system string (or "").
- on_error="raise": SDK failure propagates as SynapIntegrationError.
- Empty ST never wipes user system text.
- All three styles (structured / narrative / bullet_points) are accepted.
- Custom preamble_open / preamble_close work; both None → no wrapper tags.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_haystack.short_term import SynapShortTermContext
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(formatted: str | None, available: bool):
    resp = MagicMock()
    resp.available = available
    resp.formatted_context = formatted
    return resp


def _fake_sdk(formatted: str | None = "Customer is on Pro plan.", available: bool = True):
    sdk = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=_make_response(formatted, available)
    )
    return sdk


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapShortTermContext(None, "conv_abc")  # type: ignore[arg-type]

    def test_requires_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            SynapShortTermContext(_fake_sdk(), "")

    def test_rejects_whitespace_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            SynapShortTermContext(_fake_sdk(), "   ")

    def test_rejects_unknown_style(self):
        with pytest.raises(ValueError, match="unsupported style"):
            SynapShortTermContext(_fake_sdk(), "conv_abc", style="bogus")

    def test_rejects_invalid_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            SynapShortTermContext(_fake_sdk(), "conv_abc", on_error="ignore")  # type: ignore[arg-type]

    def test_accepts_all_valid_styles(self):
        for style in ("structured", "narrative", "bullet_points"):
            comp = SynapShortTermContext(_fake_sdk(), "conv_abc", style=style)
            assert comp.style == style

    def test_default_style_is_narrative(self):
        comp = SynapShortTermContext(_fake_sdk(), "conv_abc")
        assert comp.style == "narrative"

    def test_default_on_error_is_fallback(self):
        comp = SynapShortTermContext(_fake_sdk(), "conv_abc")
        assert comp.on_error == "fallback"

    def test_accepts_on_error_raise(self):
        comp = SynapShortTermContext(_fake_sdk(), "conv_abc", on_error="raise")
        assert comp.on_error == "raise"


# ---------------------------------------------------------------------------
# run — happy paths
# ---------------------------------------------------------------------------


class TestRun:
    def test_returns_combined_synap_st(self):
        sdk = _fake_sdk(formatted="User is happy.")
        comp = SynapShortTermContext(sdk, "conv_abc", system="You are helpful.")
        out = comp.run()
        assert "synap_st" in out
        content = out["synap_st"]
        assert "<synap_short_term_context>" in content
        assert "User is happy." in content
        assert "</synap_short_term_context>" in content
        assert "You are helpful." in content
        assert content.index("User is happy") < content.index("You are helpful")

    def test_passes_conv_and_style_to_sdk(self):
        sdk = _fake_sdk()
        comp = SynapShortTermContext(sdk, "conv_abc", style="bullet_points")
        comp.run()
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="bullet_points",
        )

    def test_passes_structured_style(self):
        sdk = _fake_sdk(formatted="Structured output here.")
        comp = SynapShortTermContext(sdk, "conv_abc", style="structured")
        comp.run()
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="structured",
        )

    def test_custom_preamble_tags(self):
        sdk = _fake_sdk(formatted="X")
        comp = SynapShortTermContext(
            sdk, "conv_abc", system="sys",
            preamble_open="[B]", preamble_close="[E]",
        )
        out = comp.run()
        assert "[B]" in out["synap_st"]
        assert "[E]" in out["synap_st"]

    def test_no_preamble_raw_concat(self):
        """Both preamble tags None → ST concatenated directly, no wrapper."""
        sdk = _fake_sdk(formatted="raw st")
        comp = SynapShortTermContext(
            sdk, "conv_abc", system="user sys",
            preamble_open=None, preamble_close=None,
        )
        out = comp.run()
        assert out["synap_st"] == "raw st\n\nuser sys"

    def test_no_preamble_st_only(self):
        sdk = _fake_sdk(formatted="just st")
        comp = SynapShortTermContext(
            sdk, "conv_abc", system="",
            preamble_open=None, preamble_close=None,
        )
        out = comp.run()
        assert out["synap_st"] == "just st"

    def test_st_only_no_system(self):
        sdk = _fake_sdk(formatted="ST block")
        comp = SynapShortTermContext(sdk, "conv_abc", system="")
        out = comp.run()
        assert "ST block" in out["synap_st"]
        assert "<synap_short_term_context>" in out["synap_st"]
        # No trailing empty double-newline
        assert not out["synap_st"].endswith("\n\n")


# ---------------------------------------------------------------------------
# run — empty / unavailable ST
# ---------------------------------------------------------------------------


class TestEmptyST:
    def test_unavailable_keeps_user_system(self):
        sdk = _fake_sdk(formatted=None, available=False)
        comp = SynapShortTermContext(sdk, "conv_abc", system="Stay polite.")
        assert comp.run()["synap_st"] == "Stay polite."

    def test_both_empty_returns_empty_string(self):
        sdk = _fake_sdk(formatted=None, available=False)
        comp = SynapShortTermContext(sdk, "conv_abc", system="")
        assert comp.run()["synap_st"] == ""

    def test_available_false_with_non_none_formatted_ignored(self):
        """Even if formatted_context has text, available=False should emit no ST."""
        sdk = _fake_sdk(formatted="some content", available=False)
        comp = SynapShortTermContext(sdk, "conv_abc", system="Sys.")
        out = comp.run()["synap_st"]
        # available=False → st_block not applied
        assert out == "Sys."

    def test_whitespace_only_formatted_context_collapsed(self):
        sdk = _fake_sdk(formatted="   ", available=True)
        comp = SynapShortTermContext(sdk, "conv_abc", system="Sys.")
        out = comp.run()["synap_st"]
        # Stripped whitespace is falsy → no ST block
        assert out == "Sys."


# ---------------------------------------------------------------------------
# run — error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_fallback_returns_user_system_only(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        comp = SynapShortTermContext(sdk, "conv_abc", system="Stay calm.")
        assert comp.run()["synap_st"] == "Stay calm."

    def test_fallback_returns_empty_when_no_system(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        comp = SynapShortTermContext(sdk, "conv_abc", system="")
        assert comp.run()["synap_st"] == ""

    def test_raise_propagates_synap_integration_error(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        comp = SynapShortTermContext(sdk, "conv_abc", system="X", on_error="raise")
        with pytest.raises(SynapIntegrationError):
            comp.run()

    def test_failing_sdk_fallback(self, failing_sdk):
        comp = SynapShortTermContext(failing_sdk, "conv_abc", system="Fallback sys.")
        out = comp.run()["synap_st"]
        assert out == "Fallback sys."

    def test_failing_sdk_raise_mode(self, failing_sdk):
        comp = SynapShortTermContext(failing_sdk, "conv_abc", on_error="raise")
        with pytest.raises(SynapIntegrationError):
            comp.run()


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_surface_exports():
    import synap_haystack
    assert hasattr(synap_haystack, "SynapShortTermContext")
    assert "SynapShortTermContext" in synap_haystack.__all__
