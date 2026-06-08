"""Tests for synap_haystack.short_term — SynapShortTermContext."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_haystack.short_term import SynapShortTermContext
from synap_integrations_common import SynapIntegrationError


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


class TestValidation:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapShortTermContext(None, "conv_abc")  # type: ignore[arg-type]

    def test_requires_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            SynapShortTermContext(_fake_sdk(), "")

    def test_rejects_unknown_style(self):
        with pytest.raises(ValueError, match="unsupported style"):
            SynapShortTermContext(_fake_sdk(), "conv_abc", style="bogus")

    def test_rejects_invalid_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            SynapShortTermContext(_fake_sdk(), "conv_abc", on_error="ignore")  # type: ignore[arg-type]


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

    def test_passes_conv_and_style(self):
        sdk = _fake_sdk()
        comp = SynapShortTermContext(sdk, "conv_abc", style="bullet_points")
        comp.run()
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="bullet_points",
        )

    def test_custom_preamble(self):
        sdk = _fake_sdk(formatted="X")
        comp = SynapShortTermContext(
            sdk, "conv_abc", system="sys",
            preamble_open="[B]", preamble_close="[E]",
        )
        out = comp.run()
        assert "[B]" in out["synap_st"] and "[E]" in out["synap_st"]


class TestEmptyST:
    def test_unavailable_keeps_user_system(self):
        sdk = _fake_sdk(formatted=None, available=False)
        comp = SynapShortTermContext(sdk, "conv_abc", system="Stay polite.")
        assert comp.run()["synap_st"] == "Stay polite."

    def test_both_empty_returns_empty_string(self):
        sdk = _fake_sdk(formatted=None, available=False)
        comp = SynapShortTermContext(sdk, "conv_abc", system="")
        assert comp.run()["synap_st"] == ""


class TestErrorHandling:
    def test_fallback_returns_user_system_only(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        comp = SynapShortTermContext(sdk, "conv_abc", system="Stay calm.")
        assert comp.run()["synap_st"] == "Stay calm."

    def test_raise_propagates(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        comp = SynapShortTermContext(sdk, "conv_abc", system="X", on_error="raise")
        with pytest.raises(SynapIntegrationError):
            comp.run()


def test_public_surface_exports():
    import synap_haystack
    assert hasattr(synap_haystack, "SynapShortTermContext")
    assert "SynapShortTermContext" in synap_haystack.__all__
