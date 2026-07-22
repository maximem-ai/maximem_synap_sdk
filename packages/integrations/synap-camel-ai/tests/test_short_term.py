"""Tests for synap_st_system_message — short-term context for CAMEL system prompts."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from synap_camel_ai import synap_st_system_message
from synap_integrations_common import SynapIntegrationError


def _unavailable(mock_sdk):
    resp = MagicMock()
    resp.available = False
    resp.formatted_context = ""
    mock_sdk.conversation.context.get_context_for_prompt = AsyncMock(return_value=resp)
    return mock_sdk


# ── validation ───────────────────────────────────────────────────────────────


def test_requires_sdk():
    with pytest.raises(ValueError):
        synap_st_system_message(None, "conv")


def test_requires_conversation_id(mock_sdk):
    with pytest.raises(ValueError):
        synap_st_system_message(mock_sdk, "")


def test_rejects_bad_style(mock_sdk):
    with pytest.raises(ValueError):
        synap_st_system_message(mock_sdk, "conv", style="haiku")


def test_rejects_bad_on_error(mock_sdk):
    with pytest.raises(ValueError):
        synap_st_system_message(mock_sdk, "conv", on_error="explode")


# ── composition ──────────────────────────────────────────────────────────────


def test_prepends_context_above_system_message(mock_sdk):
    out = synap_st_system_message(
        mock_sdk, "conv", system_message="You are a support agent."
    )
    assert "Recent conversation summary" in out
    assert "You are a support agent." in out
    assert out.index("Recent conversation summary") < out.index("You are a support agent.")
    assert "<synap_short_term_context>" in out


def test_tags_dropped_when_preamble_none(mock_sdk):
    out = synap_st_system_message(
        mock_sdk, "conv", system_message="BASE",
        preamble_open=None, preamble_close=None,
    )
    assert "<synap_short_term_context>" not in out
    assert "Recent conversation summary" in out
    assert "BASE" in out


def test_style_flows_to_sdk(mock_sdk):
    synap_st_system_message(mock_sdk, "conv", style="bullet_points")
    assert (
        mock_sdk.conversation.context.get_context_for_prompt.call_args.kwargs["style"]
        == "bullet_points"
    )


# ── empty / unavailable ──────────────────────────────────────────────────────


def test_unavailable_context_returns_bare_system_message(mock_sdk):
    sdk = _unavailable(mock_sdk)
    assert synap_st_system_message(sdk, "conv", system_message="BASE") == "BASE"


def test_unavailable_with_no_system_message_returns_empty(mock_sdk):
    sdk = _unavailable(mock_sdk)
    assert synap_st_system_message(sdk, "conv") == ""


# ── error handling ───────────────────────────────────────────────────────────


def test_fallback_swallows_sdk_failure(failing_sdk):
    out = synap_st_system_message(failing_sdk, "conv", system_message="BASE")
    assert out == "BASE"


def test_raise_propagates_sdk_failure(failing_sdk):
    with pytest.raises(SynapIntegrationError):
        synap_st_system_message(failing_sdk, "conv", system_message="BASE", on_error="raise")
