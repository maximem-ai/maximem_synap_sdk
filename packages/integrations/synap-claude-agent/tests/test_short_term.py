"""Tests for synap_claude_agent.short_term — create_synap_st_hook."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_claude_agent.short_term import create_synap_st_hook
from synap_integrations_common import SynapIntegrationError


def _make_response(formatted: str | None, available: bool):
    resp = MagicMock()
    resp.available = available
    resp.formatted_context = formatted
    return resp


def _fake_sdk(formatted: str | None = "User likes detailed answers.", available: bool = True):
    sdk = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=_make_response(formatted, available)
    )
    return sdk


def _hook_callable(hooks_dict):
    """Pull the registered async callable out of the hooks dict."""
    matchers = hooks_dict["UserPromptSubmit"]
    assert len(matchers) == 1
    # HookMatcher.hooks is a list of callables
    matcher = matchers[0]
    callables = matcher.hooks
    assert len(callables) == 1
    return callables[0]


class TestValidation:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            create_synap_st_hook(None, "conv_abc")  # type: ignore[arg-type]

    def test_requires_conversation_id(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            create_synap_st_hook(sdk, "")

    def test_rejects_unknown_style(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="unsupported style"):
            create_synap_st_hook(sdk, "conv_abc", style="bogus")


class TestCacheHit:
    @pytest.mark.asyncio
    async def test_returns_additional_context_with_st(self):
        sdk = _fake_sdk(formatted="User name is Maya.")
        hooks = create_synap_st_hook(sdk, "conv_abc")
        cb = _hook_callable(hooks)
        out = await cb({"prompt": "hi"}, None, MagicMock())
        assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        ctx = out["hookSpecificOutput"]["additionalContext"]
        assert "<synap_short_term_context>" in ctx
        assert "User name is Maya." in ctx
        assert "</synap_short_term_context>" in ctx

    @pytest.mark.asyncio
    async def test_passes_explicit_conv_to_sdk(self):
        sdk = _fake_sdk()
        cb = _hook_callable(create_synap_st_hook(sdk, "conv_abc", style="structured"))
        await cb({"prompt": "hi"}, None, MagicMock())
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="structured",
        )

    @pytest.mark.asyncio
    async def test_custom_preamble(self):
        sdk = _fake_sdk(formatted="X")
        cb = _hook_callable(
            create_synap_st_hook(
                sdk, "conv_abc",
                preamble_open="[BEGIN]",
                preamble_close="[END]",
            )
        )
        out = await cb({"prompt": "p"}, None, MagicMock())
        ctx = out["hookSpecificOutput"]["additionalContext"]
        assert "[BEGIN]" in ctx and "[END]" in ctx
        assert "<synap_short_term_context>" not in ctx

    @pytest.mark.asyncio
    async def test_no_preamble_means_raw(self):
        sdk = _fake_sdk(formatted="raw st")
        cb = _hook_callable(
            create_synap_st_hook(
                sdk, "conv_abc",
                preamble_open=None, preamble_close=None,
            )
        )
        out = await cb({"prompt": "p"}, None, MagicMock())
        assert out["hookSpecificOutput"]["additionalContext"] == "raw st"


class TestEmptyST:
    @pytest.mark.asyncio
    async def test_unavailable_returns_empty_dict(self):
        sdk = _fake_sdk(formatted=None, available=False)
        cb = _hook_callable(create_synap_st_hook(sdk, "conv_abc"))
        out = await cb({"prompt": "hi"}, None, MagicMock())
        assert out == {}

    @pytest.mark.asyncio
    async def test_blank_formatted_treated_as_unavailable(self):
        sdk = _fake_sdk(formatted="   ", available=True)
        cb = _hook_callable(create_synap_st_hook(sdk, "conv_abc"))
        out = await cb({"prompt": "hi"}, None, MagicMock())
        assert out == {}


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_fallback_returns_empty_on_sdk_failure(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        cb = _hook_callable(create_synap_st_hook(sdk, "conv_abc"))
        out = await cb({"prompt": "hi"}, None, MagicMock())
        assert out == {}

    @pytest.mark.asyncio
    async def test_raise_propagates(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        cb = _hook_callable(
            create_synap_st_hook(sdk, "conv_abc", on_error="raise")
        )
        with pytest.raises(SynapIntegrationError):
            await cb({"prompt": "hi"}, None, MagicMock())


def test_public_surface_exports():
    import synap_claude_agent
    assert hasattr(synap_claude_agent, "create_synap_st_hook")
    assert "create_synap_st_hook" in synap_claude_agent.__all__
