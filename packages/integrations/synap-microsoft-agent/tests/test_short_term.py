"""Tests for synap_microsoft_agent.short_term — SynapShortTermContextProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_microsoft_agent.short_term import SynapShortTermContextProvider
from synap_integrations_common import SynapIntegrationError


def _make_response(formatted: str | None, available: bool):
    resp = MagicMock()
    resp.available = available
    resp.formatted_context = formatted
    return resp


def _fake_sdk(formatted: str | None = "User has 3 open tickets.", available: bool = True):
    sdk = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=_make_response(formatted, available)
    )
    return sdk


def _fake_context():
    """Mimic MAF's context object exposing extend_instructions(source_id, text)."""
    ctx = MagicMock()
    ctx.extended = []  # captures (source_id, text) tuples

    def _extend(source_id, text):
        ctx.extended.append((source_id, text))

    ctx.extend_instructions = _extend
    return ctx


class TestValidation:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapShortTermContextProvider(None, "conv_abc")  # type: ignore[arg-type]

    def test_requires_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            SynapShortTermContextProvider(_fake_sdk(), "")

    def test_rejects_unknown_style(self):
        with pytest.raises(ValueError, match="unsupported style"):
            SynapShortTermContextProvider(_fake_sdk(), "conv_abc", style="bogus")

    def test_rejects_invalid_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            SynapShortTermContextProvider(  # type: ignore[arg-type]
                _fake_sdk(), "conv_abc", on_error="ignore",
            )


class TestBeforeRun:
    @pytest.mark.asyncio
    async def test_extends_instructions_with_wrapped_st(self):
        sdk = _fake_sdk(formatted="User is VIP.")
        provider = SynapShortTermContextProvider(sdk, "conv_abc")
        ctx = _fake_context()

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        assert len(ctx.extended) == 1
        source_id, text = ctx.extended[0]
        assert source_id == provider.source_id
        assert "<synap_short_term_context>" in text
        assert "User is VIP." in text
        assert "</synap_short_term_context>" in text

    @pytest.mark.asyncio
    async def test_passes_conv_and_style(self):
        sdk = _fake_sdk()
        provider = SynapShortTermContextProvider(sdk, "conv_abc", style="bullet_points")
        await provider.before_run(agent=None, session=None, context=_fake_context(), state={})
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="bullet_points",
        )

    @pytest.mark.asyncio
    async def test_no_preamble_means_raw(self):
        sdk = _fake_sdk(formatted="raw st")
        provider = SynapShortTermContextProvider(
            sdk, "conv_abc", preamble_open=None, preamble_close=None,
        )
        ctx = _fake_context()
        await provider.before_run(agent=None, session=None, context=ctx, state={})
        _, text = ctx.extended[0]
        assert text == "raw st"


class TestEmptyST:
    @pytest.mark.asyncio
    async def test_unavailable_no_extension(self):
        sdk = _fake_sdk(formatted=None, available=False)
        provider = SynapShortTermContextProvider(sdk, "conv_abc")
        ctx = _fake_context()
        await provider.before_run(agent=None, session=None, context=ctx, state={})
        assert ctx.extended == []

    @pytest.mark.asyncio
    async def test_empty_formatted_no_extension(self):
        sdk = _fake_sdk(formatted="   ", available=True)
        provider = SynapShortTermContextProvider(sdk, "conv_abc")
        ctx = _fake_context()
        await provider.before_run(agent=None, session=None, context=ctx, state={})
        assert ctx.extended == []


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_fallback_no_extension_on_failure(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        provider = SynapShortTermContextProvider(sdk, "conv_abc")
        ctx = _fake_context()
        await provider.before_run(agent=None, session=None, context=ctx, state={})
        assert ctx.extended == []

    @pytest.mark.asyncio
    async def test_raise_propagates(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        provider = SynapShortTermContextProvider(sdk, "conv_abc", on_error="raise")
        ctx = _fake_context()
        with pytest.raises(SynapIntegrationError):
            await provider.before_run(agent=None, session=None, context=ctx, state={})


def test_public_surface_exports():
    import synap_microsoft_agent
    assert hasattr(synap_microsoft_agent, "SynapShortTermContextProvider")
    assert "SynapShortTermContextProvider" in synap_microsoft_agent.__all__
