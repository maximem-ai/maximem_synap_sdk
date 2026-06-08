"""Tests for synap_pydantic_ai.short_term — register_synap_st_system_prompt.

We exercise the registered callback directly by capturing what
``@agent.system_prompt`` is called with, rather than spinning up a real
Pydantic AI Agent (which would require a model client). The captured
callable is then invoked against a fake RunContext + SynapDeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_integrations_common import SynapIntegrationError
from synap_pydantic_ai.deps import SynapDeps
from synap_pydantic_ai.short_term import register_synap_st_system_prompt


def _make_response(formatted: str | None, available: bool):
    resp = MagicMock()
    resp.available = available
    resp.formatted_context = formatted
    return resp


def _fake_sdk(formatted: str | None = "User likes Markdown.", available: bool = True):
    sdk = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=_make_response(formatted, available)
    )
    return sdk


@dataclass
class _FakeRunContext:
    """Stand-in for pydantic_ai.RunContext. Only ``deps`` is read."""
    deps: Any


class _FakeAgent:
    """Captures the @agent.system_prompt-decorated callable."""

    def __init__(self):
        self.system_prompt_callbacks: List[Callable] = []

    def system_prompt(self, fn):
        self.system_prompt_callbacks.append(fn)
        return fn


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rejects_unknown_style(self):
        agent = _FakeAgent()
        with pytest.raises(ValueError, match="unsupported style"):
            register_synap_st_system_prompt(agent, style="bogus")

    def test_rejects_invalid_on_error(self):
        agent = _FakeAgent()
        with pytest.raises(ValueError, match="on_error"):
            register_synap_st_system_prompt(agent, on_error="ignore")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cache-hit behaviour
# ---------------------------------------------------------------------------


class TestCacheHit:
    @pytest.mark.asyncio
    async def test_returns_combined_system_prompt(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, system="You are a helpful agent.")
        cb = agent.system_prompt_callbacks[0]

        sdk = _fake_sdk(formatted="User name is Sam.")
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="conv_abc")

        out = await cb(_FakeRunContext(deps=deps))
        assert "<synap_short_term_context>" in out
        assert "User name is Sam." in out
        assert "</synap_short_term_context>" in out
        assert "You are a helpful agent." in out
        assert out.index("User name is Sam") < out.index("helpful agent")

    @pytest.mark.asyncio
    async def test_passes_conv_and_style(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, style="bullet_points", system="X")
        cb = agent.system_prompt_callbacks[0]

        sdk = _fake_sdk()
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="conv_abc")
        await cb(_FakeRunContext(deps=deps))
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="bullet_points",
        )


# ---------------------------------------------------------------------------
# Missing conv_id / empty ST
# ---------------------------------------------------------------------------


class TestSkipPaths:
    @pytest.mark.asyncio
    async def test_skips_when_conv_id_absent(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, system="Stay calm.")
        cb = agent.system_prompt_callbacks[0]

        sdk = _fake_sdk()
        # No conversation_id passed in deps
        deps = SynapDeps(sdk=sdk, user_id="u1")
        out = await cb(_FakeRunContext(deps=deps))

        assert out == "Stay calm."
        sdk.conversation.context.get_context_for_prompt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unavailable_keeps_user_system(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, system="Be polite.")
        cb = agent.system_prompt_callbacks[0]

        sdk = _fake_sdk(formatted=None, available=False)
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="c1")
        out = await cb(_FakeRunContext(deps=deps))
        assert out == "Be polite."

    @pytest.mark.asyncio
    async def test_both_empty_returns_empty(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, system="")
        cb = agent.system_prompt_callbacks[0]
        sdk = _fake_sdk(formatted=None, available=False)
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="c1")
        out = await cb(_FakeRunContext(deps=deps))
        assert out == ""


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_fallback_returns_user_system_only(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, system="Calm.")
        cb = agent.system_prompt_callbacks[0]

        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="c1")
        out = await cb(_FakeRunContext(deps=deps))
        assert out == "Calm."

    @pytest.mark.asyncio
    async def test_raise_propagates(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, system="X", on_error="raise")
        cb = agent.system_prompt_callbacks[0]

        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="c1")
        with pytest.raises(SynapIntegrationError):
            await cb(_FakeRunContext(deps=deps))


def test_public_surface_exports():
    import synap_pydantic_ai
    assert hasattr(synap_pydantic_ai, "register_synap_st_system_prompt")
    assert "register_synap_st_system_prompt" in synap_pydantic_ai.__all__
