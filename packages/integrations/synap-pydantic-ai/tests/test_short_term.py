"""Tests for synap_pydantic_ai.short_term — register_synap_st_system_prompt.

We exercise the registered callback directly by capturing what
``@agent.system_prompt`` is called with, rather than spinning up a real
Pydantic AI Agent (which would require a model client). The captured
callable is then invoked against a fake RunContext + SynapDeps.

Covers:
- Validation: style and on_error guards (_validate)
- All supported style values are accepted
- Registration contract: exactly one system_prompt callback registered
- Cache-hit behaviour: ST injected with preamble tags, ordering
- Style forwarded to SDK (structured / narrative / bullet_points)
- Missing conversation_id: ST skipped, static system returned
- Unavailable ST (available=False): static system returned
- Both empty: empty string returned
- Failure path with on_error="fallback": no crash, static system returned
- Failure path with on_error="raise": SynapIntegrationError raised
- SynapIntegrationError not double-wrapped
- No-preamble mode (preamble_open=None, preamble_close=None): raw concat
- Custom preamble tags
- Only ST present (no static system): preamble wraps ST only
- Only static system (no ST): system returned as-is
- Whitespace-only ST treated as empty (no preamble injected)
- Public-surface exports
- SDK context kwargs forwarded (conversation_id, style)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_integrations_common import SynapIntegrationError
from synap_pydantic_ai.deps import SynapDeps
from synap_pydantic_ai.short_term import (
    _compose,
    _validate,
    register_synap_st_system_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(formatted: str | None, available: bool) -> MagicMock:
    resp = MagicMock()
    resp.available = available
    resp.formatted_context = formatted
    return resp


def _fake_sdk(
    formatted: str | None = "User likes Markdown.",
    available: bool = True,
) -> MagicMock:
    sdk = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=_make_response(formatted, available)
    )
    return sdk


def _failing_sdk(exc: Exception | None = None) -> MagicMock:
    sdk = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        side_effect=exc or RuntimeError("sdk boom")
    )
    return sdk


@dataclass
class _FakeRunContext:
    """Stand-in for pydantic_ai.RunContext. Only .deps is read."""
    deps: Any


class _FakeAgent:
    """Captures the @agent.system_prompt-decorated callable."""

    def __init__(self):
        self.system_prompt_callbacks: List[Callable] = []

    def system_prompt(self, fn: Callable) -> Callable:
        self.system_prompt_callbacks.append(fn)
        return fn


# ---------------------------------------------------------------------------
# _validate — unit tests for the internal validation helper
# ---------------------------------------------------------------------------


class TestValidateHelper:
    def test_rejects_unknown_style(self):
        with pytest.raises(ValueError, match="unsupported style"):
            _validate("poetic", "fallback", "test_site")

    def test_rejects_invalid_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            _validate("narrative", "ignore", "test_site")  # type: ignore[arg-type]

    @pytest.mark.parametrize("style", ["structured", "narrative", "bullet_points"])
    def test_accepts_all_valid_styles(self, style: str):
        _validate(style, "fallback", "test_site")  # should not raise

    @pytest.mark.parametrize("on_error", ["fallback", "raise"])
    def test_accepts_all_valid_on_error_values(self, on_error: str):
        _validate("narrative", on_error, "test_site")  # should not raise

    def test_error_message_includes_site(self):
        with pytest.raises(ValueError, match="my_site"):
            _validate("bogus", "fallback", "my_site")

    def test_error_message_includes_style_value(self):
        with pytest.raises(ValueError, match="bogus"):
            _validate("bogus", "fallback", "site")


# ---------------------------------------------------------------------------
# _compose — unit tests for the composition helper
# ---------------------------------------------------------------------------


class TestComposeHelper:
    def test_both_empty_returns_empty_string(self):
        assert _compose("", "", None, None) == ""

    def test_only_user_system(self):
        assert _compose("", "user sys", None, None) == "user sys"

    def test_only_st_no_preamble(self):
        assert _compose("st content", "", None, None) == "st content"

    def test_st_and_user_no_preamble(self):
        result = _compose("st content", "user sys", None, None)
        assert result == "st content\n\nuser sys"

    def test_st_before_user_system(self):
        result = _compose("st", "user sys", None, None)
        assert result.index("st") < result.index("user sys")

    def test_preamble_wraps_st(self):
        result = _compose("st", "user sys", "<open>", "<close>")
        assert "<open>\nst\n<close>" in result

    def test_preamble_and_user_system_separated_by_double_newline(self):
        result = _compose("st", "user sys", "<open>", "<close>")
        assert "<close>\n\nuser sys" in result

    def test_preamble_only_st_no_user_system(self):
        result = _compose("st", "", "<open>", "<close>")
        assert result == "<open>\nst\n<close>"

    def test_whitespace_only_st_treated_as_empty(self):
        result = _compose("   ", "user sys", "<open>", "<close>")
        assert result == "user sys"

    def test_whitespace_stripped_from_st(self):
        result = _compose("  stripped  ", "user", None, None)
        assert "stripped" in result
        assert result.startswith("stripped")


# ---------------------------------------------------------------------------
# register_synap_st_system_prompt — registration contract
# ---------------------------------------------------------------------------


class TestRegistrationContract:
    def test_registers_exactly_one_system_prompt(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent)
        assert len(agent.system_prompt_callbacks) == 1

    def test_validation_runs_at_registration_time(self):
        """Bad args must fail at registration, not at invocation time."""
        agent = _FakeAgent()
        with pytest.raises(ValueError):
            register_synap_st_system_prompt(agent, style="bogus")

    def test_validation_on_error_runs_at_registration_time(self):
        agent = _FakeAgent()
        with pytest.raises(ValueError):
            register_synap_st_system_prompt(agent, on_error="ignore")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cache-hit behaviour — ST available
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

    @pytest.mark.asyncio
    async def test_st_appears_before_user_system(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, system="Static prompt.")
        cb = agent.system_prompt_callbacks[0]

        sdk = _fake_sdk(formatted="ST content.")
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="conv_abc")
        out = await cb(_FakeRunContext(deps=deps))

        assert out.index("ST content") < out.index("Static prompt")

    @pytest.mark.asyncio
    async def test_passes_conv_id_to_sdk(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, system="X")
        cb = agent.system_prompt_callbacks[0]

        sdk = _fake_sdk()
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="conv_target")
        await cb(_FakeRunContext(deps=deps))

        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_target",
            style="narrative",
        )

    @pytest.mark.asyncio
    async def test_passes_style_to_sdk(self):
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

    @pytest.mark.asyncio
    @pytest.mark.parametrize("style", ["structured", "narrative", "bullet_points"])
    async def test_all_styles_accepted_at_runtime(self, style: str):
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, style=style)
        cb = agent.system_prompt_callbacks[0]

        sdk = _fake_sdk(formatted="Some ST content.")
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="conv_abc")
        out = await cb(_FakeRunContext(deps=deps))

        assert "Some ST content." in out


# ---------------------------------------------------------------------------
# No-preamble mode
# ---------------------------------------------------------------------------


class TestNoPreambleMode:
    @pytest.mark.asyncio
    async def test_no_preamble_raw_concat(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(
            agent,
            system="user sys",
            preamble_open=None,
            preamble_close=None,
        )
        cb = agent.system_prompt_callbacks[0]

        sdk = _fake_sdk(formatted="st content")
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="conv_abc")
        out = await cb(_FakeRunContext(deps=deps))

        assert out == "st content\n\nuser sys"

    @pytest.mark.asyncio
    async def test_no_preamble_no_xml_tags(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(
            agent,
            system="user sys",
            preamble_open=None,
            preamble_close=None,
        )
        cb = agent.system_prompt_callbacks[0]

        sdk = _fake_sdk(formatted="st content")
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="conv_abc")
        out = await cb(_FakeRunContext(deps=deps))

        assert "<synap_short_term_context>" not in out
        assert "</synap_short_term_context>" not in out


# ---------------------------------------------------------------------------
# Custom preamble tags
# ---------------------------------------------------------------------------


class TestCustomPreamble:
    @pytest.mark.asyncio
    async def test_custom_preamble_tags_appear_in_output(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(
            agent,
            system="sys",
            preamble_open="[BEGIN_MEM]",
            preamble_close="[END_MEM]",
        )
        cb = agent.system_prompt_callbacks[0]

        sdk = _fake_sdk(formatted="X")
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="c1")
        out = await cb(_FakeRunContext(deps=deps))

        assert "[BEGIN_MEM]" in out
        assert "[END_MEM]" in out

    @pytest.mark.asyncio
    async def test_custom_preamble_wraps_st_not_user_system(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(
            agent,
            system="static user sys",
            preamble_open="[B]",
            preamble_close="[E]",
        )
        cb = agent.system_prompt_callbacks[0]

        sdk = _fake_sdk(formatted="ST text")
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="c1")
        out = await cb(_FakeRunContext(deps=deps))

        b_pos = out.index("[B]")
        e_pos = out.index("[E]")
        st_pos = out.index("ST text")
        user_pos = out.index("static user sys")

        assert b_pos < st_pos < e_pos < user_pos


# ---------------------------------------------------------------------------
# Skip paths — missing conv_id, empty ST
# ---------------------------------------------------------------------------


class TestSkipPaths:
    @pytest.mark.asyncio
    async def test_skips_when_conv_id_absent(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, system="Stay calm.")
        cb = agent.system_prompt_callbacks[0]

        sdk = _fake_sdk()
        deps = SynapDeps(sdk=sdk, user_id="u1")  # no conversation_id
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

    @pytest.mark.asyncio
    async def test_none_conv_id_not_empty_string(self):
        """conversation_id=None (the default) must skip SDK, not call with None."""
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, system="system text")
        cb = agent.system_prompt_callbacks[0]

        sdk = _fake_sdk()
        deps = SynapDeps(sdk=sdk, user_id="u1")
        out = await cb(_FakeRunContext(deps=deps))

        sdk.conversation.context.get_context_for_prompt.assert_not_awaited()
        assert out == "system text"

    @pytest.mark.asyncio
    async def test_only_st_no_static_system_no_preamble(self):
        """If only ST, no static system — result is just ST (no trailing newlines)."""
        agent = _FakeAgent()
        register_synap_st_system_prompt(
            agent,
            system="",
            preamble_open=None,
            preamble_close=None,
        )
        cb = agent.system_prompt_callbacks[0]

        sdk = _fake_sdk(formatted="only st")
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="c1")
        out = await cb(_FakeRunContext(deps=deps))

        assert out == "only st"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_fallback_returns_user_system_on_sdk_failure(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, system="Calm.")
        cb = agent.system_prompt_callbacks[0]

        sdk = _failing_sdk(RuntimeError("boom"))
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="c1")
        out = await cb(_FakeRunContext(deps=deps))

        assert out == "Calm."

    @pytest.mark.asyncio
    async def test_fallback_empty_system_returns_empty_string(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, system="")
        cb = agent.system_prompt_callbacks[0]

        sdk = _failing_sdk()
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="c1")
        out = await cb(_FakeRunContext(deps=deps))

        assert out == ""

    @pytest.mark.asyncio
    async def test_raise_propagates_synap_integration_error(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, system="X", on_error="raise")
        cb = agent.system_prompt_callbacks[0]

        sdk = _failing_sdk(RuntimeError("boom"))
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="c1")

        with pytest.raises(SynapIntegrationError):
            await cb(_FakeRunContext(deps=deps))

    @pytest.mark.asyncio
    async def test_raise_preserves_original_cause(self):
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, system="X", on_error="raise")
        cb = agent.system_prompt_callbacks[0]

        original = RuntimeError("original cause")
        sdk = _failing_sdk(original)
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="c1")

        with pytest.raises(SynapIntegrationError) as exc_info:
            await cb(_FakeRunContext(deps=deps))

        assert exc_info.value.__cause__ is original

    @pytest.mark.asyncio
    async def test_synap_integration_error_not_double_wrapped_on_raise(self):
        """A SynapIntegrationError from SDK must propagate as-is under on_error='raise'."""
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, system="X", on_error="raise")
        cb = agent.system_prompt_callbacks[0]

        inner = SynapIntegrationError("upstream.op", "already wrapped")
        sdk = _failing_sdk(inner)
        deps = SynapDeps(sdk=sdk, user_id="u1", conversation_id="c1")

        with pytest.raises(SynapIntegrationError) as exc_info:
            await cb(_FakeRunContext(deps=deps))

        assert exc_info.value is inner

    @pytest.mark.asyncio
    async def test_fallback_uses_shared_harness_failing_sdk(self, failing_sdk):
        """Integration-level: shared failing_sdk fixture works with ST fallback path."""
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, system="Fallback text.")
        cb = agent.system_prompt_callbacks[0]

        deps = SynapDeps(sdk=failing_sdk, user_id="u1", conversation_id="c1")
        out = await cb(_FakeRunContext(deps=deps))

        assert out == "Fallback text."


# ---------------------------------------------------------------------------
# Public surface exports
# ---------------------------------------------------------------------------


def test_public_surface_exports():
    import synap_pydantic_ai
    assert hasattr(synap_pydantic_ai, "register_synap_st_system_prompt")
    assert "register_synap_st_system_prompt" in synap_pydantic_ai.__all__


def test_register_synap_st_system_prompt_importable_from_short_term():
    from synap_pydantic_ai.short_term import register_synap_st_system_prompt  # noqa: F401
    assert register_synap_st_system_prompt is not None
