"""Tests for synap_openai_agents.short_term — synap_st_instructions.

Covers:
- Construction validation (sdk=None, empty/whitespace conversation_id, unknown style,
  invalid on_error)
- Happy paths: ST block present, preamble wrapping, style forwarding, all 3 styles accepted
- Empty-ST safety: unavailable flag, None formatted_context, empty string
- Error policies: fallback (on_error="fallback") returns bare system, raise propagates
  SynapIntegrationError
- Callable contract: returned value is an async callable accepting (context, agent)
- Harness: failing_sdk fixture
- Public surface: __all__ exports
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_openai_agents.short_term import synap_st_instructions
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Local helpers — minimal, keep tests readable
# ---------------------------------------------------------------------------


def _make_response(formatted: str | None, available: bool = True):
    resp = MagicMock()
    resp.available = available
    resp.formatted_context = formatted
    return resp


def _fake_sdk(formatted: str | None = "User prefers concise replies.", available: bool = True):
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
            synap_st_instructions(None, "conv_abc")  # type: ignore[arg-type]

    def test_requires_nonempty_conversation_id(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            synap_st_instructions(sdk, "")

    def test_rejects_whitespace_only_conversation_id(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            synap_st_instructions(sdk, "   ")

    def test_rejects_unknown_style(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="unsupported style"):
            synap_st_instructions(sdk, "conv_abc", style="poetic")

    def test_rejects_invalid_on_error(self):
        sdk = _fake_sdk()
        with pytest.raises(ValueError, match="on_error"):
            synap_st_instructions(sdk, "conv_abc", on_error="ignore")  # type: ignore[arg-type]

    def test_accepts_all_supported_styles(self):
        sdk = _fake_sdk()
        for style in ("structured", "narrative", "bullet_points"):
            # Should NOT raise
            fn = synap_st_instructions(sdk, "conv_abc", style=style)
            assert callable(fn), f"should return callable for style={style!r}"


# ---------------------------------------------------------------------------
# Callable contract
# ---------------------------------------------------------------------------


class TestCallableContract:
    def test_returns_async_callable(self):
        sdk = _fake_sdk()
        fn = synap_st_instructions(sdk, "conv_abc")
        assert callable(fn)
        assert inspect.iscoroutinefunction(fn), "instructions must be async"

    def test_callable_is_named(self):
        sdk = _fake_sdk()
        fn = synap_st_instructions(sdk, "conv_abc")
        assert fn.__name__ == "synap_st_instructions"

    @pytest.mark.asyncio
    async def test_accepts_two_positional_args(self):
        """OpenAI Agents calls instructions(context, agent) — both args must be accepted."""
        sdk = _fake_sdk(formatted="ctx")
        fn = synap_st_instructions(sdk, "conv_abc", system="sys")
        # Should not raise with two MagicMock positional arguments
        result = await fn(MagicMock(), MagicMock())
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Happy paths — ST present
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_returns_combined_instructions_with_default_preamble(self):
        sdk = _fake_sdk(formatted="User likes Markdown.")
        instr = synap_st_instructions(sdk, "conv_abc", system="You are a coding agent.")
        result = await instr(MagicMock(), MagicMock())
        assert "<synap_short_term_context>" in result
        assert "User likes Markdown." in result
        assert "</synap_short_term_context>" in result
        assert "You are a coding agent." in result
        # ST block must appear before user system text
        assert result.index("User likes Markdown") < result.index("coding agent")

    @pytest.mark.asyncio
    async def test_passes_conversation_id_and_style_to_sdk(self):
        sdk = _fake_sdk()
        instr = synap_st_instructions(sdk, "conv_xyz", style="bullet_points", system="X")
        await instr(MagicMock(), MagicMock())
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_xyz",
            style="bullet_points",
        )

    @pytest.mark.asyncio
    async def test_default_style_is_narrative(self):
        sdk = _fake_sdk()
        instr = synap_st_instructions(sdk, "conv_abc", system="X")
        await instr(MagicMock(), MagicMock())
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="narrative",
        )

    @pytest.mark.asyncio
    async def test_custom_preamble_tags(self):
        sdk = _fake_sdk(formatted="X")
        instr = synap_st_instructions(
            sdk, "conv_abc", system="sys",
            preamble_open="[BEGIN]", preamble_close="[END]",
        )
        out = await instr(MagicMock(), MagicMock())
        assert "[BEGIN]" in out
        assert "[END]" in out
        assert "<synap_short_term_context>" not in out

    @pytest.mark.asyncio
    async def test_no_preamble_raw_concat(self):
        """When both preamble tags are None, ST and system are joined with double newline."""
        sdk = _fake_sdk(formatted="raw st")
        instr = synap_st_instructions(
            sdk, "conv_abc", system="user sys",
            preamble_open=None, preamble_close=None,
        )
        out = await instr(MagicMock(), MagicMock())
        assert out == "raw st\n\nuser sys"

    @pytest.mark.asyncio
    async def test_st_only_no_system(self):
        """When system is empty, only the ST block (with preamble) is returned."""
        sdk = _fake_sdk(formatted="context block")
        instr = synap_st_instructions(sdk, "conv_abc", system="")
        out = await instr(MagicMock(), MagicMock())
        assert "context block" in out
        assert "<synap_short_term_context>" in out

    @pytest.mark.asyncio
    async def test_structured_style_forwarded(self):
        sdk = _fake_sdk()
        instr = synap_st_instructions(sdk, "conv_abc", style="structured")
        await instr(MagicMock(), MagicMock())
        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="structured",
        )


# ---------------------------------------------------------------------------
# Empty / unavailable ST
# ---------------------------------------------------------------------------


class TestEmptyST:
    @pytest.mark.asyncio
    async def test_unavailable_returns_user_system_only(self):
        sdk = _fake_sdk(formatted=None, available=False)
        instr = synap_st_instructions(sdk, "conv_abc", system="You are friendly.")
        out = await instr(MagicMock(), MagicMock())
        assert out == "You are friendly."

    @pytest.mark.asyncio
    async def test_unavailable_empty_system_returns_empty_string(self):
        sdk = _fake_sdk(formatted=None, available=False)
        instr = synap_st_instructions(sdk, "conv_abc", system="")
        out = await instr(MagicMock(), MagicMock())
        assert out == ""

    @pytest.mark.asyncio
    async def test_none_formatted_context_with_available_true_treated_as_empty(self):
        """Even if available=True, a None formatted_context yields no ST block."""
        sdk = _fake_sdk(formatted=None, available=True)
        instr = synap_st_instructions(sdk, "conv_abc", system="sys")
        out = await instr(MagicMock(), MagicMock())
        assert out == "sys"
        assert "<synap_short_term_context>" not in out

    @pytest.mark.asyncio
    async def test_empty_string_formatted_context_treated_as_empty(self):
        sdk = _fake_sdk(formatted="", available=True)
        instr = synap_st_instructions(sdk, "conv_abc", system="sys")
        out = await instr(MagicMock(), MagicMock())
        assert out == "sys"

    @pytest.mark.asyncio
    async def test_whitespace_only_formatted_context_treated_as_empty(self):
        sdk = _fake_sdk(formatted="   ", available=True)
        instr = synap_st_instructions(sdk, "conv_abc", system="sys")
        out = await instr(MagicMock(), MagicMock())
        assert out == "sys"


# ---------------------------------------------------------------------------
# Error policies
# ---------------------------------------------------------------------------


class TestErrorPolicies:
    @pytest.mark.asyncio
    async def test_fallback_on_sdk_failure_returns_user_system(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("sdk boom")
        )
        instr = synap_st_instructions(sdk, "conv_abc", system="Stay calm.", on_error="fallback")
        out = await instr(MagicMock(), MagicMock())
        assert out == "Stay calm."

    @pytest.mark.asyncio
    async def test_fallback_empty_system_returns_empty_string(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("sdk boom")
        )
        instr = synap_st_instructions(sdk, "conv_abc", system="", on_error="fallback")
        out = await instr(MagicMock(), MagicMock())
        assert out == ""

    @pytest.mark.asyncio
    async def test_raise_propagates_synap_integration_error(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("sdk boom")
        )
        instr = synap_st_instructions(sdk, "conv_abc", system="X", on_error="raise")
        with pytest.raises(SynapIntegrationError):
            await instr(MagicMock(), MagicMock())

    @pytest.mark.asyncio
    async def test_raise_chains_original_cause(self):
        original = RuntimeError("original sdk error")
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(side_effect=original)
        instr = synap_st_instructions(sdk, "conv_abc", system="X", on_error="raise")
        with pytest.raises(SynapIntegrationError) as exc_info:
            await instr(MagicMock(), MagicMock())
        assert exc_info.value.__cause__ is original

    @pytest.mark.asyncio
    async def test_synap_integration_error_passthrough_on_raise(self):
        """A SynapIntegrationError raised by the SDK should propagate without double-wrapping."""
        sdk = MagicMock()
        original = SynapIntegrationError("op", "direct error")
        sdk.conversation.context.get_context_for_prompt = AsyncMock(side_effect=original)
        instr = synap_st_instructions(sdk, "conv_abc", system="X", on_error="raise")
        with pytest.raises(SynapIntegrationError) as exc_info:
            await instr(MagicMock(), MagicMock())
        # Should be exactly the original instance (not a new wrapper)
        assert exc_info.value is original


# ---------------------------------------------------------------------------
# failing_sdk harness fixture
# ---------------------------------------------------------------------------


class TestFailingSdkHarness:
    @pytest.mark.asyncio
    async def test_fallback_with_failing_sdk(self, failing_sdk):
        """failing_sdk triggers fallback; system text is returned unmodified."""
        instr = synap_st_instructions(
            failing_sdk, "conv_abc", system="fallback text", on_error="fallback"
        )
        out = await instr(MagicMock(), MagicMock())
        assert out == "fallback text"

    @pytest.mark.asyncio
    async def test_raise_with_failing_sdk_surfaces_error(self, failing_sdk):
        """failing_sdk with on_error='raise' must propagate as SynapIntegrationError."""
        instr = synap_st_instructions(
            failing_sdk, "conv_abc", system="X", on_error="raise"
        )
        with pytest.raises(SynapIntegrationError):
            await instr(MagicMock(), MagicMock())


# ---------------------------------------------------------------------------
# mock_sdk harness fixture
# ---------------------------------------------------------------------------


class TestMockSdkHarness:
    @pytest.mark.asyncio
    async def test_mock_sdk_happy_path(self, mock_sdk):
        """mock_sdk pre-wires get_context_for_prompt → returns a combined instructions string."""
        instr = synap_st_instructions(mock_sdk, "conv_abc", system="Agent system prompt.")
        out = await instr(MagicMock(), MagicMock())
        assert isinstance(out, str)
        # Combined output contains ST and user system text
        assert "Agent system prompt." in out


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_module_exports_synap_st_instructions(self):
        import synap_openai_agents
        assert hasattr(synap_openai_agents, "synap_st_instructions")
        assert "synap_st_instructions" in synap_openai_agents.__all__

    def test_create_search_tool_exported(self):
        import synap_openai_agents
        assert hasattr(synap_openai_agents, "create_search_tool")
        assert "create_search_tool" in synap_openai_agents.__all__

    def test_create_store_tool_exported(self):
        import synap_openai_agents
        assert hasattr(synap_openai_agents, "create_store_tool")
        assert "create_store_tool" in synap_openai_agents.__all__
