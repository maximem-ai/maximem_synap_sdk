"""Tests for synap_google_adk.short_term — synap_st_instruction.

Covers:
- Construction-time validation (sdk, conversation_id, style, on_error)
- Happy-path: ST prepended above static instruction inside preamble tags
- Exact SDK call kwargs (conversation_id + style forwarded correctly)
- Custom preamble tags
- No-preamble (raw concat) mode
- Empty/unavailable ST: static instruction returned unchanged
- Empty ST + empty instruction: empty string returned
- available=True but empty formatted_context treated as no ST
- on_error="fallback" (default): SDK failure returns static instruction only
- on_error="raise": SDK failure raises SynapIntegrationError
- SDK called fresh on every invocation (not cached by the closure)
- All three valid styles accepted
- Whitespace-only conversation_id rejected
- Returned callable is an async function named "synap_st_instruction"
- Public surface: __init__.py exports synap_st_instruction in __all__
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_google_adk.short_term import synap_st_instruction
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(formatted: str | None, available: bool) -> MagicMock:
    resp = MagicMock()
    resp.available = available
    resp.formatted_context = formatted
    return resp


def _fake_sdk(
    formatted: str | None = "User prefers concise replies.",
    available: bool = True,
) -> MagicMock:
    sdk = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=_make_response(formatted, available)
    )
    return sdk


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_requires_non_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            synap_st_instruction(None, "conv_abc")  # type: ignore[arg-type]

    def test_requires_non_empty_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            synap_st_instruction(_fake_sdk(), "")

    def test_rejects_whitespace_only_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            synap_st_instruction(_fake_sdk(), "   ")

    def test_rejects_unknown_style(self):
        with pytest.raises(ValueError, match="unsupported style"):
            synap_st_instruction(_fake_sdk(), "conv_abc", style="poetic")

    def test_rejects_invalid_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            synap_st_instruction(_fake_sdk(), "conv_abc", on_error="ignore")  # type: ignore[arg-type]

    def test_all_valid_styles_accepted(self):
        """All three documented styles must construct without error."""
        for style in ("structured", "narrative", "bullet_points"):
            synap_st_instruction(_fake_sdk(), "conv_abc", style=style)  # no raise

    def test_returns_callable(self):
        instr = synap_st_instruction(_fake_sdk(), "conv_abc")
        assert callable(instr)

    def test_returned_callable_is_async(self):
        instr = synap_st_instruction(_fake_sdk(), "conv_abc")
        assert inspect.iscoroutinefunction(instr)

    def test_returned_callable_has_correct_name(self):
        instr = synap_st_instruction(_fake_sdk(), "conv_abc")
        assert instr.__name__ == "synap_st_instruction"


# ---------------------------------------------------------------------------
# Happy-path: ST present, instruction present
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_st_prepended_above_static_instruction(self):
        """ST block appears before the static instruction in the output."""
        sdk = _fake_sdk(formatted="User is a VIP.")
        instr = synap_st_instruction(sdk, "conv_abc", instruction="You are polite.")
        out = await instr(MagicMock())

        assert "User is a VIP." in out
        assert "You are polite." in out
        assert out.index("User is a VIP.") < out.index("You are polite.")

    @pytest.mark.asyncio
    async def test_default_preamble_tags_present(self):
        """Default preamble wraps ST in <synap_short_term_context> tags."""
        sdk = _fake_sdk(formatted="Some context.")
        instr = synap_st_instruction(sdk, "conv_abc", instruction="sys")
        out = await instr(MagicMock())

        assert "<synap_short_term_context>" in out
        assert "</synap_short_term_context>" in out

    @pytest.mark.asyncio
    async def test_passes_conversation_id_and_style_to_sdk(self):
        """SDK is called with the exact conversation_id and style provided."""
        sdk = _fake_sdk()
        instr = synap_st_instruction(
            sdk, "conv_abc", style="structured", instruction="X"
        )
        await instr(MagicMock())

        sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
            conversation_id="conv_abc",
            style="structured",
        )

    @pytest.mark.asyncio
    async def test_default_style_is_narrative(self):
        """When no style is specified, SDK is called with style='narrative'."""
        sdk = _fake_sdk()
        instr = synap_st_instruction(sdk, "conv_abc")
        await instr(MagicMock())

        call_kwargs = sdk.conversation.context.get_context_for_prompt.call_args.kwargs
        assert call_kwargs["style"] == "narrative"

    @pytest.mark.asyncio
    async def test_ctx_argument_is_forwarded_to_callable(self):
        """The ADK context argument (any object) is accepted without error."""
        sdk = _fake_sdk(formatted="ctx")
        instr = synap_st_instruction(sdk, "conv_abc", instruction="I")
        ctx_mock = MagicMock()
        out = await instr(ctx_mock)
        assert isinstance(out, str)

    @pytest.mark.asyncio
    async def test_custom_preamble_tags_used(self):
        """Custom preamble_open / preamble_close replace the defaults."""
        sdk = _fake_sdk(formatted="Block content")
        instr = synap_st_instruction(
            sdk,
            "conv_abc",
            instruction="sys",
            preamble_open="[BEGIN]",
            preamble_close="[END]",
        )
        out = await instr(MagicMock())

        assert "[BEGIN]" in out
        assert "[END]" in out
        assert "<synap_short_term_context>" not in out

    @pytest.mark.asyncio
    async def test_no_preamble_raw_concat(self):
        """When both preamble_open and preamble_close are None, ST and instruction
        are joined with a blank line, no wrapping tags."""
        sdk = _fake_sdk(formatted="raw st")
        instr = synap_st_instruction(
            sdk,
            "conv_abc",
            instruction="user sys",
            preamble_open=None,
            preamble_close=None,
        )
        out = await instr(MagicMock())

        assert out == "raw st\n\nuser sys"

    @pytest.mark.asyncio
    async def test_st_only_no_static_instruction(self):
        """ST alone (no static instruction) returned inside preamble tags."""
        sdk = _fake_sdk(formatted="ST only content.")
        instr = synap_st_instruction(sdk, "conv_abc")
        out = await instr(MagicMock())

        assert "ST only content." in out
        assert "<synap_short_term_context>" in out

    @pytest.mark.asyncio
    async def test_sdk_called_fresh_each_invocation(self):
        """The callable is stateless — SDK is consulted on every call."""
        call_count = 0

        async def fresh_ctx(**_kwargs):
            nonlocal call_count
            call_count += 1
            return _make_response(f"context call {call_count}", available=True)

        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=fresh_ctx
        )
        instr = synap_st_instruction(sdk, "conv_abc")

        await instr(MagicMock())
        await instr(MagicMock())

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_bullet_points_style_content_returned(self):
        sdk = _fake_sdk(formatted="• User prefers brevity.")
        instr = synap_st_instruction(sdk, "conv_abc", style="bullet_points")
        out = await instr(MagicMock())
        assert "• User prefers brevity." in out


# ---------------------------------------------------------------------------
# Empty / unavailable ST
# ---------------------------------------------------------------------------


class TestEmptyST:
    @pytest.mark.asyncio
    async def test_unavailable_false_keeps_static_instruction(self):
        """available=False → only the static instruction is returned."""
        sdk = _fake_sdk(formatted=None, available=False)
        instr = synap_st_instruction(sdk, "conv_abc", instruction="Be precise.")
        out = await instr(MagicMock())

        assert out == "Be precise."

    @pytest.mark.asyncio
    async def test_unavailable_and_no_instruction_returns_empty(self):
        """available=False + no static instruction → empty string (never None)."""
        sdk = _fake_sdk(formatted=None, available=False)
        instr = synap_st_instruction(sdk, "conv_abc", instruction="")
        out = await instr(MagicMock())

        assert out == ""

    @pytest.mark.asyncio
    async def test_available_true_but_empty_formatted_context_acts_as_no_st(self):
        """available=True but empty formatted_context: ST block is skipped;
        static instruction returned unchanged."""
        sdk = _fake_sdk(formatted="", available=True)
        instr = synap_st_instruction(sdk, "conv_abc", instruction="Static instr")
        out = await instr(MagicMock())

        assert out == "Static instr"
        assert "<synap_short_term_context>" not in out

    @pytest.mark.asyncio
    async def test_available_true_whitespace_formatted_context_acts_as_no_st(self):
        """available=True but whitespace-only formatted_context: same as no ST."""
        sdk = _fake_sdk(formatted="   ", available=True)
        instr = synap_st_instruction(sdk, "conv_abc", instruction="Fallback")
        out = await instr(MagicMock())

        assert out == "Fallback"

    @pytest.mark.asyncio
    async def test_no_preamble_tags_when_st_empty(self):
        """Preamble tags must NOT appear in output when ST is unavailable."""
        sdk = _fake_sdk(formatted=None, available=False)
        instr = synap_st_instruction(sdk, "conv_abc", instruction="Only this.")
        out = await instr(MagicMock())

        assert "<synap_short_term_context>" not in out
        assert "</synap_short_term_context>" not in out


# ---------------------------------------------------------------------------
# Error handling — fallback (default) and raise
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_fallback_on_sdk_failure_returns_static_instruction(self):
        """on_error='fallback' (default): SDK RuntimeError → static instruction,
        no crash."""
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("network timeout")
        )
        instr = synap_st_instruction(sdk, "conv_abc", instruction="Stay calm.")
        out = await instr(MagicMock())

        assert out == "Stay calm."

    @pytest.mark.asyncio
    async def test_fallback_with_no_instruction_returns_empty_string(self):
        """on_error='fallback', no static instruction → empty string (no crash)."""
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        instr = synap_st_instruction(sdk, "conv_abc", instruction="")
        out = await instr(MagicMock())

        assert out == ""

    @pytest.mark.asyncio
    async def test_raise_mode_propagates_synap_integration_error(self):
        """on_error='raise': SDK RuntimeError → SynapIntegrationError raised."""
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        instr = synap_st_instruction(sdk, "conv_abc", instruction="X", on_error="raise")

        with pytest.raises(SynapIntegrationError):
            await instr(MagicMock())

    @pytest.mark.asyncio
    async def test_raise_mode_with_no_instruction_still_raises(self):
        """on_error='raise': even with empty instruction, SynapIntegrationError raised."""
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        instr = synap_st_instruction(sdk, "conv_abc", on_error="raise")

        with pytest.raises(SynapIntegrationError):
            await instr(MagicMock())

    @pytest.mark.asyncio
    async def test_fallback_uses_shared_failing_sdk_fixture(self, failing_sdk):
        """Verify fallback behaviour with the canonical shared failing_sdk fixture."""
        instr = synap_st_instruction(
            failing_sdk, "conv_abc", instruction="Shared fallback."
        )
        out = await instr(MagicMock())

        assert out == "Shared fallback."

    @pytest.mark.asyncio
    async def test_raise_mode_uses_shared_failing_sdk_fixture(self, failing_sdk):
        """Verify raise behaviour with the canonical shared failing_sdk fixture."""
        instr = synap_st_instruction(
            failing_sdk, "conv_abc", instruction="X", on_error="raise"
        )
        with pytest.raises(SynapIntegrationError):
            await instr(MagicMock())


# ---------------------------------------------------------------------------
# Happy-path with shared mock_sdk fixture
# ---------------------------------------------------------------------------


class TestWithSharedFixtures:
    @pytest.mark.asyncio
    async def test_returns_string_with_shared_mock_sdk(self, mock_sdk):
        """Smoke test using the canonical shared mock_sdk fixture."""
        instr = synap_st_instruction(mock_sdk, "conv-shared", instruction="Static.")
        out = await instr(MagicMock())

        assert isinstance(out, str)
        # shared fixture returns available=True with formatted text
        assert "Static." in out


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_synap_st_instruction_in_package_init(self):
        import synap_google_adk

        assert hasattr(synap_google_adk, "synap_st_instruction")

    def test_synap_st_instruction_in_all(self):
        import synap_google_adk

        assert "synap_st_instruction" in synap_google_adk.__all__

    def test_create_synap_tools_in_package_init(self):
        import synap_google_adk

        assert hasattr(synap_google_adk, "create_synap_tools")

    def test_create_synap_tools_in_all(self):
        import synap_google_adk

        assert "create_synap_tools" in synap_google_adk.__all__
