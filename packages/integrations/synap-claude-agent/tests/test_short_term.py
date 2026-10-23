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
    resp.context_block = None
    resp.history_block = None
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


# ---------------------------------------------------------------------------
# Who has the turns decides what is added
# ---------------------------------------------------------------------------

import json  # noqa: E402

from synap_claude_agent.short_term import _scan_transcript  # noqa: E402

BLOCK = "## Conversation so far (compacted 2026-10-22)\nTrip to \"Lisbon\"."


def _compacted_sdk():
    sdk = MagicMock()
    resp = MagicMock()
    resp.available = True
    resp.context_block = BLOCK
    resp.history_block = None
    resp.formatted_context = BLOCK + "\nuser: and hotels?"
    sdk.conversation.context.get_context_for_prompt = AsyncMock(return_value=resp)
    return sdk


def _transcript(tmp_path, *entries):
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(e, separators=(",", ":"), ensure_ascii=False)
                              for e in entries) + "\n")
    return str(path)


USER = {"type": "user", "message": {"role": "user", "content": "flights?"}}


class TestWhoHasTheTurns:
    @pytest.mark.asyncio
    async def test_an_empty_transcript_gets_the_whole_formatted_context(self, tmp_path):
        """A fresh query() per turn: Synap is the only record of the recent turns."""
        cb = _hook_callable(create_synap_st_hook(_compacted_sdk(), "c1"))
        out = await cb({"prompt": "hi", "transcript_path": str(tmp_path / "none.jsonl")}, None, MagicMock())
        assert "user: and hotels?" in out["hookSpecificOutput"]["additionalContext"]

    @pytest.mark.asyncio
    async def test_a_transcript_with_turns_gets_the_compaction_without_them(self, tmp_path):
        cb = _hook_callable(create_synap_st_hook(_compacted_sdk(), "c1"))
        out = await cb({"prompt": "hi", "transcript_path": _transcript(tmp_path, USER)}, None, MagicMock())
        ctx = out["hookSpecificOutput"]["additionalContext"]
        assert "Trip to" in ctx and "and hotels?" not in ctx

    @pytest.mark.asyncio
    async def test_a_transcript_that_holds_the_block_gets_nothing(self, tmp_path):
        held = {"type": "attachment", "text": BLOCK}
        cb = _hook_callable(create_synap_st_hook(_compacted_sdk(), "c1"))
        out = await cb({"prompt": "hi", "transcript_path": _transcript(tmp_path, USER, held)},
                       None, MagicMock())
        assert out == {}

    @pytest.mark.asyncio
    async def test_a_block_from_before_claude_codes_own_compaction_is_added_again(self, tmp_path):
        """What Claude Code compacted left the model's context, block included."""
        held = {"type": "attachment", "text": BLOCK}
        boundary = {"type": "system", "subtype": "compact_boundary"}
        cb = _hook_callable(create_synap_st_hook(_compacted_sdk(), "c1"))
        out = await cb({"prompt": "hi", "transcript_path": _transcript(tmp_path, USER, held, boundary)},
                       None, MagicMock())
        assert "Trip to" in out["hookSpecificOutput"]["additionalContext"]

    @pytest.mark.asyncio
    async def test_no_compaction_and_a_transcript_with_turns_adds_nothing(self, tmp_path):
        sdk = _compacted_sdk()
        sdk.conversation.context.get_context_for_prompt.return_value.context_block = None
        sdk.conversation.context.get_context_for_prompt.return_value.formatted_context = "user: x"
        cb = _hook_callable(create_synap_st_hook(sdk, "c1"))
        out = await cb({"prompt": "hi", "transcript_path": _transcript(tmp_path, USER)}, None, MagicMock())
        assert out == {}

    def test_an_unreadable_transcript_is_empty(self, tmp_path):
        assert _scan_transcript(str(tmp_path / "missing"), "x") == (False, False)
        assert _scan_transcript(None, "x") == (False, False)
