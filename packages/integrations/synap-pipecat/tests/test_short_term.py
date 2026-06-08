"""Tests for synap_pipecat.short_term — SynapShortTermProcessor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

# Defer pipecat imports so test collection still works if any subdep
# is missing — the conftest path setup already handles in-repo SDK.
from pipecat.frames.frames import TextFrame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from synap_pipecat.short_term import _ST_MARKER, SynapShortTermProcessor
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


def _transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="u1", timestamp="2026-05-27T00:00:00")


def _system_st_messages(ctx: LLMContext):
    return [
        m
        for m in ctx.get_messages()
        if isinstance(m, dict)
        and m.get("role") == "system"
        and isinstance(m.get("content"), str)
        and m["content"].startswith(_ST_MARKER)
    ]


class TestValidation:
    def test_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapShortTermProcessor(None, conversation_id="c1")  # type: ignore[arg-type]

    def test_requires_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            SynapShortTermProcessor(_fake_sdk(), conversation_id="")

    def test_rejects_unknown_style(self):
        with pytest.raises(ValueError, match="unsupported style"):
            SynapShortTermProcessor(_fake_sdk(), conversation_id="c1", style="bogus")


class TestProcessing:
    @pytest.mark.asyncio
    async def test_inserts_st_on_transcription(self):
        sdk = _fake_sdk(formatted="User is VIP.")
        ctx = LLMContext(messages=[])
        proc = SynapShortTermProcessor(sdk, conversation_id="conv_abc", context=ctx)
        # Properly start the processor to satisfy FrameProcessor's lifecycle
        from pipecat.frames.frames import StartFrame
        await proc.queue_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        # In Pipecat, process_frame is called by the pipeline runner; we call
        # it directly here with a "started" StartFrame so push_frame doesn't error.
        # The simpler path: just exercise _refresh_st directly.
        await proc._refresh_st()

        st_msgs = _system_st_messages(ctx)
        assert len(st_msgs) == 1
        assert "User is VIP." in st_msgs[0]["content"]
        assert _ST_MARKER in st_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_replaces_existing_st_message(self):
        sdk_old = _fake_sdk(formatted="Old context.")
        ctx = LLMContext(messages=[])
        proc = SynapShortTermProcessor(sdk_old, conversation_id="conv_abc", context=ctx)
        await proc._refresh_st()
        assert "Old context." in _system_st_messages(ctx)[0]["content"]

        # Replace SDK to return new content
        sdk_new = _fake_sdk(formatted="New context.")
        proc.sdk = sdk_new
        await proc._refresh_st()
        st_msgs = _system_st_messages(ctx)
        assert len(st_msgs) == 1
        assert "New context." in st_msgs[0]["content"]
        assert "Old context." not in st_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_drops_stale_when_unavailable(self):
        sdk_seed = _fake_sdk(formatted="Seed.")
        ctx = LLMContext(messages=[])
        proc = SynapShortTermProcessor(sdk_seed, conversation_id="conv_abc", context=ctx)
        await proc._refresh_st()
        assert len(_system_st_messages(ctx)) == 1

        proc.sdk = _fake_sdk(formatted=None, available=False)
        await proc._refresh_st()
        assert len(_system_st_messages(ctx)) == 0

    @pytest.mark.asyncio
    async def test_no_context_means_no_op(self):
        sdk = _fake_sdk()
        proc = SynapShortTermProcessor(sdk, conversation_id="conv_abc", context=None)
        # Should not crash; nothing to inject into.
        await proc._refresh_st()
        sdk.conversation.context.get_context_for_prompt.assert_not_awaited()


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_fallback_no_inject_on_failure(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        ctx = LLMContext(messages=[])
        proc = SynapShortTermProcessor(sdk, conversation_id="conv_abc", context=ctx)
        await proc._refresh_st()
        assert _system_st_messages(ctx) == []

    @pytest.mark.asyncio
    async def test_raise_propagates(self):
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        ctx = LLMContext(messages=[])
        proc = SynapShortTermProcessor(
            sdk, conversation_id="conv_abc", context=ctx, on_error="raise"
        )
        with pytest.raises(SynapIntegrationError):
            await proc._refresh_st()


def test_public_surface_exports():
    import synap_pipecat
    assert hasattr(synap_pipecat, "SynapShortTermProcessor")
    assert "SynapShortTermProcessor" in synap_pipecat.__all__
