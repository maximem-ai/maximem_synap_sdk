"""Tests for synap_livekit_agents.short_term — preload + refresh."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from livekit.agents import ChatContext, ChatMessage

from synap_livekit_agents.short_term import (
    _ST_MARKER,
    preload_synap_st,
    refresh_synap_st,
)
from synap_integrations_common import SynapIntegrationError


def _make_response(formatted: str | None, available: bool):
    resp = MagicMock()
    resp.available = available
    resp.formatted_context = formatted
    return resp


def _fake_sdk(formatted: str | None = "User is on a call now.", available: bool = True):
    sdk = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=_make_response(formatted, available)
    )
    return sdk


def _empty_chat_ctx() -> ChatContext:
    ctx = ChatContext()
    return ctx


def _message_text(msg: ChatMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p for p in content if isinstance(p, str))
    return ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.asyncio
    async def test_preload_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            await preload_synap_st(_empty_chat_ctx(), None, conversation_id="c1")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_preload_requires_conv_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            await preload_synap_st(_empty_chat_ctx(), _fake_sdk(), conversation_id="")

    @pytest.mark.asyncio
    async def test_refresh_requires_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            await refresh_synap_st(_empty_chat_ctx(), None, conversation_id="c1")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# preload_synap_st
# ---------------------------------------------------------------------------


class TestPreload:
    @pytest.mark.asyncio
    async def test_prepends_system_message_with_marker(self):
        ctx = _empty_chat_ctx()
        sdk = _fake_sdk(formatted="User is VIP.")
        msg = await preload_synap_st(ctx, sdk, conversation_id="conv_abc")
        assert msg is not None
        assert msg.role == "system"
        body = _message_text(msg)
        assert _ST_MARKER in body
        assert "<synap_short_term_context>" in body
        assert "User is VIP." in body
        # Inserted at head
        assert ctx.items[0] is msg

    @pytest.mark.asyncio
    async def test_no_op_when_unavailable(self):
        ctx = _empty_chat_ctx()
        sdk = _fake_sdk(formatted=None, available=False)
        msg = await preload_synap_st(ctx, sdk, conversation_id="conv_abc")
        assert msg is None
        assert len(ctx.items) == 0

    @pytest.mark.asyncio
    async def test_fallback_on_sdk_failure(self):
        ctx = _empty_chat_ctx()
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        msg = await preload_synap_st(ctx, sdk, conversation_id="conv_abc")
        assert msg is None
        assert len(ctx.items) == 0

    @pytest.mark.asyncio
    async def test_raise_propagates(self):
        ctx = _empty_chat_ctx()
        sdk = MagicMock()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        with pytest.raises(SynapIntegrationError):
            await preload_synap_st(
                ctx, sdk, conversation_id="conv_abc", on_error="raise"
            )


# ---------------------------------------------------------------------------
# refresh_synap_st
# ---------------------------------------------------------------------------


class TestRefresh:
    @pytest.mark.asyncio
    async def test_inserts_when_none_existing(self):
        ctx = _empty_chat_ctx()
        sdk = _fake_sdk(formatted="Fresh.")
        msg = await refresh_synap_st(ctx, sdk, conversation_id="conv_abc")
        assert msg is not None
        assert ctx.items[0] is msg

    @pytest.mark.asyncio
    async def test_replaces_existing_st_message(self):
        ctx = _empty_chat_ctx()
        sdk_old = _fake_sdk(formatted="Old.")
        await preload_synap_st(ctx, sdk_old, conversation_id="conv_abc")
        assert "Old." in _message_text(ctx.items[0])

        sdk_new = _fake_sdk(formatted="New.")
        await refresh_synap_st(ctx, sdk_new, conversation_id="conv_abc")
        # Still only one item, content replaced
        assert len([m for m in ctx.items if _ST_MARKER in _message_text(m)]) == 1
        assert "New." in _message_text(ctx.items[0])
        assert "Old." not in _message_text(ctx.items[0])

    @pytest.mark.asyncio
    async def test_drops_stale_when_unavailable(self):
        ctx = _empty_chat_ctx()
        sdk_seed = _fake_sdk(formatted="Seed.")
        await preload_synap_st(ctx, sdk_seed, conversation_id="conv_abc")
        assert len(ctx.items) == 1

        sdk_empty = _fake_sdk(formatted=None, available=False)
        await refresh_synap_st(ctx, sdk_empty, conversation_id="conv_abc")
        # Stale message removed
        assert len(ctx.items) == 0


def test_public_surface_exports():
    import synap_livekit_agents
    assert hasattr(synap_livekit_agents, "preload_synap_st")
    assert hasattr(synap_livekit_agents, "refresh_synap_st")
    assert "preload_synap_st" in synap_livekit_agents.__all__
    assert "refresh_synap_st" in synap_livekit_agents.__all__
