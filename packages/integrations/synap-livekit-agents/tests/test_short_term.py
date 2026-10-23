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


def _turns_ctx() -> ChatContext:
    ctx = ChatContext()
    for role, text in (("user", "turn-1"), ("assistant", "reply-1"), ("user", "now")):
        ctx.add_message(role=role, content=text)
    return ctx


def _uncompacted_sdk(extra=None):
    """Synap holds turn-1, reply-1, now (and `extra`), and has not compacted."""
    from types import SimpleNamespace

    recent = [("user", "turn-1"), ("assistant", "reply-1"), ("user", "now")]
    if extra:
        recent.insert(0, extra)
    sdk = _fake_sdk()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=SimpleNamespace(
            available=True,
            formatted_context="\n".join(f"{r}: {t}" for r, t in recent),
            context_block="",
            history_block="",
            recent_messages=[SimpleNamespace(role=r, content=t) for r, t in recent],
            compaction_id=None,
        )
    )
    return sdk


def _compacted_sdk():
    from types import SimpleNamespace

    sdk = _fake_sdk(formatted="ST block with turn-1 and reply-1")
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=SimpleNamespace(
            available=True,
            formatted_context="ST block with turn-1 and reply-1",
            context_block="SUMMARY: refund",
            history_block="",
            recent_messages=[SimpleNamespace(role="user", content="turn-3")],
            compaction_id="c1",
        )
    )
    return sdk


class TestNeverSendsATurnTwice:
    """The ST block carries recent turns. Beside a context that already holds
    them, that is every turn sent twice."""

    @pytest.mark.asyncio
    async def test_refresh_adds_no_block_to_a_context_that_holds_the_turns(self):
        ctx = _turns_ctx()
        msg = await refresh_synap_st(ctx, _uncompacted_sdk(), conversation_id="c1")
        assert msg is None
        assert [m.text_content for m in ctx.items] == ["turn-1", "reply-1", "now"]

    @pytest.mark.asyncio
    async def test_refresh_drops_a_block_left_from_before_the_turns_arrived(self):
        ctx = _empty_chat_ctx()
        await preload_synap_st(ctx, _uncompacted_sdk(), conversation_id="c1")
        for role, text in (("user", "turn-1"), ("assistant", "reply-1"), ("user", "now")):
            ctx.add_message(role=role, content=text)
        await refresh_synap_st(ctx, _uncompacted_sdk(), conversation_id="c1")
        assert [m.text_content for m in ctx.items] == ["turn-1", "reply-1", "now"]

    @pytest.mark.asyncio
    async def test_the_block_lands_after_the_instructions_not_before_them(self):
        ctx = ChatContext()
        ctx.add_message(role="system", content="be nice")
        await preload_synap_st(ctx, _uncompacted_sdk(), conversation_id="c1")
        texts = [m.text_content for m in ctx.items]
        assert texts[0] == "be nice" and texts[1].startswith(_ST_MARKER)

    @pytest.mark.asyncio
    async def test_a_resumed_conversation_still_gets_the_turns_the_context_lacks(self):
        ctx = _empty_chat_ctx()
        await preload_synap_st(ctx, _uncompacted_sdk(), conversation_id="c1")
        assert len(ctx.items) == 1 and "turn-1" in ctx.items[0].text_content
        # A live turn arrives; the block keeps only what the context lacks.
        ctx.add_message(role="user", content="turn-1")
        ctx.add_message(role="assistant", content="reply-1")
        await refresh_synap_st(ctx, _uncompacted_sdk(extra=("user", "from last week")),
                               conversation_id="c1")
        block = [m for m in ctx.items if m.text_content.startswith(_ST_MARKER)]
        assert len(block) == 1
        assert "from last week" in block[0].text_content
        assert "reply-1" not in block[0].text_content

    @pytest.mark.asyncio
    async def test_once_compacted_refresh_swaps_the_earlier_turns_for_the_compaction(self):
        ctx = _turns_ctx()
        msg = await refresh_synap_st(ctx, _compacted_sdk(), conversation_id="c1")
        texts = [m.text_content for m in ctx.items]
        assert msg is not None and msg.text_content.startswith("<synap_conversation_history>")
        assert texts[0] == msg.text_content
        assert "turn-1" not in texts and "reply-1" not in texts
        assert texts[-1] == "now"
        assert not any(_ST_MARKER in t for t in texts)

    @pytest.mark.asyncio
    async def test_refreshing_twice_keeps_one_compaction(self):
        ctx = _turns_ctx()
        await refresh_synap_st(ctx, _compacted_sdk(), conversation_id="c1")
        await refresh_synap_st(ctx, _compacted_sdk(), conversation_id="c1")
        texts = [m.text_content for m in ctx.items]
        assert sum(t.startswith("<synap_conversation_history>") for t in texts) == 1

    @pytest.mark.asyncio
    async def test_preload_on_a_populated_context_splices_instead_of_stacking(self):
        ctx = _turns_ctx()
        await preload_synap_st(ctx, _compacted_sdk(), conversation_id="c1")
        texts = [m.text_content for m in ctx.items]
        assert "turn-1" not in texts
        assert not any(_ST_MARKER in t for t in texts)


def test_public_surface_exports():
    import synap_livekit_agents
    assert hasattr(synap_livekit_agents, "preload_synap_st")
    assert hasattr(synap_livekit_agents, "refresh_synap_st")
    assert "preload_synap_st" in synap_livekit_agents.__all__
    assert "refresh_synap_st" in synap_livekit_agents.__all__
