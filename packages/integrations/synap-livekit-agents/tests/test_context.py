"""Tests for synap_livekit_agents.context — preload_synap_context.

Documented error-handling contract (from context.py docstring):
- Read-side failures degrade silently: SDK raises → log ERROR, return None,
  ChatContext untouched.
- Empty or whitespace-only formatted_context → None (no blank message injected).
- Non-None result is always a system ChatMessage inserted at ctx.items[0].
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from livekit.agents import ChatContext, ChatMessage

from synap_livekit_agents.context import preload_synap_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_sdk(formatted_context: str | None = "User is a senior engineer."):
    sdk = MagicMock()
    resp = MagicMock()
    resp.formatted_context = formatted_context
    sdk.fetch = AsyncMock(return_value=resp)
    return sdk


def _empty_ctx() -> ChatContext:
    return ChatContext()


def _msg_text(msg: ChatMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p for p in content if isinstance(p, str))
    return ""


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.asyncio
    async def test_requires_non_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            await preload_synap_context(_empty_ctx(), None, user_id="u1")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_requires_non_empty_user_id(self):
        with pytest.raises(ValueError, match="non-empty user_id"):
            await preload_synap_context(_empty_ctx(), _fake_sdk(), user_id="")

    @pytest.mark.asyncio
    async def test_requires_non_whitespace_user_id_coerced(self):
        # Empty string is caught; the guard is `if not user_id` so whitespace passes
        # through to SDK — that is the documented behaviour (SDK enforces scope rules).
        sdk = _fake_sdk(formatted_context="ctx")
        ctx = _empty_ctx()
        # Should NOT raise — the product only guards against falsy user_id
        msg = await preload_synap_context(ctx, sdk, user_id=" ")
        # The call went through; SDK received the whitespace user_id unchanged
        assert sdk.fetch.await_count == 1


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_prepends_system_message_at_head(self):
        ctx = _empty_ctx()
        sdk = _fake_sdk("User is a senior engineer.")
        msg = await preload_synap_context(ctx, sdk, user_id="u1")
        assert msg is not None
        assert msg.role == "system"
        assert ctx.items[0] is msg

    @pytest.mark.asyncio
    async def test_message_content_contains_formatted_context(self):
        ctx = _empty_ctx()
        sdk = _fake_sdk("User prefers Python.")
        msg = await preload_synap_context(ctx, sdk, user_id="u1")
        body = _msg_text(msg)
        assert "User prefers Python." in body

    @pytest.mark.asyncio
    async def test_message_content_contains_preamble(self):
        ctx = _empty_ctx()
        sdk = _fake_sdk("some memory")
        msg = await preload_synap_context(ctx, sdk, user_id="u1")
        body = _msg_text(msg)
        assert "Relevant long-term memory" in body

    @pytest.mark.asyncio
    async def test_message_inserted_before_existing_items(self):
        ctx = _empty_ctx()
        # Pre-populate the context with a user message
        existing = ChatMessage(role="user", content=["hello"])
        ctx.items.append(existing)
        sdk = _fake_sdk("memory facts")
        msg = await preload_synap_context(ctx, sdk, user_id="u1")
        # Synap message must be at position 0, user message at position 1
        assert ctx.items[0] is msg
        assert ctx.items[1] is existing

    @pytest.mark.asyncio
    async def test_returns_chat_message_instance(self):
        ctx = _empty_ctx()
        sdk = _fake_sdk("facts")
        msg = await preload_synap_context(ctx, sdk, user_id="u1")
        assert isinstance(msg, ChatMessage)

    @pytest.mark.asyncio
    async def test_sdk_fetch_called_with_user_id(self):
        ctx = _empty_ctx()
        sdk = _fake_sdk("context")
        await preload_synap_context(ctx, sdk, user_id="user-abc")
        assert sdk.fetch.await_count == 1
        kw = sdk.fetch.call_args.kwargs
        assert kw["user_id"] == "user-abc"

    @pytest.mark.asyncio
    async def test_sdk_fetch_called_with_customer_id(self):
        ctx = _empty_ctx()
        sdk = _fake_sdk("context")
        await preload_synap_context(ctx, sdk, user_id="u1", customer_id="cust-99")
        kw = sdk.fetch.call_args.kwargs
        assert kw["customer_id"] == "cust-99"

    @pytest.mark.asyncio
    async def test_empty_customer_id_passed_as_none_to_sdk(self):
        """Empty customer_id is converted to None before sdk.fetch call."""
        ctx = _empty_ctx()
        sdk = _fake_sdk("context")
        await preload_synap_context(ctx, sdk, user_id="u1", customer_id="")
        kw = sdk.fetch.call_args.kwargs
        assert kw["customer_id"] is None

    @pytest.mark.asyncio
    async def test_sdk_fetch_called_with_mode(self):
        ctx = _empty_ctx()
        sdk = _fake_sdk("context")
        await preload_synap_context(ctx, sdk, user_id="u1", mode="fast")
        kw = sdk.fetch.call_args.kwargs
        assert kw["mode"] == "fast"

    @pytest.mark.asyncio
    async def test_sdk_fetch_called_with_max_results(self):
        ctx = _empty_ctx()
        sdk = _fake_sdk("context")
        await preload_synap_context(ctx, sdk, user_id="u1", max_results=5)
        kw = sdk.fetch.call_args.kwargs
        assert kw["max_results"] == 5

    @pytest.mark.asyncio
    async def test_search_query_passed_as_list(self):
        ctx = _empty_ctx()
        sdk = _fake_sdk("results for query")
        await preload_synap_context(ctx, sdk, user_id="u1", search_query="coffee")
        kw = sdk.fetch.call_args.kwargs
        assert kw["search_query"] == ["coffee"]

    @pytest.mark.asyncio
    async def test_no_search_query_passes_none(self):
        ctx = _empty_ctx()
        sdk = _fake_sdk("ambient context")
        await preload_synap_context(ctx, sdk, user_id="u1")
        kw = sdk.fetch.call_args.kwargs
        assert kw["search_query"] is None

    @pytest.mark.asyncio
    async def test_include_conversation_context_forwarded(self):
        ctx = _empty_ctx()
        sdk = _fake_sdk("context")
        await preload_synap_context(
            ctx, sdk, user_id="u1", include_conversation_context=True
        )
        kw = sdk.fetch.call_args.kwargs
        assert kw["include_conversation_context"] is True

    @pytest.mark.asyncio
    async def test_default_mode_is_accurate(self):
        ctx = _empty_ctx()
        sdk = _fake_sdk("ctx")
        await preload_synap_context(ctx, sdk, user_id="u1")
        kw = sdk.fetch.call_args.kwargs
        assert kw["mode"] == "accurate"

    @pytest.mark.asyncio
    async def test_default_max_results_is_20(self):
        ctx = _empty_ctx()
        sdk = _fake_sdk("ctx")
        await preload_synap_context(ctx, sdk, user_id="u1")
        kw = sdk.fetch.call_args.kwargs
        assert kw["max_results"] == 20


# ---------------------------------------------------------------------------
# No-op / empty context paths
# ---------------------------------------------------------------------------


class TestNoOp:
    @pytest.mark.asyncio
    async def test_returns_none_when_formatted_context_is_none(self):
        ctx = _empty_ctx()
        sdk = _fake_sdk(formatted_context=None)
        msg = await preload_synap_context(ctx, sdk, user_id="u1")
        assert msg is None
        assert len(ctx.items) == 0

    @pytest.mark.asyncio
    async def test_returns_none_when_formatted_context_is_empty_string(self):
        ctx = _empty_ctx()
        sdk = _fake_sdk(formatted_context="")
        msg = await preload_synap_context(ctx, sdk, user_id="u1")
        assert msg is None

    @pytest.mark.asyncio
    async def test_returns_none_when_formatted_context_is_whitespace_only(self):
        ctx = _empty_ctx()
        sdk = _fake_sdk(formatted_context="   \n  ")
        msg = await preload_synap_context(ctx, sdk, user_id="u1")
        assert msg is None
        assert len(ctx.items) == 0

    @pytest.mark.asyncio
    async def test_no_message_inserted_when_empty(self):
        ctx = _empty_ctx()
        ctx.items.append(ChatMessage(role="user", content=["hi"]))
        sdk = _fake_sdk(formatted_context="")
        await preload_synap_context(ctx, sdk, user_id="u1")
        # Only the original user message should remain
        assert len(ctx.items) == 1

    @pytest.mark.asyncio
    async def test_response_missing_formatted_context_attr_treated_as_empty(self):
        """Response object that has no formatted_context attr → no message injected."""
        ctx = _empty_ctx()
        sdk = MagicMock()
        resp = MagicMock(spec=[])  # no formatted_context
        sdk.fetch = AsyncMock(return_value=resp)
        msg = await preload_synap_context(ctx, sdk, user_id="u1")
        assert msg is None


# ---------------------------------------------------------------------------
# Failure / degradation paths
# ---------------------------------------------------------------------------


class TestFailureDegradation:
    @pytest.mark.asyncio
    async def test_sdk_fetch_exception_returns_none(self):
        ctx = _empty_ctx()
        sdk = MagicMock()
        sdk.fetch = AsyncMock(side_effect=RuntimeError("network down"))
        msg = await preload_synap_context(ctx, sdk, user_id="u1")
        assert msg is None

    @pytest.mark.asyncio
    async def test_sdk_fetch_exception_leaves_ctx_untouched(self):
        ctx = _empty_ctx()
        existing = ChatMessage(role="user", content=["hello"])
        ctx.items.append(existing)
        sdk = MagicMock()
        sdk.fetch = AsyncMock(side_effect=RuntimeError("network down"))
        await preload_synap_context(ctx, sdk, user_id="u1")
        # Context must be untouched
        assert len(ctx.items) == 1
        assert ctx.items[0] is existing

    @pytest.mark.asyncio
    async def test_sdk_fetch_exception_logs_error(self, caplog):
        ctx = _empty_ctx()
        sdk = MagicMock()
        sdk.fetch = AsyncMock(side_effect=RuntimeError("timeout"))
        with caplog.at_level(logging.ERROR, logger="synap_livekit_agents.context"):
            await preload_synap_context(ctx, sdk, user_id="u1")
        assert any("u1" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_sdk_fetch_value_error_also_swallowed(self):
        """ValueError from SDK is also caught and degraded gracefully."""
        ctx = _empty_ctx()
        sdk = MagicMock()
        sdk.fetch = AsyncMock(side_effect=ValueError("bad config"))
        msg = await preload_synap_context(ctx, sdk, user_id="u1")
        assert msg is None

    @pytest.mark.asyncio
    async def test_failing_sdk_fixture_degrades(self, failing_sdk):
        """Shared failing_sdk fixture — sdk.fetch raises — context left intact."""
        ctx = _empty_ctx()
        msg = await preload_synap_context(ctx, failing_sdk, user_id="u1")
        assert msg is None
        assert len(ctx.items) == 0


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_surface_exports():
    import synap_livekit_agents
    assert hasattr(synap_livekit_agents, "preload_synap_context")
    assert "preload_synap_context" in synap_livekit_agents.__all__
