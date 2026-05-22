"""Unit tests for the Phase 2 SDK helpers — _should_skip_server_st and
_merge_local_st_into_response."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from maximem_synap.cache.short_term_store import (
    CachedShortTermContext,
    ShortTermContextStore,
)
from maximem_synap.sdk import _merge_local_st_into_response, _should_skip_server_st
from maximem_synap.models.context import (
    ContextResponse,
    ConversationContextModel,
    ResponseMetadata,
)


def _make_sdk(*, authoritative: bool, store: ShortTermContextStore) -> SimpleNamespace:
    sdk = SimpleNamespace(_st_store=store)
    sdk._is_st_authoritative = lambda: authoritative
    return sdk


def _warm_store(conversation_id: str) -> ShortTermContextStore:
    store = ShortTermContextStore()
    store.apply_compaction({
        "_anticipation_conversation_id": conversation_id,
        "conversation_context": {
            "conversation_id": conversation_id,
            "summary": "cached summary",
            "compaction_id": "comp-cache-1",
            "current_state": {"status": "active"},
            "key_extractions": {"facts": [{"content": "x"}]},
            "end_timestamp": "2026-05-22T10:00:00+00:00",
        },
    })
    return store


def _cold_store_with_only_turns(conversation_id: str) -> ShortTermContextStore:
    store = ShortTermContextStore()
    store.append_turn(conversation_id, "user", "hi")
    return store


def _empty_response() -> ContextResponse:
    md = ResponseMetadata(
        correlation_id="corr-test",
        ttl_seconds=300,
        source="cloud",
        retrieved_at=datetime.now(timezone.utc),
    )
    return ContextResponse(metadata=md)


class TestShouldSkipServerST:
    def test_flag_off_never_skips(self):
        store = _warm_store("c1")
        sdk = _make_sdk(authoritative=False, store=store)
        assert _should_skip_server_st(sdk, "c1") is False

    def test_no_conversation_id_never_skips(self):
        sdk = _make_sdk(authoritative=True, store=_warm_store("c1"))
        assert _should_skip_server_st(sdk, None) is False
        assert _should_skip_server_st(sdk, "") is False

    def test_cold_cache_never_skips(self):
        sdk = _make_sdk(authoritative=True, store=ShortTermContextStore())
        assert _should_skip_server_st(sdk, "c1") is False

    def test_turns_only_no_compaction_does_not_skip(self):
        # If we only have raw turns but no compaction summary, we still
        # need the server's summary.
        sdk = _make_sdk(authoritative=True, store=_cold_store_with_only_turns("c1"))
        assert _should_skip_server_st(sdk, "c1") is False

    def test_warm_cache_skips(self):
        sdk = _make_sdk(authoritative=True, store=_warm_store("c1"))
        assert _should_skip_server_st(sdk, "c1") is True


class TestMergeLocalSTIntoResponse:
    def test_no_op_without_conversation_id(self):
        resp = _empty_response()
        _merge_local_st_into_response(_make_sdk(authoritative=True, store=_warm_store("c1")), resp, None)
        assert resp.conversation_context is None

    def test_no_op_when_entry_missing(self):
        resp = _empty_response()
        sdk = _make_sdk(authoritative=True, store=ShortTermContextStore())
        _merge_local_st_into_response(sdk, resp, "c1")
        assert resp.conversation_context is None

    def test_merges_summary_and_extractions(self):
        resp = _empty_response()
        store = _warm_store("c1")
        sdk = _make_sdk(authoritative=True, store=store)
        _merge_local_st_into_response(sdk, resp, "c1")
        assert isinstance(resp.conversation_context, ConversationContextModel)
        assert resp.conversation_context.summary == "cached summary"
        assert resp.conversation_context.compaction_id == "comp-cache-1"
        assert resp.conversation_context.key_extractions["facts"][0]["content"] == "x"

    def test_merges_recent_turns(self):
        store = _warm_store("c1")
        # add a turn that arrived after the compaction
        store.append_turn(
            "c1",
            "user",
            "post-compaction turn",
            timestamp=datetime(2026, 5, 22, 11, 0, 0, tzinfo=timezone.utc),
        )
        resp = _empty_response()
        sdk = _make_sdk(authoritative=True, store=store)
        _merge_local_st_into_response(sdk, resp, "c1")
        turns = resp.conversation_context.recent_turns
        assert len(turns) == 1
        assert turns[0]["content"] == "post-compaction turn"
        assert turns[0]["role"] == "user"

    def test_overwrites_existing_conversation_context(self):
        # If the response already had a (stale) conversation_context (e.g.
        # when an older server returned one despite the skip flag), the
        # local merge should replace it.
        resp = _empty_response()
        resp.conversation_context = ConversationContextModel(
            summary="STALE", conversation_id="c1"
        )
        sdk = _make_sdk(authoritative=True, store=_warm_store("c1"))
        _merge_local_st_into_response(sdk, resp, "c1")
        assert resp.conversation_context.summary == "cached summary"
