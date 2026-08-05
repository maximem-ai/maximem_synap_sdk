"""Unit tests for the Phase 2 SDK helpers — _should_skip_server_st and
_overlay_local_recent_turns.

``_merge_local_st_into_response`` was renamed to ``_overlay_local_recent_turns``
when the verbatim-overlay design landed, and this module was not updated. The
stale import made it uncollectable, so pytest aborted the entire run on a
collection error instead of reporting a failure. The rename also narrowed the
contract: the local store supplies the verbatim tail, and the server stays
authoritative for whichever compacted fields it did send."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from maximem_synap.cache.short_term_store import (
    CachedShortTermContext,
    ShortTermContextStore,
)
from maximem_synap.sdk import _overlay_local_recent_turns, _should_skip_server_st
from maximem_synap.models.context import (
    ContextResponse,
    ConversationContextModel,
    ResponseMetadata,
)


# The short-term store evicts an entry on read once its last_activity_at is
# older than max_age (12h). append_turn() sets last_activity_at from the
# timestamp it is given, so a literal past date silently emptied the store and
# the overlay assertions below became unreachable. Anchor to the current run.
_NOW = datetime.now(timezone.utc)


def _ago(**offset) -> datetime:
    return _NOW - timedelta(**offset)


def _make_sdk(
    *,
    authoritative: bool,
    store: ShortTermContextStore,
    verbatim_overlay: bool = True,
) -> SimpleNamespace:
    sdk = SimpleNamespace(_st_store=store)
    sdk._is_st_authoritative = lambda: authoritative
    # Kill-switch consulted by _overlay_local_recent_turns. Absent from the
    # stub, every overlay attempt died in the helper's defensive except and
    # logged instead of asserting.
    sdk._is_st_verbatim_overlay = lambda: verbatim_overlay
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
            "end_timestamp": _ago(hours=2).isoformat(),
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
        _overlay_local_recent_turns(_make_sdk(authoritative=True, store=_warm_store("c1")), resp, None)
        assert resp.conversation_context is None

    def test_no_op_when_entry_missing(self):
        resp = _empty_response()
        sdk = _make_sdk(authoritative=True, store=ShortTermContextStore())
        _overlay_local_recent_turns(sdk, resp, "c1")
        assert resp.conversation_context is None

    def test_merges_summary_and_extractions(self):
        resp = _empty_response()
        store = _warm_store("c1")
        sdk = _make_sdk(authoritative=True, store=store)
        _overlay_local_recent_turns(sdk, resp, "c1")
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
            timestamp=_ago(hours=1),
        )
        resp = _empty_response()
        sdk = _make_sdk(authoritative=True, store=store)
        _overlay_local_recent_turns(sdk, resp, "c1")
        turns = resp.conversation_context.recent_turns
        assert len(turns) == 1
        assert turns[0]["content"] == "post-compaction turn"
        assert turns[0]["role"] == "user"

    def test_server_compacted_fields_survive_the_overlay(self):
        # The server's conversation_context is authoritative for compacted
        # fields; only the verbatim tail is replaced from the local store.
        store = _warm_store("c1")
        store.append_turn(
            "c1",
            "user",
            "post-compaction turn",
            timestamp=_ago(hours=1),
        )
        resp = _empty_response()
        resp.conversation_context = ConversationContextModel(
            summary="from server", conversation_id="c1"
        )
        sdk = _make_sdk(authoritative=True, store=store)
        _overlay_local_recent_turns(sdk, resp, "c1")
        assert resp.conversation_context.summary == "from server"
        assert [t["content"] for t in resp.conversation_context.recent_turns] == [
            "post-compaction turn"
        ]

    def test_kill_switch_leaves_a_server_context_untouched(self):
        resp = _empty_response()
        resp.conversation_context = ConversationContextModel(
            summary="from server", conversation_id="c1"
        )
        sdk = _make_sdk(
            authoritative=True, store=_warm_store("c1"), verbatim_overlay=False
        )
        _overlay_local_recent_turns(sdk, resp, "c1")
        assert resp.conversation_context.summary == "from server"
        assert resp.conversation_context.recent_turns == []

    def test_local_fills_the_fields_the_server_omitted(self):
        # Skip-server-ST path: no conversation_context at all, so the local
        # entry supplies the compacted fields as well as the tail.
        resp = _empty_response()
        sdk = _make_sdk(authoritative=True, store=_warm_store("c1"))
        _overlay_local_recent_turns(sdk, resp, "c1")
        assert resp.conversation_context.summary == "cached summary"
        assert resp.conversation_context.compaction_id == "comp-cache-1"
