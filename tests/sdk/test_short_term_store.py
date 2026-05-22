"""Unit tests for ShortTermContextStore."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from maximem_synap.cache.short_term_store import (
    CachedShortTermContext,
    ShortTermContextStore,
)


def _bundle(
    conversation_id: str,
    summary: str = "test summary",
    compaction_id: str = "comp-1",
    end_timestamp: str | None = None,
    extras: dict | None = None,
) -> dict:
    """Build a minimal compaction_update bundle dict."""
    cc = {
        "conversation_id": conversation_id,
        "summary": summary,
        "compaction_id": compaction_id,
        "current_state": {"status": "active"},
        "key_extractions": {"facts": ["x"]},
    }
    if end_timestamp is not None:
        cc["end_timestamp"] = end_timestamp
        cc["compacted_at"] = end_timestamp
    if extras:
        cc.update(extras)
    return {
        "_bundle_type": "compaction_update",
        "_anticipation_conversation_id": conversation_id,
        "conversation_context": cc,
    }


class TestAppendTurn:
    def test_first_append_creates_entry(self):
        store = ShortTermContextStore()
        store.append_turn("c1", "user", "hello")
        e = store.get("c1")
        assert e is not None
        assert e.conversation_id == "c1"
        assert len(e.recent_turns) == 1
        assert e.recent_turns[0]["content"] == "hello"
        assert e.summary is None  # not yet compacted

    def test_appends_preserve_order(self):
        store = ShortTermContextStore()
        store.append_turn("c1", "user", "one")
        store.append_turn("c1", "assistant", "two")
        store.append_turn("c1", "user", "three")
        contents = [t["content"] for t in store.get("c1").recent_turns]
        assert contents == ["one", "two", "three"]

    def test_explicit_timestamp_is_used(self):
        store = ShortTermContextStore()
        ts = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
        store.append_turn("c1", "user", "hi", timestamp=ts)
        assert store.get("c1").recent_turns[0]["timestamp"] == ts.isoformat()

    def test_empty_conversation_id_is_noop(self):
        store = ShortTermContextStore()
        store.append_turn("", "user", "hi")
        assert store.size() == 0


class TestApplyCompaction:
    def test_empty_cache_creates_entry(self):
        store = ShortTermContextStore()
        store.apply_compaction(_bundle("c1", summary="hello"))
        e = store.get("c1")
        assert e is not None
        assert e.summary == "hello"
        assert e.compaction_id == "comp-1"
        assert e.recent_turns == []

    def test_replaces_summary_keeps_unmatched_turns(self):
        store = ShortTermContextStore()
        ts_old = datetime(2026, 5, 22, 10, 0, 0, tzinfo=timezone.utc)
        ts_new = datetime(2026, 5, 22, 11, 0, 0, tzinfo=timezone.utc)
        store.append_turn("c1", "user", "old turn", timestamp=ts_old)
        store.append_turn("c1", "user", "new turn", timestamp=ts_new)

        cutoff = datetime(2026, 5, 22, 10, 30, 0, tzinfo=timezone.utc)
        store.apply_compaction(_bundle("c1", end_timestamp=cutoff.isoformat()))

        e = store.get("c1")
        assert e.summary == "test summary"
        # Old turn dropped, new turn kept
        assert len(e.recent_turns) == 1
        assert e.recent_turns[0]["content"] == "new turn"

    def test_cutoff_before_all_turns_keeps_all(self):
        store = ShortTermContextStore()
        ts = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
        store.append_turn("c1", "user", "a", timestamp=ts)
        store.append_turn("c1", "user", "b", timestamp=ts + timedelta(seconds=1))

        cutoff = datetime(2026, 5, 22, 11, 0, 0, tzinfo=timezone.utc)
        store.apply_compaction(_bundle("c1", end_timestamp=cutoff.isoformat()))
        assert len(store.get("c1").recent_turns) == 2

    def test_cutoff_after_all_turns_drops_all(self):
        store = ShortTermContextStore()
        ts = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
        store.append_turn("c1", "user", "a", timestamp=ts)
        store.append_turn("c1", "user", "b", timestamp=ts + timedelta(seconds=1))

        cutoff = datetime(2026, 5, 22, 13, 0, 0, tzinfo=timezone.utc)
        store.apply_compaction(_bundle("c1", end_timestamp=cutoff.isoformat()))
        assert store.get("c1").recent_turns == []

    def test_cutoff_exactly_equal_is_dropped(self):
        # The reconciliation rule is "drop ≤ end_timestamp", not "<".
        store = ShortTermContextStore()
        ts = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
        store.append_turn("c1", "user", "boundary", timestamp=ts)
        store.apply_compaction(_bundle("c1", end_timestamp=ts.isoformat()))
        assert store.get("c1").recent_turns == []

    def test_bundle_with_no_conversation_id_is_noop(self):
        store = ShortTermContextStore()
        store.apply_compaction({"_bundle_type": "compaction_update"})
        assert store.size() == 0

    def test_falls_back_to_anticipation_conversation_id(self):
        store = ShortTermContextStore()
        bundle = {
            "_bundle_type": "compaction_update",
            "_anticipation_conversation_id": "fallback-cid",
            "conversation_context": {
                "summary": "from-fallback",
                "compaction_id": "c-1",
            },
        }
        store.apply_compaction(bundle)
        assert store.get("fallback-cid") is not None
        assert store.get("fallback-cid").summary == "from-fallback"

    def test_handles_z_suffixed_iso_timestamps(self):
        store = ShortTermContextStore()
        store.append_turn(
            "c1",
            "user",
            "z-form",
            timestamp=datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc),
        )
        store.apply_compaction(_bundle("c1", end_timestamp="2026-05-22T13:00:00Z"))
        assert store.get("c1").recent_turns == []


class TestEviction:
    def test_lru_evicts_oldest(self):
        store = ShortTermContextStore(max_conversations=3)
        store.append_turn("c1", "user", "x")
        store.append_turn("c2", "user", "x")
        store.append_turn("c3", "user", "x")
        # Touch c1 so c2 is now LRU
        store.get("c1")
        store.append_turn("c4", "user", "x")
        assert store.get("c2") is None
        assert store.get("c1") is not None
        assert store.get("c3") is not None
        assert store.get("c4") is not None

    def test_age_eviction(self):
        store = ShortTermContextStore(max_age=timedelta(seconds=0))
        store.append_turn("c1", "user", "x")
        # Anything > 0s old is evicted on read; force the activity ts back
        e = store._cache["c1"]
        e.last_activity_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert store.get("c1") is None


class TestInvalidate:
    def test_invalidate_removes(self):
        store = ShortTermContextStore()
        store.append_turn("c1", "user", "x")
        store.invalidate("c1")
        assert store.get("c1") is None

    def test_clear_removes_all(self):
        store = ShortTermContextStore()
        store.append_turn("c1", "user", "x")
        store.append_turn("c2", "user", "x")
        store.clear()
        assert store.size() == 0


class TestCachedShortTermContextSerialization:
    def test_to_dict_round_trips_compaction(self):
        store = ShortTermContextStore()
        store.apply_compaction(
            _bundle("c1", summary="s", end_timestamp="2026-05-22T10:00:00+00:00")
        )
        d = store.get("c1").to_dict()
        assert d["conversation_id"] == "c1"
        assert d["summary"] == "s"
        assert d["compaction_id"] == "comp-1"
        assert d["end_timestamp"] == "2026-05-22T10:00:00+00:00"
