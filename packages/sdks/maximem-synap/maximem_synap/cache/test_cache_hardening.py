"""2026-07-16 eval-register hardening: absolute entry age cap (#6),
scope-qualified item dedup (#9), and write-path invalidation (#8).

Contracts pinned here:
- The sliding lease (hit renews ``stored_at``) is now bounded by an absolute
  cap on ``created_at`` — default 2x TTL, ``SYNAP_SDK_CACHE_MAX_ENTRY_AGE``
  overrides, ``0`` disables. Before this, a bundle hit at least once per TTL
  window never expired.
- Item dedup is keyed per bundle scope, so two visitors' byte-identical seed
  facts both index (previously the second visitor's copy was swallowed and
  scope-excluded forever).
- ``invalidate_entity`` drops exactly the given scope's bundles; the SDK
  write paths call it only when SYNAP_SDK_CACHE_INVALIDATE_ON_WRITE is on
  (default off).
"""

import time

from maximem_synap.cache.anticipation_cache import (
    AnticipationCache,
    invalidate_on_write_enabled,
)


def _bundle(bundle_id, entity, content, queries=("account file records",)):
    return {
        "bundle_id": bundle_id,
        "_anticipation_user_id": entity,
        "_anticipation_conversation_id": None,
        "items_by_type": {
            "facts": [{"content": content, "confidence": 0.9}],
        },
        "search_queries": list(queries),
    }


SEED_FACT = "the company refund policy allows requests within thirty days"
QUERY = "refund policy requests thirty days"


class TestAbsoluteAgeCap:
    def test_hit_renewal_cannot_outlive_the_cap(self, monkeypatch):
        monkeypatch.delenv("SYNAP_SDK_CACHE_MAX_ENTRY_AGE", raising=False)
        monkeypatch.delenv("SYNAP_SDK_CACHE_HONOR_TTL_HINT", raising=False)
        cache = AnticipationCache(ttl_seconds=300)
        cache.store(_bundle("b1", "alice", SEED_FACT))
        entry = cache._entries["b1"]
        # Simulate a bundle kept alive by steady hits: lease is fresh but the
        # entry was created beyond the 2x-TTL cap.
        entry.stored_at = time.monotonic()
        entry.created_at = time.monotonic() - 601
        assert cache.lookup(search_query=[QUERY], entity_id="alice") is None
        assert "b1" not in cache._entries

    def test_env_zero_disables_the_cap(self, monkeypatch):
        monkeypatch.setenv("SYNAP_SDK_CACHE_MAX_ENTRY_AGE", "0")
        monkeypatch.delenv("SYNAP_SDK_CACHE_HONOR_TTL_HINT", raising=False)
        cache = AnticipationCache(ttl_seconds=300)
        cache.store(_bundle("b1", "alice", SEED_FACT))
        entry = cache._entries["b1"]
        entry.stored_at = time.monotonic()
        entry.created_at = time.monotonic() - 10_000
        assert cache.lookup(search_query=[QUERY], entity_id="alice") is not None

    def test_fresh_entry_unaffected(self, monkeypatch):
        monkeypatch.delenv("SYNAP_SDK_CACHE_MAX_ENTRY_AGE", raising=False)
        cache = AnticipationCache(ttl_seconds=300)
        cache.store(_bundle("b1", "alice", SEED_FACT))
        assert cache.lookup(search_query=[QUERY], entity_id="alice") is not None

    def test_pre_field_entries_fall_back_to_stored_at(self, monkeypatch):
        """Entries with created_at=0.0 (pre-upgrade) must not insta-expire."""
        monkeypatch.delenv("SYNAP_SDK_CACHE_MAX_ENTRY_AGE", raising=False)
        cache = AnticipationCache(ttl_seconds=300)
        cache.store(_bundle("b1", "alice", SEED_FACT))
        cache._entries["b1"].created_at = 0.0
        assert cache.lookup(search_query=[QUERY], entity_id="alice") is not None


class TestScopedDedup:
    def test_identical_seed_facts_index_for_both_visitors(self):
        cache = AnticipationCache(ttl_seconds=300)
        cache.store(_bundle("bundle_a", "visitor_a", SEED_FACT))
        cache.store(_bundle("bundle_b", "visitor_b", SEED_FACT))
        assert len(cache._items) == 2
        for visitor in ("visitor_a", "visitor_b"):
            result = cache.lookup(search_query=[QUERY], entity_id=visitor)
            assert result is not None, f"{visitor} should hit its own copy"

    def test_within_scope_dedup_still_applies(self):
        cache = AnticipationCache(ttl_seconds=300)
        cache.store(_bundle("bundle_1", "visitor_a", SEED_FACT))
        cache.store(_bundle("bundle_2", "visitor_a", SEED_FACT))
        assert len(cache._items) == 1

    def test_rebuild_preserves_scoped_keys(self):
        cache = AnticipationCache(ttl_seconds=300)
        cache.store(_bundle("bundle_a", "visitor_a", SEED_FACT))
        cache.store(_bundle("bundle_b", "visitor_b", SEED_FACT))
        cache._remove_bundle("bundle_a")
        # After the rebuild, visitor_b's copy must survive and still hit.
        assert cache.lookup(search_query=[QUERY], entity_id="visitor_b") is not None


class TestInvalidateEntity:
    def test_drops_only_the_given_scope(self):
        cache = AnticipationCache(ttl_seconds=300)
        cache.store(_bundle("bundle_a", "visitor_a", SEED_FACT))
        cache.store(_bundle("bundle_b", "visitor_b", SEED_FACT))
        dropped = cache.invalidate_entity("visitor_a")
        assert dropped == 1
        assert cache.lookup(search_query=[QUERY], entity_id="visitor_a") is None
        assert cache.lookup(search_query=[QUERY], entity_id="visitor_b") is not None

    def test_any_scope_bundles_survive(self):
        cache = AnticipationCache(ttl_seconds=300)
        shared = _bundle("bundle_shared", None, SEED_FACT)
        shared.pop("_anticipation_user_id")
        cache.store(shared)  # lands under "_any"
        assert cache.invalidate_entity("visitor_a") == 0
        assert "bundle_shared" in cache._entries

    def test_empty_entity_is_a_noop(self):
        cache = AnticipationCache(ttl_seconds=300)
        cache.store(_bundle("bundle_a", "visitor_a", SEED_FACT))
        assert cache.invalidate_entity("") == 0
        assert len(cache._entries) == 1


class TestInvalidateOnWriteFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("SYNAP_SDK_CACHE_INVALIDATE_ON_WRITE", raising=False)
        assert invalidate_on_write_enabled() is False

    def test_env_on(self, monkeypatch):
        monkeypatch.setenv("SYNAP_SDK_CACHE_INVALIDATE_ON_WRITE", "true")
        assert invalidate_on_write_enabled() is True
