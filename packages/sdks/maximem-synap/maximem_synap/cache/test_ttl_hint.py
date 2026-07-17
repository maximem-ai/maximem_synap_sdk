"""TTL-hint honoring (WS3, SYNAP_SDK_CACHE_HONOR_TTL_HINT — default OFF).

Anticipation bundles are pre-knowledge-update snapshots. Today a stale
snapshot lives for the global TTL (30 min) and every cache hit renews its
lease, so a retired value could be served indefinitely under steady
traffic. With the flag on: a bundle's server-sent ttl_hint_seconds bounds
its lifetime (shorten-only — a hint above the global TTL is capped), and
hits stop renewing hinted bundles.

Zero-regression contract pinned here: flag OFF (default) ⇒ eviction and
hit-refresh behavior byte-identical to today, hints ignored; and even with
the flag ON, un-hinted bundles behave exactly as today.
"""

import time
from unittest import mock

from maximem_synap.cache.anticipation_cache import AnticipationCache


def _push(cache, *, bundle_id, ttl_hint=None, content="alice likes window seats"):
    bundle = {
        "bundle_id": bundle_id,
        "_anticipation_user_id": "alice",
        "_anticipation_conversation_id": None,
        "items_by_type": {"facts": [{"content": content, "confidence": 0.9}]},
        "search_queries": ["seats"],
    }
    if ttl_hint is not None:
        bundle["_ttl_hint_seconds"] = ttl_hint
    cache.store(bundle)


def _age(cache, bundle_id, seconds):
    cache._entries[bundle_id].stored_at = time.monotonic() - seconds


def _hit(cache):
    return cache.lookup(search_query=["window seats"], entity_id="alice")


class TestFlagOffUnchanged:
    def test_hint_ignored_by_default(self, monkeypatch):
        """A short hint does not evict when the flag is off."""
        monkeypatch.delenv("SYNAP_SDK_CACHE_HONOR_TTL_HINT", raising=False)
        cache = AnticipationCache(ttl_seconds=1800)
        _push(cache, bundle_id="b1", ttl_hint=60)
        _age(cache, "b1", 120)  # older than hint, younger than global TTL
        assert _hit(cache) is not None

    def test_hit_still_refreshes_lease_by_default(self, monkeypatch):
        monkeypatch.delenv("SYNAP_SDK_CACHE_HONOR_TTL_HINT", raising=False)
        cache = AnticipationCache(ttl_seconds=1800)
        _push(cache, bundle_id="b1", ttl_hint=60)
        _age(cache, "b1", 120)
        before = cache._entries["b1"].stored_at
        assert _hit(cache) is not None
        assert cache._entries["b1"].stored_at > before  # LRU refresh kept


class TestFlagOnHonorsHint:
    def test_hinted_bundle_expires_on_hint(self, monkeypatch):
        monkeypatch.setenv("SYNAP_SDK_CACHE_HONOR_TTL_HINT", "true")
        cache = AnticipationCache(ttl_seconds=1800)
        _push(cache, bundle_id="b1", ttl_hint=60)
        _age(cache, "b1", 120)
        assert _hit(cache) is None
        assert "b1" not in cache._entries

    def test_hinted_bundle_alive_within_hint(self, monkeypatch):
        monkeypatch.setenv("SYNAP_SDK_CACHE_HONOR_TTL_HINT", "true")
        cache = AnticipationCache(ttl_seconds=1800)
        _push(cache, bundle_id="b1", ttl_hint=60)
        _age(cache, "b1", 30)
        assert _hit(cache) is not None

    def test_unhinted_bundle_keeps_global_ttl(self, monkeypatch):
        """Flag on, no hint: today's behavior exactly."""
        monkeypatch.setenv("SYNAP_SDK_CACHE_HONOR_TTL_HINT", "true")
        cache = AnticipationCache(ttl_seconds=1800)
        _push(cache, bundle_id="b1", ttl_hint=None)
        _age(cache, "b1", 120)
        before = cache._entries["b1"].stored_at
        assert _hit(cache) is not None
        assert cache._entries["b1"].stored_at > before  # refresh kept too

    def test_hint_can_only_shorten_never_extend(self, monkeypatch):
        monkeypatch.setenv("SYNAP_SDK_CACHE_HONOR_TTL_HINT", "true")
        cache = AnticipationCache(ttl_seconds=100)
        _push(cache, bundle_id="b1", ttl_hint=9999)
        _age(cache, "b1", 150)  # beyond global TTL
        assert _hit(cache) is None

    def test_hit_does_not_renew_hinted_lease(self, monkeypatch):
        """The lease-renewal kill: repeated hits must not keep a hinted
        snapshot alive past its hint."""
        monkeypatch.setenv("SYNAP_SDK_CACHE_HONOR_TTL_HINT", "true")
        cache = AnticipationCache(ttl_seconds=1800)
        _push(cache, bundle_id="b1", ttl_hint=60)
        _age(cache, "b1", 30)
        stored_before = cache._entries["b1"].stored_at
        assert _hit(cache) is not None
        assert cache._entries["b1"].stored_at == stored_before  # no renewal
        _age(cache, "b1", 120)
        assert _hit(cache) is None  # dead on schedule despite the hit


class TestServerHintHelper:
    def test_default_unchanged_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("SYNAP_ANTICIPATION_BUNDLE_TTL_HINT_S", raising=False)
        from synap.cloud.agents.anticipation_agent.tools import _bundle_ttl_hint_seconds
        assert _bundle_ttl_hint_seconds(1800) == 1800
        assert _bundle_ttl_hint_seconds(None) == 300

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("SYNAP_ANTICIPATION_BUNDLE_TTL_HINT_S", "120")
        from synap.cloud.agents.anticipation_agent.tools import _bundle_ttl_hint_seconds
        assert _bundle_ttl_hint_seconds(1800) == 120

    def test_malformed_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("SYNAP_ANTICIPATION_BUNDLE_TTL_HINT_S", "soon")
        from synap.cloud.agents.anticipation_agent.tools import _bundle_ttl_hint_seconds
        assert _bundle_ttl_hint_seconds(1800) == 1800
