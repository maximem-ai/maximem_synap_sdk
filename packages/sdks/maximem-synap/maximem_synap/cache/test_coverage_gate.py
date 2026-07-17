"""Coverage gate (WS3 of the echo-poisoning plan) — telemetry-first.

A bundle can clear the BM25 threshold on one shared token ("account") while
lacking the asked-for fact entirely — the false HIT behind the eval's
"forgot the name" failure. `lookup()` now always computes query-stem
coverage over the picked items and reports it via the lookup hook / log
lines; it ENFORCES a minimum only when SYNAP_SDK_CACHE_COVERAGE_MIN is
explicitly set.

Zero-regression contract, pinned here: with the env unset (default),
lookup results are identical to before — the only change is the additive
`coverage` field in hook payloads.
"""

from maximem_synap.cache.anticipation_cache import AnticipationCache


def _cache_with_bundle():
    cache = AnticipationCache(ttl_seconds=300)
    cache.store({
        "bundle_id": "b1",
        "_anticipation_user_id": "alice",
        "_anticipation_conversation_id": None,
        "items_by_type": {
            "facts": [
                {"content": "alice monthly ride budget is 300 dollars on account SYN-1", "confidence": 0.9},
                {"content": "tipping is optional and goes directly to drivers", "confidence": 0.9},
                {"content": "surge pricing applies during peak demand hours", "confidence": 0.9},
                {"content": "pets are allowed at the driver discretion", "confidence": 0.9},
            ],
        },
        "search_queries": ["budget"],
    })
    return cache


def _lookup(cache, query, hooks=None):
    if hooks is not None:
        cache.register_lookup_hook(hooks.append)
    return cache.lookup(search_query=[query], entity_id="alice")


# Query whose stems partially overlap one item: "account"/"budget" match,
# the asked-for concept ("holder name") does not — the eval's failure shape.
PARTIAL_QUERY = "account budget holder name"
# Query fully covered by item stems.
COVERED_QUERY = "alice ride budget account"


class TestDefaultUnchanged:
    def test_partial_coverage_still_hits_by_default(self, monkeypatch):
        """Env unset: today's behavior — BM25-only decision."""
        monkeypatch.delenv("SYNAP_SDK_CACHE_COVERAGE_MIN", raising=False)
        cache = _cache_with_bundle()
        hooks = []
        result = _lookup(cache, PARTIAL_QUERY, hooks)
        assert result is not None
        assert hooks and hooks[-1]["hit"] is True

    def test_coverage_reported_in_hook(self, monkeypatch):
        monkeypatch.delenv("SYNAP_SDK_CACHE_COVERAGE_MIN", raising=False)
        cache = _cache_with_bundle()
        hooks = []
        _lookup(cache, PARTIAL_QUERY, hooks)
        cov = hooks[-1]["coverage"]
        assert 0.0 < cov < 1.0  # "account" covered; "holder"/"name" not

    def test_full_coverage_reports_one(self, monkeypatch):
        monkeypatch.delenv("SYNAP_SDK_CACHE_COVERAGE_MIN", raising=False)
        cache = _cache_with_bundle()
        hooks = []
        result = _lookup(cache, COVERED_QUERY, hooks)
        assert result is not None
        assert hooks[-1]["coverage"] == 1.0


class TestGateEnforced:
    def test_low_coverage_misses_when_gate_set(self, monkeypatch):
        monkeypatch.setenv("SYNAP_SDK_CACHE_COVERAGE_MIN", "0.95")
        cache = _cache_with_bundle()
        hooks = []
        result = _lookup(cache, PARTIAL_QUERY, hooks)
        assert result is None
        assert hooks[-1]["exit_reason"] == "coverage_gate"
        assert hooks[-1]["hit"] is False

    def test_gate_rejection_leaves_state_untouched(self, monkeypatch):
        """A gated MISS must not refresh bundle stored_at — clearing the env
        immediately restores the exact pre-gate HIT."""
        monkeypatch.setenv("SYNAP_SDK_CACHE_COVERAGE_MIN", "0.95")
        cache = _cache_with_bundle()
        before = {bid: e.stored_at for bid, e in cache._entries.items()}
        assert _lookup(cache, PARTIAL_QUERY) is None
        after = {bid: e.stored_at for bid, e in cache._entries.items()}
        assert before == after
        monkeypatch.delenv("SYNAP_SDK_CACHE_COVERAGE_MIN", raising=False)
        assert cache.lookup(search_query=[PARTIAL_QUERY], entity_id="alice") is not None

    def test_covered_query_passes_gate(self, monkeypatch):
        monkeypatch.setenv("SYNAP_SDK_CACHE_COVERAGE_MIN", "0.95")
        cache = _cache_with_bundle()
        assert _lookup(cache, COVERED_QUERY) is not None

    def test_low_threshold_passes(self, monkeypatch):
        monkeypatch.setenv("SYNAP_SDK_CACHE_COVERAGE_MIN", "0.05")
        cache = _cache_with_bundle()
        assert _lookup(cache, PARTIAL_QUERY) is not None

    def test_malformed_threshold_is_observe_only(self, monkeypatch):
        monkeypatch.setenv("SYNAP_SDK_CACHE_COVERAGE_MIN", "not-a-number")
        cache = _cache_with_bundle()
        assert _lookup(cache, PARTIAL_QUERY) is not None
