"""WS4 recall bypass (SYNAP_SDK_CACHE_RECALL_BYPASS) — playground query hygiene.

A recall-shaped question ("remind me…", "what do you have on file…", "where am
I located again?") asks the agent to go LOOK SOMETHING UP; serving it from a
pre-fetched anticipation bundle risks the eval-measured failure where the
agent denies a fact the store holds (false HIT at score 2.86 / coverage 1.00 —
no score-side gate can catch it). With the flag on, such queries always miss
the anticipation cache and fall through to a fresh fetch.

Zero-regression contract, pinned here: with the env unset (default), lookup
results are byte-identical to before.
"""

from maximem_synap.cache.anticipation_cache import (
    AnticipationCache,
    is_recall_query,
)


def _cache_with_bundle():
    cache = AnticipationCache(ttl_seconds=300)
    cache.store({
        "bundle_id": "b1",
        "_anticipation_user_id": "alice",
        "_anticipation_conversation_id": None,
        "items_by_type": {
            "facts": [
                {"content": "alice is located in california and her account file is SYN-1", "confidence": 0.9},
                {"content": "alice ride refund case is under review with the records team", "confidence": 0.9},
                {"content": "surge pricing applies during peak demand hours", "confidence": 0.9},
                {"content": "tipping is optional and goes directly to drivers", "confidence": 0.9},
            ],
        },
        "search_queries": ["located account file records"],
    })
    return cache


def _lookup(cache, query, hooks=None):
    if hooks is not None:
        cache.register_lookup_hook(hooks.append)
    return cache.lookup(search_query=[query], entity_id="alice")


# Recall-shaped AND stem-overlapping the stored items, so that DEFAULT
# behavior is a genuine HIT — which makes the flag-off pin meaningful and
# the flag-on bypass observable.
RECALL_QUERY = "where am i located again? check my account file records"
# Task-shaped query with the same stem overlap — must keep hitting with the
# flag ON (the bypass is about question shape, not vocabulary).
TASK_QUERY = "book a ride to the california office and bill the account file"


class TestDefaultUnchanged:
    def test_recall_query_still_hits_by_default(self, monkeypatch):
        monkeypatch.delenv("SYNAP_SDK_CACHE_RECALL_BYPASS", raising=False)
        cache = _cache_with_bundle()
        hooks = []
        result = _lookup(cache, RECALL_QUERY, hooks)
        assert result is not None
        assert hooks and hooks[-1]["hit"] is True

    def test_flag_explicit_false_unchanged(self, monkeypatch):
        monkeypatch.setenv("SYNAP_SDK_CACHE_RECALL_BYPASS", "false")
        cache = _cache_with_bundle()
        assert _lookup(cache, RECALL_QUERY) is not None


class TestBypassOn:
    def test_recall_query_bypasses(self, monkeypatch):
        monkeypatch.setenv("SYNAP_SDK_CACHE_RECALL_BYPASS", "true")
        cache = _cache_with_bundle()
        hooks = []
        result = _lookup(cache, RECALL_QUERY, hooks)
        assert result is None
        assert hooks[-1]["hit"] is False
        assert hooks[-1]["exit_reason"] == "recall_bypass"

    def test_task_query_still_hits(self, monkeypatch):
        monkeypatch.setenv("SYNAP_SDK_CACHE_RECALL_BYPASS", "true")
        cache = _cache_with_bundle()
        hooks = []
        result = _lookup(cache, TASK_QUERY, hooks)
        assert result is not None
        assert hooks[-1]["hit"] is True

    def test_empty_query_unaffected(self, monkeypatch):
        """No-query lookups route to the freshness path exactly as before."""
        monkeypatch.setenv("SYNAP_SDK_CACHE_RECALL_BYPASS", "true")
        cache = _cache_with_bundle()
        # Must not raise; freshness lookup semantics are out of scope here.
        cache.lookup(search_query=[], entity_id="alice")
        cache.lookup(search_query=None, entity_id="alice")

    def test_multi_query_any_recall_bypasses(self, monkeypatch):
        """One recall-shaped query in the list is enough to bypass."""
        monkeypatch.setenv("SYNAP_SDK_CACHE_RECALL_BYPASS", "true")
        cache = _cache_with_bundle()
        result = cache.lookup(
            search_query=[TASK_QUERY, "remind me of my account file records"],
            entity_id="alice",
        )
        assert result is None


class TestRecallHeuristic:
    RECALL = [
        "remind me what account number you have on file for me?",
        "where am I located again?",
        "how long have i been using Uber now?",
        "which of my open ride issues should we address first?",
        "what's my local sales tax then?",
        "what email will you use to contact me?",
        "do you remember my seating preference?",
        "what did I say about my budget earlier?",
        "is my address still in your records?",
    ]
    NOT_RECALL = [
        "book a ride to the airport",
        "book me a ride again",  # bare "again" is not a recall marker
        "tell me about surge pricing",
        "i want to dispute a charge on my last trip",
        "can you waive the cancellation fee?",
        "my email is alice@example.com",
    ]

    def test_recall_positives(self):
        for q in self.RECALL:
            assert is_recall_query([q]), f"should be recall-shaped: {q!r}"

    def test_non_recall_negatives(self):
        for q in self.NOT_RECALL:
            assert not is_recall_query([q]), f"should NOT be recall-shaped: {q!r}"

    def test_none_and_empty(self):
        assert not is_recall_query(None)
        assert not is_recall_query([])
        assert not is_recall_query([None, ""])
