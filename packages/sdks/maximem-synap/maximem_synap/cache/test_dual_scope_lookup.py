"""Regression tests for AnticipationCache dual-scope conversation lookup.

A bundle pushed without an `_anticipation_conversation_id` (e.g. a
session-start profile bundle keyed by user_id only) used to be rejected
the moment a consumer queried with a specific conversation_id. The
dual-scope lookup keeps the cross-conversation privacy guarantee (a
bundle keyed for conv A is still rejected on a conv B lookup) while
treating user-scope bundles (no conv_id at push time) as applicable to
any conversation owned by the same user.
"""

from maximem_synap.cache.anticipation_cache import AnticipationCache


def _push_bundle(cache: AnticipationCache, *, user_id, conv_id, bundle_id, content):
    cache.store({
        "bundle_id": bundle_id,
        "_anticipation_user_id": user_id,
        "_anticipation_conversation_id": conv_id,
        "items_by_type": {
            "facts": [{"content": content, "confidence": 0.9}],
        },
        "search_queries": [content.split()[0]],
    })


def test_user_scope_bundle_matches_when_consumer_asks_with_conv_id():
    """A bundle pushed without conv_id (user-scope) is applicable to any
    conversation owned by that same user."""
    cache = AnticipationCache(ttl_seconds=300)
    _push_bundle(cache, user_id="alice", conv_id=None, bundle_id="b1",
                 content="alice likes window seats on flights")

    result = cache.lookup(
        search_query=["window seat preferences"],
        entity_id="alice",
        conversation_id="conv-42",
    )
    assert result is not None, (
        "user-scope bundle should match conversation-scoped lookup for same user"
    )
    items = result.get("items_by_type", {}).get("facts", [])
    assert any("window seat" in i["content"] for i in items)


def test_exact_conv_match_still_hits():
    cache = AnticipationCache(ttl_seconds=300)
    _push_bundle(cache, user_id="alice", conv_id="conv-42", bundle_id="b1",
                 content="alice asked about delayed flight")

    result = cache.lookup(
        search_query=["delayed flight status"],
        entity_id="alice",
        conversation_id="conv-42",
    )
    assert result is not None


def test_different_conv_id_still_rejected():
    """Cross-conversation isolation must hold. A bundle pushed for
    conv-42 must NOT leak into a lookup for conv-99."""
    cache = AnticipationCache(ttl_seconds=300)
    _push_bundle(cache, user_id="alice", conv_id="conv-42", bundle_id="b1",
                 content="confidential conv-42 only payload")

    result = cache.lookup(
        search_query=["confidential payload"],
        entity_id="alice",
        conversation_id="conv-99",
    )
    assert result is None, "bundle keyed to conv-42 must not match conv-99 lookup"


def test_different_user_still_rejected():
    """Cross-user isolation must hold (Section 15 privacy hardening)."""
    cache = AnticipationCache(ttl_seconds=300)
    _push_bundle(cache, user_id="alice", conv_id=None, bundle_id="b1",
                 content="alice's private memory content")

    result = cache.lookup(
        search_query=["alice private memory"],
        entity_id="bob",
        conversation_id="conv-42",
    )
    assert result is None, "bundle for alice must not leak into bob's lookup"
