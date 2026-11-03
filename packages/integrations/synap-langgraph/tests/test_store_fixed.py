"""The inverse of the characterisation proofs for SynapStore.

Each test here is the FIXED behaviour of a test that pinned the wrong one
(``synap_langgraph_proofs/test_proofs.py``, ``test_created_at.py``,
``test_integration_impact.py``); the name says which.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import pytest
from langgraph.store.base import SearchItem

from maximem_synap.models.context import ContextResponse, ResponseMetadata, UnifiedContextResponse
from synap_integrations_common import SynapIntegrationError, coerce_conversation_id
from synap_langgraph import SynapStore
from synap_langgraph.store import UNKNOWN_DATE

from ._support import (
    SAID, Sdk, UUIDS, fact, ranked, realistic_response, response_of, tagged,
)


def run(coro):
    return asyncio.run(coro)


def store_for(sdk, **kw):
    return SynapStore(sdk, user_id="alice", customer_id="acme", **kw)


# ── S-1 inverted: every provenance field survives, the client item is told apart ──

def test_S1_inverted_client_scope_item_is_labelled_and_distinguishable():
    results = run(store_for(Sdk()).asearch(("alice",), query="refund policy", limit=10))
    by_text = {r.value["_raw"]: r for r in results}
    policy = by_text["Refunds must be filed within 30 days"]
    ride = by_text["The user takes a monthly ride on the 10th"]
    assert policy.value["_synap"]["scope"] == "client"
    assert ride.value["_synap"]["scope"] == "user"
    assert policy.value["_synap"]["scope"] != ride.value["_synap"]["scope"]


def test_S1_inverted_the_synap_block_has_the_listed_fields():
    resp = response_of(ranked(
        "f9", 1, "facts", "Moved to Paris", 0.77, "user",
        source="conv", said_at=SAID, event_date=datetime(2023, 6, 11, tzinfo=timezone.utc),
        valid_until=datetime(2030, 1, 1, tzinfo=timezone.utc),
        metadata={"assertion_source": "assistant", "subject_id": "s1"}))
    item = run(store_for(Sdk(resp)).asearch(("alice",), query="q"))[0]
    meta = item.value["_synap"]
    assert meta == {
        "id": "f9", "category": "facts", "type": "fact", "scope": "user",
        "assertion_source": "assistant",
        "said_at": SAID.isoformat(), "event_date": "2023-06-11T00:00:00+00:00",
        "valid_until": "2030-01-01T00:00:00+00:00", "dates_known": True,
        "source": "conv", "rank": 1, "rank_score": 0.77, "confidence": 0.5,
        "metadata": {"subject_id": "s1"},
    }
    assert item.value["_raw"] == "Moved to Paris"
    assert "lg_store" not in str(meta)


def test_S1_inverted_scope_falls_back_to_the_responses_scope_map():
    it = ranked("f1", 1, "facts", "x", 0.5, "user")
    it.metadata.pop("scope")
    resp = response_of(it)
    resp.scope_map = {"f1": "customer"}
    assert run(store_for(Sdk(resp)).asearch(("a",), query="q"))[0].value["_synap"]["scope"] == "customer"


# ── S-2 inverted: ranked order, rank score ──────────────────────────────────

def test_S2_inverted_limit_returns_the_best_ranked_items_first():
    results = run(store_for(Sdk()).asearch(("alice",), query="anything", limit=2))
    assert [r.value["_raw"] for r in results] == [
        "Prefers email contact", "Refunds must be filed within 30 days"]
    assert "The user takes a monthly ride on the 10th" not in [r.value["_raw"] for r in results]


def test_S2_inverted_score_is_rank_score_and_order_is_the_servers():
    hi_conf = fact("a", "low-ranked but confident", "user", conf=0.99)
    lo_conf = fact("b", "top-ranked but unsure", "user", conf=0.10)
    resp = UnifiedContextResponse(
        facts=[hi_conf, lo_conf],
        items=[ranked("b", 1, "facts", lo_conf.content, 0.95, "user"),
               ranked("a", 2, "facts", hi_conf.content, 0.05, "user")],
        scopes_queried=["user"], total_items=2)
    results = run(store_for(Sdk(resp)).asearch(("alice",), query="x", limit=10))
    assert [(r.value["_raw"], r.score) for r in results] == [
        ("top-ranked but unsure", 0.95), ("low-ranked but confident", 0.05)]


def test_S2_inverted_offset_and_limit_apply_to_the_ranked_list():
    results = run(store_for(Sdk()).asearch(("alice",), query="q", limit=1, offset=1))
    assert [r.value["_raw"] for r in results] == ["Refunds must be filed within 30 days"]
    sdk = Sdk()
    run(store_for(sdk).asearch(("alice",), query="q", limit=8, offset=3))
    assert sdk.last_kwargs("fetch")["max_results"] == 11


def test_S2_inverted_a_response_without_a_ranked_list_falls_back_to_buckets_with_no_rank():
    resp = realistic_response()
    resp.items = []
    results = run(store_for(Sdk(resp)).asearch(("alice",), query="q", limit=10))
    assert [r.value["_synap"]["rank"] for r in results] == [None, None, None]
    assert all(r.score is None for r in results)


# ── S-3 / S-4 inverted: the store can ask what the SDK can ──────────────────

def test_S3_inverted_fetch_options_reach_sdk_fetch():
    sdk = Sdk()
    store = store_for(sdk, fetch_options={
        "precision_level": "medium", "types": ["facts"], "max_results_per_query": 4,
        "conversation_id": "conv-1", "mode": "fast", "scopes": ["user", "client"]})
    run(store.asearch(("alice",), query="refund policy", limit=5))
    kw = sdk.last_kwargs("fetch")
    assert kw["response_shape"] == "ranked"
    assert kw["precision_level"] == "medium" and kw["types"] == ["facts"]
    assert kw["max_results_per_query"] == 4 and kw["conversation_id"] == coerce_conversation_id("conv-1")
    assert kw["mode"] == "fast"                         # overrides the constructor default
    assert kw["scopes"] == ["user", "client"]           # client scope is reachable
    assert kw["search_query"] == ["refund policy"]


def test_S3_inverted_scopes_are_not_hard_coded():
    sdk = Sdk()
    run(store_for(sdk).asearch(("alice",), query="q"))
    assert "scopes" not in sdk.last_kwargs("fetch")


def test_S3_inverted_fetch_options_apply_to_get_too():
    sdk = Sdk()
    run(store_for(sdk, fetch_options={"precision_level": "medium"}).aget(("n",), "k"))
    assert sdk.last_kwargs("fetch")["precision_level"] == "medium"


def test_S4_inverted_customer_only_store_does_not_pin_scopes():
    sdk = Sdk()
    run(SynapStore(sdk, customer_id="acme").asearch(("acme",), query="q"))
    kw = sdk.last_kwargs("fetch")
    assert "scopes" not in kw and kw["user_id"] is None and kw["customer_id"] == "acme"


# ── S-5 inverted: one tagged item no longer hides the rest ──────────────────

def test_S5_inverted_one_tagged_item_among_three_untagged_returns_all_four():
    resp = realistic_response()
    resp.items.append(tagged("m1", 4, 0.5, "alice/prefs", "language", '{"value": "English"}'))
    results = run(store_for(Sdk(resp)).asearch(("alice",), query="anything", limit=10))
    assert len(results) == 4
    assert {r.key for r in results} == {"p1", "f2", "f1", "language"}


def test_S5_inverted_semantic_fallback_false_is_the_strict_opt_out():
    resp = realistic_response()
    resp.items.append(tagged("m1", 4, 0.5, "alice/prefs", "language"))
    results = run(store_for(Sdk(resp), semantic_fallback=False).asearch(("alice",), query="q", limit=10))
    assert [r.key for r in results] == ["language"]


def test_S5_inverted_no_markers_stripped_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="synap_langgraph.store"):
        run(store_for(Sdk()).asearch(("alice",), query="q"))
        run(store_for(Sdk()).aget(("alice",), "k"))
    assert [r for r in caplog.records if "marker" in r.message.lower() or "strip" in r.message.lower()] == []


# ── S-7 inverted: an outage can be told from an empty memory ────────────────

def test_S7_inverted_default_still_degrades_with_an_error_log(caplog):
    sdk = Sdk().fails(RuntimeError("503 from Synap"))
    with caplog.at_level(logging.ERROR, logger="synap_langgraph.store"):
        assert run(store_for(sdk).asearch(("alice",), query="q")) == []
    assert any("sdk.fetch failed" in r.message for r in caplog.records)


def test_S7_inverted_on_error_raise_makes_the_failure_visible():
    sdk = Sdk().fails(RuntimeError("503 from Synap"))
    with pytest.raises(SynapIntegrationError, match="503 from Synap") as ei:
        run(store_for(sdk, on_error="raise").asearch(("alice",), query="q"))
    assert ei.value.operation == "langgraph.store.search"
    assert isinstance(ei.value.__cause__, RuntimeError)
    # ...and an empty memory is still just empty (every scope answered, with nothing):
    empty = UnifiedContextResponse(scopes_queried=["user", "customer"])
    assert run(store_for(Sdk(empty), on_error="raise").asearch(("alice",), query="q")) == []


def test_S7_inverted_get_honours_on_error_too():
    sdk = Sdk().fails(RuntimeError("down"))
    assert run(store_for(sdk).aget(("n",), "k")) is None
    with pytest.raises(SynapIntegrationError, match="down"):
        run(store_for(sdk, on_error="raise").aget(("n",), "k"))


# ── S-14 inverted: the memory's own date ────────────────────────────────────

def test_created_at_inverted_is_the_day_it_was_said_not_the_fetch_clock():
    meta = ResponseMetadata(correlation_id="c", ttl_seconds=0, source="cloud",
                            retrieved_at=datetime.now(timezone.utc))
    cr = ContextResponse.from_cloud_response({"facts": [{
        "id": "f1", "content": "The user takes a monthly ride", "confidence": 0.6,
        "source": "conv", "said_at": "2023-06-10T00:00:00+00:00",
        "event_date": "2023-06-12T00:00:00+00:00", "metadata": {"scope": "user"}}]}, meta)
    unified = UnifiedContextResponse.merge([("user", cr)])
    # the SDK fills extracted_at with the fetch time; the store must not use it
    assert unified.facts[0].extracted_at.year == datetime.now().year
    hit = run(store_for(Sdk(unified)).asearch(("alice",), query="ride"))[0]
    assert hit.created_at == datetime(2023, 6, 10, tzinfo=timezone.utc)      # said_at wins
    assert hit.updated_at == hit.created_at
    assert hit.value["_synap"]["dates_known"] is True


def test_created_at_falls_back_to_event_date_then_to_a_documented_sentinel():
    only_event = ranked("e", 1, "facts", "x", 0.5, "user", event_date=datetime(2022, 1, 2, tzinfo=timezone.utc))
    none = ranked("n", 2, "facts", "y", 0.4, "user")
    a, b = run(store_for(Sdk(response_of(only_event, none))).asearch(("alice",), query="q"))
    assert a.created_at == datetime(2022, 1, 2, tzinfo=timezone.utc) and a.value["_synap"]["dates_known"] is True
    assert b.created_at == b.updated_at == UNKNOWN_DATE == datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert b.value["_synap"]["dates_known"] is False


def test_get_item_dates_and_labels():
    it = tagged("t1", 1, 0.9, "prefs", "lang")
    it.said_at = SAID
    item = run(store_for(Sdk(response_of(it))).aget(("prefs",), "lang"))
    assert item.created_at == SAID and item.value["_synap"]["scope"] == "user"


# ── P-5 inverted: delete reaches the SDK ────────────────────────────────────

def test_P5_inverted_delete_finds_the_memory_by_its_tag_and_deletes_it():
    sdk = Sdk(response_of(tagged(UUIDS[1], 1, 0.9, "alice/prefs", "language")))
    run(store_for(sdk).adelete(("alice", "prefs"), "language"))
    from uuid import UUID
    sdk.memories.delete.assert_awaited_once_with(UUID(UUIDS[1]))
    kw = sdk.last_kwargs("fetch")
    assert kw["scopes"] == ["user"]                    # this store's own level only
    assert kw["search_query"] == ["alice/prefs language"]


def test_delete_via_put_none_and_the_base_class_delete():
    sdk = Sdk(response_of(tagged(UUIDS[1], 1, 0.9, "prefs", "language")))
    run(store_for(sdk).adelete(("prefs",), "language"))
    assert sdk.memories.delete.await_count == 1


def test_delete_removes_every_memory_carrying_the_tag():
    sdk = Sdk(response_of(tagged(UUIDS[1], 1, 0.9, "p", "k"), tagged(UUIDS[2], 2, 0.8, "p", "k"),
                          tagged(UUIDS[3], 3, 0.7, "p", "other")))
    run(store_for(sdk).adelete(("p",), "k"))
    deleted = {c.args[0].hex[-1] for c in sdk.memories.delete.await_args_list}
    assert deleted == {"1", "2"}


def test_delete_not_found_raises_a_clear_error_and_deletes_nothing():
    sdk = Sdk(response_of(tagged(UUIDS[1], 1, 0.9, "p", "other")))
    with pytest.raises(SynapIntegrationError, match="no memory found.*Nothing was deleted"):
        run(store_for(sdk).adelete(("p",), "k"))
    sdk.memories.delete.assert_not_awaited()


def test_delete_ignores_untagged_items_and_non_uuid_ids():
    sdk = Sdk(response_of(ranked("f1", 1, "facts", "untagged", 0.9, "user"),
                          tagged("chunk_abc", 2, 0.8, "p", "k")))
    with pytest.raises(SynapIntegrationError, match="no memory found"):
        run(store_for(sdk).adelete(("p",), "k"))
    sdk.memories.delete.assert_not_awaited()


def test_delete_never_touches_a_level_above_the_stores_own():
    customer_level = tagged(UUIDS[1], 1, 0.9, "p", "k", scope="customer")
    sdk = Sdk(response_of(customer_level))
    with pytest.raises(SynapIntegrationError, match="no memory found"):
        run(store_for(sdk).adelete(("p",), "k"))
    sdk.memories.delete.assert_not_awaited()


def test_delete_lookup_failure_raises_even_with_on_error_fallback():
    sdk = Sdk().fails(RuntimeError("down"))
    with pytest.raises(SynapIntegrationError, match="down"):
        run(store_for(sdk).adelete(("p",), "k"))


def test_delete_sdk_failure_raises():
    sdk = Sdk(response_of(tagged(UUIDS[1], 1, 0.9, "p", "k")))
    sdk.memories.delete.side_effect = RuntimeError("500")
    with pytest.raises(SynapIntegrationError, match="500"):
        run(store_for(sdk).adelete(("p",), "k"))


def test_delete_in_a_shared_store_uses_the_identity_and_the_sub_namespace():
    sdk = Sdk(response_of(tagged(UUIDS[1], 1, 0.9, "prefs", "language", ident=("acme", "alice"))))
    run(SynapStore(sdk).adelete(("acme", "alice", "prefs"), "language"))
    kw = sdk.last_kwargs("fetch")
    assert kw["user_id"] == "alice" and kw["customer_id"] == "acme" and kw["scopes"] == ["user"]
    assert kw["search_query"] == ["prefs language"]
    assert sdk.memories.delete.await_count == 1


def test_customer_pool_store_deletes_at_customer_scope():
    sdk = Sdk(response_of(tagged(UUIDS[1], 1, 0.9, "policies", "refund", scope="customer", ident=("acme",))))
    run(SynapStore(sdk, level="customer").adelete(("acme", "policies"), "refund"))
    assert sdk.last_kwargs("fetch")["scopes"] == ["customer"]
    assert sdk.memories.delete.await_count == 1


def test_no_stale_no_public_delete_api_text_remains():
    import pathlib
    pkg = pathlib.Path(__file__).resolve().parents[1]
    offenders = [str(p) for p in list(pkg.rglob("*.py")) + list(pkg.rglob("*.md"))
                 if "no public delete api" in p.read_text().lower()
                 # SynapCheckpointSaver (deprecated, deliberately unchanged) keeps
                 # its own delete_thread note
                 and p.name not in ("test_store_fixed.py", "checkpointer.py", "test_checkpointer.py")]
    assert offenders == []


def test_list_namespaces_is_still_a_documented_no_op(caplog):
    with caplog.at_level(logging.WARNING, logger="synap_langgraph.store"):
        assert run(store_for(Sdk()).alist_namespaces()) == []
    assert any("list_namespaces" in r.message for r in caplog.records)


# ── integration impact inverted (package-level) ─────────────────────────────

def test_integration_impact_inverted_one_call_ranked_with_everything_the_prompt_needs():
    """What the playground's prompt builders needed and the old store dropped:
    scope per item, who said it, the dates, the ranked order, the true date."""
    ride = ranked("f1", 3, "facts", "The user takes a monthly ride", 0.62, "user",
                  said_at=SAID, event_date=SAID, metadata={"assertion_source": "user"})
    guess = ranked("f3", 4, "facts", "The exact date mentioned for the ride issue is June 10", 0.55,
                   "user", metadata={"assertion_source": "assistant"})
    policy = ranked("f2", 2, "facts", "Refunds must be filed within 30 days", 0.80, "client")
    email = ranked("p1", 1, "preferences", "Prefers email contact", 0.91, "user")
    sdk = Sdk(response_of(email, policy, ride, guess))
    hits = run(store_for(sdk).asearch(("alice",), query="state on file", limit=10))
    assert len(sdk.calls_to("fetch")) == 1
    assert hits[0].value["_raw"] == "Prefers email contact"            # 1st stays 1st
    metas = {h.value["_raw"]: h.value["_synap"] for h in hits}
    assert metas["Refunds must be filed within 30 days"]["scope"] == "client"
    assert metas["The exact date mentioned for the ride issue is June 10"]["assertion_source"] == "assistant"
    assert metas["The user takes a monthly ride"]["said_at"].startswith("2023-06-10")
    assert metas["Prefers email contact"]["type"] == "preference"      # not [FACT]
    assert all(isinstance(h, SearchItem) for h in hits)
