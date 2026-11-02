"""Shared SynapStore: the identity comes from the namespace, on every call.

Built on ``RecordingSDK`` so the exact keyword arguments of every SDK call are
asserted, and on the SDK's real ``ScopeLadder`` (``make_ladder``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import GetOp, PutOp, SearchOp

from maximem_synap.models.errors import InvalidInputError
from synap_integrations_common import SynapIntegrationError
from synap_integrations_common.testing import make_ladder
from synap_langgraph import SynapStore

from ._support import Sdk, response_of, tagged


def run(coro):
    return asyncio.run(coro)


# ── N entries of identity, set by the ladder and `level` ────────────────────

def test_b2b_default_reads_customer_and_user_from_the_first_two_entries():
    sdk = Sdk()
    run(SynapStore(sdk).asearch(("acme", "alice", "prefs"), query="q", limit=5))
    kw = sdk.last_kwargs("fetch")
    assert kw["customer_id"] == "acme" and kw["user_id"] == "alice"
    assert "scope_path" not in kw          # default ladder: ids only, never a path
    assert "scopes" not in kw
    assert kw["response_shape"] == "ranked" and kw["search_query"] == ["q"]


def test_b2c_default_takes_one_entry():
    sdk = Sdk(ladder="b2c")
    run(SynapStore(sdk).asearch(("alice", "prefs"), query="q"))
    kw = sdk.last_kwargs("fetch")
    assert kw["user_id"] == "alice" and "customer_id" not in kw


def test_deeper_ladder_takes_three_entries_and_sends_the_path():
    sdk = Sdk(ladder="deep")
    run(SynapStore(sdk).asearch(("acme", "payments", "dana", "notes"), query="q"))
    kw = sdk.last_kwargs("fetch")
    assert kw["user_id"] == "dana" and kw["customer_id"] == "acme"
    assert kw["scope_path"] == {"customer": "acme", "team": "payments", "user": "dana"}


def test_put_on_a_deeper_ladder_writes_with_scope():
    sdk = Sdk(ladder="deep")
    run(SynapStore(sdk).aput(("acme", "payments", "dana", "prefs"), "lang", {"v": "en"}))
    kw = sdk.last_kwargs("memories.create")
    assert kw["scope"] == {"customer": "acme", "team": "payments", "user": "dana"}
    assert kw["user_id"] == "dana" and kw["customer_id"] == "acme"
    # the tag holds the SUB-namespace only: the identity lives in the scope
    assert kw["metadata"] == {"lg_store": True, "lg_store_ns": "prefs", "lg_store_key": "lang",
                              "lg_store_id": '["acme", "payments", "dana"]'}
    assert "scope_path" not in kw


def test_put_on_default_ladder_writes_ids_only():
    sdk = Sdk()
    run(SynapStore(sdk).aput(("acme", "alice", "prefs"), "lang", {"v": "en"}))
    kw = sdk.last_kwargs("memories.create")
    assert kw == {
        "document": '{"v": "en"}', "user_id": "alice", "customer_id": "acme",
        "metadata": {"lg_store": True, "lg_store_ns": "prefs", "lg_store_key": "lang",
                     "lg_store_id": '["acme", "alice"]'},
    }


def test_level_customer_is_a_second_store_with_one_identity_entry():
    sdk = Sdk()
    pool = SynapStore(sdk, level="customer")
    run(pool.asearch(("acme", "policies"), query="refunds"))
    kw = sdk.last_kwargs("fetch")
    assert kw["customer_id"] == "acme" and "user_id" not in kw
    run(pool.aput(("acme", "policies"), "refund", {"days": 30}))
    w = sdk.last_kwargs("memories.create")
    assert w["customer_id"] == "acme" and "user_id" not in w


def test_level_middle_of_a_deep_ladder():
    sdk = Sdk(ladder="deep")
    run(SynapStore(sdk, level="team").asearch(("acme", "payments"), query="q"))
    kw = sdk.last_kwargs("fetch")
    assert kw["scope_path"] == {"customer": "acme", "team": "payments"}
    assert "user_id" not in kw


# ── fail closed ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ns", [(), ("acme",)])
def test_namespace_shorter_than_n_raises_and_asks_nothing(ns):
    sdk = Sdk()
    with pytest.raises(SynapIntegrationError, match="too short"):
        run(SynapStore(sdk).asearch(ns, query="q"))
    assert sdk.calls_to("fetch") == []


def test_every_operation_fails_closed_on_a_short_namespace():
    sdk = Sdk()
    store = SynapStore(sdk)
    for coro in (store.aget(("acme",), "k"), store.aput(("acme",), "k", {"a": 1}),
                 store.adelete(("acme",), "k")):
        with pytest.raises(SynapIntegrationError, match="too short"):
            run(coro)
    assert sdk.calls_to("fetch") == [] and sdk.calls_to("memories.create") == []


def test_short_namespace_raises_even_with_on_error_fallback():
    with pytest.raises(SynapIntegrationError):
        run(SynapStore(Sdk(), on_error="fallback").asearch(("acme",), query="q"))


def test_unknown_level_raises_and_names_the_levels():
    sdk = Sdk()
    with pytest.raises(SynapIntegrationError, match=r"unknown level 'team'.*\['customer', 'user'\]"):
        run(SynapStore(sdk, level="team").asearch(("acme", "x"), query="q"))
    assert sdk.calls_to("fetch") == []


def test_b2c_tenant_level_has_no_pool_of_its_own():
    with pytest.raises(SynapIntegrationError, match="unknown level 'customer'"):
        run(SynapStore(Sdk(ladder="b2c"), level="customer").asearch(("x",), query="q"))


def test_no_ladder_raises():
    with pytest.raises(SynapIntegrationError, match="could not be determined"):
        run(SynapStore(Sdk(ladder=None)).asearch(("acme", "alice"), query="q"))


def test_a_failed_ladder_read_raises():
    with pytest.raises(SynapIntegrationError):
        run(SynapStore(Sdk(ladder=RuntimeError("whoami down"))).asearch(("a", "b"), query="q"))


def test_an_sdk_without_get_scope_ladder_raises_an_upgrade_message():
    class Old:
        pass
    with pytest.raises(SynapIntegrationError, match="get_scope_ladder"):
        run(SynapStore(Old()).asearch(("acme", "alice"), query="q"))


def test_empty_string_in_the_identity_raises():
    with pytest.raises(InvalidInputError):
        run(SynapStore(Sdk()).abatch([SearchOp(("acme", ""), "q", None, 10, 0)]))


def test_deeper_ladder_that_is_not_ratified_raises_before_any_call():
    sdk = Sdk(ladder=make_ladder("deep", ratified=False))
    with pytest.raises(InvalidInputError, match="refuse a scope path"):
        run(SynapStore(sdk).asearch(("acme", "payments", "dana"), query="q"))
    assert sdk.calls_to("fetch") == []


def test_non_tuple_or_non_string_namespace_raises():
    with pytest.raises(SynapIntegrationError, match="tuple of strings"):
        run(SynapStore(Sdk()).abatch([SearchOp(("acme", 5), "q", None, 10, 0)]))  # type: ignore[arg-type]


# ── client level needs an explicit option ───────────────────────────────────

def test_client_level_is_refused_without_the_option():
    with pytest.raises(ValueError, match="allow_client_level"):
        SynapStore(Sdk(), level="client")


def test_client_level_with_the_option_reads_and_writes_the_account_level():
    sdk = Sdk()
    store = SynapStore(sdk, level="client", allow_client_level=True)
    run(store.asearch(("kb",), query="q"))
    kw = sdk.last_kwargs("fetch")
    assert kw["scopes"] == ["client"] and "user_id" not in kw and "customer_id" not in kw
    run(store.aput(("kb",), "k", {"a": 1}))
    assert "user_id" not in sdk.last_kwargs("memories.create")
    assert sdk.calls_to("get_scope_ladder") == []     # no identity, no ladder


# ── constructor validation ──────────────────────────────────────────────────

def test_level_with_ids_is_rejected():
    with pytest.raises(ValueError, match="level"):
        SynapStore(Sdk(), user_id="u", level="customer")


def test_scope_path_on_a_shared_store_is_rejected():
    with pytest.raises(ValueError, match="scope_path"):
        SynapStore(Sdk(), scope_path={"user": "u"})


def test_on_error_is_validated():
    with pytest.raises(ValueError, match="on_error"):
        SynapStore(Sdk(), on_error="ignore")


def test_fetch_options_may_not_override_the_stores_own_arguments():
    for key in ("user_id", "customer_id", "scope_path", "search_query", "response_shape", "max_results"):
        with pytest.raises(ValueError, match=key):
            SynapStore(Sdk(), fetch_options={key: 1})


# ── proof P-3 inverted: one store compiled once, run for two users ──────────

def test_P3_inverted_one_compiled_store_asks_about_each_user():
    class S(TypedDict, total=False):
        out: str

    sdk = Sdk()
    store = SynapStore(sdk)

    async def recall(state: S, config, *, store):
        who = config["configurable"]["user_id"]
        await store.asearch(("acme", who), query="what do I like")
        return {"out": who}

    g = StateGraph(S)
    g.add_node("recall", recall)
    g.add_edge(START, "recall")
    g.add_edge("recall", END)
    app = g.compile(store=store)

    run(app.ainvoke({}, config={"configurable": {"user_id": "alice"}}))
    run(app.ainvoke({}, config={"configurable": {"user_id": "bob"}}))
    asked = [(c["customer_id"], c["user_id"]) for c in sdk.kwargs_of("fetch")]
    assert asked == [("acme", "alice"), ("acme", "bob")]


def test_concurrent_users_do_not_mix():
    sdk = Sdk()
    store = SynapStore(sdk)

    async def go():
        await asyncio.gather(*[store.asearch(("acme", f"u{i}"), query="q") for i in range(20)])

    run(go())
    assert sorted(c["user_id"] for c in sdk.kwargs_of("fetch")) == sorted(f"u{i}" for i in range(20))


# ── sub-namespace rule (issue 3) ────────────────────────────────────────────

def test_tagged_items_follow_the_sub_namespace_prefix_untagged_are_always_returned():
    resp = response_of(
        tagged("t1", 1, 0.9, "prefs", "language", ident=("acme", "alice")),
        tagged("t2", 2, 0.8, "other", "color", ident=("acme", "alice")),
        tagged("t3", 3, 0.7, "prefs/ui", "theme", ident=("acme", "alice")),
    )
    from ._support import ranked
    resp.items.append(ranked("c1", 4, "facts", "Lives in Paris", 0.5, "user"))
    sdk = Sdk(resp)
    hits = run(SynapStore(sdk).asearch(("acme", "alice", "prefs"), query="q"))
    assert [h.key for h in hits] == ["language", "theme", "c1"]
    assert hits[0].namespace == ("acme", "alice", "prefs")
    assert hits[1].namespace == ("acme", "alice", "prefs", "ui")
    assert hits[2].namespace == ("acme", "alice", "prefs")   # untagged: the namespace asked for


def test_get_in_a_shared_store_matches_the_exact_sub_namespace_and_key():
    ident = ("acme", "alice")
    resp = response_of(tagged("t1", 1, 0.9, "prefs", "language", value='{"v": "en"}', ident=ident),
                       tagged("t2", 2, 0.8, "other", "language", value='{"v": "fr"}', ident=ident))
    sdk = Sdk(resp)
    item = run(SynapStore(sdk).aget(("acme", "alice", "prefs"), "language"))
    assert item.value["v"] == "en" and item.namespace == ("acme", "alice", "prefs")
    assert run(SynapStore(sdk).aget(("acme", "alice", "nope"), "language")) is None


# ── record_message on a shared store ────────────────────────────────────────

def test_record_message_on_a_shared_store_needs_the_namespace():
    sdk = Sdk()
    store = SynapStore(sdk)
    with pytest.raises(SynapIntegrationError, match="namespace"):
        run(store.arecord_message("c1", "user", "hi"))
    run(store.arecord_message("c1", "user", "hi", namespace=("acme", "alice")))
    kw = sdk.last_kwargs("instance.send_message")
    assert kw["user_id"] == "alice" and kw["customer_id"] == "acme" and "scope_path" not in kw


def test_record_message_on_a_deeper_ladder_carries_the_path():
    sdk = Sdk(ladder="deep")
    run(SynapStore(sdk).arecord_message("c1", "user", "hi", namespace=("acme", "payments", "dana")))
    assert sdk.last_kwargs("instance.send_message")["scope_path"] == {
        "customer": "acme", "team": "payments", "user": "dana"}


def test_record_message_on_a_customer_pool_store_skips_with_a_warning(caplog):
    sdk = Sdk()
    with caplog.at_level(logging.WARNING, logger="synap_langgraph.store"):
        run(SynapStore(sdk, level="customer").arecord_message("c1", "user", "hi", namespace=("acme",)))
    assert sdk.calls_to("instance.send_message") == []
    assert any("user_id" in r.message for r in caplog.records)


# ── single-user store: unchanged where no issue applies ─────────────────────

def test_single_user_store_fetch_kwargs_exactly():
    sdk = Sdk()
    run(SynapStore(sdk, user_id="alice", customer_id="acme").asearch(("prefs",), query="q", limit=3))
    assert sdk.last_kwargs("fetch") == {
        "mode": "accurate", "include_conversation_context": False,
        "user_id": "alice", "customer_id": "acme",
        "search_query": ["q"], "max_results": 10, "response_shape": "ranked",
    }
    assert sdk.calls_to("get_scope_ladder") == []       # no extra network call


def test_single_user_store_put_kwargs_exactly():
    sdk = Sdk()
    run(SynapStore(sdk, user_id="alice", customer_id="acme").aput(("a", "b"), "k", {"x": 1}))
    assert sdk.last_kwargs("memories.create") == {
        "document": '{"x": 1}', "user_id": "alice", "customer_id": "acme",
        "metadata": {"lg_store": True, "lg_store_ns": "a/b", "lg_store_key": "k"},
    }


def test_single_user_store_with_scope_path():
    sdk = Sdk(ladder="deep")
    path = {"customer": "acme", "team": "payments", "user": "dana"}
    store = SynapStore(sdk, user_id="dana", customer_id="acme", scope_path=path)
    run(store.asearch(("n",), query="q"))
    run(store.aput(("n",), "k", {"a": 1}))
    run(store.arecord_message("c1", "user", "hi"))
    assert sdk.last_kwargs("fetch")["scope_path"] == path
    assert sdk.last_kwargs("memories.create")["scope"] == path
    assert sdk.last_kwargs("instance.send_message")["scope_path"] == path


def test_single_user_store_without_scope_path_never_sends_one():
    sdk = Sdk(accept_scope_path=False)
    store = SynapStore(sdk, user_id="alice", customer_id="acme")
    run(store.asearch(("n",), query="q"))
    run(store.aput(("n",), "k", {"a": 1}))
    run(store.arecord_message("c1", "user", "hi"))
    assert "scope_path" not in sdk.last_kwargs("fetch")
    assert "scope" not in sdk.last_kwargs("memories.create")
