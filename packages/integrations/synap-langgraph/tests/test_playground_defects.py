"""Defects found by running the playground agents on the package (one test group each)."""

from __future__ import annotations

import asyncio
import logging

import pytest

from synap_integrations_common import SynapIntegrationError
from synap_langgraph import SynapStore

from ._support import RealFetchSdk


def run(coro):
    return asyncio.run(coro)


# ── P-SWALLOW ────────────────────────────────────────────────────────────────

def test_swallow_real_sdk_fetch_drops_a_failed_scope_from_scopes_queried():
    """Premise: the SDK swallows a failed scope and says so only by leaving it out
    of ``scopes_queried``."""
    sdk = RealFetchSdk(outage={"user"})
    resp = run(sdk.fetch(user_id="alice", customer_id="acme", search_query=["x"]))
    assert resp.scopes_queried == ["customer"]
    assert resp.items == [] and resp.facts == []


def test_swallow_raise_mode_raises_when_a_scope_failed_search():
    store = SynapStore(RealFetchSdk(outage={"user"}), user_id="alice", customer_id="acme",
                       on_error="raise")
    with pytest.raises(SynapIntegrationError) as err:
        run(store.asearch(("memories",), query="tea"))
    assert "user" in str(err.value)
    assert err.value.operation == "langgraph.store.search"


def test_swallow_raise_mode_raises_when_every_scope_failed_search_shared_store():
    store = SynapStore(RealFetchSdk(outage={"user", "customer"}), on_error="raise")
    with pytest.raises(SynapIntegrationError):
        run(store.asearch(("acme", "alice"), query="tea"))




def test_swallow_raise_mode_is_quiet_when_every_scope_answered():
    store = SynapStore(RealFetchSdk(), user_id="alice", customer_id="acme", on_error="raise")
    hits = run(store.asearch(("memories",), query="tea"))
    assert [h.value["_raw"] for h in hits] == ["Likes tea"]


def test_swallow_fallback_mode_keeps_the_partial_answer_and_logs(caplog):
    """Default behaviour is unchanged: the working scope's items are returned."""
    store = SynapStore(RealFetchSdk(outage={"customer"}), user_id="alice", customer_id="acme")
    with caplog.at_level(logging.WARNING):
        hits = run(store.asearch(("memories",), query="tea"))
    assert [h.value["_raw"] for h in hits] == ["Likes tea"]
    assert any("customer" in r.getMessage() and "incomplete" in r.getMessage()
               for r in caplog.records)


def test_swallow_only_the_scopes_asked_for_count():
    """``scopes`` limits what is queried; a scope never asked for is not an outage."""
    store = SynapStore(RealFetchSdk(outage={"customer"}), user_id="alice", customer_id="acme",
                       on_error="raise", fetch_options={"scopes": ["user"]})
    assert run(store.asearch(("memories",), query="tea"))




# ── shared helpers ───────────────────────────────────────────────────────────

from maximem_synap.models.context import Fact, RankedItem, UnifiedContextResponse  # noqa: E402

from ._support import NOW, Sdk, response_of, tagged  # noqa: E402


def text_response(*texts):
    items = [RankedItem(id=f"x{i}", rank=i + 1, category="facts", content=t, rank_score=0.9 - i / 100,
                        metadata={"scope": "user"}) for i, t in enumerate(texts)]
    return response_of(*items)


# ── P-MINFETCH ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("limit,offset,asked", [(8, 0, 8), (3, 0, 3), (1, 0, 1), (5, 4, 9), (20, 0, 20)])
def test_minfetch_asks_the_server_for_what_the_caller_asked(limit, offset, asked):
    sdk = Sdk()
    store = SynapStore(sdk, user_id="alice", customer_id="acme")
    run(store.asearch((), query="tea", limit=limit, offset=offset))
    assert sdk.last_kwargs("fetch")["max_results"] == asked


@pytest.mark.parametrize("kw", [
    {"namespace_prefix": ("prefs",)},                # items are dropped by sub-namespace
    {"namespace_prefix": (), "filter": {"x": 1}},     # ...or by metadata filter
])
def test_minfetch_over_fetches_only_when_the_store_filters_afterwards(kw):
    sdk = Sdk()
    store = SynapStore(sdk, user_id="alice", customer_id="acme")
    run(store.asearch(kw.pop("namespace_prefix"), query="tea", limit=3, **kw))
    assert sdk.last_kwargs("fetch")["max_results"] == 10


def test_minfetch_strict_namespace_mode_over_fetches_too():
    sdk = Sdk()
    store = SynapStore(sdk, user_id="alice", customer_id="acme", semantic_fallback=False)
    run(store.asearch((), query="tea", limit=3))
    assert sdk.last_kwargs("fetch")["max_results"] == 10


# ── P-JSON ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "1.50", "true", "null", "007", "1e3", '"quoted"', "[1, 2]", "12", "-0.0", "1_000",
    "  42", "Infinity", "NaN",
])
def test_json_memory_text_comes_back_exactly(text):
    store = SynapStore(Sdk(text_response(text)), user_id="alice", customer_id="acme")
    (hit,) = run(store.asearch((), query="q"))
    assert hit.value["_raw"] == text
    assert "_value" not in hit.value


def test_json_object_still_round_trips_as_a_dict_and_its_exact_text_is_kept():
    text = '{"v": 1.50, "ok": true}'
    store = SynapStore(Sdk(text_response(text)), user_id="alice", customer_id="acme")
    (hit,) = run(store.asearch((), query="q"))
    assert hit.value["v"] == 1.5 and hit.value["ok"] is True and "_raw" not in hit.value
    assert hit.value["_synap"]["text"] == text


# ── P-CONVID ─────────────────────────────────────────────────────────────────

import uuid  # noqa: E402

from synap_integrations_common.testing import RecordingSDK  # noqa: E402
from synap_langgraph import create_synap_st_node, create_synap_turn_node, synap_st_prompt  # noqa: E402

PG = "pgv2_0123456789abcdef0123456789abcdef"
PG_UUID = str(uuid.uuid5(uuid.NAMESPACE_URL, PG))
CTX = "conversation.context.get_context_for_prompt"


def strict(**kw):
    return RecordingSDK(strict_conversation_ids=True, **kw)


def test_convid_st_node_reads_a_playground_conversation_id():
    sdk = strict()
    out = run(create_synap_st_node(sdk, PG, on_error="raise")({}))
    assert out == {"synap_st": ""}
    assert sdk.last_kwargs(CTX)["conversation_id"] == PG_UUID


def test_convid_st_node_takes_the_id_from_the_run_config_too():
    sdk = strict()
    run(create_synap_st_node(sdk, on_error="raise")({}, {"configurable": {"conversation_id": PG}}))
    assert sdk.last_kwargs(CTX)["conversation_id"] == PG_UUID


def test_convid_prompt_callable_reads_a_playground_conversation_id():
    sdk = strict()
    run(synap_st_prompt(sdk, PG, on_error="raise")({"messages": []}))
    assert sdk.last_kwargs(CTX)["conversation_id"] == PG_UUID


def test_convid_a_uuid_is_sent_unchanged():
    sdk = strict()
    u = "123e4567-e89b-12d3-a456-426614174000"
    run(create_synap_st_node(sdk, u)({}))
    assert sdk.last_kwargs(CTX)["conversation_id"] == u


def test_convid_turn_node_rest_fallback_records_a_playground_turn():
    sdk = strict(listening=False)
    sdk.instance.listen = None
    node = create_synap_turn_node(sdk, role="user", content="content", user_id="u",
                                  customer_id="c", conversation_id=PG, on_error="raise")
    run(node({"content": "hello"}, {"configurable": {}}))
    assert sdk.last_kwargs("conversation.record_message")["conversation_id"] == PG_UUID


def test_convid_turn_on_the_stream_keeps_the_callers_own_id():
    sdk = strict()
    node = create_synap_turn_node(sdk, role="user", content="content", user_id="u",
                                  customer_id="c", conversation_id=PG, on_error="raise")
    run(node({"content": "hello"}, {"configurable": {}}))
    assert sdk.last_kwargs("instance.send_message")["conversation_id"] == PG


def test_convid_a_failed_short_term_read_is_never_silent(caplog):
    sdk = strict()
    sdk.errors[CTX] = RuntimeError("down")
    with caplog.at_level(logging.ERROR):
        out = run(create_synap_st_node(sdk, PG)({}))
    assert out == {"synap_st": ""}
    assert any(r.levelno >= logging.ERROR and "down" in r.getMessage() for r in caplog.records)


def test_convid_a_missing_conversation_id_is_logged(caplog):
    with caplog.at_level(logging.ERROR):
        out = run(create_synap_st_node(strict())({}, {"configurable": {}}))
    assert out == {"synap_st": ""}
    assert any("no conversation id" in r.getMessage() for r in caplog.records)


def test_convid_store_fetch_conversation_id_option_is_derived():
    sdk = strict()
    store = SynapStore(sdk, user_id="alice", customer_id="acme",
                       fetch_options={"conversation_id": PG})
    run(store.asearch((), query="tea"))
    assert sdk.last_kwargs("fetch")["conversation_id"] == PG_UUID


# ── P-QUERIES ────────────────────────────────────────────────────────────────

def one_fetch(sdk):
    assert len(sdk.calls_to("fetch")) == 1, "several queries must cost ONE server call"
    return sdk.last_kwargs("fetch")


def test_queries_several_queries_are_one_fetch():
    sdk = Sdk()
    store = SynapStore(sdk, user_id="alice", customer_id="acme")
    run(store.asearch((), query=["tea", "coffee", "cab"], limit=6))
    kw = one_fetch(sdk)
    assert kw["search_query"] == ["tea", "coffee", "cab"] and kw["max_results"] == 6


def test_queries_a_tuple_and_the_sync_method_work_and_empty_entries_are_dropped():
    sdk = Sdk()
    store = SynapStore(sdk, user_id="alice", customer_id="acme")
    store.search((), query=("tea", " ", "tea", "cab"))
    assert one_fetch(sdk)["search_query"] == ["tea", "cab"]


def test_queries_one_string_and_no_query_are_as_before():
    sdk = Sdk()
    store = SynapStore(sdk, user_id="alice", customer_id="acme")
    run(store.asearch((), query="tea"))
    assert sdk.last_kwargs("fetch")["search_query"] == ["tea"]
    run(store.asearch((), query=[]))
    assert sdk.last_kwargs("fetch")["search_query"] is None
    run(store.asearch(("prefs",), query=None))
    assert sdk.last_kwargs("fetch")["search_query"] == ["prefs"]


def test_queries_non_text_entries_are_refused():
    store = SynapStore(Sdk(), user_id="alice", customer_id="acme")
    with pytest.raises(SynapIntegrationError, match="query"):
        run(store.asearch((), query=["tea", 3]))


def test_queries_per_query_cap_is_a_fetch_option():
    sdk = Sdk()
    store = SynapStore(sdk, user_id="alice", customer_id="acme",
                       fetch_options={"max_results_per_query": 2})
    run(store.asearch((), query=["a", "b"], limit=6))
    assert one_fetch(sdk)["max_results_per_query"] == 2


# ── P-PERCALL ────────────────────────────────────────────────────────────────

OPTS = {"mode": "fast", "precision_level": "medium", "types": ["facts", "preferences"],
        "max_results_per_query": 3, "query_date": "2023-03-11", "conversation_id": PG}


def test_percall_options_apply_to_that_search_only():
    sdk = strict()
    store = SynapStore(sdk, user_id="alice", customer_id="acme", mode="accurate")
    run(store.asearch_with((), query=["a", "b"], fetch_options=OPTS, limit=5))
    kw = one_fetch(sdk)
    assert (kw["mode"], kw["precision_level"], kw["types"], kw["max_results_per_query"],
            kw["query_date"]) == ("fast", "medium", ["facts", "preferences"], 3, "2023-03-11")
    assert kw["conversation_id"] == PG_UUID and kw["max_results"] == 5
    sdk.reset()
    run(store.asearch((), query="a"))                      # the next call: the store's own
    kw = sdk.last_kwargs("fetch")
    assert kw["mode"] == "accurate" and "types" not in kw and "conversation_id" not in kw


def test_percall_sync_variant():
    sdk = Sdk()
    store = SynapStore(sdk, user_id="alice", customer_id="acme")
    store.search_with((), query="a", fetch_options={"mode": "fast"})
    assert sdk.last_kwargs("fetch")["mode"] == "fast"


def test_percall_context_manager_and_isolation_between_concurrent_calls():
    sdk = Sdk()
    store = SynapStore(sdk, user_id="alice", customer_id="acme")

    async def one(mode):
        with store.with_fetch_options(mode=mode):
            await asyncio.sleep(0.01)
            await store.asearch((), query=mode)

    async def both():
        await asyncio.gather(one("fast"), one("accurate"))
        await store.asearch((), query="plain")

    run(both())
    seen = {c.kwargs["search_query"][0]: c.kwargs["mode"] for c in sdk.calls_to("fetch")}
    assert seen == {"fast": "fast", "accurate": "accurate", "plain": "accurate"}


def test_percall_options_override_the_constructors():
    sdk = Sdk()
    store = SynapStore(sdk, user_id="alice", customer_id="acme",
                       fetch_options={"precision_level": "high", "types": ["facts"]})
    with store.with_fetch_options(precision_level="medium"):
        run(store.asearch(("n",), query="k"))
    kw = sdk.last_kwargs("fetch")
    assert kw["precision_level"] == "medium" and kw["types"] == ["facts"]


def test_percall_run_config_key_reaches_a_store_call_inside_a_node():
    from typing import TypedDict
    from langgraph.graph import END, START, StateGraph

    sdk = Sdk()
    store = SynapStore(sdk, user_id="alice", customer_id="acme", mode="accurate")

    class S(TypedDict, total=False):
        n: int

    async def node(state, config, *, store):
        await store.asearch((), query="tea")
        return {}

    g = StateGraph(S)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    app = g.compile(store=store)
    run(app.ainvoke({}, config={"configurable": {"synap_fetch_options": {"mode": "fast"}}}))
    assert sdk.last_kwargs("fetch")["mode"] == "fast"
    run(app.ainvoke({}))
    assert sdk.last_kwargs("fetch")["mode"] == "accurate"


@pytest.mark.parametrize("bad", ["user_id", "customer_id", "scope_path", "search_query",
                                 "response_shape", "max_results"])
def test_percall_the_stores_own_arguments_cannot_be_overridden(bad):
    store = SynapStore(Sdk(), user_id="alice", customer_id="acme")
    with pytest.raises(ValueError, match=bad):
        with store.with_fetch_options(**{bad: "x"}):
            pass
    with pytest.raises(ValueError, match=bad):
        run(store.asearch_with((), query="a", fetch_options={bad: "x"}))


# ── P-SDK ────────────────────────────────────────────────────────────────────

import contextvars  # noqa: E402

CURRENT = contextvars.ContextVar("current_sdk")


def test_sdk_is_resolved_on_every_call_from_a_provider():
    a, b = Sdk(), Sdk()
    store = SynapStore(sdk_provider=CURRENT.get)          # built once, no SDK yet
    CURRENT.set(a)
    run(store.asearch(("acme", "alice"), query="x"))
    CURRENT.set(b)
    run(store.asearch(("acme", "alice"), query="y"))
    store.search(("acme", "alice"), query="z")
    assert [len(a.calls_to("fetch")), len(b.calls_to("fetch"))] == [1, 2]


def test_sdk_concurrent_calls_each_use_their_own_sdk():
    sdks = {n: Sdk() for n in "ab"}
    store = SynapStore(sdk_provider=lambda: sdks[CURRENT.get()])

    async def one(name):
        CURRENT.set(name)
        await asyncio.sleep(0.01)
        await store.asearch(("acme", "alice"), query=name)

    async def both():
        await asyncio.gather(*(asyncio.create_task(one(n)) for n in "ab"))

    run(both())
    assert [c.kwargs["search_query"] for c in sdks["a"].calls_to("fetch")] == [["a"]]
    assert [c.kwargs["search_query"] for c in sdks["b"].calls_to("fetch")] == [["b"]]


def test_sdk_provider_may_be_async_and_serves_writes_and_the_ladder_too():
    a = Sdk()

    async def provider():
        return a

    store = SynapStore(sdk_provider=provider)
    run(store.aput(("acme", "alice", "prefs"), "k", {"v": 1}))
    assert a.calls_to("memories.create") and a.calls_to("get_scope_ladder")
    run(store.arecord_message(PG, "user", "hi", namespace=("acme", "alice")))
    assert a.calls_to("instance.send_message")


def test_sdk_provider_returning_nothing_is_an_error_not_a_guess():
    store = SynapStore(sdk_provider=lambda: None)
    with pytest.raises(SynapIntegrationError, match="sdk_provider"):
        run(store.asearch(("acme", "alice"), query="x"))


def test_sdk_the_simple_form_is_unchanged():
    sdk = Sdk()
    store = SynapStore(sdk)
    assert store.sdk is sdk
    run(store.asearch(("acme", "alice"), query="x"))
    assert sdk.calls_to("fetch")
    with pytest.raises(ValueError):
        SynapStore()
    with pytest.raises(ValueError):
        SynapStore(sdk, sdk_provider=lambda: sdk)
    with pytest.raises(SynapIntegrationError):
        _ = SynapStore(sdk_provider=lambda: sdk).sdk      # no SDK outside a call


def test_sdk_sync_methods_resolve_the_provider_in_the_callers_thread():
    import threading

    sdk, threads = Sdk(), []

    def provider():
        threads.append(threading.get_ident())
        return sdk

    store = SynapStore(sdk_provider=provider)
    store.search(("acme", "alice"), query="x")
    store.record_message(PG, "user", "hi", namespace=("acme", "alice"))
    assert threads and set(threads) == {threading.get_ident()}


def test_percall_blocks_nest_and_the_inner_one_wins():
    sdk = Sdk()
    store = SynapStore(sdk, user_id="alice", customer_id="acme")
    with store.with_fetch_options(mode="fast", precision_level="medium"):
        with store.with_fetch_options(mode="accurate"):
            run(store.asearch((), query="a"))
        run(store.asearch((), query="b"))
    first, second = (c.kwargs for c in sdk.calls_to("fetch"))
    assert (first["mode"], first["precision_level"]) == ("accurate", "medium")
    assert (second["mode"], second["precision_level"]) == ("fast", "medium")


# ── P-TRANSPORT / P-STEP / P-AUTOLISTEN / P-SDK on nodes ─────────────────────

from synap_langgraph import create_synap_reasoning_node, create_synap_tool_call_node  # noqa: E402


def turn_node(sdk, **kw):
    return create_synap_turn_node(sdk, role="user", content="q", user_id="u", customer_id="c",
                                  conversation_id=PG, **kw)


def test_transport_the_turn_node_says_how_the_turn_travelled():
    live = strict()
    assert run(turn_node(live, transport_key="t")({"q": "hi"}, {})) == {"t": "stream"}
    down = strict(listening=False)
    down.instance.listen = None
    assert run(turn_node(down, transport_key="t")({"q": "hi"}, {})) == {"t": "rest"}
    assert run(turn_node(live, transport_key="t")({"q": "  "}, {})) == {"t": "skipped"}
    down.errors["conversation.record_message"] = RuntimeError("x")
    assert run(turn_node(down, transport_key="t")({"q": "hi"}, {})) == {"t": "failed"}


def test_transport_default_returns_nothing_as_before():
    assert run(turn_node(strict())({"q": "hi"}, {})) == {}


def test_transport_event_nodes_say_whether_they_were_sent():
    live, down = strict(), strict(listening=False)
    down.instance.listen = None
    mk = lambda sdk: create_synap_tool_call_node(  # noqa: E731
        sdk, tool_name="t", user_id="u", conversation_id=PG, transport_key="t")
    assert run(mk(live)({}, {})) == {"t": "stream"}
    assert run(mk(down)({}, {})) == {"t": "none"}


def test_step_reasoning_node_takes_step_index_and_type_from_the_state():
    sdk = strict()
    node = create_synap_reasoning_node(
        sdk, content="plan", user_id="u", conversation_id=PG,
        step_index=lambda st: st["n"], thought_type=lambda st: st["kind"])
    run(node({"plan": "x", "n": 3, "kind": "replan"}, {}))
    kw = sdk.last_kwargs("instance.record_thinking")
    assert kw["step_index"] == 3 and kw["thought_type"] == "replan"
    node2 = create_synap_reasoning_node(sdk, content="plan", user_id="u", conversation_id=PG,
                                        step_index=2, thought_type="plan")
    run(node2({"plan": "x"}, {}))
    kw = sdk.last_kwargs("instance.record_thinking")
    assert kw["step_index"] == 2 and kw["thought_type"] == "plan"


def test_autolisten_false_keeps_rest_and_never_opens_a_stream():
    sdk = strict(listening=False)
    run(turn_node(sdk, auto_listen=False)({"q": "hi"}, {}))
    assert not sdk.instance.is_listening
    assert sdk.last_kwargs("conversation.record_message")["conversation_id"] == PG_UUID
    assert not sdk.calls_to("instance.send_message")


def test_autolisten_default_opens_the_stream_as_the_sdk_does():
    sdk = strict(listening=False)
    run(turn_node(sdk)({"q": "hi"}, {}))
    assert sdk.instance.is_listening and sdk.calls_to("instance.send_message")


def test_sdk_provider_for_nodes_resolves_on_every_run():
    a, b = strict(), strict()
    cur = {"sdk": a}
    node = create_synap_turn_node(sdk_provider=lambda: cur["sdk"], role="user", content="q",
                                  user_id="u", conversation_id=PG)
    run(node({"q": "1"}, {}))
    cur["sdk"] = b
    run(node({"q": "2"}, {}))
    assert len(a.calls_to("instance.send_message")) == 1 and len(b.calls_to("instance.send_message")) == 1
    st = create_synap_st_node(sdk_provider=lambda: cur["sdk"])
    run(st({}, {"configurable": {"conversation_id": PG}}))
    assert b.calls_to(CTX) and not a.calls_to(CTX)
    prompt = synap_st_prompt(sdk_provider=lambda: a)
    run(prompt({"messages": []}, {"configurable": {"conversation_id": PG}}))
    assert a.calls_to(CTX)
    with pytest.raises(ValueError):
        create_synap_st_node()
    with pytest.raises(ValueError):
        create_synap_turn_node(a, sdk_provider=lambda: a, role="user", content="q")
    with pytest.raises(SynapIntegrationError):
        run(create_synap_st_node(sdk_provider=lambda: None, on_error="raise")(
            {}, {"configurable": {"conversation_id": PG}}))


# ── P-SWALLOW guards ─────────────────────────────────────────────────────────

def test_swallow_conversation_scope_is_expected_only_with_a_conversation_id_and_a_query():
    """No query: the conversation's state is read through get_context_for_prompt, not a
    sub-fetch, so its failure (which the SDK only debug-logs) is not a scope outage."""
    sdk = RealFetchSdk()
    sdk.conversation.context.get_context_for_prompt.side_effect = RuntimeError("compaction down")
    store = SynapStore(sdk, user_id="alice", customer_id="acme", on_error="raise",
                       fetch_options={"conversation_id": PG_UUID})
    run(store.asearch((), query=None))                  # no query: must not raise
    assert sdk.scope_calls == [] or all(c[0] != "conversation" for c in sdk.scope_calls)


def test_swallow_conversation_scope_with_a_query_is_expected_and_its_failure_raises():
    sdk = RealFetchSdk(outage={"conversation"})
    store = SynapStore(sdk, user_id="alice", customer_id="acme", on_error="raise",
                       fetch_options={"conversation_id": PG_UUID})
    with pytest.raises(SynapIntegrationError) as err:
        run(store.asearch((), query="tea"))
    assert "conversation" in str(err.value)


def test_swallow_unit_guard_of_unanswered_scopes():
    from types import SimpleNamespace

    from synap_langgraph.store import _unanswered_scopes

    kw = {"user_id": "u", "customer_id": "c", "conversation_id": PG_UUID}
    done = SimpleNamespace(scopes_queried=["user", "customer"])
    assert _unanswered_scopes({**kw, "search_query": None}, done) == []
    assert _unanswered_scopes({**kw, "search_query": ["q"]}, done) == ["conversation"]
    assert _unanswered_scopes({**kw, "search_query": [" "]}, done) == []      # blank is no query


def test_swallow_empty_scopes_queried_means_all_failed_for_the_real_sdk():
    """Evidence: with every asked scope failing, the REAL sdk.fetch returns scopes_queried == []."""
    sdk = RealFetchSdk(outage={"user", "customer"})
    resp = run(sdk.fetch(user_id="alice", customer_id="acme", search_query=["x"]))
    assert resp.scopes_queried == []
    store = SynapStore(sdk, user_id="alice", customer_id="acme", on_error="raise")
    with pytest.raises(SynapIntegrationError):
        run(store.asearch((), query="x"))


def test_swallow_nothing_asked_is_not_an_outage_and_a_healthy_empty_memory_is_not_either():
    sdk = RealFetchSdk(items=[])
    store = SynapStore(sdk, user_id="alice", customer_id="acme", on_error="raise")
    assert run(store.asearch((), query="x")) == []      # both scopes answered, with nothing


def test_swallow_a_stand_in_with_items_and_an_empty_scope_list_is_taken_as_complete():
    resp = response_of(*text_response("Likes tea").items, scopes=())
    assert resp.scopes_queried == [] and resp.items
    store = SynapStore(Sdk(resp), user_id="alice", customer_id="acme", on_error="raise")
    assert [h.value["_raw"] for h in run(store.asearch((), query="x"))] == ["Likes tea"]


def test_swallow_a_stand_in_with_no_items_and_an_empty_scope_list_is_reported():
    store = SynapStore(Sdk(response_of(scopes=())), user_id="alice", customer_id="acme",
                       on_error="raise")
    with pytest.raises(SynapIntegrationError):
        run(store.asearch((), query="x"))
