"""Delete / get must never reach a level above the store's own, and other
defects found by independent verification of WP5."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from synap_integrations_common import SynapIntegrationError
from synap_langgraph import SynapStore

from ._support import Sdk, SimSdk, UUIDS, ranked, response_of, tagged


def run(coro):
    return asyncio.run(coro)


# ── S1: a team store cannot touch the customer level ────────────────────────

def deep_world():
    sim = SimSdk(echo_caller_metadata=True)
    cust, team = SynapStore(sim, level="customer"), SynapStore(sim, level="team")
    run(cust.aput(("acme", "policies"), "k", {"o": "CUSTOMER"}))
    run(team.aput(("acme", "payments", "policies"), "k", {"o": "TEAM"}))
    return sim, cust, team








def test_S1_search_gives_each_item_its_true_namespace():
    sim, cust, team = deep_world()
    hits = run(team.asearch(("acme", "payments", "policies"), query="q"))
    assert {h.value["o"]: h.namespace for h in hits} == {
        "TEAM": ("acme", "payments", "policies"), "CUSTOMER": ("acme", "policies")}






# ── S3: never trust scope_map ───────────────────────────────────────────────



# ── S2: a single-user store's delete is documented and warns once ───────────





# ── 4: read-modify-write does not store the _synap block ────────────────────



# ── 5: get prefers the store's own level (single-user store too) ────────────





# ── 6: import does not need resolve_address ─────────────────────────────────

def test_6_the_package_imports_without_resolve_address():
    import subprocess, sys, os
    code = (
        "import synap_integrations_common as c\n"
        "del c.resolve_address\n"
        "import synap_integrations_common.scope as s\n"
        "import synap_langgraph\n"
        "from synap_langgraph import SynapStore, synap_st_prompt\n"
        "SynapStore(object(), user_id='u')\n"
        "print('ok')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=dict(os.environ))
    assert out.stdout.strip() == "ok", out.stderr


# ── 7: an SDK that cannot take the arguments is a setup error ───────────────

class OldFetchSdk:
    """An SDK whose fetch has no response_shape / scope_path."""

    async def fetch(self, conversation_id=None, user_id=None, customer_id=None, search_query=None,
                    max_results=20, mode="fast", include_conversation_context=True, scopes=None):
        raise AssertionError("must not be called")


@pytest.mark.parametrize("on_error", ["fallback", "raise"])
def test_7_old_sdk_raises_whatever_on_error_is(on_error):
    store = SynapStore(OldFetchSdk(), user_id="u", on_error=on_error)
    for coro in (store.asearch(("n",), query="q"),):
        with pytest.raises(SynapIntegrationError, match="response_shape"):
            run(coro)


def test_7_scope_path_support_is_checked_when_a_path_is_used():
    class NoPath(OldFetchSdk):
        async def fetch(self, user_id=None, customer_id=None, search_query=None, max_results=20,
                        mode="fast", include_conversation_context=True, response_shape="by_type"):
            raise AssertionError("must not be called")

    store = SynapStore(NoPath(), user_id="u", scope_path={"customer": "a", "user": "u"})
    with pytest.raises(SynapIntegrationError, match="scope_path"):
        run(store.asearch(("n",), query="q"))
    # without a path the same SDK is fine to ask
    ok = SynapStore(NoPath(), user_id="u", on_error="raise")
    with pytest.raises(SynapIntegrationError, match="must not be called"):   # reached the SDK
        run(ok.asearch(("n",), query="q"))


# ── 10: namespace type, logging ─────────────────────────────────────────────

@pytest.mark.parametrize("shared", [True, False])
def test_10_a_non_tuple_namespace_is_an_integration_error_not_a_type_error(shared):
    from langgraph.store.base import SearchOp
    store = SynapStore(Sdk()) if shared else SynapStore(Sdk(), user_id="u")
    with pytest.raises(SynapIntegrationError, match="tuple of strings"):
        run(store.abatch([SearchOp(None, "q", None, 10, 0)]))   # type: ignore[arg-type]


def test_10_errors_do_not_log_the_query_or_the_user(caplog):
    sdk = Sdk().fails(RuntimeError("boom"))
    with caplog.at_level(logging.ERROR, logger="synap_langgraph.store"):
        run(SynapStore(sdk).asearch(("acme", "alice-secret-id", "prefs"), query="my medical condition"))
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "sdk.fetch failed" in text
    for secret in ("my medical condition", "alice-secret-id", "prefs"):
        assert secret not in text


def test_10_put_error_context_is_fingerprints():
    sdk = Sdk()
    sdk.errors["memories.create"] = RuntimeError("500")
    with pytest.raises(SynapIntegrationError) as ei:
        run(SynapStore(sdk).aput(("acme", "alice", "prefs"), "diagnosis", {"v": 1}))
    assert "diagnosis" not in str(ei.value.context) and "prefs" not in str(ei.value.context)


# ── 8: the handler and the nodes are alternatives ───────────────────────────

def test_8_handler_plus_turn_nodes_report_every_turn_twice():
    """DOCUMENTED, not deduped: use ONE of them. See the README."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.graph import END, START, MessagesState, StateGraph
    from synap_langgraph import SynapLangGraphCallbackHandler, create_synap_turn_node

    sdk = Sdk()
    model = GenericFakeChatModel(messages=iter([AIMessage(content="an answer")]))

    async def answer(state: MessagesState):
        return {"messages": [await model.ainvoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("emit_user", create_synap_turn_node(sdk, role="user", content="messages"))
    g.add_node("answer", answer)
    g.add_node("emit_assistant", create_synap_turn_node(sdk, role="assistant", content="messages"))
    g.add_edge(START, "emit_user"); g.add_edge("emit_user", "answer")
    g.add_edge("answer", "emit_assistant"); g.add_edge("emit_assistant", END)
    handler = SynapLangGraphCallbackHandler(sdk=sdk, conversation_id="c1", user_id="alice")
    conf = {"callbacks": [handler], "configurable": {"user_id": "alice", "conversation_id": "c1"}}
    run(g.compile().ainvoke({"messages": [HumanMessage("hi")]}, config=conf))
    assert [c["role"] for c in sdk.kwargs_of("instance.send_message")].count("user") == 2
    assert [c["role"] for c in sdk.kwargs_of("instance.send_message")].count("assistant") == 2


# ── 3: a single-user store pinned by a scope_path is a level, tagged like one ─

TEAM = {"customer": "acme", "team": "payments"}


def pinned_world():
    sim = SimSdk(echo_caller_metadata=True)
    pool = SynapStore(sim, level="customer")
    team = SynapStore(sim, customer_id="acme", scope_path=TEAM)
    run(pool.aput(("acme", "policies"), "k", {"o": "CUSTOMER"}))
    run(team.aput(("policies",), "k", {"o": "TEAM"}))
    return sim, pool, team


def test_3_pinned_store_writes_the_identity_tag():
    sim, pool, team = pinned_world()
    assert sim.rows[1][2]["lg_store_id"] == '["acme", "payments"]'      # same as a shared team store








# ── 4: a missing resolve_address is an integration error with a hint ────────

def test_4_shared_store_without_resolve_address_raises_an_integration_error():
    import subprocess, sys, os
    code = (
        "import asyncio, synap_integrations_common as c\n"
        "del c.resolve_address\n"
        "from synap_langgraph import SynapStore\n"
        "from synap_integrations_common import SynapIntegrationError\n"
        "from tests._support import Sdk\n"
        "try:\n"
        "    asyncio.run(SynapStore(Sdk()).asearch(('acme','alice'), query='q'))\n"
        "except SynapIntegrationError as e:\n"
        "    print('OK', 'resolve_address' in str(e), 'Upgrade' in str(e))\n")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=here, env=dict(os.environ))
    assert out.stdout.strip() == "OK True True", out.stderr


# ── 5: the four mutants ─────────────────────────────────────────────────────

def test_T24_the_identity_tag_is_computed_per_call():
    sdk = Sdk()
    store = SynapStore(sdk)
    run(store.aput(("acme", "alice", "p"), "k", {"a": 1}))
    run(store.aput(("acme", "bob", "p"), "k", {"a": 1}))
    tags = [kw["metadata"]["lg_store_id"] for kw in sdk.kwargs_of("memories.create")]
    assert tags == ['["acme", "alice"]', '["acme", "bob"]']




class _SwitchingSdk:
    """fetch() has no scope_path; the ladder starts default and later turns deeper."""

    def __init__(self):
        from synap_integrations_common.testing import make_ladder
        self.make = make_ladder
        self.ladder = make_ladder("b2b")
        self.fetched = 0

    async def get_scope_ladder(self, refresh=False):
        return self.ladder

    async def fetch(self, user_id=None, customer_id=None, search_query=None, max_results=20,
                    mode="fast", include_conversation_context=True, response_shape="by_type"):
        self.fetched += 1
        return response_of()


def test_T15_scope_path_support_is_checked_on_every_call_that_uses_a_path():
    sdk = _SwitchingSdk()
    store = SynapStore(sdk, on_error="raise")
    run(store.asearch(("acme", "alice"), query="q"))          # no path: fine, check passes once
    assert sdk.fetched == 1
    sdk.ladder = sdk.make("deep")                              # now every address carries a path
    with pytest.raises(SynapIntegrationError, match=r"does not accept \['scope_path'\]"):
        run(store.asearch(("acme", "payments", "dana"), query="q"))
    assert sdk.fetched == 1


def test_T23_the_put_error_log_does_not_contain_the_raw_namespace_or_key(caplog):
    sdk = Sdk()
    sdk.errors["memories.create"] = RuntimeError("500")
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SynapIntegrationError):
            run(SynapStore(sdk).aput(("acme", "alice", "medical-prefs"), "diagnosis-key", {"a": 1}))
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "put failed" in text or "failed" in text
    assert "medical-prefs" not in text and "diagnosis-key" not in text
