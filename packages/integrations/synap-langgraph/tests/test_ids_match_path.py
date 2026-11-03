"""Ids must agree with a fixed ``scope_path``: the server lets the path's
``user`` / ``customer`` win over flat ids, so a mismatch would read and file as
the path's user.

* Constructors raise before any SDK call.
* Reporting nodes read ids and path per run from the config; a run for another
  user is skipped and logged at ERROR (``on_error="raise"`` raises).
* A shared SynapStore takes ids and path from ONE ladder address, so they agree.
"""

from __future__ import annotations

import asyncio
import inspect
import logging

import pytest

from synap_integrations_common import SynapIntegrationError, ensure_ids_match_path
from synap_langgraph import (
    SynapCheckpointSaver,
    SynapLangGraphCallbackHandler,
    SynapStore,
    create_synap_reasoning_node,
    create_synap_tool_call_node,
    create_synap_tool_result_node,
    create_synap_turn_node,
)

from ._support import Sdk


def run(coro):
    return asyncio.run(coro)


PATH = {"customer": "acme", "team": "payments", "user": "dana"}
B2C = {"customer": "dana", "user": "dana"}


# ── SynapStore, single-user ─────────────────────────────────────────────────

def test_store_other_user_refused_with_zero_calls():
    sdk = Sdk(ladder="deep")
    with pytest.raises(SynapIntegrationError) as ei:
        SynapStore(sdk, user_id="bob", customer_id="acme", scope_path=PATH)
    assert "bob" not in str(ei.value) and sdk.calls == []


def test_store_other_customer_refused_with_zero_calls():
    sdk = Sdk(ladder="deep")
    with pytest.raises(SynapIntegrationError):
        SynapStore(sdk, user_id="dana", customer_id="globex", scope_path=PATH)
    assert sdk.calls == []


def test_store_matching_ids_accepted_and_sent_unchanged():
    sdk = Sdk(ladder="deep")
    store = SynapStore(sdk, user_id="dana", customer_id="acme", scope_path=PATH)
    run(store.asearch(("prefs",), query="q"))
    kw = sdk.last_kwargs("fetch")
    assert kw["user_id"] == "dana" and kw["customer_id"] == "acme" and kw["scope_path"] == PATH


def test_store_customer_only_with_a_customer_level_path_accepted():
    SynapStore(Sdk(ladder="deep"), customer_id="acme",
               scope_path={"customer": "acme", "team": "payments"})


def test_store_empty_customer_id_accepted():
    SynapStore(Sdk(ladder="deep"), user_id="dana", customer_id="", scope_path=PATH)


def test_store_b2c_collapse_accepted():
    SynapStore(Sdk(ladder="deep"), user_id="dana", scope_path=B2C)


def test_store_without_path_accepts_any_ids_and_sends_no_path():
    sdk = Sdk()
    store = SynapStore(sdk, user_id="bob", customer_id="globex")
    run(store.asearch(("prefs",), query="q"))
    kw = sdk.last_kwargs("fetch")
    assert kw["user_id"] == "bob" and kw["customer_id"] == "globex" and "scope_path" not in kw


# ── SynapStore, shared: ids and path come from one address ──────────────────

def test_shared_store_every_call_has_ids_that_agree_with_its_path():
    sdk = Sdk(ladder="deep")
    store = SynapStore(sdk)
    run(store.asearch(("acme", "payments", "dana", "n"), query="q"))
    run(store.aput(("acme", "payments", "dana", "n"), "k", {"v": 1}))
    run(store.arecord_message("c1", "user", "hi", namespace=("acme", "payments", "dana")))
    seen = 0
    for name, key in (("fetch", "scope_path"), ("memories.create", "scope"),
                      ("instance.send_message", "scope_path")):
        for kw in sdk.kwargs_of(name):
            ensure_ids_match_path(kw[key], user_id=kw.get("user_id"),
                                  customer_id=kw.get("customer_id"), where=name)
            seen += 1
    assert seen >= 3


# ── callback handler ────────────────────────────────────────────────────────

def test_handler_other_user_refused_with_zero_calls():
    sdk = Sdk(ladder="deep")
    with pytest.raises(SynapIntegrationError):
        SynapLangGraphCallbackHandler(sdk=sdk, conversation_id="c1", user_id="bob",
                                      customer_id="acme", scope_path=PATH)
    assert sdk.calls == []


def test_handler_matching_empty_customer_and_b2c_accepted():
    sdk = Sdk(ladder="deep")
    SynapLangGraphCallbackHandler(sdk=sdk, conversation_id="c1", user_id="dana",
                                  customer_id="acme", scope_path=PATH)
    SynapLangGraphCallbackHandler(sdk=sdk, conversation_id="c1", user_id="dana",
                                  customer_id="", scope_path=PATH)
    SynapLangGraphCallbackHandler(sdk=sdk, conversation_id="c1", user_id="dana", scope_path=B2C)


def test_handler_without_path_accepts_any_ids():
    SynapLangGraphCallbackHandler(sdk=Sdk(), conversation_id="c1", user_id="bob", customer_id="globex")


# ── reporting nodes ─────────────────────────────────────────────────────────

def conf(**over):
    return {"configurable": {"user_id": "dana", "customer_id": "acme",
                             "conversation_id": "c1", "scope_path": PATH, **over}}


def turn_node(sdk, **kw):
    return create_synap_turn_node(sdk, role="user", content="q", **kw)


def test_turn_node_other_user_in_config_is_skipped_and_logged_with_zero_calls(caplog):
    sdk = Sdk(ladder="deep")
    with caplog.at_level(logging.ERROR):
        out = run(turn_node(sdk)({"q": "hi"}, conf(user_id="bob")))
    assert out == {} and sdk.calls == []
    errs = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errs and "bob" not in errs[0].getMessage()


def test_turn_node_other_customer_in_config_is_skipped_with_zero_calls():
    sdk = Sdk(ladder="deep")
    run(turn_node(sdk)({"q": "hi"}, conf(customer_id="globex")))
    assert sdk.calls == []


def test_turn_node_on_error_raise_raises_with_zero_calls():
    sdk = Sdk(ladder="deep")
    with pytest.raises(SynapIntegrationError):
        run(turn_node(sdk, on_error="raise")({"q": "hi"}, conf(user_id="bob")))
    assert sdk.calls == []


@pytest.mark.parametrize("make", [
    lambda sdk: create_synap_tool_call_node(sdk, tool_name="t"),
    lambda sdk: create_synap_tool_result_node(sdk, result="q"),
    lambda sdk: create_synap_reasoning_node(sdk, content="q"),
])
def test_other_reporting_nodes_also_skip_a_run_for_another_user(make):
    sdk = Sdk(ladder="deep")
    run(make(sdk)({"q": "hi"}, conf(user_id="bob")))
    assert sdk.calls == []


def test_turn_node_matching_config_reports_with_the_path():
    sdk = Sdk(ladder="deep")
    run(turn_node(sdk)({"q": "hi"}, conf()))
    kw = sdk.last_kwargs("instance.send_message")
    assert kw["user_id"] == "dana" and kw["scope_path"] == PATH


def test_turn_node_empty_customer_in_config_accepted():
    sdk = Sdk(ladder="deep")
    run(turn_node(sdk)({"q": "hi"}, conf(customer_id="")))
    assert len(sdk.kwargs_of("instance.send_message")) == 1


def test_turn_node_b2c_collapse_accepted():
    sdk = Sdk(ladder="deep")
    run(turn_node(sdk)({"q": "hi"}, conf(scope_path=B2C, customer_id="", user_id="dana")))
    assert len(sdk.kwargs_of("instance.send_message")) == 1


def test_turn_node_factory_arguments_that_disagree_raise_at_construction():
    sdk = Sdk(ladder="deep")
    with pytest.raises(SynapIntegrationError):
        turn_node(sdk, user_id="bob", scope_path=PATH)
    assert sdk.calls == []
    turn_node(sdk, user_id="dana", customer_id="acme", scope_path=PATH)


def test_turn_node_without_path_sends_any_ids_unchanged():
    sdk = Sdk()
    run(turn_node(sdk)({"q": "hi"}, {"configurable": {
        "user_id": "bob", "customer_id": "globex", "conversation_id": "c1"}}))
    kw = sdk.last_kwargs("instance.send_message")
    assert kw["user_id"] == "bob" and kw["customer_id"] == "globex" and "scope_path" not in kw


# ── checkpointer takes no scope_path: nothing to check ──────────────────────

def test_checkpointer_has_no_scope_path_parameter():
    assert "scope_path" not in inspect.signature(SynapCheckpointSaver.__init__).parameters
