"""What the store does against TODAY'S server, which returns only a fixed list
of item metadata and never the metadata a caller passed to memories.create
(``SimSdk()`` default). The tag logic itself is proven with
``SimSdk(echo_caller_metadata=True)`` in test_store_security.py."""

from __future__ import annotations

import asyncio
import json

import pytest

from synap_integrations_common import SynapIntegrationError
from synap_langgraph import SynapStore

from ._support import SimSdk


def run(coro):
    return asyncio.run(coro)


def world():
    sim = SimSdk(ladder="b2b", depth=2)
    store = SynapStore(sim)
    run(store.aput(("acme", "alice", "prefs"), "language", {"v": "en"}))
    return sim, store


def test_put_works_and_sends_the_tag():
    sim, _ = world()
    assert len(sim.rows) == 1
    md = sim.rows[0][2]
    assert md["lg_store_key"] == "language" and md["lg_store_id"] == '["acme", "alice"]'


def test_search_works_and_returns_the_memory_as_an_ordinary_one():
    sim, store = world()
    hits = run(store.asearch(("acme", "alice", "prefs"), query="language"))
    assert [h.value["v"] for h in hits] == ["en"]
    assert hits[0].namespace == ("acme", "alice", "prefs")       # the namespace asked for
    assert hits[0].value["_synap"]["id"] == sim.rows[0][0]       # the id to delete by


def test_search_does_not_filter_by_sub_namespace_because_nothing_is_tagged():
    sim, store = world()
    run(store.aput(("acme", "alice", "other"), "color", {"v": "red"}))
    assert len(run(store.asearch(("acme", "alice", "prefs"), query="q"))) == 2






def test_deleting_by_the_id_from_search_works():
    sim, store = world()
    memory_id = run(store.asearch(("acme", "alice", "prefs"), query="q"))[0].value["_synap"]["id"]
    run(sim.memories.delete(memory_id))
    assert sim.rows == []
