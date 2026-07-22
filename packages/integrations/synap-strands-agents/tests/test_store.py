"""Tests for synap_strands_agents.store — SynapMemoryStore.

Contract:
- satisfies the Strands MemoryStore protocol (5 config attrs + search/add/add_messages).
- search maps Synap structured items -> MemoryEntry, honors max_search_results,
  and degrades to [] on SDK failure.
- add / add_messages ingest via memories.create (NOT record_messages_batch),
  add_messages uses a stable document_id, and both raise on failure.
"""

from __future__ import annotations

import pytest

from strands.memory import MemoryEntry, MemoryManager

from synap_integrations_common import SynapIntegrationError
from synap_integrations_common.testing import (
    make_episode,
    make_fact,
    make_preference,
    make_unified_response,
)

from synap_strands_agents.store import SynapMemoryStore


def _user(text):
    return {"role": "user", "content": [{"text": text}]}


def _assistant(text):
    return {"role": "assistant", "content": [{"text": text}]}


# ── construction / protocol shape ────────────────────────────────────────────


def test_requires_sdk():
    with pytest.raises(ValueError):
        SynapMemoryStore(None, "alice")


def test_requires_user_id(mock_sdk):
    with pytest.raises(ValueError):
        SynapMemoryStore(mock_sdk, "")


def test_exposes_protocol_config_fields(mock_sdk):
    store = SynapMemoryStore(mock_sdk, "alice", max_search_results=7)
    assert store.name == "synap"
    assert store.description
    assert store.max_search_results == 7
    assert store.writable is True
    assert store.extraction is True


def test_satisfies_memory_manager(mock_sdk):
    store = SynapMemoryStore(mock_sdk, "alice")
    mgr = MemoryManager(stores=[store])  # must not raise
    assert mgr is not None


# ── search ───────────────────────────────────────────────────────────────────


async def test_search_maps_structured_items_to_entries(mock_sdk):
    store = SynapMemoryStore(mock_sdk, "alice", customer_id="acme")
    entries = await store.search("who am i")

    assert all(isinstance(e, MemoryEntry) for e in entries)
    contents = [e.content for e in entries]
    assert "User is an engineer" in contents          # from make_fact()
    assert "Prefers dark mode" in contents             # from make_preference()
    types = {e.metadata["synap_type"] for e in entries}
    assert {"fact", "preference"} <= types

    kwargs = mock_sdk.fetch.await_args.kwargs
    assert kwargs["user_id"] == "alice"
    assert kwargs["customer_id"] == "acme"
    assert kwargs["search_query"] == ["who am i"]


async def test_search_covers_all_categories(mock_sdk):
    mock_sdk.fetch.return_value = make_unified_response(
        facts=[make_fact(content="fact one")],
        preferences=[make_preference(content="pref one")],
        episodes=[make_episode(summary="episode one")],
    )
    store = SynapMemoryStore(mock_sdk, "alice", max_search_results=10)
    entries = await store.search("q")
    contents = [e.content for e in entries]
    assert "episode one" in contents  # episodes mapped via .summary


async def test_search_honors_max_search_results(mock_sdk):
    store = SynapMemoryStore(mock_sdk, "alice", max_search_results=5)
    entries = await store.search("q", options={"max_search_results": 1})

    assert len(entries) == 1
    assert mock_sdk.fetch.await_args.kwargs["max_results"] == 1


async def test_search_degrades_on_failure(failing_sdk):
    store = SynapMemoryStore(failing_sdk, "alice")
    assert await store.search("q") == []


# ── add ──────────────────────────────────────────────────────────────────────


async def test_add_ingests_via_memories_create(mock_sdk):
    store = SynapMemoryStore(mock_sdk, "alice", customer_id="acme")
    await store.add("User likes tea", metadata={"k": "v"})

    mock_sdk.memories.create.assert_awaited_once()
    kwargs = mock_sdk.memories.create.await_args.kwargs
    assert kwargs["document"] == "User likes tea"
    assert kwargs["user_id"] == "alice"
    assert kwargs["customer_id"] == "acme"
    assert kwargs["metadata"] == {"k": "v"}


async def test_add_raises_on_failure(failing_sdk):
    store = SynapMemoryStore(failing_sdk, "alice")
    with pytest.raises(SynapIntegrationError):
        await store.add("boom")


# ── add_messages ─────────────────────────────────────────────────────────────


async def test_add_messages_renders_transcript_and_ingests(mock_sdk):
    store = SynapMemoryStore(mock_sdk, "alice", conversation_id="conv_1")
    await store.add_messages(
        [
            _user("what's the weather"),
            _assistant("sunny"),
            {"role": "user", "content": [{"toolResult": {"x": 1}}]},  # no text
        ]
    )

    kwargs = mock_sdk.memories.create.await_args.kwargs
    assert kwargs["document"] == "user: what's the weather\nassistant: sunny"
    assert kwargs["document_type"] == "ai-chat-conversation"
    assert kwargs["document_id"] == "strands-conv_1"


async def test_add_messages_document_id_is_stable(mock_sdk):
    store = SynapMemoryStore(mock_sdk, "alice", conversation_id="conv_1")
    await store.add_messages([_user("one")])
    await store.add_messages([_user("one"), _assistant("two")])

    ids = {c.kwargs["document_id"] for c in mock_sdk.memories.create.await_args_list}
    assert ids == {"strands-conv_1"}


async def test_add_messages_empty_batch_is_noop(mock_sdk):
    store = SynapMemoryStore(mock_sdk, "alice", conversation_id="conv_1")
    await store.add_messages([{"role": "user", "content": [{"toolResult": {}}]}])
    mock_sdk.memories.create.assert_not_awaited()


async def test_add_messages_raises_on_failure(failing_sdk):
    store = SynapMemoryStore(failing_sdk, "alice", conversation_id="conv_1")
    with pytest.raises(SynapIntegrationError):
        await store.add_messages([_user("boom")])
