"""Tests for synap_strands_agents.store — SynapMemoryStore.

Contract:
- satisfies the Strands MemoryStore protocol (5 config attrs + search/add/add_messages).
- search maps Synap structured items -> MemoryEntry, honors max_search_results,
  and degrades to [] on SDK failure.
- add ingests via memories.create; add_messages records turns (no transcript
  ingest, no double write next to SynapStreamHook), and both raise on failure.
"""

from __future__ import annotations

import pytest

from strands.memory import MemoryEntry, MemoryManager

from synap_integrations_common import SynapIntegrationError, set_auto_listen
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
#
# The batch is recorded as conversation turns, not ingested as a transcript with
# memories.create: Synap extracts from the conversation it keeps, and a
# transcript on top of turns SynapStreamHook already wrote is extracted twice.


@pytest.fixture(autouse=True)
def _rest_only(mock_sdk):
    set_auto_listen(mock_sdk, False)  # the stream path is test_history's


def _written(sdk):
    return [(c.kwargs["role"], c.kwargs["content"]) for c in sdk.conversation.record_message.await_args_list]


async def test_add_messages_records_the_turns_and_does_not_ingest_a_transcript(mock_sdk):
    store = SynapMemoryStore(mock_sdk, "alice", customer_id="acme", conversation_id="conv_1")
    await store.add_messages(
        [
            _user("what's the weather"),
            _assistant("sunny"),
            {"role": "user", "content": [{"toolResult": {"x": 1}}]},  # no text
        ]
    )

    assert _written(mock_sdk) == [("user", "what's the weather"), ("assistant", "sunny")]
    kwargs = mock_sdk.conversation.record_message.await_args.kwargs
    assert kwargs["conversation_id"] == "conv_1"
    assert kwargs["user_id"] == "alice"
    assert kwargs["customer_id"] == "acme"
    mock_sdk.memories.create.assert_not_awaited()


async def test_add_messages_without_a_conversation_id_keeps_one_per_store(mock_sdk):
    store = SynapMemoryStore(mock_sdk, "alice")
    await store.add_messages([_user("one")])
    await store.add_messages([_assistant("two")])

    ids = {c.kwargs["conversation_id"] for c in mock_sdk.conversation.record_message.await_args_list}
    assert len(ids) == 1 and next(iter(ids)).startswith("strands-")


async def test_add_messages_skips_what_the_stream_hook_already_wrote(mock_sdk):
    from unittest.mock import MagicMock
    from strands.hooks import MessageAddedEvent
    from synap_strands_agents.stream import SynapStreamHook

    hook = SynapStreamHook(mock_sdk, conversation_id="conv_1", user_id="alice")
    for role, text in (("user", "hi"), ("assistant", "hello"), ("user", "ok")):
        await hook._on_message(MessageAddedEvent(
            agent=MagicMock(), message={"role": role, "content": [{"text": text}]}))
    assert len(_written(mock_sdk)) == 3

    store = SynapMemoryStore(mock_sdk, "alice", conversation_id="conv_1")
    await store.add_messages([_user("hi"), _assistant("hello"), _user("ok"), _user("ok")])

    # only the second "ok" is new: one write consumed one write
    assert _written(mock_sdk)[3:] == [("user", "ok")]
    mock_sdk.memories.create.assert_not_awaited()


async def test_add_messages_empty_batch_is_noop(mock_sdk):
    store = SynapMemoryStore(mock_sdk, "alice", conversation_id="conv_1")
    await store.add_messages([{"role": "user", "content": [{"toolResult": {}}]}])
    mock_sdk.conversation.record_message.assert_not_awaited()
    mock_sdk.memories.create.assert_not_awaited()


async def test_add_messages_never_records_synap_context_as_a_turn(mock_sdk):
    store = SynapMemoryStore(mock_sdk, "alice", conversation_id="conv_1")
    await store.add_messages([{"role": "user", "content": [
        {"text": "<synap_short_term_context>\nx\n</synap_short_term_context>"}, {"text": "hello"}]}])
    assert _written(mock_sdk) == [("user", "hello")]


async def test_add_messages_raises_on_failure_and_a_retry_does_not_repeat_what_went_through(mock_sdk):
    store = SynapMemoryStore(mock_sdk, "alice", conversation_id="conv_1")
    sent = mock_sdk.conversation.record_message
    sent.side_effect = [None, RuntimeError("down")]
    batch = [_user("one"), _assistant("two")]
    with pytest.raises(SynapIntegrationError):
        await store.add_messages(batch)

    sent.side_effect = None
    await store.add_messages(batch)
    contents = [c.kwargs["content"] for c in sent.await_args_list]
    assert contents == ["one", "two", "two"]  # "one" went through the first time
