"""Tests for synap_agno.db — SynapDb and _fact_to_user_memory.

Documented error-handling contract (from db.py docstring):
- reads (get_user_memory, get_user_memories, get_all_memory_topics): degrade
  gracefully — log at ERROR, return empty result. SDK blips must not crash
  an agent turn.
- writes (upsert_user_memory, upsert_memories): surface SynapIntegrationError
  so ingestion outages are observable.
- deletes (delete_user_memory, delete_user_memories, clear_memories): warn
  once and no-op — Synap has no public delete API.
- get_user_memory_stats: warn once and return ([], 0) — Synap has no
  aggregate stats endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agno.db.in_memory.in_memory_db import InMemoryDb
from agno.db.schemas.memory import UserMemory

from synap_agno.db import SynapDb, _fact_to_user_memory, _MARKER
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sdk(
    *,
    fetch_response=None,
    fetch_error: Exception | None = None,
    create_error: Exception | None = None,
) -> MagicMock:
    """Build a minimal SDK mock suitable for SynapDb tests."""
    sdk = MagicMock()
    sdk.memories = MagicMock()

    if fetch_error is not None:
        sdk.fetch = AsyncMock(side_effect=fetch_error)
    else:
        resp = fetch_response if fetch_response is not None else _empty_response()
        sdk.fetch = AsyncMock(return_value=resp)

    if create_error is not None:
        sdk.memories.create = AsyncMock(side_effect=create_error)
    else:
        result = MagicMock()
        result.ingestion_id = "ing-001"
        sdk.memories.create = AsyncMock(return_value=result)

    return sdk


def _empty_response() -> MagicMock:
    resp = MagicMock()
    resp.facts = []
    return resp


def _response_with_facts(*facts) -> MagicMock:
    resp = MagicMock()
    resp.facts = list(facts)
    return resp


def _make_marked_fact(
    memory_id: str = "mem-001",
    user_id: str = "alice",
    content: str = "User likes dark mode",
    agent_id: str | None = None,
    team_id: str | None = None,
    topics: list[str] | None = None,
    input_: str | None = None,
    feedback: str | None = None,
    created_at: int = 1000,
    updated_at: int = 2000,
) -> MagicMock:
    """Build a Fact mock tagged with the agno_user_memory marker."""
    fact = MagicMock()
    fact.content = content
    fact.id = memory_id
    fact.metadata = {
        _MARKER: True,
        "memory_id": memory_id,
        "user_id": user_id,
        "agent_id": agent_id,
        "team_id": team_id,
        "topics": topics or [],
        "input": input_,
        "feedback": feedback,
        "created_at": created_at,
        "updated_at": updated_at,
    }
    return fact


def _make_untagged_fact(memory_id: str = "bare-001") -> MagicMock:
    """A fact with no agno_user_memory marker — external/non-agno origin."""
    fact = MagicMock()
    fact.content = "external content"
    fact.id = memory_id
    fact.metadata = {"some_other_key": True, "memory_id": memory_id}
    return fact


def _run_sync(coro):
    """Drive a coroutine to completion in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Patch run_async to use a fresh event loop (avoids nest_asyncio in test context).
_patch_run_async = patch("synap_agno.db.run_async", side_effect=_run_sync)


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestSynapDbConstruction:
    def test_requires_non_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapDb(None)  # type: ignore[arg-type]

    def test_defaults(self):
        sdk = _make_sdk()
        db = SynapDb(sdk)
        assert db.customer_id == ""
        assert db.mode == "accurate"
        assert db.max_results == 50
        assert db._delete_warned is False
        assert db._stats_warned is False

    def test_custom_params(self):
        sdk = _make_sdk()
        db = SynapDb(sdk, customer_id="acme", mode="fast", max_results=10)
        assert db.customer_id == "acme"
        assert db.mode == "fast"
        assert db.max_results == 10

    def test_is_subclass_of_in_memory_db(self):
        """SynapDb must extend InMemoryDb so Agno's 46+ non-memory methods work."""
        assert issubclass(SynapDb, InMemoryDb)

    def test_public_export(self):
        import synap_agno
        assert hasattr(synap_agno, "SynapDb")
        assert "SynapDb" in synap_agno.__all__


# ---------------------------------------------------------------------------
# _fact_to_user_memory helper
# ---------------------------------------------------------------------------


class TestFactToUserMemory:
    def test_maps_content_and_metadata_fields(self):
        fact = _make_marked_fact(
            memory_id="mem-xyz",
            user_id="bob",
            content="Bob's preference",
            agent_id="agent-1",
            team_id="team-1",
            topics=["cooking", "travel"],
            input_="I love cooking",
            feedback="positive",
            created_at=1100,
            updated_at=2200,
        )
        mem = _fact_to_user_memory(fact)
        assert mem.memory == "Bob's preference"
        assert mem.memory_id == "mem-xyz"
        assert mem.user_id == "bob"
        assert mem.agent_id == "agent-1"
        assert mem.team_id == "team-1"
        assert mem.topics == ["cooking", "travel"]
        assert mem.input == "I love cooking"
        assert mem.feedback == "positive"
        assert mem.created_at == 1100
        assert mem.updated_at == 2200

    def test_falls_back_to_fact_id_when_no_memory_id_in_metadata(self):
        fact = MagicMock()
        fact.content = "bare"
        fact.id = "fact-id-fallback"
        fact.metadata = {_MARKER: True}  # no 'memory_id' key
        mem = _fact_to_user_memory(fact)
        assert mem.memory_id == "fact-id-fallback"

    def test_none_metadata_returns_memory_from_content(self):
        class _BareObj:
            content = "content only"
            id = "bare-id"
            metadata = None

        mem = _fact_to_user_memory(_BareObj())
        assert mem.memory == "content only"

    def test_absent_metadata_attr_returns_memory_from_content(self):
        class _NoMeta:
            content = "no meta"
            id = "no-meta-id"

        mem = _fact_to_user_memory(_NoMeta())
        assert mem.memory == "no meta"

    def test_topics_must_be_list_or_none(self):
        # topics present as list → list
        fact = _make_marked_fact(topics=["a", "b"])
        mem = _fact_to_user_memory(fact)
        assert isinstance(mem.topics, list)
        assert mem.topics == ["a", "b"]

    def test_non_list_topics_in_metadata_maps_to_none(self):
        fact = MagicMock()
        fact.content = "c"
        fact.id = "id"
        fact.metadata = {_MARKER: True, "topics": "not-a-list"}
        mem = _fact_to_user_memory(fact)
        # non-list topics value → None (safe fallback)
        assert mem.topics is None


# ---------------------------------------------------------------------------
# upsert_user_memory — happy path
# ---------------------------------------------------------------------------


class TestUpsertUserMemory:
    def test_calls_sdk_memories_create(self):
        sdk = _make_sdk()
        with _patch_run_async:
            db = SynapDb(sdk, customer_id="acme")
            m = UserMemory(memory="User prefers dark mode", user_id="alice")
            db.upsert_user_memory(m)
        sdk.memories.create.assert_awaited_once()

    def test_passes_document_user_id_customer_id(self):
        sdk = _make_sdk()
        with _patch_run_async:
            db = SynapDb(sdk, customer_id="acme")
            m = UserMemory(memory="Tea lover", user_id="alice")
            db.upsert_user_memory(m)
        kw = sdk.memories.create.call_args.kwargs
        assert kw["document"] == "Tea lover"
        assert kw["user_id"] == "alice"
        assert kw["customer_id"] == "acme"

    def test_empty_customer_id_becomes_none_in_sdk_call(self):
        sdk = _make_sdk()
        with _patch_run_async:
            db = SynapDb(sdk, customer_id="")
            m = UserMemory(memory="test")
            db.upsert_user_memory(m)
        kw = sdk.memories.create.call_args.kwargs
        assert kw["customer_id"] is None

    def test_metadata_carries_agno_marker(self):
        sdk = _make_sdk()
        with _patch_run_async:
            db = SynapDb(sdk)
            m = UserMemory(memory="x")
            db.upsert_user_memory(m)
        metadata = sdk.memories.create.call_args.kwargs["metadata"]
        assert metadata[_MARKER] is True

    def test_metadata_carries_all_memory_fields(self):
        sdk = _make_sdk()
        with _patch_run_async:
            db = SynapDb(sdk, customer_id="acme")
            m = UserMemory(
                memory="test",
                user_id="alice",
                agent_id="agent-1",
                team_id="team-1",
                topics=["coding", "python"],
                input="original input",
                feedback="liked",
            )
            db.upsert_user_memory(m)
        metadata = sdk.memories.create.call_args.kwargs["metadata"]
        assert metadata["user_id"] == "alice"
        assert metadata["agent_id"] == "agent-1"
        assert metadata["team_id"] == "team-1"
        assert metadata["topics"] == ["coding", "python"]
        assert metadata["input"] == "original input"
        assert metadata["feedback"] == "liked"

    def test_auto_assigns_memory_id_when_none(self):
        sdk = _make_sdk()
        with _patch_run_async:
            db = SynapDb(sdk)
            m = UserMemory(memory="test")
            assert m.memory_id is None
            result = db.upsert_user_memory(m)
        assert result.memory_id is not None
        assert len(result.memory_id) > 0

    def test_preserves_pre_assigned_memory_id(self):
        sdk = _make_sdk()
        with _patch_run_async:
            db = SynapDb(sdk)
            m = UserMemory(memory="test", memory_id="pre-assigned-id")
            result = db.upsert_user_memory(m)
        assert result.memory_id == "pre-assigned-id"

    def test_sets_timestamps(self):
        sdk = _make_sdk()
        before = int(time.time())
        with _patch_run_async:
            db = SynapDb(sdk)
            m = UserMemory(memory="test")
            result = db.upsert_user_memory(m)
        after = int(time.time())
        assert before <= result.created_at <= after
        assert before <= result.updated_at <= after

    def test_returns_user_memory_when_deserialize_true(self):
        sdk = _make_sdk()
        with _patch_run_async:
            db = SynapDb(sdk)
            m = UserMemory(memory="test")
            result = db.upsert_user_memory(m, deserialize=True)
        assert isinstance(result, UserMemory)

    def test_returns_dict_when_deserialize_false(self):
        sdk = _make_sdk()
        with _patch_run_async:
            db = SynapDb(sdk)
            m = UserMemory(memory="test")
            result = db.upsert_user_memory(m, deserialize=False)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# upsert_user_memory — failure path
# ---------------------------------------------------------------------------


class TestUpsertUserMemoryFailure:
    def test_raises_synap_integration_error_on_sdk_failure(self):
        sdk = _make_sdk(create_error=RuntimeError("create failed"))
        with _patch_run_async:
            db = SynapDb(sdk)
            with pytest.raises(SynapIntegrationError):
                db.upsert_user_memory(UserMemory(memory="test"))

    def test_error_chains_original_cause(self):
        original = RuntimeError("root cause")
        sdk = _make_sdk(create_error=original)
        with _patch_run_async:
            db = SynapDb(sdk)
            with pytest.raises(SynapIntegrationError) as exc_info:
                db.upsert_user_memory(UserMemory(memory="test"))
        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# upsert_memories (batch) — happy path
# ---------------------------------------------------------------------------


class TestUpsertMemories:
    def test_calls_create_once_per_memory(self):
        sdk = _make_sdk()
        with _patch_run_async:
            db = SynapDb(sdk)
            mems = [UserMemory(memory=f"memory {i}") for i in range(3)]
            db.upsert_memories(mems)
        assert sdk.memories.create.await_count == 3

    def test_returns_list_of_user_memories_when_deserialize_true(self):
        sdk = _make_sdk()
        with _patch_run_async:
            db = SynapDb(sdk)
            mems = [UserMemory(memory="a"), UserMemory(memory="b")]
            result = db.upsert_memories(mems)
        assert len(result) == 2
        assert all(isinstance(r, UserMemory) for r in result)

    def test_returns_list_of_dicts_when_deserialize_false(self):
        sdk = _make_sdk()
        with _patch_run_async:
            db = SynapDb(sdk)
            mems = [UserMemory(memory="a"), UserMemory(memory="b")]
            result = db.upsert_memories(mems, deserialize=False)
        assert all(isinstance(r, dict) for r in result)

    def test_preserve_updated_at_keeps_original_timestamp(self):
        sdk = _make_sdk()
        with _patch_run_async:
            db = SynapDb(sdk)
            m = UserMemory(memory="test", updated_at=9999)
            result = db.upsert_memories([m], preserve_updated_at=True)
        assert result[0].updated_at == 9999

    def test_default_overwrites_updated_at(self):
        sdk = _make_sdk()
        before = int(time.time())
        with _patch_run_async:
            db = SynapDb(sdk)
            m = UserMemory(memory="test", updated_at=9999)
            result = db.upsert_memories([m], preserve_updated_at=False)
        after = int(time.time())
        assert before <= result[0].updated_at <= after

    def test_auto_assigns_memory_id_for_memories_without_id(self):
        sdk = _make_sdk()
        with _patch_run_async:
            db = SynapDb(sdk)
            mems = [UserMemory(memory="x"), UserMemory(memory="y")]
            result = db.upsert_memories(mems)
        assert all(m.memory_id is not None for m in result)

    def test_empty_list_returns_empty_list(self):
        sdk = _make_sdk()
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.upsert_memories([])
        assert result == []
        sdk.memories.create.assert_not_awaited()

    def test_raises_synap_integration_error_on_sdk_failure(self):
        sdk = _make_sdk(create_error=RuntimeError("batch failed"))
        with _patch_run_async:
            db = SynapDb(sdk)
            with pytest.raises(SynapIntegrationError):
                db.upsert_memories([UserMemory(memory="x")])


# ---------------------------------------------------------------------------
# get_user_memory — happy path
# ---------------------------------------------------------------------------


class TestGetUserMemory:
    def test_returns_user_memory_for_existing_id(self):
        fact = _make_marked_fact("mem-001", user_id="alice", content="dark mode")
        sdk = _make_sdk(fetch_response=_response_with_facts(fact))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memory("mem-001")
        assert isinstance(result, UserMemory)
        assert result.memory_id == "mem-001"
        assert result.memory == "dark mode"

    def test_returns_none_when_memory_id_not_found(self):
        fact = _make_marked_fact("mem-001")
        sdk = _make_sdk(fetch_response=_response_with_facts(fact))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memory("mem-999")
        assert result is None

    def test_user_id_filter_returns_none_for_wrong_user(self):
        fact = _make_marked_fact("mem-001", user_id="alice")
        sdk = _make_sdk(fetch_response=_response_with_facts(fact))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memory("mem-001", user_id="bob")
        assert result is None

    def test_user_id_filter_returns_memory_for_correct_user(self):
        fact = _make_marked_fact("mem-001", user_id="alice")
        sdk = _make_sdk(fetch_response=_response_with_facts(fact))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memory("mem-001", user_id="alice")
        assert result is not None
        assert result.user_id == "alice"

    def test_skips_untagged_facts(self):
        tagged = _make_marked_fact("mem-001")
        untagged = _make_untagged_fact("mem-002")
        sdk = _make_sdk(fetch_response=_response_with_facts(tagged, untagged))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memory("mem-002")
        assert result is None  # untagged fact should not match

    def test_returns_dict_when_deserialize_false(self):
        fact = _make_marked_fact("mem-001")
        sdk = _make_sdk(fetch_response=_response_with_facts(fact))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memory("mem-001", deserialize=False)
        assert isinstance(result, dict)

    def test_passes_memory_id_as_search_query_to_sdk(self):
        sdk = _make_sdk(fetch_response=_empty_response())
        with _patch_run_async:
            db = SynapDb(sdk, customer_id="acme")
            db.get_user_memory("mem-target")
        kw = sdk.fetch.call_args.kwargs
        assert "mem-target" in kw["search_query"]
        assert kw["customer_id"] == "acme"


# ---------------------------------------------------------------------------
# get_user_memory — failure path (graceful degrade)
# ---------------------------------------------------------------------------


class TestGetUserMemoryFailure:
    def test_returns_none_on_sdk_error(self):
        sdk = _make_sdk(fetch_error=RuntimeError("sdk boom"))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memory("mem-001")
        assert result is None  # graceful degrade, not a crash

    def test_logs_error_on_sdk_failure(self, caplog):
        sdk = _make_sdk(fetch_error=RuntimeError("sdk boom"))
        with caplog.at_level(logging.ERROR, logger="synap_agno.db"):
            with _patch_run_async:
                db = SynapDb(sdk)
                db.get_user_memory("mem-001")
        assert any("sdk.fetch failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# get_user_memories — happy path
# ---------------------------------------------------------------------------


class TestGetUserMemories:
    def _db_with_facts(self, *facts) -> tuple[SynapDb, MagicMock]:
        sdk = _make_sdk(fetch_response=_response_with_facts(*facts))
        return SynapDb(sdk), sdk

    def test_returns_all_tagged_memories(self):
        f1 = _make_marked_fact("m1", user_id="alice")
        f2 = _make_marked_fact("m2", user_id="bob")
        sdk = _make_sdk(fetch_response=_response_with_facts(f1, f2))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memories()
        assert len(result) == 2

    def test_skips_untagged_facts(self):
        tagged = _make_marked_fact("m1")
        untagged = _make_untagged_fact("m2")
        sdk = _make_sdk(fetch_response=_response_with_facts(tagged, untagged))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memories()
        assert len(result) == 1
        assert result[0].memory_id == "m1"

    def test_filters_by_user_id(self):
        f1 = _make_marked_fact("m1", user_id="alice")
        f2 = _make_marked_fact("m2", user_id="bob")
        sdk = _make_sdk(fetch_response=_response_with_facts(f1, f2))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memories(user_id="alice")
        assert len(result) == 1
        assert result[0].user_id == "alice"

    def test_filters_by_agent_id(self):
        f1 = _make_marked_fact("m1", agent_id="agent-1")
        f2 = _make_marked_fact("m2", agent_id="agent-2")
        sdk = _make_sdk(fetch_response=_response_with_facts(f1, f2))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memories(agent_id="agent-1")
        assert len(result) == 1
        assert result[0].memory_id == "m1"

    def test_filters_by_team_id(self):
        f1 = _make_marked_fact("m1", team_id="team-a")
        f2 = _make_marked_fact("m2", team_id="team-b")
        sdk = _make_sdk(fetch_response=_response_with_facts(f1, f2))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memories(team_id="team-a")
        assert len(result) == 1

    def test_filters_by_topic_intersection(self):
        f1 = _make_marked_fact("m1", topics=["coding", "python"])
        f2 = _make_marked_fact("m2", topics=["cooking"])
        sdk = _make_sdk(fetch_response=_response_with_facts(f1, f2))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memories(topics=["coding"])
        assert len(result) == 1
        assert result[0].memory_id == "m1"

    def test_passes_search_content_as_search_query(self):
        sdk = _make_sdk(fetch_response=_empty_response())
        with _patch_run_async:
            db = SynapDb(sdk)
            db.get_user_memories(search_content="dark mode")
        kw = sdk.fetch.call_args.kwargs
        assert kw["search_query"] == ["dark mode"]

    def test_no_search_content_passes_none_query(self):
        sdk = _make_sdk(fetch_response=_empty_response())
        with _patch_run_async:
            db = SynapDb(sdk)
            db.get_user_memories()
        kw = sdk.fetch.call_args.kwargs
        assert kw["search_query"] is None

    def test_passes_correct_fetch_kwargs(self):
        sdk = _make_sdk(fetch_response=_empty_response())
        with _patch_run_async:
            db = SynapDb(sdk, customer_id="acme", mode="fast", max_results=10)
            db.get_user_memories(user_id="alice")
        kw = sdk.fetch.call_args.kwargs
        assert kw["user_id"] == "alice"
        assert kw["customer_id"] == "acme"
        assert kw["mode"] == "fast"
        assert kw["include_conversation_context"] is False

    def test_sort_by_created_at_desc_default(self):
        f1 = _make_marked_fact("m1", created_at=1000)
        f2 = _make_marked_fact("m2", created_at=3000)
        f3 = _make_marked_fact("m3", created_at=2000)
        sdk = _make_sdk(fetch_response=_response_with_facts(f1, f2, f3))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memories(sort_by="created_at")
        ids = [m.memory_id for m in result]
        assert ids == ["m2", "m3", "m1"]

    def test_sort_by_created_at_asc(self):
        f1 = _make_marked_fact("m1", created_at=1000)
        f2 = _make_marked_fact("m2", created_at=3000)
        f3 = _make_marked_fact("m3", created_at=2000)
        sdk = _make_sdk(fetch_response=_response_with_facts(f1, f2, f3))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memories(sort_by="created_at", sort_order="asc")
        ids = [m.memory_id for m in result]
        assert ids == ["m1", "m3", "m2"]

    def test_sort_by_updated_at_desc(self):
        f1 = _make_marked_fact("m1", updated_at=5000)
        f2 = _make_marked_fact("m2", updated_at=1000)
        sdk = _make_sdk(fetch_response=_response_with_facts(f1, f2))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memories(sort_by="updated_at")
        assert result[0].memory_id == "m1"

    def test_pagination_limit_and_page(self):
        facts = [_make_marked_fact(f"m{i}", created_at=i) for i in range(5)]
        sdk = _make_sdk(fetch_response=_response_with_facts(*facts))
        with _patch_run_async:
            db = SynapDb(sdk)
            # Page 0, limit 2 → first 2 items (after sort desc)
            page0 = db.get_user_memories(sort_by="created_at", limit=2, page=0)
            page1 = db.get_user_memories(sort_by="created_at", limit=2, page=1)
        assert len(page0) == 2
        assert len(page1) == 2
        # Ensure no overlap
        ids0 = {m.memory_id for m in page0}
        ids1 = {m.memory_id for m in page1}
        assert ids0.isdisjoint(ids1)

    def test_returns_list_of_user_memories_when_deserialize_true(self):
        f = _make_marked_fact("m1")
        sdk = _make_sdk(fetch_response=_response_with_facts(f))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memories(deserialize=True)
        assert isinstance(result, list)
        assert all(isinstance(m, UserMemory) for m in result)

    def test_returns_tuple_of_dicts_and_count_when_deserialize_false(self):
        f = _make_marked_fact("m1")
        sdk = _make_sdk(fetch_response=_response_with_facts(f))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memories(deserialize=False)
        assert isinstance(result, tuple)
        dicts, count = result
        assert isinstance(dicts, list)
        assert count == 1
        assert isinstance(dicts[0], dict)

    def test_empty_response_returns_empty_list(self):
        sdk = _make_sdk(fetch_response=_empty_response())
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memories()
        assert result == []


# ---------------------------------------------------------------------------
# get_user_memories — failure path (graceful degrade)
# ---------------------------------------------------------------------------


class TestGetUserMemoriesFailure:
    def test_returns_empty_list_on_sdk_error(self):
        sdk = _make_sdk(fetch_error=RuntimeError("sdk boom"))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_user_memories()
        assert result == []

    def test_logs_error_on_sdk_failure(self, caplog):
        sdk = _make_sdk(fetch_error=RuntimeError("sdk boom"))
        with caplog.at_level(logging.ERROR, logger="synap_agno.db"):
            with _patch_run_async:
                db = SynapDb(sdk)
                db.get_user_memories(user_id="alice")
        assert any("sdk.fetch failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# get_user_memory_stats — warns once, returns ([], 0)
# ---------------------------------------------------------------------------


class TestGetUserMemoryStats:
    def test_returns_empty_tuple(self):
        sdk = _make_sdk()
        db = SynapDb(sdk)
        result = db.get_user_memory_stats()
        assert result == ([], 0)

    def test_return_type_is_tuple_of_list_and_int(self):
        sdk = _make_sdk()
        db = SynapDb(sdk)
        items, count = db.get_user_memory_stats()
        assert isinstance(items, list)
        assert isinstance(count, int)

    def test_warns_once_then_silent(self, caplog):
        sdk = _make_sdk()
        db = SynapDb(sdk)
        with caplog.at_level(logging.WARNING, logger="synap_agno.db"):
            db.get_user_memory_stats()
            db.get_user_memory_stats()
            db.get_user_memory_stats()
        warning_msgs = [r for r in caplog.records if "aggregate" in r.message]
        assert len(warning_msgs) == 1, "must warn exactly once regardless of call count"

    def test_sets_stats_warned_flag(self):
        sdk = _make_sdk()
        db = SynapDb(sdk)
        assert db._stats_warned is False
        db.get_user_memory_stats()
        assert db._stats_warned is True

    def test_accepts_limit_page_user_id_kwargs(self):
        """Signature must accept these Agno-standard kwargs without crashing."""
        sdk = _make_sdk()
        db = SynapDb(sdk)
        result = db.get_user_memory_stats(limit=10, page=0, user_id="alice")
        assert result == ([], 0)


# ---------------------------------------------------------------------------
# get_all_memory_topics
# ---------------------------------------------------------------------------


class TestGetAllMemoryTopics:
    def test_returns_sorted_unique_topics(self):
        f1 = _make_marked_fact("m1", topics=["python", "coding"])
        f2 = _make_marked_fact("m2", topics=["cooking", "python"])  # python appears twice
        sdk = _make_sdk(fetch_response=_response_with_facts(f1, f2))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_all_memory_topics()
        assert result == sorted(set(result))  # sorted
        assert len(result) == len(set(result))  # unique
        assert "python" in result

    def test_filters_topics_by_user_id(self):
        f1 = _make_marked_fact("m1", user_id="alice", topics=["music"])
        f2 = _make_marked_fact("m2", user_id="bob", topics=["gaming"])
        sdk = _make_sdk(fetch_response=_response_with_facts(f1, f2))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_all_memory_topics(user_id="alice")
        assert result == ["music"]

    def test_skips_non_agno_facts(self):
        tagged = _make_marked_fact("m1", topics=["included"])
        untagged = _make_untagged_fact("m2")
        untagged.metadata["topics"] = ["excluded"]
        sdk = _make_sdk(fetch_response=_response_with_facts(tagged, untagged))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_all_memory_topics()
        assert "excluded" not in result
        assert "included" in result

    def test_ignores_non_string_topic_values(self):
        fact = MagicMock()
        fact.content = "c"
        fact.id = "m1"
        fact.metadata = {
            _MARKER: True,
            "user_id": "alice",
            "topics": ["valid", 123, None, "also-valid"],
        }
        sdk = _make_sdk(fetch_response=_response_with_facts(fact))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_all_memory_topics()
        assert "valid" in result
        assert "also-valid" in result
        # non-strings must be filtered out (only isinstance(t, str) are added)
        assert 123 not in result
        assert None not in result

    def test_returns_empty_list_when_no_facts(self):
        sdk = _make_sdk(fetch_response=_empty_response())
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_all_memory_topics()
        assert result == []

    def test_degrades_gracefully_on_sdk_error(self):
        sdk = _make_sdk(fetch_error=RuntimeError("boom"))
        with _patch_run_async:
            db = SynapDb(sdk)
            result = db.get_all_memory_topics()
        assert result == []


# ---------------------------------------------------------------------------
# delete methods — warn once + no-op
# ---------------------------------------------------------------------------


class TestDeleteMethods:
    def test_delete_user_memory_warns_once(self, caplog):
        sdk = _make_sdk()
        db = SynapDb(sdk)
        with caplog.at_level(logging.WARNING, logger="synap_agno.db"):
            db.delete_user_memory("mem-001")
            db.delete_user_memory("mem-002")
            db.delete_user_memory("mem-003")
        warn_msgs = [r for r in caplog.records if "no public delete" in r.message]
        assert len(warn_msgs) == 1, "must warn exactly once regardless of call count"

    def test_delete_user_memories_warns_once(self, caplog):
        sdk = _make_sdk()
        db = SynapDb(sdk)
        with caplog.at_level(logging.WARNING, logger="synap_agno.db"):
            db.delete_user_memories(["mem-001", "mem-002"])
            db.delete_user_memories(["mem-003"])
        warn_msgs = [r for r in caplog.records if "no public delete" in r.message]
        assert len(warn_msgs) == 1

    def test_clear_memories_warns_once(self, caplog):
        sdk = _make_sdk()
        db = SynapDb(sdk)
        with caplog.at_level(logging.WARNING, logger="synap_agno.db"):
            db.clear_memories()
            db.clear_memories()
        warn_msgs = [r for r in caplog.records if "no public delete" in r.message]
        assert len(warn_msgs) == 1

    def test_warn_deduplication_across_all_delete_variants(self, caplog):
        """All three delete variants share the same warn-once flag."""
        sdk = _make_sdk()
        db = SynapDb(sdk)
        with caplog.at_level(logging.WARNING, logger="synap_agno.db"):
            db.delete_user_memory("m1")          # triggers warn
            db.delete_user_memories(["m2"])       # suppressed
            db.clear_memories()                   # suppressed
        warn_msgs = [r for r in caplog.records if "no public delete" in r.message]
        assert len(warn_msgs) == 1

    def test_sets_delete_warned_flag(self):
        sdk = _make_sdk()
        db = SynapDb(sdk)
        assert db._delete_warned is False
        db.delete_user_memory("mem-001")
        assert db._delete_warned is True

    def test_delete_user_memory_does_not_call_sdk(self):
        """Deletes are no-ops — SDK must not be called."""
        sdk = _make_sdk()
        db = SynapDb(sdk)
        db.delete_user_memory("mem-001")
        sdk.memories.create.assert_not_awaited()
        sdk.fetch.assert_not_awaited()

    def test_delete_user_memories_does_not_call_sdk(self):
        sdk = _make_sdk()
        db = SynapDb(sdk)
        db.delete_user_memories(["mem-001", "mem-002"])
        sdk.memories.create.assert_not_awaited()
        sdk.fetch.assert_not_awaited()

    def test_clear_memories_does_not_call_sdk(self):
        sdk = _make_sdk()
        db = SynapDb(sdk)
        db.clear_memories()
        sdk.memories.create.assert_not_awaited()
        sdk.fetch.assert_not_awaited()


# ---------------------------------------------------------------------------
# Public surface — __init__.py exports
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_synapdb_exported(self):
        import synap_agno
        assert hasattr(synap_agno, "SynapDb")
        assert "SynapDb" in synap_agno.__all__

    def test_synap_st_instructions_exported(self):
        import synap_agno
        assert hasattr(synap_agno, "synap_st_instructions")
        assert "synap_st_instructions" in synap_agno.__all__

    def test_no_extra_public_exports(self):
        """__all__ should contain exactly the two documented exports."""
        import synap_agno
        assert set(synap_agno.__all__) == {"SynapDb", "synap_st_instructions"}
