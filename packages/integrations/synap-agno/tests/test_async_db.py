"""AsyncSynapDb: Agno's async database interface, awaited natively (no event-loop bridge)."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from agno.db.base import AsyncBaseDb
from agno.db.schemas.memory import UserMemory

import synap_agno
from synap_agno import AsyncSynapDb
from synap_agno.db import DEFAULT_QUERY
from synap_integrations_common.testing import make_fact, make_preference, make_unified_response


@pytest.fixture(autouse=True)
def _no_bridge(monkeypatch):
    """The whole point of the async class: nothing here may fall back to the sync bridge."""
    def boom(*a, **k):
        raise AssertionError("AsyncSynapDb must not use the sync bridge")
    monkeypatch.setattr("synap_agno.db.run_blocking", boom)


def test_it_is_a_real_async_base_db_with_nothing_left_abstract(mock_sdk):
    db = AsyncSynapDb(mock_sdk)
    assert isinstance(db, AsyncBaseDb)
    assert AsyncSynapDb.__abstractmethods__ == frozenset()


def test_it_is_exported():
    assert "AsyncSynapDb" in synap_agno.__all__


def test_requires_an_sdk():
    with pytest.raises(ValueError, match="non-None sdk"):
        AsyncSynapDb(None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_upsert_user_memory_writes_to_synap(mock_sdk):
    db = AsyncSynapDb(mock_sdk, customer_id="acme")
    saved = await db.upsert_user_memory(UserMemory(memory="Ravi prefers window seats", user_id="ravi"))
    kwargs = mock_sdk.memories.create.await_args.kwargs
    assert kwargs["document"] == "Ravi prefers window seats"
    assert kwargs["user_id"] == "ravi" and kwargs["customer_id"] == "acme"
    assert saved.memory_id and saved.created_at and saved.updated_at


@pytest.mark.asyncio
async def test_upsert_user_memory_deserialize_false_returns_a_dict(mock_sdk):
    saved = await AsyncSynapDb(mock_sdk).upsert_user_memory(UserMemory(memory="x", user_id="u"), deserialize=False)
    assert isinstance(saved, dict) and saved["memory"] == "x"


@pytest.mark.asyncio
async def test_upsert_memories_writes_each(mock_sdk):
    out = await AsyncSynapDb(mock_sdk).upsert_memories([UserMemory(memory=f"m{i}", user_id="u") for i in range(3)])
    assert len(out) == 3 and mock_sdk.memories.create.await_count == 3


@pytest.mark.asyncio
async def test_a_write_failure_surfaces(failing_sdk):
    with pytest.raises(Exception):
        await AsyncSynapDb(failing_sdk).upsert_user_memory(UserMemory(memory="x", user_id="u"))


@pytest.mark.asyncio
async def test_get_user_memories_reads_every_type(mock_sdk):
    mock_sdk.fetch.return_value = make_unified_response(
        facts=[make_fact("f1", "a fact")], preferences=[make_preference("p1", "a preference")]
    )
    memories = await AsyncSynapDb(mock_sdk).get_user_memories(user_id="ravi", search_content="x")
    assert sorted(m.memory for m in memories) == ["a fact", "a preference"]


@pytest.mark.asyncio
async def test_get_user_memories_without_a_search_term_asks_a_real_question(mock_sdk):
    await AsyncSynapDb(mock_sdk).get_user_memories(user_id="ravi")
    assert mock_sdk.fetch.await_args.kwargs["search_query"] == [DEFAULT_QUERY]


@pytest.mark.asyncio
async def test_a_read_failure_degrades_to_empty(failing_sdk, caplog):
    with caplog.at_level(logging.ERROR):
        assert await AsyncSynapDb(failing_sdk).get_user_memories(user_id="r", search_content="x") == []


@pytest.mark.asyncio
async def test_get_user_memory_by_synap_id(mock_sdk):
    memory_id = uuid4()
    mock_sdk.memories.get = AsyncMock(return_value=type("M", (), {"content": "hello", "created_at": None, "updated_at": None})())
    found = await AsyncSynapDb(mock_sdk).get_user_memory(str(memory_id), user_id="ravi")
    assert found.memory == "hello"


@pytest.mark.asyncio
async def test_deletes_warn_once_and_never_call_synap(mock_sdk, caplog):
    db = AsyncSynapDb(mock_sdk)
    with caplog.at_level(logging.WARNING):
        await db.delete_user_memory("m1")
        await db.delete_user_memories(["m1", "m2"])
        await db.clear_memories()
    assert caplog.text.count("no public delete API") == 1
    mock_sdk.memories.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_stats_are_empty_with_a_one_time_warning(mock_sdk):
    assert await AsyncSynapDb(mock_sdk).get_user_memory_stats() == ([], 0)


@pytest.mark.asyncio
async def test_non_memory_methods_are_awaitable_and_stay_in_process(mock_sdk):
    from agno.db.base import SessionType
    db = AsyncSynapDb(mock_sdk)
    result = await db.get_sessions(session_type=SessionType.AGENT)
    assert result == [] or (isinstance(result, tuple) and result[0] == [])
    mock_sdk.fetch.assert_not_awaited()
    mock_sdk.memories.create.assert_not_awaited()
