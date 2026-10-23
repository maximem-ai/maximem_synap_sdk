"""SynapDb read path, against real SDK response objects.

Synap returns preferences, episodes, emotions and dated events as well as facts, and it does
not echo back the custom metadata written with ``memories.create``. The old read path only
looked at facts and dropped any hit without an Agno marker in its metadata, so it could never
return anything a real Synap produced.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from synap_agno import SynapDb
from synap_agno.db import DEFAULT_QUERY
from synap_integrations_common.testing import (
    make_emotion, make_episode, make_fact, make_preference, make_temporal_event, make_unified_response,
)


def _db(sdk, **kw):
    return SynapDb(sdk, **kw)


def _all_types():
    return make_unified_response(
        facts=[make_fact("f1", "Ravi works at Acme", 0.9)],
        preferences=[make_preference("p1", "Ravi prefers window seats", 0.8)],
        episodes=[make_episode("e1", "Ravi flew to Delhi", 0.7)],
        emotions=[make_emotion("m1", "joy", "after the launch", 0.6)],
        temporal_events=[make_temporal_event("t1", "Trip on 12 March")],
    )


def test_every_memory_type_comes_back_as_a_user_memory(mock_sdk):
    mock_sdk.fetch.return_value = _all_types()
    memories = _db(mock_sdk).get_user_memories(user_id="ravi", search_content="travel")
    assert sorted(m.memory for m in memories) == sorted([
        "Ravi works at Acme", "Ravi prefers window seats", "Ravi flew to Delhi",
        "joy: after the launch", "Trip on 12 March",
    ])
    assert {m.memory_id for m in memories} == {"f1", "p1", "e1", "m1", "t1"}
    assert all(m.user_id == "ravi" for m in memories)


def test_hits_need_no_agno_marker_in_metadata(mock_sdk):
    """Synap does not echo back custom metadata, so nothing may be dropped for lacking it."""
    fact = make_fact("f1", "Ravi works at Acme")
    assert not getattr(fact, "metadata", None)
    mock_sdk.fetch.return_value = make_unified_response(facts=[fact], preferences=[])
    assert [m.memory for m in _db(mock_sdk).get_user_memories(user_id="ravi", search_content="x")] == ["Ravi works at Acme"]


def test_timestamps_are_filled_from_the_hit(mock_sdk):
    mock_sdk.fetch.return_value = make_unified_response(facts=[make_fact()], preferences=[])
    (memory,) = _db(mock_sdk).get_user_memories(user_id="ravi", search_content="x")
    assert isinstance(memory.created_at, int) and memory.created_at > 1_600_000_000


def test_no_search_term_asks_the_default_question_not_none(mock_sdk):
    """Synap returns nothing for a fetch with no query."""
    _db(mock_sdk).get_user_memories(user_id="ravi")
    assert mock_sdk.fetch.await_args.kwargs["search_query"] == [DEFAULT_QUERY]


def test_a_custom_default_question_is_used(mock_sdk):
    _db(mock_sdk, default_query="Ravi's travel habits").get_user_memories(user_id="ravi")
    assert mock_sdk.fetch.await_args.kwargs["search_query"] == ["Ravi's travel habits"]


def test_an_empty_default_question_is_rejected(mock_sdk):
    with pytest.raises(ValueError, match="default_query"):
        _db(mock_sdk, default_query="  ")


def test_search_content_is_passed_through(mock_sdk):
    _db(mock_sdk).get_user_memories(user_id="ravi", search_content="window seats")
    assert mock_sdk.fetch.await_args.kwargs["search_query"] == ["window seats"]
    assert mock_sdk.fetch.await_args.kwargs["user_id"] == "ravi"


def test_customer_id_is_forwarded_only_when_set(mock_sdk):
    _db(mock_sdk).get_user_memories(user_id="ravi")
    assert mock_sdk.fetch.await_args.kwargs["customer_id"] is None
    _db(mock_sdk, customer_id="acme").get_user_memories(user_id="ravi")
    assert mock_sdk.fetch.await_args.kwargs["customer_id"] == "acme"


def test_agent_team_and_topic_filters_apply_only_when_the_hit_carries_the_field(mock_sdk):
    tagged = make_fact("f1", "tagged for another agent")
    tagged.metadata = {"agent_id": "other", "topics": ["food"]}
    untagged = make_fact("f2", "no metadata at all")
    mock_sdk.fetch.return_value = make_unified_response(facts=[tagged, untagged], preferences=[])
    db = _db(mock_sdk)
    assert [m.memory for m in db.get_user_memories(user_id="r", search_content="x", agent_id="mine")] == ["no metadata at all"]
    assert [m.memory for m in db.get_user_memories(user_id="r", search_content="x", topics=["travel"])] == ["no metadata at all"]
    assert len(db.get_user_memories(user_id="r", search_content="x", topics=["food"])) == 2


def test_sort_and_pagination(mock_sdk):
    old, new = make_fact("f1", "old"), make_fact("f2", "new")
    old.extracted_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    new.extracted_at = datetime(2026, 6, 1, tzinfo=timezone.utc)
    mock_sdk.fetch.return_value = make_unified_response(facts=[old, new], preferences=[])
    db = _db(mock_sdk)
    assert [m.memory for m in db.get_user_memories(user_id="r", search_content="x", sort_by="created_at", sort_order="desc")] == ["new", "old"]
    assert [m.memory for m in db.get_user_memories(user_id="r", search_content="x", sort_by="created_at", sort_order="asc", limit=1)] == ["old"]
    assert [m.memory for m in db.get_user_memories(user_id="r", search_content="x", sort_by="created_at", sort_order="asc", limit=1, page=1)] == ["new"]


def test_deserialize_false_returns_dicts_and_a_count(mock_sdk):
    mock_sdk.fetch.return_value = _all_types()
    rows, count = _db(mock_sdk).get_user_memories(user_id="r", search_content="x", deserialize=False)
    assert count == 5 and all(isinstance(r, dict) for r in rows)


def test_a_synap_read_failure_degrades_to_empty_and_logs(failing_sdk, caplog):
    with caplog.at_level(logging.ERROR):
        assert _db(failing_sdk).get_user_memories(user_id="r", search_content="x") == []
    assert "sdk.fetch failed" in caplog.text


def test_topics_come_back_only_when_hits_carry_them(mock_sdk):
    tagged = make_fact("f1", "x"); tagged.metadata = {"topics": ["travel", "food", 7]}
    mock_sdk.fetch.return_value = make_unified_response(facts=[tagged, make_fact("f2", "y")], preferences=[])
    assert _db(mock_sdk).get_all_memory_topics(user_id="r") == ["food", "travel"]
    mock_sdk.fetch.return_value = make_unified_response(facts=[make_fact()], preferences=[])
    assert _db(mock_sdk).get_all_memory_topics(user_id="r") == []


def test_get_user_memory_looks_up_a_synap_id(mock_sdk):
    memory_id = uuid4()
    found = type("M", (), {"content": "Ravi prefers window seats", "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc), "updated_at": None})()
    mock_sdk.memories.get = AsyncMock(return_value=found)
    memory = _db(mock_sdk).get_user_memory(str(memory_id), user_id="ravi")
    assert memory.memory == "Ravi prefers window seats" and memory.memory_id == str(memory_id)
    assert mock_sdk.memories.get.await_args.args[0] == memory_id


def test_get_user_memory_with_a_non_synap_id_is_none_without_a_call(mock_sdk):
    mock_sdk.memories.get = AsyncMock()
    assert _db(mock_sdk).get_user_memory("an-id-agno-minted", user_id="ravi") is None
    mock_sdk.memories.get.assert_not_awaited()


def test_get_user_memory_failure_is_none(mock_sdk):
    mock_sdk.memories.get = AsyncMock(side_effect=RuntimeError("boom"))
    assert _db(mock_sdk).get_user_memory(str(uuid4()), user_id="ravi") is None


def test_sync_reads_work_from_a_background_thread(mock_sdk):
    mock_sdk.fetch.return_value = _all_types()
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("n", len(_db(mock_sdk).get_user_memories(user_id="r", search_content="x"))))
    t.start(); t.join(30)
    assert out["n"] == 5
