"""Tests for SynapMemoryStore (Haystack)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from haystack.dataclasses import ChatMessage

from synap_haystack import SynapMemoryStore
from synap_integrations_common import SynapIntegrationError


@pytest.fixture
def mock_sdk():
    sdk = MagicMock()
    sdk.instance_id = "test-instance"
    sdk.fetch = AsyncMock()
    sdk.conversation.record_message = AsyncMock(return_value={"message_id": "m1"})
    return sdk


@pytest.fixture
def store(mock_sdk):
    return SynapMemoryStore(
        mock_sdk, user_id="u1", customer_id="c1", conversation_id="conv-1"
    )


def test_requires_scope(mock_sdk):
    with pytest.raises(ValueError):
        SynapMemoryStore(mock_sdk)


def test_add_memories_records_user_and_assistant(store, mock_sdk):
    results = store.add_memories(messages=[
        ChatMessage.from_user("hello"),
        ChatMessage.from_assistant("hi there"),
    ])
    assert mock_sdk.conversation.record_message.await_count == 2
    assert [r["status"] for r in results] == ["written", "written"]
    assert results[0]["message_id"] == "m1"


def test_add_memories_skips_system_role(store, mock_sdk):
    results = store.add_memories(messages=[ChatMessage.from_system("be nice")])
    assert mock_sdk.conversation.record_message.await_count == 0
    assert results[0]["status"] == "skipped"


def test_add_memories_requires_conversation_id(mock_sdk):
    store = SynapMemoryStore(mock_sdk, user_id="u1")
    with pytest.raises(ValueError):
        store.add_memories(messages=[ChatMessage.from_user("hello")])


def test_add_memories_raises_on_total_failure(store, mock_sdk):
    mock_sdk.conversation.record_message.side_effect = RuntimeError("boom")
    with pytest.raises(SynapIntegrationError):
        store.add_memories(messages=[ChatMessage.from_user("hello")])


def test_search_memories_returns_chat_messages(store, mock_sdk):
    fact = MagicMock(content="likes coffee", id="f1", confidence=0.9)
    mock_sdk.fetch.return_value = MagicMock(
        facts=[fact], preferences=[], episodes=[],
        emotions=[], temporal_events=[], scope_map={"f1": "user"},
    )
    msgs = store.search_memories(query="coffee")
    assert len(msgs) == 1
    assert isinstance(msgs[0], ChatMessage)
    assert msgs[0].text == "likes coffee"
    assert msgs[0].meta["type"] == "fact"
    assert msgs[0].role.value == "assistant"


def test_search_degrades_to_empty_on_error(store, mock_sdk):
    mock_sdk.fetch.side_effect = RuntimeError("down")
    assert store.search_memories(query="x") == []


def test_search_as_single_message(store, mock_sdk):
    fact = MagicMock(content="likes coffee", id="f1", confidence=0.9)
    pref = MagicMock(content="prefers dark mode", id="p1", strength=0.8)
    mock_sdk.fetch.return_value = MagicMock(
        facts=[fact], preferences=[pref], episodes=[],
        emotions=[], temporal_events=[], scope_map={},
    )
    msg = store.search_memories_as_single_message(query="x")
    assert msg.role.value == "system"
    assert "likes coffee" in msg.text
    assert "prefers dark mode" in msg.text


def test_search_as_single_message_none_when_empty(store, mock_sdk):
    mock_sdk.fetch.return_value = MagicMock(
        facts=[], preferences=[], episodes=[], emotions=[],
        temporal_events=[], scope_map={},
    )
    assert store.search_memories_as_single_message(query="x") is None


def test_delete_is_noop(store, mock_sdk):
    store.delete_memory("m1")
    store.delete_all_memories()  # warns once, does not raise


def test_to_dict_omits_sdk(store):
    d = store.to_dict()
    params = d["init_parameters"]
    assert params["instance_id"] == "test-instance"
    assert params["user_id"] == "u1"
    assert params["conversation_id"] == "conv-1"
    assert "sdk" not in params
