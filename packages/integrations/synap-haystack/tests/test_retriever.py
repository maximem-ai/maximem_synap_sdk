"""Tests for SynapRetriever (Haystack)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from haystack.dataclasses import ChatMessage

from synap_haystack.retriever import SynapMemoryRetriever, SynapRetriever


@pytest.fixture
def mock_sdk():
    sdk = MagicMock()
    sdk.instance_id = "test"
    sdk.fetch = AsyncMock()
    return sdk


@pytest.fixture
def retriever(mock_sdk):
    return SynapRetriever(sdk=mock_sdk, user_id="u1")


def test_import():
    from synap_haystack import (
        SynapMemoryRetriever,
        SynapMemoryStore,
        SynapMemoryWriter,
        SynapRetriever,
    )
    assert SynapRetriever is not None
    assert SynapMemoryRetriever is not None
    assert SynapMemoryStore is not None
    assert SynapMemoryWriter is not None


def test_memory_retriever_returns_messages(mock_sdk):
    mock_fact = MagicMock(content="likes tea", id="f1", confidence=0.8)
    mock_sdk.fetch.return_value = MagicMock(
        facts=[mock_fact], preferences=[], episodes=[],
        emotions=[], temporal_events=[], scope_map={"f1": "user"},
    )
    retriever = SynapMemoryRetriever(sdk=mock_sdk, user_id="u1")
    result = retriever.run(query="tea")
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], ChatMessage)
    assert result["messages"][0].text == "likes tea"


def test_run_calls_fetch(retriever, mock_sdk):
    mock_fact = MagicMock(
        content="likes coffee", id="f1", confidence=0.9,
        source="chat", valid_until=None, temporal_category=None,
    )
    mock_sdk.fetch.return_value = MagicMock(
        facts=[mock_fact], preferences=[], episodes=[],
        emotions=[], temporal_events=[], scope_map={"f1": "user"},
    )

    result = retriever.run(query="coffee")

    assert "documents" in result
    assert len(result["documents"]) == 1
    assert result["documents"][0].content == "likes coffee"
    mock_sdk.fetch.assert_awaited_once()
