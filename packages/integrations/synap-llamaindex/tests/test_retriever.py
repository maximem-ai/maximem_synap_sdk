"""Tests for SynapRetriever (LlamaIndex).

Documented error-handling contract (from retriever.py):
- _aretrieve: wraps SDK errors as SynapIntegrationError and re-raises via
  wrap_sdk_errors_async. Callers decide how to handle; never silently discards.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from llama_index.core.schema import QueryBundle, TextNode, NodeWithScore

from synap_llamaindex.retriever import SynapRetriever
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sdk():
    sdk = MagicMock()
    sdk.fetch = AsyncMock()
    return sdk


@pytest.fixture
def retriever(mock_sdk):
    return SynapRetriever(
        sdk=mock_sdk, user_id="user-1", customer_id="cust-1",
        mode="accurate", max_results=20, types=None, conversation_id=None,
    )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_surface_exports():
    import synap_llamaindex
    assert hasattr(synap_llamaindex, "SynapRetriever")
    assert "SynapRetriever" in synap_llamaindex.__all__


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


def test_init_raises_on_none_sdk():
    with pytest.raises(ValueError, match="non-None sdk"):
        SynapRetriever(sdk=None, user_id="u")


def test_init_raises_on_empty_user_id():
    sdk = MagicMock()
    with pytest.raises(ValueError, match="non-empty user_id"):
        SynapRetriever(sdk=sdk, user_id="")


def test_init_stores_params(mock_sdk):
    r = SynapRetriever(
        sdk=mock_sdk, user_id="u1", customer_id="c1",
        conversation_id="conv-1", mode="fast", max_results=5,
        types=["fact"],
    )
    assert r._user_id == "u1"
    assert r._customer_id == "c1"
    assert r._conversation_id == "conv-1"
    assert r._mode == "fast"
    assert r._max_results == 5
    assert r._types == ["fact"]


# ---------------------------------------------------------------------------
# Happy path — all memory item types
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aretrieve_fact_returns_node_with_correct_metadata(retriever, mock_sdk):
    """Facts are mapped to NodeWithScore with correct text, score, and metadata."""
    mock_fact = MagicMock(
        content="likes tea", id="f1", confidence=0.85,
        source="chat", temporal_category="permanent",
    )
    mock_sdk.fetch.return_value = MagicMock(
        facts=[mock_fact], preferences=[], episodes=[],
        emotions=[], temporal_events=[], scope_map={"f1": "user"},
    )

    nodes = await retriever._aretrieve(QueryBundle(query_str="tea"))

    assert len(nodes) == 1
    n = nodes[0]
    assert n.node.text == "likes tea"
    assert n.node.id_ == "f1"
    assert n.score == 0.85
    assert n.node.metadata["type"] == "fact"
    assert n.node.metadata["confidence"] == 0.85
    assert n.node.metadata["source"] == "chat"
    assert n.node.metadata["scope"] == "user"
    assert n.node.metadata["temporal_category"] == "permanent"


@pytest.mark.asyncio
async def test_aretrieve_preference_returns_node_with_correct_metadata(retriever, mock_sdk):
    """Preferences are mapped with strength as score and category in metadata."""
    mock_pref = MagicMock(
        content="dark mode", id="p1", strength=0.8, category="ui",
    )
    mock_sdk.fetch.return_value = MagicMock(
        facts=[], preferences=[mock_pref], episodes=[],
        emotions=[], temporal_events=[], scope_map={"p1": "user"},
    )

    nodes = await retriever._aretrieve(QueryBundle(query_str="theme"))

    assert len(nodes) == 1
    n = nodes[0]
    assert n.node.text == "dark mode"
    assert n.node.id_ == "p1"
    assert n.score == 0.8
    assert n.node.metadata["type"] == "preference"
    assert n.node.metadata["strength"] == 0.8
    assert n.node.metadata["category"] == "ui"
    assert n.node.metadata["scope"] == "user"


@pytest.mark.asyncio
async def test_aretrieve_episode_returns_node_with_summary_and_significance(retriever, mock_sdk):
    """Episodes use summary as text and significance as score."""
    mock_ep = MagicMock(
        summary="Had a support call", id="e1", significance=0.7,
        occurred_at="2024-01-15T10:00:00Z",
    )
    mock_sdk.fetch.return_value = MagicMock(
        facts=[], preferences=[], episodes=[mock_ep],
        emotions=[], temporal_events=[], scope_map={"e1": "user"},
    )

    nodes = await retriever._aretrieve(QueryBundle(query_str="support"))

    assert len(nodes) == 1
    n = nodes[0]
    assert n.node.text == "Had a support call"
    assert n.node.id_ == "e1"
    assert n.score == 0.7
    assert n.node.metadata["type"] == "episode"
    assert n.node.metadata["significance"] == 0.7
    assert n.node.metadata["scope"] == "user"
    assert "occurred_at" in n.node.metadata


@pytest.mark.asyncio
async def test_aretrieve_emotion_returns_node_with_type_context_format(retriever, mock_sdk):
    """Emotions use 'emotion_type: context' as text and intensity as score."""
    mock_em = MagicMock(
        emotion_type="frustrated", context="Long support wait",
        id="em1", intensity=0.7,
    )
    mock_sdk.fetch.return_value = MagicMock(
        facts=[], preferences=[], episodes=[],
        emotions=[mock_em], temporal_events=[], scope_map={"em1": "user"},
    )

    nodes = await retriever._aretrieve(QueryBundle(query_str="emotion"))

    assert len(nodes) == 1
    n = nodes[0]
    assert n.node.text == "frustrated: Long support wait"
    assert n.node.id_ == "em1"
    assert n.score == 0.7
    assert n.node.metadata["type"] == "emotion"
    assert n.node.metadata["emotion_type"] == "frustrated"
    assert n.node.metadata["intensity"] == 0.7
    assert n.node.metadata["scope"] == "user"


@pytest.mark.asyncio
async def test_aretrieve_temporal_event_returns_node_with_correct_metadata(retriever, mock_sdk):
    """Temporal events use temporal_confidence as score and include event_date."""
    mock_te = MagicMock(
        content="Trial expires April 15", id="t1",
        event_date="2024-04-15T00:00:00Z",
        valid_until=None, temporal_confidence=0.9,
    )
    mock_sdk.fetch.return_value = MagicMock(
        facts=[], preferences=[], episodes=[], emotions=[],
        temporal_events=[mock_te], scope_map={"t1": "user"},
    )

    nodes = await retriever._aretrieve(QueryBundle(query_str="trial"))

    assert len(nodes) == 1
    n = nodes[0]
    assert n.node.text == "Trial expires April 15"
    assert n.node.id_ == "t1"
    assert n.score == 0.9
    assert n.node.metadata["type"] == "temporal_event"
    assert "event_date" in n.node.metadata
    assert n.node.metadata["valid_until"] is None
    assert n.node.metadata["scope"] == "user"


@pytest.mark.asyncio
async def test_aretrieve_all_five_item_types_in_single_response(retriever, mock_sdk):
    """All five memory item types can appear in one response; all become nodes."""
    mock_sdk.fetch.return_value = MagicMock(
        facts=[MagicMock(content="f", id="f1", confidence=0.9, source="s", temporal_category="p")],
        preferences=[MagicMock(content="p", id="p1", strength=0.8, category="c")],
        episodes=[MagicMock(summary="e", id="e1", significance=0.7, occurred_at="2024-01-01")],
        emotions=[MagicMock(emotion_type="calm", context="ctx", id="em1", intensity=0.5)],
        temporal_events=[MagicMock(content="t", id="t1", event_date="2024-04-01",
                                    valid_until=None, temporal_confidence=0.6)],
        scope_map={"f1": "user", "p1": "user", "e1": "user", "em1": "user", "t1": "user"},
    )

    nodes = await retriever._aretrieve(QueryBundle(query_str="everything"))

    assert len(nodes) == 5
    types = {n.node.metadata["type"] for n in nodes}
    assert types == {"fact", "preference", "episode", "emotion", "temporal_event"}


@pytest.mark.asyncio
async def test_aretrieve_empty_response_returns_empty_list(retriever, mock_sdk):
    """Empty response → empty NodeWithScore list (no crash)."""
    mock_sdk.fetch.return_value = MagicMock(
        facts=[], preferences=[], episodes=[], emotions=[], temporal_events=[],
        scope_map={},
    )

    nodes = await retriever._aretrieve(QueryBundle(query_str="nothing"))

    assert nodes == []


@pytest.mark.asyncio
async def test_aretrieve_forwards_correct_fetch_kwargs(retriever, mock_sdk):
    """The SDK fetch call receives the right parameters."""
    mock_sdk.fetch.return_value = MagicMock(
        facts=[], preferences=[], episodes=[], emotions=[], temporal_events=[],
        scope_map={},
    )

    await retriever._aretrieve(QueryBundle(query_str="coffee preferences"))

    mock_sdk.fetch.assert_awaited_once_with(
        conversation_id=None,
        user_id="user-1",
        customer_id="cust-1",
        search_query=["coffee preferences"],
        max_results=20,
        types=None,
        mode="accurate",
        include_conversation_context=False,
    )


@pytest.mark.asyncio
async def test_aretrieve_passes_custom_params():
    """Custom mode, max_results, types, conversation_id are forwarded."""
    sdk = MagicMock()
    sdk.fetch = AsyncMock(return_value=MagicMock(
        facts=[], preferences=[], episodes=[], emotions=[], temporal_events=[],
        scope_map={},
    ))
    r = SynapRetriever(
        sdk=sdk, user_id="u", customer_id="c",
        conversation_id="conv-xyz", mode="fast", max_results=5,
        types=["fact", "preference"],
    )

    await r._aretrieve(QueryBundle(query_str="q"))

    sdk.fetch.assert_awaited_once_with(
        conversation_id="conv-xyz",
        user_id="u",
        customer_id="c",
        search_query=["q"],
        max_results=5,
        types=["fact", "preference"],
        mode="fast",
        include_conversation_context=False,
    )


@pytest.mark.asyncio
async def test_aretrieve_item_missing_from_scope_map_gets_empty_scope(retriever, mock_sdk):
    """Items absent from scope_map receive an empty string for 'scope'."""
    mock_fact = MagicMock(
        content="unknown scope fact", id="f99", confidence=0.5,
        source="s", temporal_category=None,
    )
    mock_sdk.fetch.return_value = MagicMock(
        facts=[mock_fact], preferences=[], episodes=[], emotions=[], temporal_events=[],
        scope_map={},  # f99 absent
    )

    nodes = await retriever._aretrieve(QueryBundle(query_str="q"))

    assert nodes[0].node.metadata["scope"] == ""


@pytest.mark.asyncio
async def test_aretrieve_nodes_sorted_by_score_descending(retriever, mock_sdk):
    """Nodes are returned sorted by score descending (highest confidence first)."""
    mock_sdk.fetch.return_value = MagicMock(
        facts=[
            MagicMock(content="low", id="f1", confidence=0.3, source="s", temporal_category=None),
            MagicMock(content="high", id="f2", confidence=0.9, source="s", temporal_category=None),
            MagicMock(content="mid", id="f3", confidence=0.6, source="s", temporal_category=None),
        ],
        preferences=[], episodes=[], emotions=[], temporal_events=[],
        scope_map={},
    )

    nodes = await retriever._aretrieve(QueryBundle(query_str="sort test"))

    scores = [n.score for n in nodes]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 0.9
    assert scores[-1] == 0.3


# ---------------------------------------------------------------------------
# Failure path — SDK raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aretrieve_raises_synap_integration_error_on_sdk_failure(retriever, mock_sdk):
    """SDK failure inside _aretrieve is wrapped as SynapIntegrationError.

    wrap_sdk_errors_async is used in retriever.py — any SDK error is surfaced
    as SynapIntegrationError so callers have a typed exception to handle.
    """
    mock_sdk.fetch.side_effect = RuntimeError("sdk boom")

    with pytest.raises(SynapIntegrationError):
        await retriever._aretrieve(QueryBundle(query_str="query"))


@pytest.mark.asyncio
async def test_aretrieve_error_preserves_original_cause(retriever, mock_sdk):
    """The SynapIntegrationError must chain the original SDK exception as __cause__."""
    original = RuntimeError("original sdk error")
    mock_sdk.fetch.side_effect = original

    with pytest.raises(SynapIntegrationError) as exc_info:
        await retriever._aretrieve(QueryBundle(query_str="q"))

    assert exc_info.value.__cause__ is original


@pytest.mark.asyncio
async def test_retrieve_sync_raises_synap_integration_error_on_sdk_failure(retriever, mock_sdk):
    """The sync _retrieve wrapper also surfaces SynapIntegrationError on failure."""
    mock_sdk.fetch.side_effect = RuntimeError("sdk boom")

    with pytest.raises(SynapIntegrationError):
        retriever._retrieve(QueryBundle(query_str="query"))


# ---------------------------------------------------------------------------
# Using shared failing_sdk fixture
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aretrieve_with_failing_sdk_raises(failing_sdk):
    """failing_sdk fixture confirms integration always surfaces errors from a broken SDK."""
    r = SynapRetriever(sdk=failing_sdk, user_id="u1")

    with pytest.raises(SynapIntegrationError):
        await r._aretrieve(QueryBundle(query_str="q"))
