"""Tests for SynapRetriever.

Documented error-handling contract (from retriever.py / wrap_sdk_errors_async):
- _aget_relevant_documents wraps SDK errors as SynapIntegrationError and re-raises.
- Callers decide how to handle; the integration never silently discards failures.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.documents import Document

from synap_langchain.retriever import SynapRetriever
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sdk():
    from maximem_synap import MaximemSynapSDK
    sdk = MagicMock(spec=MaximemSynapSDK)
    sdk.fetch = AsyncMock()
    return sdk


@pytest.fixture
def retriever(mock_sdk):
    return SynapRetriever.model_construct(
        sdk=mock_sdk,
        user_id="user-1",
        customer_id="cust-1",
        mode="accurate",
        max_results=20,
        types=None,
        conversation_id=None,
    )


# ---------------------------------------------------------------------------
# Happy path — all memory item types
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retriever_returns_fact_and_preference_documents(retriever, mock_sdk):
    """Facts and Preferences are mapped to Documents with correct metadata."""
    mock_fact = MagicMock(
        content="likes coffee", id="f1", confidence=0.9,
        source="chat", valid_until=None, temporal_category=None,
    )
    mock_pref = MagicMock(
        content="dark mode", id="p1", strength=0.8, category="ui",
    )
    mock_response = MagicMock(
        facts=[mock_fact],
        preferences=[mock_pref],
        episodes=[],
        emotions=[],
        temporal_events=[],
        scope_map={"f1": "user", "p1": "user"},
    )
    mock_sdk.fetch.return_value = mock_response

    run_manager = MagicMock()
    docs = await retriever._aget_relevant_documents("coffee", run_manager=run_manager)

    assert len(docs) == 2

    fact_doc = docs[0]
    assert fact_doc.page_content == "likes coffee"
    assert fact_doc.metadata["type"] == "fact"
    assert fact_doc.metadata["id"] == "f1"
    assert fact_doc.metadata["confidence"] == 0.9
    assert fact_doc.metadata["scope"] == "user"

    pref_doc = docs[1]
    assert pref_doc.page_content == "dark mode"
    assert pref_doc.metadata["type"] == "preference"
    assert pref_doc.metadata["id"] == "p1"
    assert pref_doc.metadata["strength"] == 0.8
    assert pref_doc.metadata["category"] == "ui"


@pytest.mark.asyncio
async def test_retriever_maps_episodes_to_documents(retriever, mock_sdk):
    """Episodes are mapped to Documents with summary as page_content."""
    mock_ep = MagicMock(
        summary="Had a support call about billing",
        id="e1",
        significance=0.75,
        occurred_at="2024-01-15T10:00:00Z",
    )
    mock_response = MagicMock(
        facts=[],
        preferences=[],
        episodes=[mock_ep],
        emotions=[],
        temporal_events=[],
        scope_map={"e1": "user"},
    )
    mock_sdk.fetch.return_value = mock_response

    run_manager = MagicMock()
    docs = await retriever._aget_relevant_documents("support call", run_manager=run_manager)

    assert len(docs) == 1
    ep_doc = docs[0]
    assert ep_doc.page_content == "Had a support call about billing"
    assert ep_doc.metadata["type"] == "episode"
    assert ep_doc.metadata["id"] == "e1"
    assert ep_doc.metadata["significance"] == 0.75
    assert ep_doc.metadata["scope"] == "user"
    assert "occurred_at" in ep_doc.metadata


@pytest.mark.asyncio
async def test_retriever_maps_emotions_to_documents(retriever, mock_sdk):
    """Emotions are mapped to Documents with 'emotion_type: context' format."""
    mock_em = MagicMock(
        emotion_type="frustrated",
        context="Long support wait",
        id="em1",
        intensity=0.7,
        detected_at="2024-01-15T10:00:00Z",
    )
    mock_response = MagicMock(
        facts=[],
        preferences=[],
        episodes=[],
        emotions=[mock_em],
        temporal_events=[],
        scope_map={"em1": "user"},
    )
    mock_sdk.fetch.return_value = mock_response

    run_manager = MagicMock()
    docs = await retriever._aget_relevant_documents("emotions", run_manager=run_manager)

    assert len(docs) == 1
    em_doc = docs[0]
    assert em_doc.page_content == "frustrated: Long support wait"
    assert em_doc.metadata["type"] == "emotion"
    assert em_doc.metadata["id"] == "em1"
    assert em_doc.metadata["intensity"] == 0.7
    assert em_doc.metadata["emotion_type"] == "frustrated"


@pytest.mark.asyncio
async def test_retriever_maps_temporal_events_to_documents(retriever, mock_sdk):
    """Temporal events are mapped to Documents with correct metadata."""
    mock_te = MagicMock(
        content="Trial expires April 15",
        id="t1",
        event_date="2024-04-15T00:00:00Z",
        valid_until=None,
        temporal_category="temporal_fact",
    )
    mock_response = MagicMock(
        facts=[],
        preferences=[],
        episodes=[],
        emotions=[],
        temporal_events=[mock_te],
        scope_map={"t1": "user"},
    )
    mock_sdk.fetch.return_value = mock_response

    run_manager = MagicMock()
    docs = await retriever._aget_relevant_documents("trial", run_manager=run_manager)

    assert len(docs) == 1
    te_doc = docs[0]
    assert te_doc.page_content == "Trial expires April 15"
    assert te_doc.metadata["type"] == "temporal_event"
    assert te_doc.metadata["id"] == "t1"
    assert te_doc.metadata["temporal_category"] == "temporal_fact"


@pytest.mark.asyncio
async def test_retriever_all_item_types_in_single_response(retriever, mock_sdk):
    """All five memory item types can be returned in a single response."""
    mock_response = MagicMock(
        facts=[MagicMock(content="f", id="f1", confidence=0.9, source="s", valid_until=None, temporal_category=None)],
        preferences=[MagicMock(content="p", id="p1", strength=0.8, category="c")],
        episodes=[MagicMock(summary="e", id="e1", significance=0.7, occurred_at="2024-01-01")],
        emotions=[MagicMock(emotion_type="calm", context="ctx", id="em1", intensity=0.5, detected_at="2024-01-01")],
        temporal_events=[MagicMock(content="t", id="t1", event_date="2024-04-01", valid_until=None, temporal_category="temporal_fact")],
        scope_map={"f1": "user", "p1": "user", "e1": "user", "em1": "user", "t1": "user"},
    )
    mock_sdk.fetch.return_value = mock_response

    run_manager = MagicMock()
    docs = await retriever._aget_relevant_documents("everything", run_manager=run_manager)

    assert len(docs) == 5
    types = [d.metadata["type"] for d in docs]
    assert "fact" in types
    assert "preference" in types
    assert "episode" in types
    assert "emotion" in types
    assert "temporal_event" in types


@pytest.mark.asyncio
async def test_retriever_empty_response_returns_empty_list(retriever, mock_sdk):
    """Empty response → empty Document list (no crash)."""
    mock_sdk.fetch.return_value = MagicMock(
        facts=[], preferences=[], episodes=[], emotions=[], temporal_events=[],
        scope_map={},
    )
    run_manager = MagicMock()
    docs = await retriever._aget_relevant_documents("nothing", run_manager=run_manager)
    assert docs == []


@pytest.mark.asyncio
async def test_retriever_forwards_correct_fetch_kwargs(retriever, mock_sdk):
    """The SDK fetch call receives the right parameters."""
    mock_sdk.fetch.return_value = MagicMock(
        facts=[], preferences=[], episodes=[], emotions=[], temporal_events=[], scope_map={},
    )
    run_manager = MagicMock()
    await retriever._aget_relevant_documents("coffee preferences", run_manager=run_manager)

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
async def test_retriever_fact_missing_from_scope_map_gives_empty_scope(retriever, mock_sdk):
    """Items not in scope_map get an empty string for 'scope'."""
    mock_fact = MagicMock(
        content="fact not in scope_map", id="f99", confidence=0.5,
        source="s", valid_until=None, temporal_category=None,
    )
    mock_response = MagicMock(
        facts=[mock_fact], preferences=[], episodes=[], emotions=[], temporal_events=[],
        scope_map={},  # f99 absent
    )
    mock_sdk.fetch.return_value = mock_response

    run_manager = MagicMock()
    docs = await retriever._aget_relevant_documents("q", run_manager=run_manager)

    assert docs[0].metadata["scope"] == ""


# ---------------------------------------------------------------------------
# Failure path — SDK raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retriever_raises_synap_integration_error_on_sdk_failure(retriever, mock_sdk):
    """SDK failure inside _aget_relevant_documents is wrapped as SynapIntegrationError.

    wrap_sdk_errors_async is used in retriever.py — any SDK error is surfaced as
    SynapIntegrationError so callers have a typed exception to handle.
    """
    mock_sdk.fetch.side_effect = RuntimeError("sdk boom")

    run_manager = MagicMock()
    with pytest.raises(SynapIntegrationError):
        await retriever._aget_relevant_documents("query", run_manager=run_manager)
