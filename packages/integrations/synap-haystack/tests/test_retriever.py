"""Tests for SynapRetriever and SynapMemoryRetriever (Haystack).

Documented error-handling contract (from retriever.py / store.py):
- Both components delegate to SynapMemoryStore.
- Search degrades gracefully to [] on SDK failure (read-side tolerance).
- Construction requires (store) OR (sdk + scope); both missing → ValueError.
- scope requires at least one of user_id / customer_id.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from haystack import Document
from haystack.dataclasses import ChatMessage

from synap_haystack import SynapMemoryRetriever, SynapRetriever, SynapMemoryStore
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(**kw):
    defaults = dict(
        facts=[], preferences=[], episodes=[], emotions=[], temporal_events=[],
        scope_map={},
    )
    defaults.update(kw)
    return MagicMock(**defaults)


# ---------------------------------------------------------------------------
# Public surface — all exports present
# ---------------------------------------------------------------------------


def test_public_surface_all_exports():
    import synap_haystack
    for name in ("SynapMemoryStore", "SynapMemoryRetriever", "SynapRetriever",
                  "SynapMemoryWriter", "SynapShortTermContext"):
        assert hasattr(synap_haystack, name), f"Missing export: {name}"
        assert name in synap_haystack.__all__, f"Missing from __all__: {name}"


# ---------------------------------------------------------------------------
# SynapRetriever — construction
# ---------------------------------------------------------------------------


class TestSynapRetrieverConstruction:
    def test_builds_from_sdk_and_user_id(self, mock_sdk):
        r = SynapRetriever(sdk=mock_sdk, user_id="u1")
        assert r.store is not None

    def test_builds_from_sdk_and_customer_id(self, mock_sdk):
        r = SynapRetriever(sdk=mock_sdk, customer_id="c1")
        assert r.store is not None

    def test_accepts_store_kwarg(self, mock_sdk):
        store = SynapMemoryStore(mock_sdk, user_id="u1")
        r = SynapRetriever(store=store)
        assert r.store is store

    def test_requires_store_or_sdk(self):
        with pytest.raises(ValueError):
            SynapRetriever()

    def test_requires_scope_when_sdk_provided(self, mock_sdk):
        with pytest.raises(ValueError):
            SynapRetriever(sdk=mock_sdk)


# ---------------------------------------------------------------------------
# SynapRetriever — run (RAG path, returns Documents)
# ---------------------------------------------------------------------------


class TestSynapRetrieverRun:
    @pytest.fixture
    def retriever(self, mock_sdk):
        return SynapRetriever(sdk=mock_sdk, user_id="u1")

    def test_run_returns_documents_key(self, retriever, mock_sdk):
        mock_sdk.fetch.return_value = _make_response(
            facts=[MagicMock(content="likes coffee", id="f1", confidence=0.9)],
            scope_map={"f1": "user"},
        )
        result = retriever.run(query="coffee")
        assert "documents" in result

    def test_run_maps_fact_to_document(self, retriever, mock_sdk):
        fact = MagicMock(content="likes coffee", id="f1", confidence=0.9)
        mock_sdk.fetch.return_value = _make_response(
            facts=[fact], scope_map={"f1": "user"}
        )
        docs = retriever.run(query="coffee")["documents"]
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].content == "likes coffee"
        assert docs[0].meta["type"] == "fact"
        assert docs[0].meta["id"] == "f1"
        assert docs[0].meta["confidence"] == 0.9
        assert docs[0].meta["scope"] == "user"

    def test_run_maps_preference_to_document(self, retriever, mock_sdk):
        pref = MagicMock(content="dark mode", id="p1", strength=0.8)
        mock_sdk.fetch.return_value = _make_response(
            preferences=[pref], scope_map={"p1": "user"}
        )
        docs = retriever.run(query="mode")["documents"]
        assert docs[0].meta["type"] == "preference"
        assert docs[0].meta["strength"] == 0.8

    def test_run_maps_episode_to_document(self, retriever, mock_sdk):
        ep = MagicMock(summary="Had a support call", id="e1", significance=0.7)
        mock_sdk.fetch.return_value = _make_response(
            episodes=[ep], scope_map={"e1": "user"}
        )
        docs = retriever.run(query="support")["documents"]
        assert docs[0].content == "Had a support call"
        assert docs[0].meta["type"] == "episode"
        assert docs[0].meta["significance"] == 0.7

    def test_run_maps_emotion_to_document(self, retriever, mock_sdk):
        em = MagicMock(emotion_type="frustrated", context="Long wait", id="em1", intensity=0.6)
        mock_sdk.fetch.return_value = _make_response(
            emotions=[em], scope_map={"em1": "user"}
        )
        docs = retriever.run(query="feeling")["documents"]
        assert "frustrated" in docs[0].content
        assert "Long wait" in docs[0].content
        assert docs[0].meta["type"] == "emotion"
        assert docs[0].meta["intensity"] == 0.6

    def test_run_maps_temporal_event_to_document(self, retriever, mock_sdk):
        te = MagicMock(content="Trial expires April 15", id="t1")
        mock_sdk.fetch.return_value = _make_response(
            temporal_events=[te], scope_map={"t1": "user"}
        )
        docs = retriever.run(query="trial")["documents"]
        assert docs[0].content == "Trial expires April 15"
        assert docs[0].meta["type"] == "temporal_event"

    def test_run_all_five_types(self, retriever, mock_sdk):
        mock_sdk.fetch.return_value = _make_response(
            facts=[MagicMock(content="f", id="f1", confidence=0.9)],
            preferences=[MagicMock(content="p", id="p1", strength=0.8)],
            episodes=[MagicMock(summary="e", id="e1", significance=0.7)],
            emotions=[MagicMock(emotion_type="calm", context="ok", id="em1", intensity=0.5)],
            temporal_events=[MagicMock(content="t", id="t1")],
            scope_map={},
        )
        docs = retriever.run(query="all")["documents"]
        assert len(docs) == 5
        types = {d.meta["type"] for d in docs}
        assert types == {"fact", "preference", "episode", "emotion", "temporal_event"}

    def test_run_empty_response_returns_empty_list(self, retriever, mock_sdk):
        mock_sdk.fetch.return_value = _make_response()
        result = retriever.run(query="nothing")
        assert result["documents"] == []

    def test_run_calls_sdk_fetch(self, retriever, mock_sdk):
        mock_sdk.fetch.return_value = _make_response()
        retriever.run(query="coffee")
        mock_sdk.fetch.assert_awaited_once()

    def test_run_degrades_to_empty_on_sdk_failure(self, retriever, mock_sdk):
        """Read-side degrades gracefully — never crashes the pipeline."""
        mock_sdk.fetch.side_effect = RuntimeError("sdk boom")
        result = retriever.run(query="coffee")
        assert result["documents"] == []

    def test_failing_sdk_degrades_to_empty(self, failing_sdk):
        retriever = SynapRetriever(sdk=failing_sdk, user_id="u1")
        result = retriever.run(query="coffee")
        assert result["documents"] == []


# ---------------------------------------------------------------------------
# SynapMemoryRetriever — construction
# ---------------------------------------------------------------------------


class TestSynapMemoryRetrieverConstruction:
    def test_builds_from_sdk_and_user_id(self, mock_sdk):
        r = SynapMemoryRetriever(sdk=mock_sdk, user_id="u1")
        assert r.store is not None

    def test_accepts_store_kwarg(self, mock_sdk):
        store = SynapMemoryStore(mock_sdk, user_id="u1")
        r = SynapMemoryRetriever(store=store)
        assert r.store is store

    def test_requires_store_or_sdk(self):
        with pytest.raises(ValueError):
            SynapMemoryRetriever()

    def test_requires_scope_when_sdk_provided(self, mock_sdk):
        with pytest.raises(ValueError):
            SynapMemoryRetriever(sdk=mock_sdk)


# ---------------------------------------------------------------------------
# SynapMemoryRetriever — run (chat path, returns ChatMessages)
# ---------------------------------------------------------------------------


class TestSynapMemoryRetrieverRun:
    @pytest.fixture
    def retriever(self, mock_sdk):
        return SynapMemoryRetriever(sdk=mock_sdk, user_id="u1")

    def test_run_returns_messages_key(self, retriever, mock_sdk):
        mock_sdk.fetch.return_value = _make_response(
            facts=[MagicMock(content="likes tea", id="f1", confidence=0.8)],
            scope_map={"f1": "user"},
        )
        result = retriever.run(query="tea")
        assert "messages" in result

    def test_run_returns_chat_messages(self, retriever, mock_sdk):
        fact = MagicMock(content="likes tea", id="f1", confidence=0.8)
        mock_sdk.fetch.return_value = _make_response(
            facts=[fact], scope_map={"f1": "user"}
        )
        msgs = retriever.run(query="tea")["messages"]
        assert len(msgs) == 1
        assert isinstance(msgs[0], ChatMessage)
        assert msgs[0].text == "likes tea"
        assert msgs[0].role.value == "assistant"
        assert msgs[0].meta["type"] == "fact"

    def test_run_preference_as_chat_message(self, retriever, mock_sdk):
        pref = MagicMock(content="dark mode", id="p1", strength=0.8)
        mock_sdk.fetch.return_value = _make_response(
            preferences=[pref], scope_map={}
        )
        msgs = retriever.run(query="mode")["messages"]
        assert msgs[0].meta["type"] == "preference"

    def test_run_all_five_types_as_messages(self, retriever, mock_sdk):
        mock_sdk.fetch.return_value = _make_response(
            facts=[MagicMock(content="f", id="f1", confidence=0.9)],
            preferences=[MagicMock(content="p", id="p1", strength=0.8)],
            episodes=[MagicMock(summary="e", id="e1", significance=0.7)],
            emotions=[MagicMock(emotion_type="sad", context="loss", id="em1", intensity=0.5)],
            temporal_events=[MagicMock(content="t", id="t1")],
            scope_map={},
        )
        msgs = retriever.run(query="all")["messages"]
        assert len(msgs) == 5
        types = {m.meta["type"] for m in msgs}
        assert types == {"fact", "preference", "episode", "emotion", "temporal_event"}

    def test_run_empty_response_returns_empty_list(self, retriever, mock_sdk):
        mock_sdk.fetch.return_value = _make_response()
        result = retriever.run(query="nothing")
        assert result["messages"] == []

    def test_run_degrades_to_empty_on_sdk_failure(self, retriever, mock_sdk):
        """Read-side degrades gracefully — never crashes the pipeline."""
        mock_sdk.fetch.side_effect = RuntimeError("sdk boom")
        result = retriever.run(query="coffee")
        assert result["messages"] == []

    def test_failing_sdk_degrades_to_empty(self, failing_sdk):
        retriever = SynapMemoryRetriever(sdk=failing_sdk, user_id="u1")
        result = retriever.run(query="tea")
        assert result["messages"] == []


# ---------------------------------------------------------------------------
# Both retrievers — shared store is reused (not rebuilt)
# ---------------------------------------------------------------------------


class TestStoreReuse:
    def test_synap_retriever_store_is_passed_store(self, mock_sdk):
        store = SynapMemoryStore(mock_sdk, user_id="u1")
        r = SynapRetriever(store=store)
        assert r.store is store

    def test_memory_retriever_store_is_passed_store(self, mock_sdk):
        store = SynapMemoryStore(mock_sdk, user_id="u1")
        r = SynapMemoryRetriever(store=store)
        assert r.store is store
