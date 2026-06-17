"""Tests for SynapStorageBackend — synap_crewai.storage.

Covers every public method and documented behavior:

save / asave
    - happy path: sdk.memories.create called with correct args per record
    - multiple records: all records ingested
    - record persisted to _records cache after save
    - SDK failure: SynapIntegrationError raised (strict — save is explicit)

search / asearch
    - happy path: query_text via metadata_filter['_query_text']
    - returns List[Tuple[MemoryRecord, float]]
    - all response types mapped: facts, preferences, episodes, emotions, temporal_events
    - missing query_text raises SynapIntegrationError
    - min_score filter applied
    - limit applied
    - categories mapping: fact→facts, preference→preferences, episode→episodes, emotion→emotions
    - unknown categories ignored in type mapping
    - query_embedding ignored (Synap embeds server-side)
    - SDK fetch called with correct parameters

delete / adelete
    - no-op: returns 0, no exception
    - warning logged

update
    - no-op: no exception
    - warning logged

reset
    - clears all _records when no scope_prefix
    - clears only matching scope when scope_prefix given
    - server-side memories NOT affected (no SDK call)

Session-local views (list_records, count, get_record, get_scope_info, list_scopes, list_categories)
    - populated from saves
    - session-only warning logged on first access
    - warning logged only once (deduplicated)

Construction
    - None sdk raises ValueError
    - empty user_id raises ValueError
    - optional customer_id, conversation_id, mode

failing_sdk paths
    - asave with failing_sdk raises SynapIntegrationError
    - asearch with failing_sdk raises SynapIntegrationError
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from crewai.memory.types import MemoryRecord, ScopeInfo
from synap_integrations_common import SynapIntegrationError
from synap_integrations_common.testing import (
    failing_sdk,  # noqa: F401
    make_emotion,
    make_episode,
    make_fact,
    make_preference,
    make_temporal_event,
    make_unified_response,
    mock_sdk,  # noqa: F401
)

from synap_crewai.storage import SynapStorageBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    id: str = "r1",
    content: str = "User likes coffee",
    scope: str = "/user/u1",
    categories: List[str] | None = None,
    importance: float = 0.8,
) -> MemoryRecord:
    return MemoryRecord(
        id=id,
        content=content,
        scope=scope,
        categories=categories or ["preference"],
        importance=importance,
        metadata={},
        created_at=datetime.now(timezone.utc),
    )


def _make_fetch_response(
    facts=None,
    preferences=None,
    episodes=None,
    emotions=None,
    temporal_events=None,
    scope_map=None,
):
    return make_unified_response(
        facts=facts or [],
        preferences=preferences or [],
        episodes=episodes or [],
        emotions=emotions or [],
        temporal_events=temporal_events or [],
        scope_map=scope_map or {},
    )


def _backend(sdk, user_id="u1", customer_id="c1", conversation_id=None, mode="fast"):
    return SynapStorageBackend(
        sdk=sdk,
        user_id=user_id,
        customer_id=customer_id,
        conversation_id=conversation_id,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_raises_on_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapStorageBackend(sdk=None, user_id="u1")  # type: ignore[arg-type]

    def test_raises_on_empty_user_id(self, mock_sdk):
        with pytest.raises(ValueError, match="non-empty user_id"):
            SynapStorageBackend(sdk=mock_sdk, user_id="")

    def test_optional_customer_id_defaults_to_empty(self, mock_sdk):
        b = SynapStorageBackend(sdk=mock_sdk, user_id="u1")
        assert b.customer_id == ""

    def test_optional_conversation_id_defaults_to_none(self, mock_sdk):
        b = SynapStorageBackend(sdk=mock_sdk, user_id="u1")
        assert b.conversation_id is None

    def test_mode_stored(self, mock_sdk):
        b = SynapStorageBackend(sdk=mock_sdk, user_id="u1", mode="accurate")
        assert b.mode == "accurate"

    def test_records_initially_empty(self, mock_sdk):
        b = _backend(mock_sdk)
        assert len(b._records) == 0

    def test_session_warning_not_logged_at_init(self, mock_sdk, caplog):
        with caplog.at_level(logging.WARNING, logger="synap_crewai.storage"):
            _backend(mock_sdk)
        # No session-only warning at construction
        assert not any("session-local" in r.message.lower() or
                        "listing" in r.message for r in caplog.records)

    def test_import_from_package(self):
        from synap_crewai import SynapStorageBackend as Cls
        assert Cls is SynapStorageBackend

    def test_public_surface_in_all(self):
        import synap_crewai
        assert "SynapStorageBackend" in synap_crewai.__all__


# ---------------------------------------------------------------------------
# save / asave
# ---------------------------------------------------------------------------


class TestSave:
    @pytest.mark.asyncio
    async def test_asave_calls_memories_create_once_per_record(self, mock_sdk):
        b = _backend(mock_sdk)
        r1 = _make_record(id="r1", content="Coffee fan")
        r2 = _make_record(id="r2", content="Night owl")
        await b.asave([r1, r2])
        assert mock_sdk.memories.create.await_count == 2

    @pytest.mark.asyncio
    async def test_asave_passes_document_as_content(self, mock_sdk):
        b = _backend(mock_sdk)
        r = _make_record(id="r1", content="User likes Python")
        await b.asave([r])
        kw = mock_sdk.memories.create.call_args.kwargs
        assert kw["document"] == "User likes Python"

    @pytest.mark.asyncio
    async def test_asave_passes_user_id(self, mock_sdk):
        b = _backend(mock_sdk, user_id="user-42")
        r = _make_record(id="r1")
        await b.asave([r])
        kw = mock_sdk.memories.create.call_args.kwargs
        assert kw["user_id"] == "user-42"

    @pytest.mark.asyncio
    async def test_asave_passes_customer_id(self, mock_sdk):
        b = _backend(mock_sdk, customer_id="cust-77")
        r = _make_record(id="r1")
        await b.asave([r])
        kw = mock_sdk.memories.create.call_args.kwargs
        assert kw["customer_id"] == "cust-77"

    @pytest.mark.asyncio
    async def test_asave_includes_record_id_in_metadata(self, mock_sdk):
        b = _backend(mock_sdk)
        r = _make_record(id="r-special")
        await b.asave([r])
        meta = mock_sdk.memories.create.call_args.kwargs["metadata"]
        assert meta["crewai_record_id"] == "r-special"

    @pytest.mark.asyncio
    async def test_asave_includes_importance_in_metadata(self, mock_sdk):
        b = _backend(mock_sdk)
        r = _make_record(id="r1", importance=0.95)
        await b.asave([r])
        meta = mock_sdk.memories.create.call_args.kwargs["metadata"]
        assert meta["importance"] == 0.95

    @pytest.mark.asyncio
    async def test_asave_caches_record_locally(self, mock_sdk):
        b = _backend(mock_sdk)
        r = _make_record(id="r1")
        await b.asave([r])
        assert "r1" in b._records
        assert b._records["r1"] is r

    @pytest.mark.asyncio
    async def test_asave_empty_list_is_noop(self, mock_sdk):
        b = _backend(mock_sdk)
        await b.asave([])
        mock_sdk.memories.create.assert_not_awaited()

    def test_save_sync_calls_asave(self, mock_sdk):
        """Sync save must delegate to asave via run_async."""
        b = _backend(mock_sdk)
        r = _make_record(id="r1")
        b.save([r])  # must not raise
        assert "r1" in b._records

    @pytest.mark.asyncio
    async def test_asave_raises_synap_integration_error_on_sdk_failure(self, failing_sdk):
        """Save is explicit — SDK errors must raise SynapIntegrationError, not be swallowed."""
        b = _backend(failing_sdk)
        r = _make_record(id="r1")
        with pytest.raises(SynapIntegrationError):
            await b.asave([r])


# ---------------------------------------------------------------------------
# search / asearch
# ---------------------------------------------------------------------------


class TestSearch:
    @pytest.mark.asyncio
    async def test_asearch_requires_query_text_in_metadata_filter(self, mock_sdk):
        b = _backend(mock_sdk)
        with pytest.raises(SynapIntegrationError, match="query_text missing"):
            await b.asearch(query_embedding=[])

    @pytest.mark.asyncio
    async def test_asearch_raises_without_query_text(self, mock_sdk):
        b = _backend(mock_sdk)
        with pytest.raises(SynapIntegrationError):
            await b.asearch(query_embedding=[], metadata_filter={})

    @pytest.mark.asyncio
    async def test_asearch_with_query_text_calls_sdk_fetch(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response()
        b = _backend(mock_sdk)
        results = await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "coffee preference"},
        )
        mock_sdk.fetch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_asearch_passes_query_text_as_search_query(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response()
        b = _backend(mock_sdk)
        await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "dark mode"},
        )
        kw = mock_sdk.fetch.call_args.kwargs
        assert kw["search_query"] == ["dark mode"]

    @pytest.mark.asyncio
    async def test_asearch_passes_limit_to_fetch(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response()
        b = _backend(mock_sdk)
        await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "q"},
            limit=7,
        )
        kw = mock_sdk.fetch.call_args.kwargs
        assert kw["max_results"] == 7

    @pytest.mark.asyncio
    async def test_asearch_passes_mode_to_fetch(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response()
        b = _backend(mock_sdk, mode="accurate")
        await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "q"},
        )
        kw = mock_sdk.fetch.call_args.kwargs
        assert kw["mode"] == "accurate"

    @pytest.mark.asyncio
    async def test_asearch_passes_user_id_to_fetch(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response()
        b = _backend(mock_sdk, user_id="user-99")
        await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "q"},
        )
        kw = mock_sdk.fetch.call_args.kwargs
        assert kw["user_id"] == "user-99"

    @pytest.mark.asyncio
    async def test_asearch_returns_list(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response()
        b = _backend(mock_sdk)
        results = await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "q"},
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_asearch_returns_tuples_of_record_and_score(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response(
            facts=[make_fact(id="f1", confidence=0.9)]
        )
        b = _backend(mock_sdk)
        results = await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "q"},
        )
        assert len(results) == 1
        rec, score = results[0]
        assert isinstance(rec, MemoryRecord)
        assert isinstance(score, float)

    @pytest.mark.asyncio
    async def test_asearch_maps_facts_to_category_fact(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response(
            facts=[make_fact(id="f1", content="User is an engineer", confidence=0.9)]
        )
        b = _backend(mock_sdk)
        results = await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "q"},
        )
        assert len(results) == 1
        rec, score = results[0]
        assert "fact" in rec.categories
        assert rec.content == "User is an engineer"
        assert score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_asearch_maps_preferences_to_category_preference(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response(
            preferences=[make_preference(id="p1", content="Prefers dark mode", strength=0.8)]
        )
        b = _backend(mock_sdk)
        results = await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "q"},
        )
        assert len(results) == 1
        rec, score = results[0]
        assert "preference" in rec.categories
        assert rec.content == "Prefers dark mode"
        assert score == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_asearch_maps_episodes_to_category_episode(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response(
            episodes=[make_episode(id="e1", summary="Support call", significance=0.7)]
        )
        b = _backend(mock_sdk)
        results = await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "q"},
        )
        assert len(results) == 1
        rec, score = results[0]
        assert "episode" in rec.categories
        assert rec.content == "Support call"
        assert score == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_asearch_maps_emotions_to_category_emotion(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response(
            emotions=[make_emotion(id="em1", emotion_type="happy", context="Good news", intensity=0.6)]
        )
        b = _backend(mock_sdk)
        results = await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "q"},
        )
        assert len(results) == 1
        rec, score = results[0]
        assert "emotion" in rec.categories
        assert "happy" in rec.content
        assert "Good news" in rec.content
        assert score == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_asearch_maps_temporal_events_to_category_temporal_event(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response(
            temporal_events=[make_temporal_event(id="t1", content="Trial expires April 15")]
        )
        b = _backend(mock_sdk)
        results = await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "q"},
        )
        assert len(results) == 1
        rec, score = results[0]
        assert "temporal_event" in rec.categories
        assert rec.content == "Trial expires April 15"

    @pytest.mark.asyncio
    async def test_asearch_mixed_types_all_included(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response(
            facts=[make_fact(id="f1")],
            preferences=[make_preference(id="p1")],
            episodes=[make_episode(id="e1")],
            emotions=[make_emotion(id="em1")],
            temporal_events=[make_temporal_event(id="t1")],
        )
        b = _backend(mock_sdk)
        results = await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "q"},
        )
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_asearch_min_score_filters_results(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response(
            facts=[
                make_fact(id="f1", confidence=0.9),
                make_fact(id="f2", confidence=0.3),
            ]
        )
        b = _backend(mock_sdk)
        results = await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "q"},
            min_score=0.5,
        )
        assert len(results) == 1
        assert results[0][1] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_asearch_results_sorted_by_score_descending(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response(
            facts=[
                make_fact(id="f1", confidence=0.4),
                make_fact(id="f2", confidence=0.9),
                make_fact(id="f3", confidence=0.7),
            ]
        )
        b = _backend(mock_sdk)
        results = await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "q"},
        )
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_asearch_limit_applied_after_sort(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response(
            facts=[make_fact(id=f"f{i}", confidence=float(i) / 10)
                   for i in range(1, 8)]  # 7 facts
        )
        b = _backend(mock_sdk)
        results = await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "q"},
            limit=3,
        )
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_asearch_query_embedding_ignored(self, mock_sdk):
        """query_embedding is accepted for protocol compliance but NOT forwarded to SDK."""
        mock_sdk.fetch.return_value = _make_fetch_response()
        b = _backend(mock_sdk)
        await b.asearch(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            metadata_filter={"_query_text": "some query"},
        )
        kw = mock_sdk.fetch.call_args.kwargs
        # SDK fetch must not receive embedding
        assert "embedding" not in kw
        assert "query_embedding" not in kw

    @pytest.mark.asyncio
    async def test_asearch_category_fact_mapped_to_facts_type(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response()
        b = _backend(mock_sdk)
        await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "q"},
            categories=["fact"],
        )
        kw = mock_sdk.fetch.call_args.kwargs
        assert kw.get("types") == ["facts"]

    @pytest.mark.asyncio
    async def test_asearch_category_preference_mapped(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response()
        b = _backend(mock_sdk)
        await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "q"},
            categories=["preference"],
        )
        kw = mock_sdk.fetch.call_args.kwargs
        assert kw.get("types") == ["preferences"]

    @pytest.mark.asyncio
    async def test_asearch_unknown_categories_dropped(self, mock_sdk):
        mock_sdk.fetch.return_value = _make_fetch_response()
        b = _backend(mock_sdk)
        await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "q"},
            categories=["completely_unknown"],
        )
        kw = mock_sdk.fetch.call_args.kwargs
        assert kw.get("types") is None

    @pytest.mark.asyncio
    async def test_asearch_strips_query_text_from_metadata_filter(self, mock_sdk):
        """_query_text must be popped from metadata_filter before forwarding."""
        mock_sdk.fetch.return_value = _make_fetch_response()
        b = _backend(mock_sdk)
        mf = {"_query_text": "my search", "other_key": "value"}
        await b.asearch(query_embedding=[], metadata_filter=mf)
        # _query_text popped from the dict
        assert "_query_text" not in mf

    def test_search_sync_delegates_to_asearch(self, mock_sdk):
        """Sync search must delegate to asearch via run_async."""
        mock_sdk.fetch.return_value = _make_fetch_response(
            facts=[make_fact(id="f1", confidence=0.9)]
        )
        b = _backend(mock_sdk)
        results = b.search(
            query_embedding=[],
            metadata_filter={"_query_text": "sync query"},
        )
        assert isinstance(results, list)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_asearch_raises_synap_integration_error_on_sdk_failure(self, failing_sdk):
        """SDK failure during search must raise SynapIntegrationError."""
        b = _backend(failing_sdk)
        with pytest.raises(SynapIntegrationError):
            await b.asearch(
                query_embedding=[],
                metadata_filter={"_query_text": "some query"},
            )


# ---------------------------------------------------------------------------
# delete / adelete — unsupported mutations (no-op with warning)
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_returns_zero(self, mock_sdk):
        b = _backend(mock_sdk)
        result = b.delete()
        assert result == 0

    def test_delete_does_not_raise(self, mock_sdk):
        b = _backend(mock_sdk)
        b.delete(scope_prefix="/user/u1", record_ids=["r1"])  # must not raise

    def test_delete_does_not_call_sdk(self, mock_sdk):
        b = _backend(mock_sdk)
        b.delete()
        mock_sdk.fetch.assert_not_awaited()
        mock_sdk.memories.create.assert_not_awaited()

    def test_delete_logs_warning(self, mock_sdk, caplog):
        b = _backend(mock_sdk)
        with caplog.at_level(logging.WARNING, logger="synap_crewai.storage"):
            b.delete()
        assert any("delete" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_adelete_returns_zero(self, mock_sdk):
        b = _backend(mock_sdk)
        result = await b.adelete()
        assert result == 0

    @pytest.mark.asyncio
    async def test_adelete_does_not_raise(self, mock_sdk):
        b = _backend(mock_sdk)
        await b.adelete(scope_prefix="/user/u1")  # must not raise


# ---------------------------------------------------------------------------
# update — unsupported mutation (no-op with warning)
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_does_not_raise(self, mock_sdk):
        b = _backend(mock_sdk)
        r = _make_record(id="r1")
        b.update(r)  # must not raise

    def test_update_logs_warning(self, mock_sdk, caplog):
        b = _backend(mock_sdk)
        r = _make_record(id="r1")
        with caplog.at_level(logging.WARNING, logger="synap_crewai.storage"):
            b.update(r)
        assert any("update" in msg.lower()
                   for msg in [rec.message for rec in caplog.records])


# ---------------------------------------------------------------------------
# reset — clears local cache only
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_all_records(self, mock_sdk):
        b = _backend(mock_sdk)
        b._records["r1"] = _make_record(id="r1")
        b._records["r2"] = _make_record(id="r2")
        b.reset()
        assert len(b._records) == 0

    def test_reset_with_scope_prefix_clears_matching_only(self, mock_sdk):
        b = _backend(mock_sdk)
        b._records["r1"] = _make_record(id="r1", scope="/user/u1/pref")
        b._records["r2"] = _make_record(id="r2", scope="/user/u2/fact")
        b.reset(scope_prefix="/user/u1")
        assert "r1" not in b._records
        assert "r2" in b._records

    def test_reset_does_not_call_sdk(self, mock_sdk):
        b = _backend(mock_sdk)
        b._records["r1"] = _make_record(id="r1")
        b.reset()
        mock_sdk.fetch.assert_not_awaited()
        mock_sdk.memories.create.assert_not_awaited()

    def test_reset_logs_warning(self, mock_sdk, caplog):
        b = _backend(mock_sdk)
        with caplog.at_level(logging.WARNING, logger="synap_crewai.storage"):
            b.reset()
        assert len(caplog.records) >= 1

    def test_reset_empty_scope_prefix_keeps_non_matching(self, mock_sdk):
        """Scope-prefix reset should keep records outside the prefix."""
        b = _backend(mock_sdk)
        b._records["a"] = _make_record(id="a", scope="/alpha/x")
        b._records["b"] = _make_record(id="b", scope="/beta/x")
        b.reset(scope_prefix="/alpha")
        assert "a" not in b._records
        assert "b" in b._records


# ---------------------------------------------------------------------------
# Session-local read views
# ---------------------------------------------------------------------------


class TestSessionLocalViews:
    @pytest.mark.asyncio
    async def test_count_zero_initially(self, mock_sdk):
        b = _backend(mock_sdk)
        assert b.count() == 0

    @pytest.mark.asyncio
    async def test_count_reflects_saves(self, mock_sdk):
        b = _backend(mock_sdk)
        await b.asave([_make_record("r1"), _make_record("r2")])
        assert b.count() == 2

    @pytest.mark.asyncio
    async def test_count_with_scope_prefix(self, mock_sdk):
        b = _backend(mock_sdk)
        r1 = _make_record(id="r1", scope="/user/u1")
        r2 = _make_record(id="r2", scope="/user/u2")
        await b.asave([r1, r2])
        assert b.count(scope_prefix="/user/u1") == 1
        assert b.count(scope_prefix="/user") == 2

    def test_get_record_missing_returns_none(self, mock_sdk):
        b = _backend(mock_sdk)
        assert b.get_record("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_record_returns_saved_record(self, mock_sdk):
        b = _backend(mock_sdk)
        r = _make_record(id="r1")
        await b.asave([r])
        retrieved = b.get_record("r1")
        assert retrieved is r

    @pytest.mark.asyncio
    async def test_list_records_returns_all_saved(self, mock_sdk):
        b = _backend(mock_sdk)
        r1 = _make_record(id="r1", scope="/user/u1")
        r2 = _make_record(id="r2", scope="/user/u1")
        await b.asave([r1, r2])
        records = b.list_records()
        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_list_records_scope_filter(self, mock_sdk):
        b = _backend(mock_sdk)
        r1 = _make_record(id="r1", scope="/user/u1")
        r2 = _make_record(id="r2", scope="/user/u2")
        await b.asave([r1, r2])
        records = b.list_records(scope_prefix="/user/u1")
        assert len(records) == 1
        assert records[0].id == "r1"

    @pytest.mark.asyncio
    async def test_list_records_offset_and_limit(self, mock_sdk):
        b = _backend(mock_sdk)
        records_to_save = [_make_record(id=f"r{i}", scope="/user/u1") for i in range(5)]
        await b.asave(records_to_save)
        page = b.list_records(limit=2, offset=1)
        assert len(page) == 2

    @pytest.mark.asyncio
    async def test_get_scope_info_returns_scope_info(self, mock_sdk):
        b = _backend(mock_sdk)
        r = _make_record(id="r1", scope="/user/u1", categories=["fact"])
        await b.asave([r])
        info = b.get_scope_info("/user/u1")
        assert isinstance(info, ScopeInfo)
        assert info.record_count == 1
        assert "fact" in info.categories

    @pytest.mark.asyncio
    async def test_get_scope_info_empty_scope(self, mock_sdk):
        b = _backend(mock_sdk)
        info = b.get_scope_info("/nonexistent")
        assert info.record_count == 0
        assert info.categories == []

    @pytest.mark.asyncio
    async def test_list_scopes_returns_immediate_children(self, mock_sdk):
        b = _backend(mock_sdk)
        r1 = _make_record(id="r1", scope="/user/u1")
        r2 = _make_record(id="r2", scope="/user/u2")
        await b.asave([r1, r2])
        scopes = b.list_scopes(parent="/user/")
        assert "/user/u1" in scopes
        assert "/user/u2" in scopes

    @pytest.mark.asyncio
    async def test_list_categories_returns_counts(self, mock_sdk):
        b = _backend(mock_sdk)
        r1 = _make_record(id="r1", categories=["fact"])
        r2 = _make_record(id="r2", categories=["preference"])
        r3 = _make_record(id="r3", categories=["fact"])
        await b.asave([r1, r2, r3])
        cats = b.list_categories()
        assert cats.get("fact") == 2
        assert cats.get("preference") == 1

    @pytest.mark.asyncio
    async def test_list_categories_scope_filter(self, mock_sdk):
        b = _backend(mock_sdk)
        r1 = _make_record(id="r1", scope="/user/u1", categories=["fact"])
        r2 = _make_record(id="r2", scope="/user/u2", categories=["preference"])
        await b.asave([r1, r2])
        cats = b.list_categories(scope_prefix="/user/u1")
        assert cats.get("fact") == 1
        assert "preference" not in cats

    def test_session_only_warning_logged_on_first_access(self, mock_sdk, caplog):
        b = _backend(mock_sdk)
        with caplog.at_level(logging.WARNING, logger="synap_crewai.storage"):
            b.count()
        assert any("session" in r.message.lower() or "SynapStorageBackend" in r.message
                   for r in caplog.records)

    def test_session_only_warning_logged_only_once(self, mock_sdk, caplog):
        """Warning must be deduplicated — only emitted on first access."""
        b = _backend(mock_sdk)
        with caplog.at_level(logging.WARNING, logger="synap_crewai.storage"):
            b.count()
            b.count()
            b.list_records()
        warning_count = sum(
            1 for r in caplog.records
            if "session" in r.message.lower() or "listing" in r.message
        )
        assert warning_count == 1


# ---------------------------------------------------------------------------
# End-to-end: save then search
# ---------------------------------------------------------------------------


class TestSaveThenSearch:
    @pytest.mark.asyncio
    async def test_save_then_search_finds_saved_record(self, mock_sdk):
        """After saving, the local cache reflects the record even before search."""
        b = _backend(mock_sdk)
        r = _make_record(id="r1", content="User is a Python developer")
        await b.asave([r])

        # Verify locally cached
        assert b.get_record("r1") is r
        assert b.count() == 1

        # Simulate search returning the saved fact
        mock_sdk.fetch.return_value = _make_fetch_response(
            facts=[make_fact(id="f_search", content="Python developer", confidence=0.88)]
        )
        results = await b.asearch(
            query_embedding=[],
            metadata_filter={"_query_text": "Python"},
        )
        assert len(results) == 1
        rec, score = results[0]
        assert rec.content == "Python developer"
        assert score == pytest.approx(0.88)
