"""Tests for SynapMemoryEditor — the NAT MemoryEditor backed by Synap.

Public surface + failure paths, using the shared harness (``mock_sdk`` /
``failing_sdk`` re-exported via conftest). Error policy under test:

- writes (``add_items``) surface failures as ``SynapIntegrationError``;
- reads (``search``) degrade to ``[]`` so an agent turn never crashes;
- deletes (``remove_items``) warn once and no-op (no public delete API).

These bind ``nat`` classes at import time; this module sorts before
``test_short_term.py`` (which stubs ``nat`` in sys.modules), so the real
toolkit is in scope here.
"""

from __future__ import annotations

import logging

import pytest
from nat.memory.interfaces import MemoryEditor
from nat.memory.models import MemoryItem

from synap_integrations_common import SynapIntegrationError
from synap_integrations_common.testing import (
    make_emotion, make_episode, make_fact, make_preference, make_temporal_event, make_unified_response,
)
from synap_nemo_agent_toolkit.editor import SynapMemoryEditor


@pytest.fixture
def editor(mock_sdk):
    return SynapMemoryEditor(sdk=mock_sdk)


class TestConstruction:
    def test_requires_non_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapMemoryEditor(sdk=None)  # type: ignore[arg-type]

    def test_defaults(self, mock_sdk):
        ed = SynapMemoryEditor(sdk=mock_sdk)
        assert ed.customer_id == ""
        assert ed.mode == "accurate"
        assert ed.document_type == "ai-chat-conversation"

    def test_is_a_memory_editor(self, editor):
        assert isinstance(editor, MemoryEditor)


class TestAddItems:
    @pytest.mark.asyncio
    async def test_empty_list_is_noop(self, editor, mock_sdk):
        await editor.add_items([])
        mock_sdk.memories.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_writes_one_create_per_item(self, editor, mock_sdk):
        items = [
            MemoryItem(user_id="u1", memory="likes coffee", tags=["pref"], metadata={}),
            MemoryItem(user_id="u1", memory="is an engineer", tags=[], metadata={}),
        ]
        await editor.add_items(items)
        assert mock_sdk.memories.create.await_count == 2

    @pytest.mark.asyncio
    async def test_create_args_and_marker_metadata(self, editor, mock_sdk):
        item = MemoryItem(user_id="u7", memory="likes coffee", tags=["a"], metadata={"k": "v"})
        await editor.add_items([item])

        kwargs = mock_sdk.memories.create.call_args.kwargs
        assert kwargs["document"] == "likes coffee"
        assert kwargs["user_id"] == "u7"
        assert kwargs["document_type"] == "ai-chat-conversation"
        md = kwargs["metadata"]
        assert md["nat_memory_item"] is True
        assert md["tags"] == ["a"]
        assert md["k"] == "v"

    @pytest.mark.asyncio
    async def test_conversation_only_item_is_joined(self, editor, mock_sdk):
        convo = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        item = MemoryItem(user_id="u1", memory=None, conversation=convo, tags=[], metadata={})
        await editor.add_items([item])

        kwargs = mock_sdk.memories.create.call_args.kwargs
        assert "user: hi" in kwargs["document"]
        assert "assistant: hello" in kwargs["document"]
        assert kwargs["metadata"]["conversation"] == convo

    @pytest.mark.asyncio
    async def test_item_with_no_content_is_skipped(self, editor, mock_sdk):
        item = MemoryItem(user_id="u1", memory=None, conversation=None, tags=[], metadata={})
        await editor.add_items([item])
        mock_sdk.memories.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_user_id_raises(self, editor):
        item = MemoryItem(user_id="", memory="x", tags=[], metadata={})
        with pytest.raises(ValueError, match="user_id is required"):
            await editor.add_items([item])

    @pytest.mark.asyncio
    async def test_customer_id_propagated(self, mock_sdk):
        ed = SynapMemoryEditor(sdk=mock_sdk, customer_id="acme")
        await ed.add_items([MemoryItem(user_id="u1", memory="x", tags=[], metadata={})])
        assert mock_sdk.memories.create.call_args.kwargs["customer_id"] == "acme"

    @pytest.mark.asyncio
    async def test_write_failure_surfaces_wrapped(self, failing_sdk):
        ed = SynapMemoryEditor(sdk=failing_sdk)
        with pytest.raises(SynapIntegrationError):
            await ed.add_items([MemoryItem(user_id="u1", memory="x", tags=[], metadata={})])


class TestSearch:
    @pytest.mark.asyncio
    async def test_requires_user_id_in_kwargs(self, editor):
        with pytest.raises(ValueError, match="requires user_id"):
            await editor.search("coffee", top_k=3)

    @pytest.mark.asyncio
    async def test_maps_facts_to_memory_items(self, editor):
        # The default response holds a fact AND a preference; both must come back.
        items = await editor.search("coffee", top_k=3, user_id="u1")
        assert [i.memory for i in items] == ["User is an engineer", "Prefers dark mode"]
        assert all(isinstance(i, MemoryItem) for i in items)
        fact = items[0]
        assert fact.user_id == "u1"
        assert fact.similarity_score == 0.9
        assert fact.metadata["synap_type"] == "fact"

    @pytest.mark.asyncio
    async def test_a_stored_preference_comes_back(self, mock_sdk):
        """The reported bug: preferences were dropped, so a stored preference never returned."""
        mock_sdk.fetch.return_value = make_unified_response(
            facts=[], preferences=[make_preference("p1", "Prefers window seats", 0.8)]
        )
        items = await SynapMemoryEditor(sdk=mock_sdk).search("seats", top_k=5, user_id="u1")
        assert [i.memory for i in items] == ["Prefers window seats"]
        assert items[0].similarity_score == 0.8
        assert items[0].metadata["synap_type"] == "preference"

    @pytest.mark.asyncio
    async def test_every_memory_type_is_returned_with_its_score(self, mock_sdk):
        mock_sdk.fetch.return_value = make_unified_response(
            facts=[make_fact("f1", "A fact", 0.5)],
            preferences=[make_preference("p1", "A preference", 0.6)],
            episodes=[make_episode("e1", "An episode summary", 0.7)],
            emotions=[make_emotion("m1", "joy", "after the launch", 0.8)],
            temporal_events=[make_temporal_event("t1", "A dated event")],
        )
        items = await SynapMemoryEditor(sdk=mock_sdk).search("anything", top_k=10, user_id="u1")
        by_type = {i.metadata["synap_type"]: i for i in items}
        assert set(by_type) == {"fact", "preference", "episode", "emotion", "temporal_event"}
        assert by_type["episode"].memory == "An episode summary" and by_type["episode"].similarity_score == 0.7
        assert by_type["emotion"].memory == "joy: after the launch" and by_type["emotion"].similarity_score == 0.8
        assert by_type["temporal_event"].memory == "A dated event" and by_type["temporal_event"].similarity_score == 0.9

    @pytest.mark.asyncio
    async def test_results_are_ranked_by_score_across_types_and_capped_at_top_k(self, mock_sdk):
        mock_sdk.fetch.return_value = make_unified_response(
            facts=[make_fact("f1", "low fact", 0.2)],
            preferences=[make_preference("p1", "top preference", 0.95)],
            episodes=[make_episode("e1", "mid episode", 0.6)],
            emotions=[], temporal_events=[],
        )
        items = await SynapMemoryEditor(sdk=mock_sdk).search("q", top_k=2, user_id="u1")
        assert [i.memory for i in items] == ["top preference", "mid episode"]

    @pytest.mark.asyncio
    async def test_tag_filter_applies_to_every_type(self, mock_sdk):
        pref = make_preference("p1", "tagged preference", 0.8)
        pref.metadata = {"tags": ["travel"]}
        mock_sdk.fetch.return_value = make_unified_response(facts=[make_fact()], preferences=[pref])
        items = await SynapMemoryEditor(sdk=mock_sdk).search("q", top_k=5, user_id="u1", tag_filter=["travel"])
        assert [i.memory for i in items] == ["tagged preference"]

    @pytest.mark.asyncio
    async def test_fetch_args(self, editor, mock_sdk):
        await editor.search("coffee", top_k=3, user_id="u1")
        kwargs = mock_sdk.fetch.call_args.kwargs
        assert kwargs["search_query"] == ["coffee"]
        assert kwargs["max_results"] == 3
        assert kwargs["mode"] == "accurate"
        assert kwargs["user_id"] == "u1"
        assert kwargs["include_conversation_context"] is False

    @pytest.mark.asyncio
    async def test_top_k_floor_is_one(self, editor, mock_sdk):
        await editor.search("q", top_k=0, user_id="u1")
        assert mock_sdk.fetch.call_args.kwargs["max_results"] == 1

    @pytest.mark.asyncio
    async def test_empty_query_passes_none(self, editor, mock_sdk):
        await editor.search("", top_k=3, user_id="u1")
        assert mock_sdk.fetch.call_args.kwargs["search_query"] is None

    @pytest.mark.asyncio
    async def test_tag_filter_excludes_unmatched(self, editor):
        # The default fact carries no tags metadata, so a tag_filter cannot
        # match and the hit is dropped — yields an empty result, not a crash.
        items = await editor.search("coffee", top_k=3, user_id="u1", tag_filter=["nope"])
        assert items == []

    @pytest.mark.asyncio
    async def test_read_failure_degrades_to_empty(self, failing_sdk):
        ed = SynapMemoryEditor(sdk=failing_sdk)
        result = await ed.search("coffee", top_k=3, user_id="u1")
        assert result == []


class TestRemoveItems:
    @pytest.mark.asyncio
    async def test_warns_once_then_noops(self, editor, caplog):
        with caplog.at_level(logging.WARNING):
            await editor.remove_items(user_id="u1")
            await editor.remove_items(user_id="u1")
        delete_warnings = [r for r in caplog.records if "no public delete" in r.getMessage()]
        assert len(delete_warnings) == 1
