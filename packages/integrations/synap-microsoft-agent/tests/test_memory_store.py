"""Tests for ``SynapMemoryStore`` — the Tier A harness memory store.

The load-bearing behaviours here are not the happy paths. They are:

- ``get_topic`` returns the record **exactly**, because ``_merge_memory`` does
  read-modify-write on it and any drift compounds every turn.
- ``get_topic`` raises ``FileNotFoundError`` when there is no record, because
  that is MAF's not-found contract and the only thing that keeps a rewritten
  record out of the merge loop.
- ``get_index_text`` never raises, because it feeds the system prompt on every
  run.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from tests._harness import (
    make_record,
    make_session,
    requires_harness,
    wire_ingestion,
)

pytestmark = requires_harness


def build(sdk, **kwargs):
    from synap_microsoft_agent import SynapMemoryStore

    kwargs.setdefault("user_id", "alice")
    kwargs.setdefault("customer_id", "acme")
    return SynapMemoryStore(sdk, **kwargs)


class TestConstruction:
    def test_rejects_none_sdk(self):
        from synap_microsoft_agent import SynapMemoryStore

        with pytest.raises(ValueError, match="non-None sdk"):
            SynapMemoryStore(None, user_id="alice")

    def test_rejects_missing_scope(self, mock_sdk):
        from synap_microsoft_agent import SynapMemoryStore

        with pytest.raises(ValueError, match="user_id or customer_id"):
            SynapMemoryStore(mock_sdk)

    def test_customer_only_scope_is_allowed(self, mock_sdk):
        from synap_microsoft_agent import SynapMemoryStore

        store = SynapMemoryStore(mock_sdk, customer_id="acme")
        assert store.get_owner_id(make_session()) == "/acme"

    def test_satisfies_the_abc(self):
        from synap_microsoft_agent import SynapMemoryStore

        assert not SynapMemoryStore.__abstractmethods__


class TestTopicFidelity:
    """The reason the record store exists."""

    def test_write_then_get_returns_an_identical_record(self, mock_sdk):
        store = build(mock_sdk)
        session = make_session()
        record = make_record(memories=["one", "two", "three"])

        store.write_topic(session, record, source_id="memory")
        loaded = store.get_topic(session, source_id="memory", topic="deployment workflow")

        assert loaded.to_dict() == record.to_dict()

    def test_lookup_by_slug_also_works(self, mock_sdk):
        store = build(mock_sdk)
        session = make_session()
        store.write_topic(session, make_record(), source_id="memory")

        loaded = store.get_topic(session, source_id="memory", topic="deployment-workflow")
        assert loaded.topic == "deployment workflow"

    def test_mutating_the_returned_record_cannot_corrupt_the_store(self, mock_sdk):
        store = build(mock_sdk)
        session = make_session()
        store.write_topic(session, make_record(memories=["one"]), source_id="memory")

        first = store.get_topic(session, source_id="memory", topic="deployment workflow")
        first.memories.append("injected")

        second = store.get_topic(session, source_id="memory", topic="deployment workflow")
        assert list(second.memories) == ["one"]

    def test_missing_topic_raises_file_not_found(self, mock_sdk):
        store = build(mock_sdk)
        with pytest.raises(FileNotFoundError):
            store.get_topic(make_session(), source_id="memory", topic="nothing here")

    def test_rewrite_replaces_rather_than_appends(self, mock_sdk):
        store = build(mock_sdk)
        session = make_session()
        store.write_topic(session, make_record(memories=["one"]), source_id="memory")
        store.write_topic(session, make_record(memories=["one", "two"]), source_id="memory")

        loaded = store.get_topic(session, source_id="memory", topic="deployment workflow")
        assert list(loaded.memories) == ["one", "two"]
        assert len(store.list_topics(session, source_id="memory")) == 1


class TestSynapWrites:
    def test_write_topic_submits_content_to_synap(self, mock_sdk):
        store = build(mock_sdk)
        store.write_topic(make_session(), make_record(), source_id="memory")

        mock_sdk.memories.create.assert_awaited_once()
        kwargs = mock_sdk.memories.create.await_args.kwargs
        assert "Tags a release" in kwargs["document"]
        assert kwargs["user_id"] == "alice"
        assert kwargs["customer_id"] == "acme"

    def test_content_is_submitted_as_prose_not_json(self, mock_sdk):
        """A JSON envelope adds key names the extractor stores as noise."""
        store = build(mock_sdk)
        store.write_topic(make_session(), make_record(), source_id="memory")

        document = mock_sdk.memories.create.await_args.kwargs["document"]
        assert not document.lstrip().startswith("{")
        assert '"memories"' not in document

    def test_write_failure_raises_but_keeps_the_record(self, failing_sdk):
        from synap_integrations_common import SynapIntegrationError

        store = build(failing_sdk)
        session = make_session()

        with pytest.raises(SynapIntegrationError):
            store.write_topic(session, make_record(), source_id="memory")

        # The record landed before the network call, so the agent's next turn
        # still sees what it just wrote.
        loaded = store.get_topic(session, source_id="memory", topic="deployment workflow")
        assert loaded.topic == "deployment workflow"

    def test_a_topic_with_no_memory_lines_yet_still_submits(self, mock_sdk):
        """A record created before its first memory line lands is a real state.

        MAF will not let it be empty: it rejects a blank topic outright and
        substitutes ``"No summary yet."`` for a blank summary. So there is
        always something to submit, and no guard is needed against nothing.
        """
        store = build(mock_sdk)
        store.write_topic(make_session(), make_record(memories=[]), source_id="memory")

        document = mock_sdk.memories.create.await_args.kwargs["document"]
        assert document.startswith("deployment workflow:")
        assert "\n" not in document  # no memory lines to append yet


class TestDelete:
    def test_delete_removes_record_and_synap_memories(self, mock_sdk):
        wire_ingestion(mock_sdk, ["mem-1", "mem-2"])
        store = build(mock_sdk)
        session = make_session()
        store.write_topic(session, make_record(), source_id="memory")

        store.delete_topic(session, source_id="memory", topic="deployment workflow")

        assert mock_sdk.memories.delete.await_count == 2
        with pytest.raises(FileNotFoundError):
            store.get_topic(session, source_id="memory", topic="deployment workflow")

    def test_deleting_an_unknown_topic_raises(self, mock_sdk):
        store = build(mock_sdk)
        with pytest.raises(FileNotFoundError):
            store.delete_topic(make_session(), source_id="memory", topic="nope")

    def test_delete_survives_a_failing_status_call(self, mock_sdk):
        """A purge failure must not leave the record half-deleted."""
        wire_ingestion(mock_sdk)
        mock_sdk.memories.status = AsyncMock(side_effect=RuntimeError("boom"))
        store = build(mock_sdk)
        session = make_session()
        store.write_topic(session, make_record(), source_id="memory")

        store.delete_topic(session, source_id="memory", topic="deployment workflow")

        with pytest.raises(FileNotFoundError):
            store.get_topic(session, source_id="memory", topic="deployment workflow")


class TestIndex:
    def test_index_carries_pointer_lines_and_the_recall_block(self, mock_sdk):
        store = build(mock_sdk)
        session = make_session()
        store.write_topic(session, make_record(), source_id="memory")

        text = store.get_index_text(
            session, source_id="memory", line_limit=200, line_length=150
        )

        assert "deployment workflow" in text
        assert store.recall_header in text
        assert "User is an engineer" in text  # from the shared mock's fetch

    def test_recall_block_can_be_turned_off(self, mock_sdk):
        store = build(mock_sdk, include_recall=False)
        text = store.get_index_text(
            make_session(), source_id="memory", line_limit=200, line_length=150
        )
        assert store.recall_header not in text
        mock_sdk.fetch.assert_not_awaited()

    def test_index_degrades_instead_of_raising(self, failing_sdk):
        """This text goes into the system prompt on every run."""
        store = build(failing_sdk)
        text = store.get_index_text(
            make_session(), source_id="memory", line_limit=200, line_length=150
        )
        assert "Memory Index" in text
        assert store.recall_header not in text

    def test_empty_index_says_so(self, mock_sdk):
        store = build(mock_sdk, include_recall=False)
        text = store.get_index_text(
            make_session(), source_id="memory", line_limit=200, line_length=150
        )
        assert "(no topics yet)" in text

    def test_rebuild_index_honours_the_line_limit(self, mock_sdk):
        store = build(mock_sdk)
        session = make_session()
        for index in range(5):
            store.write_topic(session, make_record(topic=f"topic {index}"), source_id="memory")

        entries = store.rebuild_index(
            session, source_id="memory", line_limit=3, line_length=150
        )
        assert len(entries) == 3

    def test_rebuild_index_writes_nothing(self, mock_sdk):
        """The index is a projection of the records, so it cannot go stale."""
        store = build(mock_sdk)
        store.rebuild_index(make_session(), source_id="memory", line_limit=10, line_length=150)
        mock_sdk.memories.create.assert_not_awaited()


class TestState:
    def test_state_round_trips(self, mock_sdk):
        store = build(mock_sdk)
        session = make_session()
        store.write_state(
            session,
            {"last_consolidated_at": "2026-08-07T00:00:00+00:00", "sessions_since_consolidation": ["a"]},
            source_id="memory",
        )
        state = store.read_state(session, source_id="memory")
        assert state["last_consolidated_at"] == "2026-08-07T00:00:00+00:00"
        assert state["sessions_since_consolidation"] == ["a"]

    def test_unset_state_has_maf_defaults(self, mock_sdk):
        state = build(mock_sdk).read_state(make_session(), source_id="memory")
        assert state == {"last_consolidated_at": None, "sessions_since_consolidation": []}

    def test_corrupt_state_is_repaired_not_propagated(self, mock_sdk):
        store = build(mock_sdk)
        session = make_session()
        store.write_state(
            session,
            {"last_consolidated_at": 17, "sessions_since_consolidation": "not-a-list"},
            source_id="memory",
        )
        state = store.read_state(session, source_id="memory")
        assert state["sessions_since_consolidation"] == []
        assert state["last_consolidated_at"] is None

    def test_state_is_not_sent_to_synap(self, mock_sdk):
        store = build(mock_sdk)
        store.write_state(make_session(), {"sessions_since_consolidation": []}, source_id="memory")
        mock_sdk.memories.create.assert_not_awaited()


class TestSearchTranscripts:
    def test_search_returns_maf_shaped_rows(self, mock_sdk):
        rows = build(mock_sdk).search_transcripts(
            make_session(), source_id="memory", query="what do they prefer"
        )
        assert rows
        assert set(rows[0]) == {"session_id", "line_number", "role", "text"}
        assert rows[0]["line_number"] == 1

    def test_search_passes_the_query_to_synap(self, mock_sdk):
        build(mock_sdk).search_transcripts(
            make_session(), source_id="memory", query="how do they deploy"
        )
        assert mock_sdk.fetch.await_args.kwargs["search_query"] == ["how do they deploy"]

    def test_search_honours_the_limit(self, mock_sdk):
        rows = build(mock_sdk).search_transcripts(
            make_session(), source_id="memory", query="anything", limit=1
        )
        assert len(rows) == 1

    def test_empty_query_raises(self, mock_sdk):
        with pytest.raises(ValueError, match="must not be empty"):
            build(mock_sdk).search_transcripts(make_session(), source_id="memory", query="  ")

    def test_search_degrades_instead_of_raising(self, failing_sdk):
        rows = build(failing_sdk).search_transcripts(
            make_session(), source_id="memory", query="anything"
        )
        assert rows == []


class TestScope:
    def test_owner_defaults_to_the_scope_path(self, mock_sdk):
        assert build(mock_sdk).get_owner_id(make_session()) == "/acme/alice"

    def test_session_state_overrides_the_default_owner(self, mock_sdk):
        store = build(mock_sdk)
        session = make_session(**{store.owner_state_key: "/acme/bob"})
        assert store.get_owner_id(session) == "/acme/bob"

    def test_scope_resolver_wins(self, mock_sdk):
        store = build(mock_sdk, scope_resolver=lambda s: "/tenant/carol")
        assert store.get_owner_id(make_session()) == "/tenant/carol"

    def test_owners_do_not_see_each_others_topics(self, mock_sdk):
        store = build(mock_sdk)
        alice = make_session("s1", **{store.owner_state_key: "/acme/alice"})
        bob = make_session("s2", **{store.owner_state_key: "/acme/bob"})

        store.write_topic(alice, make_record(), source_id="memory")
        assert len(store.list_topics(alice, source_id="memory")) == 1
        assert store.list_topics(bob, source_id="memory") == []

    def test_provider_state_round_trips_the_owner(self, mock_sdk):
        store = build(mock_sdk)
        session = make_session(**{store.owner_state_key: "/acme/bob"})
        exported = store.export_provider_state(session)

        rebuilt = make_session("s9")
        store.import_provider_state(rebuilt, state=exported)
        assert store.get_owner_id(rebuilt) == "/acme/bob"


class TestTranscriptsDirectory:
    def test_path_is_scoped_and_never_created(self, mock_sdk, tmp_path):
        store = build(mock_sdk, transcripts_root=str(tmp_path / "scratch"))
        path = store.get_transcripts_directory(make_session(), source_id="memory")

        assert not path.exists()
        assert "acme" in str(path)

    def test_two_owners_get_different_paths(self, mock_sdk, tmp_path):
        store = build(mock_sdk, transcripts_root=str(tmp_path))
        alice = make_session("s1", **{store.owner_state_key: "/acme/alice"})
        bob = make_session("s2", **{store.owner_state_key: "/acme/bob"})

        assert store.get_transcripts_directory(
            alice, source_id="memory"
        ) != store.get_transcripts_directory(bob, source_id="memory")


class TestRecordStoreSeam:
    def test_a_custom_record_store_is_used(self, mock_sdk):
        from synap_microsoft_agent import InMemoryTopicRecordStore

        shared = InMemoryTopicRecordStore()
        first = build(mock_sdk, record_store=shared)
        second = build(mock_sdk, record_store=shared)

        session = make_session()
        first.write_topic(session, make_record(), source_id="memory")

        # A second store over the same records sees them — which is what makes
        # a durable record store a drop-in upgrade.
        loaded = second.get_topic(session, source_id="memory", topic="deployment workflow")
        assert loaded.topic == "deployment workflow"

    def test_default_record_stores_are_not_shared(self, mock_sdk):
        first = build(mock_sdk)
        second = build(mock_sdk)
        session = make_session()
        first.write_topic(session, make_record(), source_id="memory")

        with pytest.raises(FileNotFoundError):
            second.get_topic(session, source_id="memory", topic="deployment workflow")
