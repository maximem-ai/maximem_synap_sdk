"""Tests for SynapMemoryStore (Haystack).

Documented error-handling contract (from store.py docstring):
- Writes raise SynapIntegrationError when *every* recordable message fails.
- Reads degrade gracefully to [] on SDK failure — never crash an agent turn.
- Delete is a no-op (Synap has no public delete API), warns once.
- All five memory types (fact/preference/episode/emotion/temporal_event) are
  mapped to ChatMessage / Document with correct meta fields.
"""

from __future__ import annotations

import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, call

from haystack import Document
from haystack.dataclasses import ChatMessage

from synap_haystack import SynapMemoryStore
from synap_integrations_common import SynapIntegrationError

from synap_integrations_common.testing import (
    make_fact,
    make_preference,
    make_episode,
    make_emotion,
    make_temporal_event,
    make_unified_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sdk():
    sdk = MagicMock()
    sdk.instance_id = "test-instance"
    sdk.fetch = AsyncMock(return_value=make_unified_response())
    sdk.conversation = MagicMock()
    sdk.conversation.record_message = AsyncMock(return_value={"message_id": "m1"})
    return sdk


@pytest.fixture
def store(mock_sdk):
    return SynapMemoryStore(
        mock_sdk, user_id="u1", customer_id="c1", conversation_id="conv-1"
    )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_none_sdk_raises(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapMemoryStore(None, user_id="u1")

    def test_no_scope_raises(self, mock_sdk):
        with pytest.raises(ValueError):
            SynapMemoryStore(mock_sdk)

    def test_user_id_only_is_valid(self, mock_sdk):
        s = SynapMemoryStore(mock_sdk, user_id="u1")
        assert s.user_id == "u1"
        assert s.customer_id == ""

    def test_customer_id_only_is_valid(self, mock_sdk):
        s = SynapMemoryStore(mock_sdk, customer_id="c1")
        assert s.user_id == ""
        assert s.customer_id == "c1"

    def test_defaults(self, mock_sdk):
        s = SynapMemoryStore(mock_sdk, user_id="u1")
        assert s.mode == "accurate"
        assert s.max_results == 20
        assert s.conversation_id is None
        assert s.include_conversation_context is False


# ---------------------------------------------------------------------------
# add_memories — happy paths
# ---------------------------------------------------------------------------


class TestAddMemories:
    def test_records_user_and_assistant(self, store, mock_sdk):
        results = store.add_memories(messages=[
            ChatMessage.from_user("hello"),
            ChatMessage.from_assistant("hi there"),
        ])
        assert mock_sdk.conversation.record_message.await_count == 2
        assert [r["status"] for r in results] == ["written", "written"]
        assert results[0]["message_id"] == "m1"

    def test_skips_system_role(self, store, mock_sdk):
        results = store.add_memories(messages=[ChatMessage.from_system("be nice")])
        assert mock_sdk.conversation.record_message.await_count == 0
        assert results[0]["status"] == "skipped"
        assert results[0]["role"] == "system"

    def test_mixed_skip_and_write(self, store, mock_sdk):
        results = store.add_memories(messages=[
            ChatMessage.from_system("sys"),
            ChatMessage.from_user("hello"),
        ])
        statuses = [r["status"] for r in results]
        assert statuses == ["skipped", "written"]

    def test_per_call_conversation_id_override(self, store, mock_sdk):
        store.add_memories(
            messages=[ChatMessage.from_user("msg")],
            conversation_id="other-conv",
        )
        kw = mock_sdk.conversation.record_message.call_args.kwargs
        assert kw["conversation_id"] == "other-conv"

    def test_per_call_user_id_override(self, store, mock_sdk):
        store.add_memories(
            messages=[ChatMessage.from_user("msg")],
            user_id="override-user",
        )
        kw = mock_sdk.conversation.record_message.call_args.kwargs
        assert kw["user_id"] == "override-user"

    def test_per_call_customer_id_override(self, store, mock_sdk):
        store.add_memories(
            messages=[ChatMessage.from_user("msg")],
            customer_id="override-cust",
        )
        kw = mock_sdk.conversation.record_message.call_args.kwargs
        assert kw["customer_id"] == "override-cust"

    def test_requires_conversation_id_raises_when_none_set(self, mock_sdk):
        s = SynapMemoryStore(mock_sdk, user_id="u1")
        with pytest.raises(ValueError, match="conversation_id"):
            s.add_memories(messages=[ChatMessage.from_user("hello")])

    def test_partial_failure_does_not_raise(self, store, mock_sdk):
        """One written, one failed — partial failure, not total. Must not raise."""
        mock_sdk.conversation.record_message.side_effect = [
            {"message_id": "m1"},
            RuntimeError("transient"),
        ]
        results = store.add_memories(messages=[
            ChatMessage.from_user("hello"),
            ChatMessage.from_assistant("reply"),
        ])
        statuses = [r["status"] for r in results]
        assert "written" in statuses
        assert "failed" in statuses

    def test_failed_result_includes_error_string(self, store, mock_sdk):
        mock_sdk.conversation.record_message.side_effect = [
            {"message_id": "m1"},
            RuntimeError("boom"),
        ]
        results = store.add_memories(messages=[
            ChatMessage.from_user("ok"),
            ChatMessage.from_assistant("fail"),
        ])
        failed = next(r for r in results if r["status"] == "failed")
        assert "error" in failed
        assert "boom" in failed["error"]


# ---------------------------------------------------------------------------
# add_memories — failure path
# ---------------------------------------------------------------------------


class TestAddMemoriesFailure:
    def test_total_failure_raises_integration_error(self, store, mock_sdk):
        mock_sdk.conversation.record_message.side_effect = RuntimeError("sdk down")
        with pytest.raises(SynapIntegrationError):
            store.add_memories(messages=[ChatMessage.from_user("hello")])

    def test_all_system_messages_no_recordable_no_raise(self, store, mock_sdk):
        """All messages are system role → none recordable → no raise."""
        results = store.add_memories(messages=[
            ChatMessage.from_system("sys1"),
            ChatMessage.from_system("sys2"),
        ])
        assert all(r["status"] == "skipped" for r in results)
        mock_sdk.conversation.record_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# search_memories — happy paths (all five memory types)
# ---------------------------------------------------------------------------


class TestSearchMemories:
    def _make_response(self, **kw):
        defaults = dict(
            facts=[], preferences=[], episodes=[], emotions=[], temporal_events=[],
            scope_map={},
        )
        defaults.update(kw)
        return MagicMock(**defaults)

    def test_returns_chat_messages(self, store, mock_sdk):
        fact = MagicMock(content="likes coffee", id="f1", confidence=0.9)
        mock_sdk.fetch.return_value = self._make_response(
            facts=[fact], scope_map={"f1": "user"}
        )
        msgs = store.search_memories(query="coffee")
        assert len(msgs) == 1
        assert isinstance(msgs[0], ChatMessage)
        assert msgs[0].text == "likes coffee"
        assert msgs[0].meta["type"] == "fact"
        assert msgs[0].meta["id"] == "f1"
        assert msgs[0].meta["confidence"] == 0.9
        assert msgs[0].meta["scope"] == "user"
        assert msgs[0].role.value == "assistant"

    def test_preference_mapped_correctly(self, store, mock_sdk):
        pref = MagicMock(content="dark mode", id="p1", strength=0.8)
        mock_sdk.fetch.return_value = self._make_response(
            preferences=[pref], scope_map={"p1": "user"}
        )
        msgs = store.search_memories(query="mode")
        assert msgs[0].meta["type"] == "preference"
        assert msgs[0].meta["strength"] == 0.8
        assert msgs[0].text == "dark mode"

    def test_episode_mapped_correctly(self, store, mock_sdk):
        ep = MagicMock(summary="Had a support call", id="e1", significance=0.7)
        mock_sdk.fetch.return_value = self._make_response(
            episodes=[ep], scope_map={"e1": "user"}
        )
        msgs = store.search_memories(query="support")
        assert msgs[0].meta["type"] == "episode"
        assert msgs[0].meta["significance"] == 0.7
        assert msgs[0].text == "Had a support call"

    def test_emotion_mapped_correctly(self, store, mock_sdk):
        em = MagicMock(emotion_type="frustrated", context="Long wait", id="em1", intensity=0.6)
        mock_sdk.fetch.return_value = self._make_response(
            emotions=[em], scope_map={"em1": "user"}
        )
        msgs = store.search_memories(query="emotion")
        assert msgs[0].meta["type"] == "emotion"
        assert msgs[0].meta["intensity"] == 0.6
        assert "frustrated" in msgs[0].text
        assert "Long wait" in msgs[0].text

    def test_temporal_event_mapped_correctly(self, store, mock_sdk):
        te = MagicMock(content="Trial expires April 15", id="t1")
        mock_sdk.fetch.return_value = self._make_response(
            temporal_events=[te], scope_map={"t1": "user"}
        )
        msgs = store.search_memories(query="trial")
        assert msgs[0].meta["type"] == "temporal_event"
        assert msgs[0].text == "Trial expires April 15"

    def test_all_five_types_in_single_response(self, store, mock_sdk):
        mock_sdk.fetch.return_value = self._make_response(
            facts=[MagicMock(content="f", id="f1", confidence=0.9)],
            preferences=[MagicMock(content="p", id="p1", strength=0.8)],
            episodes=[MagicMock(summary="e", id="e1", significance=0.7)],
            emotions=[MagicMock(emotion_type="calm", context="ok", id="em1", intensity=0.5)],
            temporal_events=[MagicMock(content="t", id="t1")],
            scope_map={"f1": "u", "p1": "u", "e1": "u", "em1": "u", "t1": "u"},
        )
        msgs = store.search_memories(query="all")
        assert len(msgs) == 5
        types = {m.meta["type"] for m in msgs}
        assert types == {"fact", "preference", "episode", "emotion", "temporal_event"}

    def test_empty_response_returns_empty_list(self, store, mock_sdk):
        mock_sdk.fetch.return_value = self._make_response()
        assert store.search_memories(query="nothing") == []

    def test_item_missing_from_scope_map_gets_empty_scope(self, store, mock_sdk):
        fact = MagicMock(content="orphan fact", id="f99", confidence=0.5)
        mock_sdk.fetch.return_value = self._make_response(
            facts=[fact], scope_map={}  # f99 absent
        )
        msgs = store.search_memories(query="q")
        assert msgs[0].meta["scope"] == ""

    def test_per_call_max_results_override(self, store, mock_sdk):
        mock_sdk.fetch.return_value = self._make_response()
        store.search_memories(query="x", max_results=5)
        kw = mock_sdk.fetch.call_args.kwargs
        assert kw["max_results"] == 5

    def test_per_call_mode_override(self, store, mock_sdk):
        mock_sdk.fetch.return_value = self._make_response()
        store.search_memories(query="x", mode="fast")
        kw = mock_sdk.fetch.call_args.kwargs
        assert kw["mode"] == "fast"


# ---------------------------------------------------------------------------
# search_memories — failure path
# ---------------------------------------------------------------------------


class TestSearchMemoriesFailure:
    def test_degrades_to_empty_on_sdk_error(self, store, mock_sdk):
        mock_sdk.fetch.side_effect = RuntimeError("down")
        assert store.search_memories(query="x") == []

    def test_logs_error_on_sdk_failure(self, store, mock_sdk, caplog):
        mock_sdk.fetch.side_effect = RuntimeError("explosion")
        with caplog.at_level(logging.ERROR, logger="synap_haystack.store"):
            store.search_memories(query="x")
        assert any("explosion" in r.message or "explosion" in str(r.exc_info)
                   for r in caplog.records)


# ---------------------------------------------------------------------------
# search_memories_as_single_message
# ---------------------------------------------------------------------------


class TestSearchAsSingleMessage:
    def _make_response(self, **kw):
        defaults = dict(
            facts=[], preferences=[], episodes=[], emotions=[], temporal_events=[],
            scope_map={},
        )
        defaults.update(kw)
        return MagicMock(**defaults)

    def test_returns_system_message_with_all_items(self, store, mock_sdk):
        fact = MagicMock(content="likes coffee", id="f1", confidence=0.9)
        pref = MagicMock(content="prefers dark mode", id="p1", strength=0.8)
        mock_sdk.fetch.return_value = self._make_response(
            facts=[fact], preferences=[pref], scope_map={}
        )
        msg = store.search_memories_as_single_message(query="x")
        assert msg is not None
        assert msg.role.value == "system"
        assert "likes coffee" in msg.text
        assert "prefers dark mode" in msg.text

    def test_returns_none_when_empty(self, store, mock_sdk):
        mock_sdk.fetch.return_value = self._make_response()
        assert store.search_memories_as_single_message(query="x") is None

    def test_degrades_to_none_on_sdk_error(self, store, mock_sdk):
        mock_sdk.fetch.side_effect = RuntimeError("down")
        # SDK error degrades → empty records → None
        result = store.search_memories_as_single_message(query="x")
        assert result is None

    def test_formatted_as_relevant_memory_preamble(self, store, mock_sdk):
        fact = MagicMock(content="engineer", id="f1", confidence=0.9)
        mock_sdk.fetch.return_value = self._make_response(facts=[fact])
        msg = store.search_memories_as_single_message(query="x")
        assert "Relevant memory" in msg.text


# ---------------------------------------------------------------------------
# search_documents (RAG path)
# ---------------------------------------------------------------------------


class TestSearchDocuments:
    def _make_response(self, **kw):
        defaults = dict(
            facts=[], preferences=[], episodes=[], emotions=[], temporal_events=[],
            scope_map={},
        )
        defaults.update(kw)
        return MagicMock(**defaults)

    def test_returns_haystack_documents(self, store, mock_sdk):
        fact = MagicMock(content="likes coffee", id="f1", confidence=0.9)
        mock_sdk.fetch.return_value = self._make_response(
            facts=[fact], scope_map={"f1": "user"}
        )
        docs = store.search_documents(query="coffee")
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].content == "likes coffee"
        assert docs[0].meta["type"] == "fact"
        assert docs[0].meta["id"] == "f1"
        assert docs[0].meta["scope"] == "user"

    def test_all_five_types_as_documents(self, store, mock_sdk):
        mock_sdk.fetch.return_value = self._make_response(
            facts=[MagicMock(content="f", id="f1", confidence=0.9)],
            preferences=[MagicMock(content="p", id="p1", strength=0.8)],
            episodes=[MagicMock(summary="e", id="e1", significance=0.7)],
            emotions=[MagicMock(emotion_type="calm", context="ok", id="em1", intensity=0.5)],
            temporal_events=[MagicMock(content="t", id="t1")],
            scope_map={},
        )
        docs = store.search_documents(query="all")
        assert len(docs) == 5
        types = {d.meta["type"] for d in docs}
        assert types == {"fact", "preference", "episode", "emotion", "temporal_event"}

    def test_degrades_to_empty_on_sdk_error(self, store, mock_sdk):
        mock_sdk.fetch.side_effect = RuntimeError("down")
        assert store.search_documents(query="x") == []

    def test_empty_response_returns_empty_list(self, store, mock_sdk):
        mock_sdk.fetch.return_value = self._make_response()
        assert store.search_documents(query="nothing") == []


# ---------------------------------------------------------------------------
# delete (no-op, warns once)
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_memory_is_noop(self, store):
        store.delete_memory("m1")  # must not raise

    def test_delete_all_memories_is_noop(self, store):
        store.delete_all_memories()  # must not raise

    def test_warns_once_on_first_delete(self, store, caplog):
        with caplog.at_level(logging.WARNING, logger="synap_haystack.store"):
            store.delete_memory("m1")
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) >= 1

    def test_warns_only_once_across_multiple_calls(self, store, caplog):
        with caplog.at_level(logging.WARNING, logger="synap_haystack.store"):
            store.delete_memory("m1")
            store.delete_memory("m2")
            store.delete_all_memories()
        # The _delete_warned flag gates repeated warnings — exactly 1 warning
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1


# ---------------------------------------------------------------------------
# Serialization — to_dict / from_dict
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_omits_sdk(self, store):
        d = store.to_dict()
        params = d["init_parameters"]
        assert "sdk" not in params
        assert params["instance_id"] == "test-instance"
        assert params["user_id"] == "u1"
        assert params["customer_id"] == "c1"
        assert params["conversation_id"] == "conv-1"
        assert params["mode"] == "accurate"
        assert params["max_results"] == 20

    def test_to_dict_round_trip(self, store, mock_sdk, monkeypatch):
        """from_dict re-builds a SynapMemoryStore with correct scalar fields."""
        from maximem_synap import MaximemSynapSDK
        # Patch MaximemSynapSDK construction so we don't need a real instance
        fake_sdk = mock_sdk
        monkeypatch.setattr(
            "synap_haystack.store.MaximemSynapSDK",
            lambda **kw: fake_sdk,
        )
        d = store.to_dict()
        restored = SynapMemoryStore.from_dict(d)
        assert restored.user_id == store.user_id
        assert restored.customer_id == store.customer_id
        assert restored.conversation_id == store.conversation_id
        assert restored.mode == store.mode
        assert restored.max_results == store.max_results

    def test_to_dict_type_field(self, store):
        d = store.to_dict()
        assert "type" in d
        assert "SynapMemoryStore" in d["type"]

    def test_user_id_none_in_to_dict_when_empty(self, mock_sdk):
        """When user_id is empty, to_dict stores None (not empty string)."""
        s = SynapMemoryStore(mock_sdk, customer_id="c1")
        d = s.to_dict()
        assert d["init_parameters"]["user_id"] is None


# ---------------------------------------------------------------------------
# Shared harness integration — mock_sdk and failing_sdk fixtures
# ---------------------------------------------------------------------------


class TestSharedHarness:
    def test_mock_sdk_fixture_works_for_search(self, mock_sdk):
        store = SynapMemoryStore(mock_sdk, user_id="u1", customer_id="c1")
        # make_unified_response has facts + preferences
        msgs = store.search_memories(query="test")
        assert len(msgs) >= 1

    def test_failing_sdk_fixture_read_degrades(self, failing_sdk):
        store = SynapMemoryStore(failing_sdk, user_id="u1", customer_id="c1")
        assert store.search_memories(query="x") == []

    def test_failing_sdk_fixture_write_raises(self, failing_sdk):
        store = SynapMemoryStore(
            failing_sdk, user_id="u1", customer_id="c1", conversation_id="conv-1"
        )
        with pytest.raises(SynapIntegrationError):
            store.add_memories(messages=[ChatMessage.from_user("hello")])
