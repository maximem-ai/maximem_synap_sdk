"""Tests for SynapMemoryWriter (Haystack).

Documented error-handling contract (from writer.py docstring):
- Documents with unrecognized roles are skipped (skipped_count increments).
- Partial failures surface in failed_count + first_error; do NOT raise.
- 100% failure raises SynapIntegrationError (propagated from the store).
- Empty document list → all zero counters, no SDK call.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from haystack import Document

from synap_haystack import SynapMemoryWriter, SynapMemoryStore
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_doc(content: str = "hello") -> Document:
    return Document(content=content, meta={"role": "user"})


def _assistant_doc(content: str = "reply") -> Document:
    return Document(content=content, meta={"role": "assistant"})


def _system_doc(content: str = "system msg") -> Document:
    return Document(content=content, meta={"role": "system"})


def _no_role_doc(content: str = "no role") -> Document:
    return Document(content=content, meta={})


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestWriterConstruction:
    def test_raises_without_conversation_id_and_no_store(self, mock_sdk):
        with pytest.raises(ValueError, match="conversation_id"):
            SynapMemoryWriter(sdk=mock_sdk, user_id="u1", customer_id="c1")

    def test_raises_without_sdk_and_no_store(self):
        with pytest.raises(ValueError):
            SynapMemoryWriter(conversation_id="c", user_id="u1")

    def test_raises_without_scope(self, mock_sdk):
        with pytest.raises(ValueError):
            SynapMemoryWriter(sdk=mock_sdk, conversation_id="c")

    def test_accepts_store_with_conversation_id(self, mock_sdk):
        store = SynapMemoryStore(
            mock_sdk, user_id="u1", conversation_id="conv-1"
        )
        w = SynapMemoryWriter(store=store)
        assert w.store is store

    def test_store_conversation_id_used_when_no_explicit(self, mock_sdk):
        store = SynapMemoryStore(
            mock_sdk, user_id="u1", conversation_id="store-conv"
        )
        w = SynapMemoryWriter(store=store)
        assert w.conversation_id == "store-conv"

    def test_explicit_conversation_id_wins_over_store(self, mock_sdk):
        store = SynapMemoryStore(
            mock_sdk, user_id="u1", conversation_id="store-conv"
        )
        w = SynapMemoryWriter(store=store, conversation_id="explicit-conv")
        assert w.conversation_id == "explicit-conv"

    def test_raises_when_store_has_no_conv_id_and_none_passed(self, mock_sdk):
        store = SynapMemoryStore(mock_sdk, user_id="u1")  # no conversation_id
        with pytest.raises(ValueError, match="conversation_id"):
            SynapMemoryWriter(store=store)

    def test_builds_from_sdk_and_scope(self, mock_sdk):
        w = SynapMemoryWriter(
            sdk=mock_sdk, conversation_id="c", user_id="u1", customer_id="c1"
        )
        assert w.store is not None


# ---------------------------------------------------------------------------
# run — happy paths
# ---------------------------------------------------------------------------


class TestWriterRun:
    @pytest.fixture
    def writer(self, mock_sdk):
        return SynapMemoryWriter(
            sdk=mock_sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
        )

    def test_records_user_document(self, writer, mock_sdk):
        result = writer.run(documents=[_user_doc("hello")])
        assert result["written_count"] == 1
        assert result["failed_count"] == 0
        assert result["skipped_count"] == 0

    def test_records_assistant_document(self, writer, mock_sdk):
        result = writer.run(documents=[_assistant_doc("reply")])
        assert result["written_count"] == 1
        assert result["failed_count"] == 0

    def test_records_user_and_assistant(self, writer, mock_sdk):
        result = writer.run(documents=[_user_doc(), _assistant_doc()])
        assert result["written_count"] == 2
        assert mock_sdk.conversation.record_message.await_count == 2

    def test_skips_system_role(self, writer, mock_sdk):
        result = writer.run(documents=[_system_doc()])
        assert result["skipped_count"] == 1
        assert result["written_count"] == 0
        mock_sdk.conversation.record_message.assert_not_awaited()

    def test_skips_missing_role_defaults_to_user(self, writer, mock_sdk):
        """A doc without 'role' in meta defaults role to 'user' (valid), not skipped."""
        # Default role is "user" in writer.py line 91: doc.meta.get("role", "user")
        result = writer.run(documents=[_no_role_doc()])
        # Default role is "user" — should be written, not skipped
        assert result["written_count"] == 1
        assert result["skipped_count"] == 0

    def test_skips_unknown_role(self, writer, mock_sdk):
        doc = Document(content="text", meta={"role": "tool"})
        result = writer.run(documents=[doc])
        assert result["skipped_count"] == 1
        assert result["written_count"] == 0

    def test_mixed_skip_and_write(self, writer, mock_sdk):
        result = writer.run(documents=[_user_doc(), _system_doc()])
        assert result["written_count"] == 1
        assert result["skipped_count"] == 1

    def test_empty_documents_returns_all_zeros(self, writer, mock_sdk):
        result = writer.run(documents=[])
        assert result["written_count"] == 0
        assert result["failed_count"] == 0
        assert result["skipped_count"] == 0
        assert result["first_error"] is None
        mock_sdk.conversation.record_message.assert_not_awaited()

    def test_first_error_is_none_on_success(self, writer, mock_sdk):
        result = writer.run(documents=[_user_doc()])
        assert result["first_error"] is None

    def test_correct_role_forwarded_to_sdk(self, writer, mock_sdk):
        writer.run(documents=[_user_doc("hi")])
        kw = mock_sdk.conversation.record_message.call_args.kwargs
        assert kw["role"] == "user"

    def test_correct_content_forwarded_to_sdk(self, writer, mock_sdk):
        writer.run(documents=[_assistant_doc("specific content")])
        kw = mock_sdk.conversation.record_message.call_args.kwargs
        assert kw["content"] == "specific content"

    def test_conversation_id_forwarded_to_sdk(self, writer, mock_sdk):
        writer.run(documents=[_user_doc()])
        kw = mock_sdk.conversation.record_message.call_args.kwargs
        assert kw["conversation_id"] == "conv-1"


# ---------------------------------------------------------------------------
# run — failure paths
# ---------------------------------------------------------------------------


class TestWriterRunFailure:
    @pytest.fixture
    def writer(self, mock_sdk):
        return SynapMemoryWriter(
            sdk=mock_sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
        )

    def test_total_failure_raises_synap_integration_error(self, writer, mock_sdk):
        mock_sdk.conversation.record_message.side_effect = RuntimeError("sdk down")
        with pytest.raises(SynapIntegrationError):
            writer.run(documents=[_user_doc()])

    def test_partial_failure_records_failed_count(self, writer, mock_sdk):
        mock_sdk.conversation.record_message.side_effect = [
            {"message_id": "m1"},
            RuntimeError("transient"),
        ]
        result = writer.run(documents=[_user_doc(), _assistant_doc()])
        assert result["written_count"] == 1
        assert result["failed_count"] == 1

    def test_partial_failure_first_error_populated(self, writer, mock_sdk):
        mock_sdk.conversation.record_message.side_effect = [
            {"message_id": "m1"},
            RuntimeError("transient boom"),
        ]
        result = writer.run(documents=[_user_doc(), _assistant_doc()])
        assert result["first_error"] is not None
        assert "transient boom" in result["first_error"]

    def test_failing_sdk_raises_integration_error(self, failing_sdk):
        writer = SynapMemoryWriter(
            sdk=failing_sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
        )
        with pytest.raises(SynapIntegrationError):
            writer.run(documents=[_user_doc()])
