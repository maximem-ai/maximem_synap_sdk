"""Tests for SynapStore — LangGraph BaseStore backed by Synap semantic memory.

Documented error-handling contracts (from store.py docstring):
- Writes (put) surface SynapIntegrationError on SDK failure — silent drops
  would hide ingestion outages.
- Reads (get / search) degrade gracefully — return None/[] with an ERROR log.
- delete / list_namespaces are no-ops (Synap has no delete API); each warns once.
- semantic_fallback=True (default): when markers are stripped, search falls back
  to scope-filtered results Synap ranked rather than returning empty.
- record_message is best-effort: SDK failure is logged and swallowed.

Coverage shape:
- SynapStore construction (user_id scopes, customer_id-only, errors)
- abatch / batch dispatching (PutOp, GetOp, SearchOp, ListNamespacesOp, unknown)
- PutOp (put): happy path + SDK failure → SynapIntegrationError
- PutOp with value=None (delete): no-op + warning fires once
- GetOp (get): match on namespace+key, miss, markers stripped → None
- SearchOp (search): marker match, namespace prefix filter, metadata filter,
  score propagation, marker-stripped fallback, semantic_fallback=False
- ListNamespacesOp: always empty + warning fires once
- arecord_message / record_message: happy path, no user_id skip, SDK failure swallowed
- batch (sync wrapper): put + get
- all_items callable path in _iter_items
- _filter_matches: equality, list membership, empty filter
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchItem,
    SearchOp,
)

from synap_integrations_common import SynapIntegrationError
from synap_langgraph.store import (
    SynapStore,
    _filter_matches,
    _iter_items,
    _matches_namespace_prefix,
    _ns_str,
    _parse_value,
    _KEY,
    _MARKER,
    _NS,
)


# ---------------------------------------------------------------------------
# Helpers — build fake SDK + fake memory responses without live services
# ---------------------------------------------------------------------------


def _make_sdk(fetch_items=None, create_raise=None, fetch_raise=None):
    """Construct a minimal SDK mock for SynapStore tests.

    Args:
        fetch_items: list of fake item objects to return from sdk.fetch.
                     Defaults to an empty response if None.
        create_raise: exception instance to raise from sdk.memories.create.
        fetch_raise: exception instance to raise from sdk.fetch.
    """
    sdk = MagicMock()
    sdk.memories = MagicMock()

    if create_raise is not None:
        sdk.memories.create = AsyncMock(side_effect=create_raise)
    else:
        sdk.memories.create = AsyncMock(return_value=MagicMock(ingestion_id="ing-001"))

    if fetch_raise is not None:
        sdk.fetch = AsyncMock(side_effect=fetch_raise)
    else:
        sdk.fetch = AsyncMock(return_value=_make_response(fetch_items or []))

    sdk.conversation = MagicMock()
    sdk.conversation.record_message = AsyncMock(return_value={"message_id": "msg-1"})
    return sdk


def _make_response(items):
    """Build a bare Python object (not MagicMock) so _iter_items uses the
    bucket path rather than the ``all_items`` callable path.

    A MagicMock's ``all_items`` attribute is always truthy+callable, which
    would shadow our bucket-union path. Using a plain class avoids that.
    """
    class R:
        def __init__(self, its):
            self.facts = [i for i in its if getattr(i, "_bucket", "facts") == "facts"]
            self.preferences = [i for i in its if getattr(i, "_bucket", "facts") == "preferences"]
            self.episodes = [i for i in its if getattr(i, "_bucket", "facts") == "episodes"]
            self.emotions = [i for i in its if getattr(i, "_bucket", "facts") == "emotions"]
            self.temporal_events = [i for i in its if getattr(i, "_bucket", "facts") == "temporal_events"]

    return R(items)


def _make_item(
    ns: str,
    key: str,
    content: str = '{"val": 1}',
    confidence: float = 0.9,
    bucket: str = "facts",
    strip_markers: bool = False,
):
    """Fake memory item with the SynapStore namespace/key markers set."""
    item = MagicMock()
    item.content = content
    item.summary = None
    item.extracted_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    item.confidence = confidence
    item.id = f"item_{key}"
    item._bucket = bucket  # used by _make_response to route to the right list
    if strip_markers:
        item.metadata = {}
    else:
        item.metadata = {_MARKER: True, _NS: ns, _KEY: key}
    return item


def _make_store(**kwargs):
    """Convenience: build a SynapStore with sdk pre-configured.

    Keyword args except ``sdk``, ``user_id``, ``customer_id`` are passed
    through to the SynapStore constructor unchanged.
    """
    sdk = kwargs.pop("sdk", None) or _make_sdk()
    user_id = kwargs.pop("user_id", "u1")
    customer_id = kwargs.pop("customer_id", "c1")
    return SynapStore(sdk, user_id=user_id, customer_id=customer_id, **kwargs)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_raises_on_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapStore(None, user_id="u1")  # type: ignore[arg-type]

    def test_raises_without_user_or_customer_id(self):
        sdk = _make_sdk()
        with pytest.raises(ValueError, match="at least one of user_id"):
            SynapStore(sdk)

    def test_user_id_only_is_valid(self):
        sdk = _make_sdk()
        store = SynapStore(sdk, user_id="u1")
        assert store.user_id == "u1"
        assert store._scopes == ["user"]

    def test_customer_id_only_is_valid_and_customer_scoped(self):
        sdk = _make_sdk()
        store = SynapStore(sdk, customer_id="c1")
        assert store.user_id == ""
        assert store._scopes == ["customer"]

    def test_both_ids_is_user_scoped(self):
        sdk = _make_sdk()
        store = SynapStore(sdk, user_id="u1", customer_id="c1")
        assert store._scopes == ["user"]

    def test_defaults(self):
        sdk = _make_sdk()
        store = SynapStore(sdk, user_id="u1")
        assert store.mode == "accurate"
        assert store.semantic_fallback is True
        assert store.include_conversation_context is False

    def test_initial_warn_flags_are_false(self):
        sdk = _make_sdk()
        store = SynapStore(sdk, user_id="u1")
        assert not store._delete_warned
        assert not store._listns_warned
        assert not store._marker_warned


# ---------------------------------------------------------------------------
# PutOp (write) — happy path
# ---------------------------------------------------------------------------


class TestPutOp:
    @pytest.mark.asyncio
    async def test_put_calls_memories_create(self):
        sdk = _make_sdk()
        store = _make_store(sdk=sdk)
        await store.abatch([PutOp(namespace=("agents",), key="k1", value={"data": 42})])
        sdk.memories.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_put_encodes_value_as_json_document(self):
        sdk = _make_sdk()
        store = _make_store(sdk=sdk)
        await store.abatch([PutOp(namespace=("a", "b"), key="k2", value={"hello": "world"})])
        doc = json.loads(sdk.memories.create.call_args.kwargs["document"])
        assert doc == {"hello": "world"}

    @pytest.mark.asyncio
    async def test_put_stamps_namespace_and_key_in_metadata(self):
        sdk = _make_sdk()
        store = _make_store(sdk=sdk)
        await store.abatch([PutOp(namespace=("ns",), key="k3", value={})])
        meta = sdk.memories.create.call_args.kwargs["metadata"]
        assert meta[_MARKER] is True
        assert meta[_NS] == "ns"
        assert meta[_KEY] == "k3"

    @pytest.mark.asyncio
    async def test_put_passes_user_and_customer_ids(self):
        sdk = _make_sdk()
        store = SynapStore(sdk, user_id="alice", customer_id="acme")
        await store.abatch([PutOp(namespace=("x",), key="y", value={"v": 1})])
        kw = sdk.memories.create.call_args.kwargs
        assert kw["user_id"] == "alice"
        assert kw["customer_id"] == "acme"

    @pytest.mark.asyncio
    async def test_put_user_id_only_passes_none_customer_id(self):
        sdk = _make_sdk()
        store = SynapStore(sdk, user_id="u1")
        await store.abatch([PutOp(namespace=("x",), key="y", value={})])
        kw = sdk.memories.create.call_args.kwargs
        assert kw["customer_id"] is None

    @pytest.mark.asyncio
    async def test_put_namespace_tuple_stringified_correctly(self):
        sdk = _make_sdk()
        store = _make_store(sdk=sdk)
        await store.abatch([PutOp(namespace=("a", "b", "c"), key="k", value={})])
        meta = sdk.memories.create.call_args.kwargs["metadata"]
        assert meta[_NS] == "a/b/c"

    @pytest.mark.asyncio
    async def test_put_returns_none_result(self):
        sdk = _make_sdk()
        store = _make_store(sdk=sdk)
        results = await store.abatch([PutOp(namespace=("ns",), key="k", value={"x": 1})])
        assert results == [None]

    @pytest.mark.asyncio
    async def test_put_sdk_failure_raises_synap_integration_error(self):
        sdk = _make_sdk(create_raise=RuntimeError("network error"))
        store = _make_store(sdk=sdk)
        with pytest.raises(SynapIntegrationError) as ei:
            await store.abatch([PutOp(namespace=("ns",), key="k", value={"x": 1})])
        assert ei.value.operation == "langgraph.store.put"
        assert isinstance(ei.value.__cause__, RuntimeError)

    def test_sync_batch_put(self):
        """Sync batch delegates to abatch (round-trips through run_async)."""
        sdk = _make_sdk()
        store = _make_store(sdk=sdk)
        results = store.batch([PutOp(namespace=("ns",), key="k", value={"x": 1})])
        assert results == [None]
        sdk.memories.create.assert_called_once()


# ---------------------------------------------------------------------------
# PutOp with value=None (delete) — no-op + warning
# ---------------------------------------------------------------------------


class TestDeleteOp:
    @pytest.mark.asyncio
    async def test_delete_is_noop_does_not_call_sdk(self, caplog):
        sdk = _make_sdk()
        store = _make_store(sdk=sdk)
        with caplog.at_level(logging.WARNING, logger="synap_langgraph.store"):
            results = await store.abatch([PutOp(namespace=("ns",), key="k", value=None)])
        # Returns None (no crash)
        assert results == [None]
        # Did NOT attempt to create a memory
        sdk.memories.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_warns_once_on_first_call(self, caplog):
        sdk = _make_sdk()
        store = _make_store(sdk=sdk)
        with caplog.at_level(logging.WARNING, logger="synap_langgraph.store"):
            await store.abatch([PutOp(namespace=("ns",), key="k", value=None)])
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "delete" in warnings[0].message.lower()

    @pytest.mark.asyncio
    async def test_delete_warning_fires_exactly_once_on_repeated_calls(self, caplog):
        sdk = _make_sdk()
        store = _make_store(sdk=sdk)
        with caplog.at_level(logging.WARNING, logger="synap_langgraph.store"):
            await store.abatch([PutOp(namespace=("ns",), key="k", value=None)])
            await store.abatch([PutOp(namespace=("ns",), key="other", value=None)])
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1  # exactly once
        assert store._delete_warned is True


# ---------------------------------------------------------------------------
# GetOp (get) — happy path + failure path
# ---------------------------------------------------------------------------


class TestGetOp:
    @pytest.mark.asyncio
    async def test_get_returns_item_when_marker_and_ns_key_match(self):
        item = _make_item("agents/u1", "prefs", content='{"dark_mode": true}')
        sdk = _make_sdk(fetch_items=[item])
        store = SynapStore(sdk, user_id="u1", customer_id="c1")
        results = await store.abatch([GetOp(namespace=("agents", "u1"), key="prefs")])
        assert results[0] is not None
        assert isinstance(results[0], Item)
        assert results[0].key == "prefs"
        assert results[0].value == {"dark_mode": True}

    @pytest.mark.asyncio
    async def test_get_returns_none_when_key_not_in_results(self):
        item = _make_item("agents/u1", "other_key")
        sdk = _make_sdk(fetch_items=[item])
        store = _make_store(sdk=sdk)
        results = await store.abatch([GetOp(namespace=("agents", "u1"), key="missing")])
        assert results[0] is None

    @pytest.mark.asyncio
    async def test_get_returns_none_when_no_results(self):
        sdk = _make_sdk(fetch_items=[])
        store = _make_store(sdk=sdk)
        results = await store.abatch([GetOp(namespace=("ns",), key="k")])
        assert results[0] is None

    @pytest.mark.asyncio
    async def test_get_matches_preference_via_summary_field(self):
        """Preferences use .summary rather than .content for text."""
        pref = _make_item("ns", "pref-k", bucket="preferences")
        pref.content = None
        pref.summary = "User prefers dark mode"
        pref.metadata = {_MARKER: True, _NS: "ns", _KEY: "pref-k"}
        sdk = _make_sdk(fetch_items=[pref])
        store = _make_store(sdk=sdk)
        results = await store.abatch([GetOp(namespace=("ns",), key="pref-k")])
        assert results[0] is not None
        # summary text goes through _parse_value as raw (non-JSON) → {"_raw": ...}
        assert results[0].value == {"_raw": "User prefers dark mode"}

    @pytest.mark.asyncio
    async def test_get_namespace_tuple_reconstructed_on_item(self):
        item = _make_item("a/b/c", "k1")
        sdk = _make_sdk(fetch_items=[item])
        store = _make_store(sdk=sdk)
        results = await store.abatch([GetOp(namespace=("a", "b", "c"), key="k1")])
        assert results[0] is not None
        assert results[0].namespace == ("a", "b", "c")

    @pytest.mark.asyncio
    async def test_get_returns_none_and_warns_when_markers_stripped(self, caplog):
        """When items exist but carry no SynapStore markers, get returns None
        (no exact key resolution possible) and emits a WARNING once."""
        item = _make_item("ns", "k", strip_markers=True)
        sdk = _make_sdk(fetch_items=[item])
        store = _make_store(sdk=sdk)
        with caplog.at_level(logging.WARNING, logger="synap_langgraph.store"):
            results = await store.abatch([GetOp(namespace=("ns",), key="k")])
        assert results[0] is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) >= 1
        assert store._marker_warned is True

    @pytest.mark.asyncio
    async def test_get_sdk_failure_degrades_to_none(self, caplog):
        """Read-side SDK failure must never crash an agent turn — return None."""
        sdk = _make_sdk(fetch_raise=RuntimeError("fetch outage"))
        store = _make_store(sdk=sdk)
        with caplog.at_level(logging.ERROR, logger="synap_langgraph.store"):
            results = await store.abatch([GetOp(namespace=("ns",), key="k")])
        assert results[0] is None
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) >= 1


# ---------------------------------------------------------------------------
# SearchOp (search) — happy path + failure path
# ---------------------------------------------------------------------------


class TestSearchOp:
    @pytest.mark.asyncio
    async def test_search_returns_matching_items(self):
        item = _make_item("agents/u1", "k1", confidence=0.85)
        sdk = _make_sdk(fetch_items=[item])
        store = SynapStore(sdk, user_id="u1", customer_id="c1")
        results = await store.abatch(
            [SearchOp(namespace_prefix=("agents",), query="test")]
        )
        assert len(results[0]) == 1
        si = results[0][0]
        assert isinstance(si, SearchItem)
        assert si.score == 0.85

    @pytest.mark.asyncio
    async def test_search_propagates_confidence_as_score(self):
        item = _make_item("ns", "k", confidence=0.72)
        sdk = _make_sdk(fetch_items=[item])
        store = _make_store(sdk=sdk)
        results = await store.abatch(
            [SearchOp(namespace_prefix=("ns",), query="q")]
        )
        assert results[0][0].score == 0.72

    @pytest.mark.asyncio
    async def test_search_none_confidence_gives_none_score(self):
        item = _make_item("ns", "k")
        item.confidence = None
        sdk = _make_sdk(fetch_items=[item])
        store = _make_store(sdk=sdk)
        results = await store.abatch(
            [SearchOp(namespace_prefix=("ns",), query="q")]
        )
        assert results[0][0].score is None

    @pytest.mark.asyncio
    async def test_search_filters_by_namespace_prefix(self):
        """Items outside the namespace prefix are excluded."""
        inside = _make_item("agents/u1", "in-ns")
        outside = _make_item("tools/u1", "out-ns")
        sdk = _make_sdk(fetch_items=[inside, outside])
        store = _make_store(sdk=sdk)
        results = await store.abatch(
            [SearchOp(namespace_prefix=("agents",), query="q")]
        )
        keys = [si.key for si in results[0]]
        assert "in-ns" in keys
        assert "out-ns" not in keys

    @pytest.mark.asyncio
    async def test_search_empty_prefix_matches_all_markers(self):
        """Empty namespace prefix matches every item regardless of namespace."""
        item1 = _make_item("a/b", "k1")
        item2 = _make_item("x/y", "k2")
        sdk = _make_sdk(fetch_items=[item1, item2])
        store = _make_store(sdk=sdk)
        results = await store.abatch(
            [SearchOp(namespace_prefix=(), query="q")]
        )
        keys = {si.key for si in results[0]}
        assert {"k1", "k2"} == keys

    @pytest.mark.asyncio
    async def test_search_metadata_filter_equality(self):
        item_yes = _make_item("ns", "yes")
        item_yes.metadata[_MARKER] = True
        item_yes.metadata["status"] = "active"
        item_no = _make_item("ns", "no")
        item_no.metadata[_MARKER] = True
        item_no.metadata["status"] = "inactive"
        sdk = _make_sdk(fetch_items=[item_yes, item_no])
        store = _make_store(sdk=sdk)
        results = await store.abatch(
            [SearchOp(namespace_prefix=("ns",), query="q", filter={"status": "active"})]
        )
        assert len(results[0]) == 1
        assert results[0][0].key == "yes"

    @pytest.mark.asyncio
    async def test_search_metadata_filter_list_membership(self):
        """filter_ with list value: metadata value must be one of the list."""
        item = _make_item("ns", "k")
        item.metadata["status"] = "pending"
        sdk = _make_sdk(fetch_items=[item])
        store = _make_store(sdk=sdk)
        results = await store.abatch(
            [SearchOp(namespace_prefix=("ns",), query="q", filter={"status": ["active", "pending"]})]
        )
        assert len(results[0]) == 1

    @pytest.mark.asyncio
    async def test_search_respects_limit_and_offset(self):
        items = [_make_item("ns", f"k{i}") for i in range(5)]
        sdk = _make_sdk(fetch_items=items)
        store = _make_store(sdk=sdk)
        results = await store.abatch(
            [SearchOp(namespace_prefix=("ns",), query="q", limit=2, offset=0)]
        )
        assert len(results[0]) == 2

    @pytest.mark.asyncio
    async def test_search_empty_response_returns_empty_list(self):
        sdk = _make_sdk(fetch_items=[])
        store = _make_store(sdk=sdk)
        results = await store.abatch(
            [SearchOp(namespace_prefix=("ns",), query="q")]
        )
        assert results[0] == []

    @pytest.mark.asyncio
    async def test_search_sdk_failure_degrades_to_empty_list(self, caplog):
        sdk = _make_sdk(fetch_raise=RuntimeError("search outage"))
        store = _make_store(sdk=sdk)
        with caplog.at_level(logging.ERROR, logger="synap_langgraph.store"):
            results = await store.abatch(
                [SearchOp(namespace_prefix=("ns",), query="q")]
            )
        assert results[0] == []
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) >= 1

    @pytest.mark.asyncio
    async def test_search_semantic_fallback_returns_items_when_markers_stripped(self):
        """When no items carry markers, semantic_fallback=True returns Synap-ranked items."""
        item = _make_item("ns", "k", strip_markers=True)
        sdk = _make_sdk(fetch_items=[item])
        store = _make_store(sdk=sdk, semantic_fallback=True)
        results = await store.abatch(
            [SearchOp(namespace_prefix=("ns",), query="q")]
        )
        assert len(results[0]) >= 1
        assert store._marker_warned is True

    @pytest.mark.asyncio
    async def test_search_semantic_fallback_false_returns_empty_when_markers_stripped(self):
        """semantic_fallback=False returns [] instead of fallback when markers are absent."""
        item = _make_item("ns", "k", strip_markers=True)
        sdk = _make_sdk(fetch_items=[item])
        store = _make_store(sdk=sdk, semantic_fallback=False)
        results = await store.abatch(
            [SearchOp(namespace_prefix=("ns",), query="q")]
        )
        assert results[0] == []

    @pytest.mark.asyncio
    async def test_search_passes_user_and_customer_ids_to_fetch(self):
        sdk = _make_sdk()
        store = SynapStore(sdk, user_id="alice", customer_id="acme")
        await store.abatch([SearchOp(namespace_prefix=("ns",), query="q")])
        kw = sdk.fetch.call_args.kwargs
        assert kw["user_id"] == "alice"
        assert kw["customer_id"] == "acme"

    @pytest.mark.asyncio
    async def test_search_passes_scopes_to_fetch(self):
        sdk = _make_sdk()
        store = SynapStore(sdk, user_id="u1")
        await store.abatch([SearchOp(namespace_prefix=("ns",), query="q")])
        kw = sdk.fetch.call_args.kwargs
        assert kw["scopes"] == ["user"]

    @pytest.mark.asyncio
    async def test_search_uses_query_as_search_token(self):
        sdk = _make_sdk()
        store = _make_store(sdk=sdk)
        await store.abatch([SearchOp(namespace_prefix=(), query="coffee preferences")])
        kw = sdk.fetch.call_args.kwargs
        assert kw["search_query"] == ["coffee preferences"]


# ---------------------------------------------------------------------------
# ListNamespacesOp — always returns empty, warns once
# ---------------------------------------------------------------------------


class TestListNamespacesOp:
    @pytest.mark.asyncio
    async def test_list_namespaces_returns_empty(self):
        sdk = _make_sdk()
        store = _make_store(sdk=sdk)
        results = await store.abatch([ListNamespacesOp()])
        assert results[0] == []

    @pytest.mark.asyncio
    async def test_list_namespaces_warns_once(self, caplog):
        sdk = _make_sdk()
        store = _make_store(sdk=sdk)
        with caplog.at_level(logging.WARNING, logger="synap_langgraph.store"):
            await store.abatch([ListNamespacesOp()])
            await store.abatch([ListNamespacesOp()])
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert store._listns_warned is True


# ---------------------------------------------------------------------------
# Unsupported op type
# ---------------------------------------------------------------------------


class TestUnsupportedOp:
    @pytest.mark.asyncio
    async def test_unsupported_op_raises_value_error(self):
        store = _make_store()

        class BogusOp:
            pass

        with pytest.raises(ValueError, match="unsupported op type"):
            await store.abatch([BogusOp()])


# ---------------------------------------------------------------------------
# record_message / arecord_message
# ---------------------------------------------------------------------------


class TestRecordMessage:
    @pytest.mark.asyncio
    async def test_arecord_message_happy_path(self):
        sdk = _make_sdk()
        store = SynapStore(sdk, user_id="u1", customer_id="c1")
        await store.arecord_message("conv-1", "user", "hello")
        sdk.conversation.record_message.assert_awaited_once()
        kw = sdk.conversation.record_message.call_args.kwargs
        assert kw["conversation_id"] == "conv-1"
        assert kw["role"] == "user"
        assert kw["content"] == "hello"
        assert kw["user_id"] == "u1"
        assert kw["customer_id"] == "c1"

    @pytest.mark.asyncio
    async def test_arecord_message_passes_optional_session_and_metadata(self):
        sdk = _make_sdk()
        store = SynapStore(sdk, user_id="u1", customer_id="c1")
        await store.arecord_message("conv-1", "user", "hi", session_id="sess-1", metadata={"key": "val"})
        kw = sdk.conversation.record_message.call_args.kwargs
        assert kw["session_id"] == "sess-1"
        assert kw["metadata"] == {"key": "val"}

    @pytest.mark.asyncio
    async def test_arecord_message_skips_when_no_user_id(self, caplog):
        sdk = _make_sdk()
        store = SynapStore(sdk, customer_id="c1")  # no user_id
        with caplog.at_level(logging.WARNING, logger="synap_langgraph.store"):
            await store.arecord_message("conv-1", "user", "hi")
        sdk.conversation.record_message.assert_not_awaited()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) >= 1

    @pytest.mark.asyncio
    async def test_arecord_message_sdk_failure_is_swallowed(self, caplog):
        """record_message is best-effort: SDK failure must never propagate."""
        sdk = _make_sdk()
        sdk.conversation.record_message = AsyncMock(side_effect=RuntimeError("record error"))
        store = SynapStore(sdk, user_id="u1", customer_id="c1")
        with caplog.at_level(logging.WARNING, logger="synap_langgraph.store"):
            await store.arecord_message("conv-1", "user", "hi")  # must not raise
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) >= 1

    def test_sync_record_message_delegates_to_arecord_message(self):
        sdk = _make_sdk()
        store = SynapStore(sdk, user_id="u1", customer_id="c1")
        store.record_message("conv-1", "user", "hello")
        sdk.conversation.record_message.assert_called_once()


# ---------------------------------------------------------------------------
# Sync wrapper batch
# ---------------------------------------------------------------------------


class TestSyncBatch:
    def test_sync_put_then_get(self):
        item = _make_item("ns", "k1", content='{"ping": "pong"}')
        sdk = _make_sdk(fetch_items=[item])
        store = _make_store(sdk=sdk)
        # sync put
        put_results = store.batch([PutOp(namespace=("ns",), key="k1", value={"ping": "pong"})])
        assert put_results == [None]
        # sync get
        get_results = store.batch([GetOp(namespace=("ns",), key="k1")])
        assert get_results[0] is not None
        assert get_results[0].key == "k1"


# ---------------------------------------------------------------------------
# _iter_items — all_items callable path
# ---------------------------------------------------------------------------


class TestIterItems:
    def test_uses_all_items_when_callable(self):
        """When response.all_items is callable, _iter_items delegates to it."""
        items = [object(), object()]

        class R:
            def all_items(self_):
                return items

        result = _iter_items(R())
        assert result == list(items)

    def test_falls_back_to_bucket_union_when_all_items_not_callable(self):
        """Without callable all_items, we union facts + preferences + episodes."""
        class R:
            facts = ["f1"]
            preferences = ["p1"]
            episodes = ["e1"]
            emotions = []
            temporal_events = []

        result = _iter_items(R())
        assert set(result) == {"f1", "p1", "e1"}

    def test_bucket_union_tolerates_none_buckets(self):
        """Missing buckets should not crash _iter_items."""
        class R:
            facts = None
            preferences = ["p1"]
            episodes = None
            emotions = None
            temporal_events = None

        result = _iter_items(R())
        assert result == ["p1"]

    def test_all_items_callable_failure_falls_back_to_buckets(self):
        """If all_items() raises, _iter_items falls back to the bucket union."""
        class R:
            def all_items(self_):
                raise RuntimeError("all_items broken")
            facts = ["fact-fallback"]
            preferences = []
            episodes = []
            emotions = []
            temporal_events = []

        result = _iter_items(R())
        assert result == ["fact-fallback"]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestNsStr:
    def test_empty_tuple_gives_empty_string(self):
        assert _ns_str(()) == ""

    def test_single_element(self):
        assert _ns_str(("agents",)) == "agents"

    def test_multi_element(self):
        assert _ns_str(("a", "b", "c")) == "a/b/c"


class TestMatchesNamespacePrefix:
    def test_empty_prefix_matches_everything(self):
        assert _matches_namespace_prefix("any/path", ())
        assert _matches_namespace_prefix("", ())

    def test_exact_match(self):
        assert _matches_namespace_prefix("a/b", ("a", "b"))

    def test_child_of_prefix_matches(self):
        assert _matches_namespace_prefix("a/b/c", ("a", "b"))

    def test_sibling_does_not_match(self):
        assert not _matches_namespace_prefix("a/bc", ("a", "b"))

    def test_parent_does_not_match(self):
        assert not _matches_namespace_prefix("a", ("a", "b"))


class TestFilterMatches:
    def test_empty_filter_always_matches(self):
        assert _filter_matches({"k": "v"}, {})
        assert _filter_matches({}, {})

    def test_equality_match(self):
        assert _filter_matches({"step": 1}, {"step": 1})

    def test_equality_no_match(self):
        assert not _filter_matches({"step": 1}, {"step": 2})

    def test_missing_key_no_match(self):
        assert not _filter_matches({}, {"step": 1})

    def test_list_value_member_matches(self):
        assert _filter_matches({"status": "active"}, {"status": ["active", "pending"]})

    def test_list_value_non_member_no_match(self):
        assert not _filter_matches({"status": "inactive"}, {"status": ["active", "pending"]})

    def test_multi_key_filter_all_must_match(self):
        assert _filter_matches({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert not _filter_matches({"a": 1, "b": 3}, {"a": 1, "b": 2})


class TestParseValue:
    def test_valid_json_dict_returned_as_is(self):
        assert _parse_value('{"x": 1}') == {"x": 1}

    def test_non_json_wrapped_as_raw(self):
        result = _parse_value("not json")
        assert result == {"_raw": "not json"}

    def test_empty_string_returns_empty_dict(self):
        assert _parse_value("") == {}

    def test_json_list_wrapped_as_value(self):
        result = _parse_value("[1, 2, 3]")
        assert result == {"_value": [1, 2, 3]}

    def test_json_scalar_wrapped_as_value(self):
        result = _parse_value("42")
        assert result == {"_value": 42}


# ---------------------------------------------------------------------------
# Public surface smoke tests
# ---------------------------------------------------------------------------


def test_public_exports():
    import synap_langgraph
    assert hasattr(synap_langgraph, "SynapStore")
    assert "SynapStore" in synap_langgraph.__all__


def test_synapstore_is_base_store_subclass():
    from langgraph.store.base import BaseStore
    assert issubclass(SynapStore, BaseStore)
