"""Tests for SynapCheckpointSaver — best-effort fuzzy BaseCheckpointSaver.

Documented error-handling contracts (from checkpointer.py docstring):
- aput / aput_writes: surface SynapIntegrationError on SDK failure (write path
  must be strict — silent drops would hide persistence outages).
- aget_tuple / alist: degrade gracefully on SDK failure — return None / empty
  iterator so a fetch outage never crashes a graph run.
- adelete_thread: no-op (Synap has no delete API); warns once.
- Corrupt checkpoint blobs are skipped with an ERROR log so one bad fact
  doesn't poison the whole thread.

Coverage shape:
- Construction validation (None sdk, empty user_id, optional customer_id)
- aput: happy path, metadata content, config return, SDK failure → SynapIntegrationError
- put (sync): delegates to aput
- aput_writes: writes each channel separately, stores task metadata, SDK failure
- put_writes (sync): delegates to aput_writes
- aget_tuple: match by thread_id, target checkpoint_id filtering, no thread_id →
  None, empty result → None, SDK failure → None, most-recent ordering
- get_tuple (sync): delegates to aget_tuple
- alist: yields tuples, most-recent-first ordering, metadata filter, limit,
  before-config sentinel, no thread_id → empty, SDK failure → empty iterator
- list (sync): delegates to alist
- adelete_thread: no-op, warning fires once regardless of repeat calls
- delete_thread (sync): delegates to adelete_thread
- Corrupt/truncated checkpoint blob skipped gracefully (ERROR log, no crash)
- Parent config populated when parent_checkpoint_id is non-empty
- _encode / _decode round-trip
- _safe_json: JSON-serializable pass-through, non-serializable coerced
- _metadata_matches_filter: equality, missing key, empty filter
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from langgraph.checkpoint.base import (
    CheckpointTuple,
    empty_checkpoint,
)

from synap_integrations_common import SynapIntegrationError
from synap_langgraph.checkpointer import (
    SynapCheckpointSaver,
    _MARKER,
    _WRITE_MARKER,
    _decode,
    _encode,
    _metadata_matches_filter,
    _safe_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sdk(create_raise=None, fetch_return=None, fetch_raise=None):
    sdk = MagicMock()
    sdk.memories = MagicMock()
    if create_raise is not None:
        sdk.memories.create = AsyncMock(side_effect=create_raise)
    else:
        sdk.memories.create = AsyncMock(return_value=MagicMock(ingestion_id="ing-1"))

    if fetch_raise is not None:
        sdk.fetch = AsyncMock(side_effect=fetch_raise)
    elif fetch_return is not None:
        sdk.fetch = AsyncMock(return_value=fetch_return)
    else:
        sdk.fetch = AsyncMock(return_value=_make_response([]))
    return sdk


def _make_response(facts):
    """Build a plain (non-MagicMock) response with the given facts list."""
    class R:
        def __init__(self, fs):
            self.facts = fs
            self.preferences = []
            self.episodes = []
            self.emotions = []
            self.temporal_events = []
    return R(facts)


def _make_saver(**kwargs):
    sdk = kwargs.pop("sdk", None) or _make_sdk()
    user_id = kwargs.pop("user_id", "u1")
    customer_id = kwargs.pop("customer_id", "c1")
    return SynapCheckpointSaver(sdk, user_id=user_id, customer_id=customer_id, **kwargs)


def _make_checkpoint_fact(
    saver: SynapCheckpointSaver,
    cp_id: str,
    thread_id: str = "th1",
    checkpoint_ns: str = "",
    parent_checkpoint_id: str = "",
    extracted_at: Any = None,
    metadata: dict | None = None,
    corrupt: bool = False,
):
    """Build a fake memory fact carrying a properly encoded checkpoint."""
    cp = empty_checkpoint()
    cp["id"] = cp_id
    if corrupt:
        doc = "not valid json at all !!"
    else:
        t, b = saver.serde.dumps_typed(cp)
        doc = _encode(t, b, metadata or {"step": 1}, {})

    class Fact:
        content = doc
        metadata_dict = {
            _MARKER: True,
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": cp_id,
            "parent_checkpoint_id": parent_checkpoint_id,
        }
        # Need attribute access not item access
        @property
        def metadata(self_):
            return self_.metadata_dict

    fact = Fact()
    fact.extracted_at = extracted_at or datetime(2024, 1, 1, tzinfo=timezone.utc)
    return fact


def _config(thread_id="th1", checkpoint_ns="", checkpoint_id=None):
    c = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
    if checkpoint_id:
        c["checkpoint_id"] = checkpoint_id
    return {"configurable": c}


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_raises_on_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapCheckpointSaver(None, user_id="u1")  # type: ignore[arg-type]

    def test_raises_on_empty_user_id(self):
        sdk = _make_sdk()
        with pytest.raises(ValueError, match="non-empty user_id"):
            SynapCheckpointSaver(sdk, user_id="")

    def test_customer_id_defaults_to_empty_string(self):
        sdk = _make_sdk()
        saver = SynapCheckpointSaver(sdk, user_id="u1")
        assert saver.customer_id == ""

    def test_mode_default_is_accurate(self):
        sdk = _make_sdk()
        saver = SynapCheckpointSaver(sdk, user_id="u1")
        assert saver.mode == "accurate"

    def test_custom_mode(self):
        sdk = _make_sdk()
        saver = SynapCheckpointSaver(sdk, user_id="u1", mode="fast")
        assert saver.mode == "fast"

    def test_delete_warned_starts_false(self):
        sdk = _make_sdk()
        saver = SynapCheckpointSaver(sdk, user_id="u1")
        assert not saver._delete_warned


# ---------------------------------------------------------------------------
# aput / put — write checkpoint
# ---------------------------------------------------------------------------


class TestAput:
    @pytest.mark.asyncio
    async def test_aput_calls_memories_create(self):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        cp = empty_checkpoint()
        cp["id"] = "cp-1"
        result = await saver.aput(_config(), cp, {"step": 1}, {})
        sdk.memories.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aput_returns_config_with_thread_and_cp_ids(self):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        cp = empty_checkpoint()
        cp["id"] = "cp-42"
        result = await saver.aput(
            _config(thread_id="th-abc", checkpoint_ns="ns"),
            cp, {}, {},
        )
        cfg = result["configurable"]
        assert cfg["thread_id"] == "th-abc"
        assert cfg["checkpoint_ns"] == "ns"
        assert cfg["checkpoint_id"] == "cp-42"

    @pytest.mark.asyncio
    async def test_aput_stamps_marker_in_metadata(self):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        cp = empty_checkpoint()
        cp["id"] = "cp-1"
        await saver.aput(_config(thread_id="th1"), cp, {}, {})
        meta = sdk.memories.create.call_args.kwargs["metadata"]
        assert meta[_MARKER] is True
        assert meta["thread_id"] == "th1"
        assert meta["checkpoint_id"] == "cp-1"

    @pytest.mark.asyncio
    async def test_aput_records_parent_checkpoint_id(self):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        cp = empty_checkpoint()
        cp["id"] = "cp-child"
        # parent_checkpoint_id comes from configurable["checkpoint_id"]
        config = {"configurable": {"thread_id": "th1", "checkpoint_id": "cp-parent"}}
        await saver.aput(config, cp, {}, {})
        meta = sdk.memories.create.call_args.kwargs["metadata"]
        assert meta["parent_checkpoint_id"] == "cp-parent"

    @pytest.mark.asyncio
    async def test_aput_encodes_checkpoint_as_json_document(self):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        cp = empty_checkpoint()
        cp["id"] = "cp-enc"
        await saver.aput(_config(), cp, {"source": "loop"}, {})
        doc = json.loads(sdk.memories.create.call_args.kwargs["document"])
        # Must have all required fields from _encode
        assert "serde_type" in doc
        assert "checkpoint_b64" in doc
        assert "metadata" in doc
        assert doc["metadata"]["source"] == "loop"

    @pytest.mark.asyncio
    async def test_aput_passes_user_and_customer_ids(self):
        sdk = _make_sdk()
        saver = SynapCheckpointSaver(sdk, user_id="alice", customer_id="acme")
        cp = empty_checkpoint()
        await saver.aput(_config(), cp, {}, {})
        kw = sdk.memories.create.call_args.kwargs
        assert kw["user_id"] == "alice"
        assert kw["customer_id"] == "acme"

    @pytest.mark.asyncio
    async def test_aput_sdk_failure_raises_synap_integration_error(self):
        sdk = _make_sdk(create_raise=RuntimeError("persistence failure"))
        saver = _make_saver(sdk=sdk)
        cp = empty_checkpoint()
        with pytest.raises(SynapIntegrationError) as ei:
            await saver.aput(_config(), cp, {}, {})
        assert ei.value.operation == "langgraph.checkpointer.put"
        assert isinstance(ei.value.__cause__, RuntimeError)

    def test_sync_put_delegates_to_aput(self):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        cp = empty_checkpoint()
        cp["id"] = "cp-sync"
        result = saver.put(_config(), cp, {}, {})
        assert result["configurable"]["checkpoint_id"] == "cp-sync"
        sdk.memories.create.assert_called_once()


# ---------------------------------------------------------------------------
# aput_writes / put_writes
# ---------------------------------------------------------------------------


class TestAputWrites:
    @pytest.mark.asyncio
    async def test_aput_writes_calls_memories_create(self):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        await saver.aput_writes(_config(), [("ch", "val")], "task-1")
        sdk.memories.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aput_writes_stamps_write_marker(self):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        await saver.aput_writes(_config(thread_id="th1", checkpoint_id="cp-1"), [("ch", "v")], "t1")
        meta = sdk.memories.create.call_args.kwargs["metadata"]
        assert meta[_WRITE_MARKER] is True
        assert meta["thread_id"] == "th1"
        assert meta["task_id"] == "t1"

    @pytest.mark.asyncio
    async def test_aput_writes_encodes_all_channels(self):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        writes = [("messages", {"role": "user", "content": "hi"}), ("output", 42)]
        await saver.aput_writes(_config(), writes, "task-1")
        doc = json.loads(sdk.memories.create.call_args.kwargs["document"])
        channels = [w["channel"] for w in doc["writes"]]
        assert channels == ["messages", "output"]
        assert doc["task_id"] == "task-1"

    @pytest.mark.asyncio
    async def test_aput_writes_base64_encodes_each_value(self):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        await saver.aput_writes(_config(), [("ch", "hello")], "t1")
        doc = json.loads(sdk.memories.create.call_args.kwargs["document"])
        write = doc["writes"][0]
        # value_b64 must be valid base64
        decoded = base64.b64decode(write["value_b64"])
        assert isinstance(decoded, bytes)

    @pytest.mark.asyncio
    async def test_aput_writes_sdk_failure_raises_synap_integration_error(self):
        sdk = _make_sdk(create_raise=RuntimeError("write fail"))
        saver = _make_saver(sdk=sdk)
        with pytest.raises(SynapIntegrationError) as ei:
            await saver.aput_writes(_config(), [("ch", "v")], "task-1")
        assert ei.value.operation == "langgraph.checkpointer.put_writes"

    def test_sync_put_writes_delegates_to_aput_writes(self):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        saver.put_writes(_config(), [("ch", "v")], "task-1")
        sdk.memories.create.assert_called_once()


# ---------------------------------------------------------------------------
# aget_tuple / get_tuple — read checkpoint
# ---------------------------------------------------------------------------


class TestAgetTuple:
    @pytest.mark.asyncio
    async def test_aget_tuple_returns_checkpoint_when_match_found(self):
        saver = _make_saver()
        fact = _make_checkpoint_fact(saver, "cp-1")
        sdk = _make_sdk(fetch_return=_make_response([fact]))
        saver.sdk = sdk
        tup = await saver.aget_tuple(_config(thread_id="th1"))
        assert tup is not None
        assert isinstance(tup, CheckpointTuple)
        assert tup.checkpoint["id"] == "cp-1"

    @pytest.mark.asyncio
    async def test_aget_tuple_returns_none_when_no_thread_id(self):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        result = await saver.aget_tuple({"configurable": {}})
        assert result is None
        sdk.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_aget_tuple_returns_none_on_empty_result(self):
        sdk = _make_sdk(fetch_return=_make_response([]))
        saver = _make_saver(sdk=sdk)
        result = await saver.aget_tuple(_config(thread_id="th1"))
        assert result is None

    @pytest.mark.asyncio
    async def test_aget_tuple_returns_none_on_sdk_failure(self):
        sdk = _make_sdk(fetch_raise=RuntimeError("fetch fail"))
        saver = _make_saver(sdk=sdk)
        result = await saver.aget_tuple(_config(thread_id="th1"))
        assert result is None

    @pytest.mark.asyncio
    async def test_aget_tuple_filters_by_target_checkpoint_id(self):
        saver = _make_saver()
        fact1 = _make_checkpoint_fact(saver, "cp-1", extracted_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
        fact2 = _make_checkpoint_fact(saver, "cp-2", extracted_at=datetime(2024, 1, 2, tzinfo=timezone.utc))
        sdk = _make_sdk(fetch_return=_make_response([fact1, fact2]))
        saver.sdk = sdk
        # Request cp-1 specifically
        tup = await saver.aget_tuple(_config(thread_id="th1", checkpoint_id="cp-1"))
        assert tup is not None
        assert tup.checkpoint["id"] == "cp-1"

    @pytest.mark.asyncio
    async def test_aget_tuple_returns_most_recent_when_no_target(self):
        saver = _make_saver()
        fact_old = _make_checkpoint_fact(saver, "cp-old", extracted_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
        fact_new = _make_checkpoint_fact(saver, "cp-new", extracted_at=datetime(2024, 1, 2, tzinfo=timezone.utc))
        sdk = _make_sdk(fetch_return=_make_response([fact_old, fact_new]))
        saver.sdk = sdk
        tup = await saver.aget_tuple(_config(thread_id="th1"))
        assert tup is not None
        assert tup.checkpoint["id"] == "cp-new"

    @pytest.mark.asyncio
    async def test_aget_tuple_config_contains_required_keys(self):
        saver = _make_saver()
        fact = _make_checkpoint_fact(saver, "cp-x", thread_id="th1", checkpoint_ns="")
        sdk = _make_sdk(fetch_return=_make_response([fact]))
        saver.sdk = sdk
        tup = await saver.aget_tuple(_config(thread_id="th1"))
        cfg = tup.config["configurable"]
        assert cfg["thread_id"] == "th1"
        assert cfg["checkpoint_id"] == "cp-x"
        assert "checkpoint_ns" in cfg

    @pytest.mark.asyncio
    async def test_aget_tuple_parent_config_set_when_parent_id_non_empty(self):
        saver = _make_saver()
        fact = _make_checkpoint_fact(saver, "cp-child", parent_checkpoint_id="cp-parent")
        sdk = _make_sdk(fetch_return=_make_response([fact]))
        saver.sdk = sdk
        tup = await saver.aget_tuple(_config(thread_id="th1"))
        assert tup is not None
        assert tup.parent_config is not None
        assert tup.parent_config["configurable"]["checkpoint_id"] == "cp-parent"

    @pytest.mark.asyncio
    async def test_aget_tuple_parent_config_none_when_parent_id_empty(self):
        saver = _make_saver()
        fact = _make_checkpoint_fact(saver, "cp-1", parent_checkpoint_id="")
        sdk = _make_sdk(fetch_return=_make_response([fact]))
        saver.sdk = sdk
        tup = await saver.aget_tuple(_config(thread_id="th1"))
        assert tup is not None
        assert tup.parent_config is None

    @pytest.mark.asyncio
    async def test_aget_tuple_skips_corrupt_fact_gracefully(self, caplog):
        saver = _make_saver()
        good = _make_checkpoint_fact(saver, "cp-good", extracted_at=datetime(2024, 1, 2, tzinfo=timezone.utc))
        corrupt = _make_checkpoint_fact(saver, "cp-bad", extracted_at=datetime(2024, 1, 3, tzinfo=timezone.utc), corrupt=True)
        sdk = _make_sdk(fetch_return=_make_response([corrupt, good]))
        saver.sdk = sdk
        with caplog.at_level(logging.ERROR, logger="synap_langgraph.checkpointer"):
            tup = await saver.aget_tuple(_config(thread_id="th1"))
        # Good fact should still be decoded and returned
        assert tup is not None
        assert tup.checkpoint["id"] == "cp-good"
        # An error log about the corrupt fact should appear
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) >= 1

    @pytest.mark.asyncio
    async def test_aget_tuple_filters_out_facts_for_other_threads(self):
        saver = _make_saver()
        fact_other_thread = _make_checkpoint_fact(saver, "cp-other", thread_id="th2")
        sdk = _make_sdk(fetch_return=_make_response([fact_other_thread]))
        saver.sdk = sdk
        tup = await saver.aget_tuple(_config(thread_id="th1"))
        assert tup is None

    def test_sync_get_tuple_delegates_to_aget_tuple(self):
        sdk = _make_sdk(fetch_return=_make_response([]))
        saver = _make_saver(sdk=sdk)
        result = saver.get_tuple(_config(thread_id="th1"))
        assert result is None


# ---------------------------------------------------------------------------
# alist / list
# ---------------------------------------------------------------------------


class TestAlist:
    @pytest.mark.asyncio
    async def test_alist_yields_checkpoints(self):
        saver = _make_saver()
        fact = _make_checkpoint_fact(saver, "cp-1")
        sdk = _make_sdk(fetch_return=_make_response([fact]))
        saver.sdk = sdk
        results = []
        async for tup in saver.alist(_config(thread_id="th1")):
            results.append(tup)
        assert len(results) == 1
        assert results[0].checkpoint["id"] == "cp-1"

    @pytest.mark.asyncio
    async def test_alist_yields_most_recent_first(self):
        saver = _make_saver()
        fact1 = _make_checkpoint_fact(saver, "cp-1", extracted_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
        fact2 = _make_checkpoint_fact(saver, "cp-2", extracted_at=datetime(2024, 1, 2, tzinfo=timezone.utc))
        fact3 = _make_checkpoint_fact(saver, "cp-3", extracted_at=datetime(2024, 1, 3, tzinfo=timezone.utc))
        sdk = _make_sdk(fetch_return=_make_response([fact1, fact2, fact3]))
        saver.sdk = sdk
        results = []
        async for tup in saver.alist(_config(thread_id="th1")):
            results.append(tup)
        ids = [t.checkpoint["id"] for t in results]
        assert ids == ["cp-3", "cp-2", "cp-1"]

    @pytest.mark.asyncio
    async def test_alist_respects_limit(self):
        saver = _make_saver()
        facts = [
            _make_checkpoint_fact(saver, f"cp-{i}", extracted_at=datetime(2024, 1, i + 1, tzinfo=timezone.utc))
            for i in range(4)
        ]
        sdk = _make_sdk(fetch_return=_make_response(facts))
        saver.sdk = sdk
        results = []
        async for tup in saver.alist(_config(thread_id="th1"), limit=2):
            results.append(tup)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_alist_applies_metadata_filter(self):
        saver = _make_saver()
        fact_match = _make_checkpoint_fact(saver, "cp-match", metadata={"step": 5, "source": "loop"})
        fact_no_match = _make_checkpoint_fact(saver, "cp-skip", metadata={"step": 3})
        sdk = _make_sdk(fetch_return=_make_response([fact_match, fact_no_match]))
        saver.sdk = sdk
        results = []
        async for tup in saver.alist(_config(thread_id="th1"), filter={"step": 5}):
            results.append(tup)
        assert len(results) == 1
        assert results[0].checkpoint["id"] == "cp-match"

    @pytest.mark.asyncio
    async def test_alist_before_stops_at_matching_checkpoint(self):
        saver = _make_saver()
        fact1 = _make_checkpoint_fact(saver, "cp-1", extracted_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
        fact2 = _make_checkpoint_fact(saver, "cp-2", extracted_at=datetime(2024, 1, 2, tzinfo=timezone.utc))
        fact3 = _make_checkpoint_fact(saver, "cp-3", extracted_at=datetime(2024, 1, 3, tzinfo=timezone.utc))
        sdk = _make_sdk(fetch_return=_make_response([fact1, fact2, fact3]))
        saver.sdk = sdk
        before_config = {"configurable": {"thread_id": "th1", "checkpoint_id": "cp-2"}}
        results = []
        async for tup in saver.alist(_config(thread_id="th1"), before=before_config):
            results.append(tup)
        ids = [t.checkpoint["id"] for t in results]
        # Sorted: cp-3, cp-2, cp-1 — before=cp-2 means we stop at cp-2 (exclusive)
        assert "cp-3" in ids
        assert "cp-2" not in ids

    @pytest.mark.asyncio
    async def test_alist_empty_when_no_thread_id(self):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        results = []
        async for tup in saver.alist({"configurable": {}}):
            results.append(tup)
        assert results == []
        sdk.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_alist_empty_when_config_is_none(self):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        results = []
        async for tup in saver.alist(None):
            results.append(tup)
        assert results == []

    @pytest.mark.asyncio
    async def test_alist_empty_on_sdk_failure(self):
        sdk = _make_sdk(fetch_raise=RuntimeError("boom"))
        saver = _make_saver(sdk=sdk)
        results = []
        async for tup in saver.alist(_config(thread_id="th1")):
            results.append(tup)
        assert results == []

    def test_sync_list_delegates_to_alist(self):
        sdk = _make_sdk(fetch_return=_make_response([]))
        saver = _make_saver(sdk=sdk)
        result = list(saver.list(_config(thread_id="th1")))
        assert result == []


# ---------------------------------------------------------------------------
# adelete_thread / delete_thread
# ---------------------------------------------------------------------------


class TestAdeleteThread:
    @pytest.mark.asyncio
    async def test_adelete_thread_is_noop_does_not_call_sdk(self):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        await saver.adelete_thread("th1")
        sdk.memories.create.assert_not_awaited()
        sdk.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_adelete_thread_warns_once(self, caplog):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        with caplog.at_level(logging.WARNING, logger="synap_langgraph.checkpointer"):
            await saver.adelete_thread("th1")
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "delete" in warnings[0].message.lower()

    @pytest.mark.asyncio
    async def test_adelete_thread_warning_fires_only_once(self, caplog):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        with caplog.at_level(logging.WARNING, logger="synap_langgraph.checkpointer"):
            await saver.adelete_thread("th1")
            await saver.adelete_thread("th1")
            await saver.adelete_thread("th2")
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert saver._delete_warned is True

    def test_sync_delete_thread_delegates_to_adelete_thread(self, caplog):
        sdk = _make_sdk()
        saver = _make_saver(sdk=sdk)
        with caplog.at_level(logging.WARNING, logger="synap_langgraph.checkpointer"):
            saver.delete_thread("th1")
        assert saver._delete_warned is True


# ---------------------------------------------------------------------------
# _encode / _decode round-trip
# ---------------------------------------------------------------------------


class TestEncodeDecode:
    def test_round_trip_preserves_checkpoint_id(self):
        saver = _make_saver()
        cp = empty_checkpoint()
        cp["id"] = "roundtrip-1"
        t, b = saver.serde.dumps_typed(cp)
        doc = _encode(t, b, {"step": 5}, {"ch": 1})
        decoded_cp, decoded_meta = _decode(doc, saver.serde)
        assert decoded_cp["id"] == "roundtrip-1"
        assert decoded_meta == {"step": 5}

    def test_encoded_document_is_valid_json(self):
        saver = _make_saver()
        cp = empty_checkpoint()
        t, b = saver.serde.dumps_typed(cp)
        doc = _encode(t, b, {}, {})
        parsed = json.loads(doc)
        assert "serde_type" in parsed
        assert "checkpoint_b64" in parsed
        assert "metadata" in parsed
        assert "new_versions" in parsed

    def test_checkpoint_b64_is_valid_base64(self):
        saver = _make_saver()
        cp = empty_checkpoint()
        t, b = saver.serde.dumps_typed(cp)
        doc = _encode(t, b, {}, {})
        payload = json.loads(doc)
        # should not raise
        base64.b64decode(payload["checkpoint_b64"])

    def test_decode_raises_on_invalid_json(self):
        saver = _make_saver()
        with pytest.raises(Exception):
            _decode("not valid json", saver.serde)


# ---------------------------------------------------------------------------
# _safe_json helper
# ---------------------------------------------------------------------------


class TestSafeJson:
    def test_serializable_dict_passes_through(self):
        d = {"a": 1, "b": "hello", "c": [1, 2, 3]}
        result = _safe_json(d)
        assert result == d
        json.dumps(result)  # must not raise

    def test_non_serializable_dict_values_coerced_to_str(self):
        d = {"a": 1, "b": object()}
        result = _safe_json(d)
        assert isinstance(result["b"], str)
        json.dumps(result)  # must not raise

    def test_non_dict_non_serializable_coerced_to_str(self):
        obj = object()
        result = _safe_json(obj)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _metadata_matches_filter helper
# ---------------------------------------------------------------------------


class TestMetadataMatchesFilter:
    def test_empty_filter_matches_anything(self):
        assert _metadata_matches_filter({"step": 1}, {})
        assert _metadata_matches_filter({}, {})

    def test_equality_match(self):
        assert _metadata_matches_filter({"step": 5}, {"step": 5})

    def test_equality_no_match(self):
        assert not _metadata_matches_filter({"step": 5}, {"step": 3})

    def test_missing_key_no_match(self):
        assert not _metadata_matches_filter({}, {"step": 1})

    def test_multi_key_all_must_match(self):
        assert _metadata_matches_filter({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert not _metadata_matches_filter({"a": 1, "b": 3}, {"a": 1, "b": 2})

    def test_none_metadata_treated_as_empty_dict(self):
        assert _metadata_matches_filter(None, {})
        assert not _metadata_matches_filter(None, {"step": 1})


# ---------------------------------------------------------------------------
# Public surface smoke test
# ---------------------------------------------------------------------------


def test_public_exports():
    import synap_langgraph
    assert hasattr(synap_langgraph, "SynapCheckpointSaver")
    assert "SynapCheckpointSaver" in synap_langgraph.__all__


def test_checkpointer_is_base_checkpoint_saver_subclass():
    from langgraph.checkpoint.base import BaseCheckpointSaver
    assert issubclass(SynapCheckpointSaver, BaseCheckpointSaver)
