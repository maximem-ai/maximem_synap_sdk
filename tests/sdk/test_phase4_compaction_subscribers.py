"""Unit tests for Phase 4 typed compaction-update subscriptions."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from maximem_synap.cache.anticipation_cache import AnticipationCache
from maximem_synap.cache.short_term_store import ShortTermContextStore
from maximem_synap.models.errors import InvalidInputError
from maximem_synap.sdk import (
    ConversationContextInterface,
    InstanceInterface,
    _dispatch_compaction_subscribers,
)


def _make_sdk():
    """A minimal SDK-shaped object the subscribe API can sit on."""
    sdk = MagicMock()
    sdk._compaction_subscribers = {}
    import threading
    sdk._compaction_subscribers_lock = threading.RLock()
    sdk._st_store = ShortTermContextStore()
    sdk._anticipation_cache = AnticipationCache()
    sdk._cache_manager = MagicMock()
    sdk._cache_manager.delete = MagicMock()
    sdk._is_st_authoritative = lambda: True
    return sdk


def _compaction_bundle(conversation_id: str = "c1") -> dict:
    return {
        "_bundle_type": "compaction_update",
        "_anticipation_conversation_id": conversation_id,
        "conversation_context": {
            "conversation_id": conversation_id,
            "summary": "the summary",
            "compaction_id": "comp-x",
            "end_timestamp": "2026-05-22T10:00:00+00:00",
        },
    }


class TestSubscribeBasic:
    def test_subscribe_requires_conversation_id(self):
        sdk = _make_sdk()
        cci = ConversationContextInterface(sdk)
        with pytest.raises(InvalidInputError):
            cci.subscribe_to_compaction_updates("", lambda b: None)

    def test_subscribe_requires_callback(self):
        sdk = _make_sdk()
        cci = ConversationContextInterface(sdk)
        with pytest.raises(InvalidInputError):
            cci.subscribe_to_compaction_updates("c1", None)  # type: ignore[arg-type]

    def test_subscribe_and_dispatch_fires_sync_callback(self):
        sdk = _make_sdk()
        cci = ConversationContextInterface(sdk)
        received = []
        cci.subscribe_to_compaction_updates("c1", lambda b: received.append(b))
        _dispatch_compaction_subscribers(sdk, "c1", _compaction_bundle("c1"))
        assert len(received) == 1
        assert received[0]["conversation_context"]["summary"] == "the summary"

    def test_dispatch_to_other_conversation_is_noop(self):
        sdk = _make_sdk()
        cci = ConversationContextInterface(sdk)
        received = []
        cci.subscribe_to_compaction_updates("c1", lambda b: received.append(b))
        _dispatch_compaction_subscribers(sdk, "c2", _compaction_bundle("c2"))
        assert received == []

    def test_multiple_subscribers_fire_in_order(self):
        sdk = _make_sdk()
        cci = ConversationContextInterface(sdk)
        order = []
        cci.subscribe_to_compaction_updates("c1", lambda b: order.append("a"))
        cci.subscribe_to_compaction_updates("c1", lambda b: order.append("b"))
        cci.subscribe_to_compaction_updates("c1", lambda b: order.append("c"))
        _dispatch_compaction_subscribers(sdk, "c1", _compaction_bundle())
        assert order == ["a", "b", "c"]

    def test_unsubscribe_removes_only_that_callback(self):
        sdk = _make_sdk()
        cci = ConversationContextInterface(sdk)
        a_calls = []
        b_calls = []
        un_a = cci.subscribe_to_compaction_updates("c1", lambda b: a_calls.append(1))
        cci.subscribe_to_compaction_updates("c1", lambda b: b_calls.append(1))
        un_a()
        _dispatch_compaction_subscribers(sdk, "c1", _compaction_bundle())
        assert a_calls == []
        assert b_calls == [1]

    def test_unsubscribe_is_idempotent(self):
        sdk = _make_sdk()
        cci = ConversationContextInterface(sdk)
        un = cci.subscribe_to_compaction_updates("c1", lambda b: None)
        un()
        un()  # should not raise
        # Re-dispatching shouldn't fire anything
        received = []
        un2 = cci.subscribe_to_compaction_updates("c1", lambda b: received.append(1))
        _dispatch_compaction_subscribers(sdk, "c1", _compaction_bundle())
        assert received == [1]

    def test_subscriber_exception_does_not_break_others(self):
        sdk = _make_sdk()
        cci = ConversationContextInterface(sdk)
        bad_called = []
        good_called = []

        def bad(_b):
            bad_called.append(1)
            raise RuntimeError("boom")

        cci.subscribe_to_compaction_updates("c1", bad)
        cci.subscribe_to_compaction_updates("c1", lambda b: good_called.append(1))
        _dispatch_compaction_subscribers(sdk, "c1", _compaction_bundle())
        assert bad_called == [1]
        assert good_called == [1]


class TestUnsubscribeAll:
    def test_unsubscribe_all_for_conv(self):
        sdk = _make_sdk()
        cci = ConversationContextInterface(sdk)
        cci.subscribe_to_compaction_updates("c1", lambda b: None)
        cci.subscribe_to_compaction_updates("c1", lambda b: None)
        cci.subscribe_to_compaction_updates("c2", lambda b: None)
        removed = cci.unsubscribe_all_compaction_updates("c1")
        assert removed == 2
        assert "c1" not in sdk._compaction_subscribers
        assert len(sdk._compaction_subscribers.get("c2", [])) == 1

    def test_unsubscribe_all_global(self):
        sdk = _make_sdk()
        cci = ConversationContextInterface(sdk)
        cci.subscribe_to_compaction_updates("c1", lambda b: None)
        cci.subscribe_to_compaction_updates("c2", lambda b: None)
        cci.subscribe_to_compaction_updates("c3", lambda b: None)
        removed = cci.unsubscribe_all_compaction_updates()
        assert removed == 3
        assert sdk._compaction_subscribers == {}


class TestAsyncCallback:
    def test_async_callback_dispatched_via_create_task(self):
        sdk = _make_sdk()
        cci = ConversationContextInterface(sdk)
        received = []

        async def cb(bundle):
            received.append(bundle)

        async def run():
            cci.subscribe_to_compaction_updates("c1", cb)
            _dispatch_compaction_subscribers(sdk, "c1", _compaction_bundle())
            await asyncio.sleep(0)  # let the task run
            assert len(received) == 1

        asyncio.run(run())


class TestEndToEndViaInstanceHandler:
    """Exercise the full path: bundle arrives at InstanceInterface._handle_anticipated_bundle,
    which routes compaction_update through the dispatcher."""

    def test_handler_dispatches_to_subscribers(self):
        # Use the SDK directly so the wiring under test is real, not mocked.
        from maximem_synap import MaximemSynapSDK

        sdk = MaximemSynapSDK(instance_id="inst-x", api_key="dummy", _force_new=True)
        cci = ConversationContextInterface(sdk)
        instance = InstanceInterface(sdk)
        instance._on_context_callback = None

        fired = []
        cci.subscribe_to_compaction_updates("c1", lambda b: fired.append(b))

        instance._handle_anticipated_bundle(_compaction_bundle("c1"))
        assert len(fired) == 1
        assert fired[0]["conversation_context"]["compaction_id"] == "comp-x"

    def test_handler_skips_non_compaction_bundles(self):
        from maximem_synap import MaximemSynapSDK

        sdk = MaximemSynapSDK(instance_id="inst-y", api_key="dummy", _force_new=True)
        cci = ConversationContextInterface(sdk)
        instance = InstanceInterface(sdk)
        instance._on_context_callback = None

        fired = []
        cci.subscribe_to_compaction_updates("c1", lambda b: fired.append(b))

        anticipation_bundle = {
            "_bundle_type": "anticipation",
            "_anticipation_conversation_id": "c1",
            "items": [],
        }
        instance._handle_anticipated_bundle(anticipation_bundle)
        assert fired == []
