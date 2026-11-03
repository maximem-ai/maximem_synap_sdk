"""The scope ladder path through synap-crewai, against ``RecordingSDK``.

A call without ``scope_path`` carries the keywords it always did (and the
record annotation is still ``/customer/user``); with one it reaches every SDK
call (``memories.create``, ``fetch``, the listener's user and assistant turns,
``report_user_turn``) and the annotation is built from the path's values in the
order given. Conversation-only short-term reads take no path.
"""

from __future__ import annotations

from synap_integrations_common.conversation_id import coerce_conversation_id
import asyncio
import time

import pytest
from crewai.events import CrewKickoffCompletedEvent, CrewKickoffStartedEvent, crewai_event_bus
from crewai.memory.types import MemoryRecord

from synap_crewai import (
    SynapCrewAIListener,
    SynapStorageBackend,
    build_synap_st_backstory,
    report_user_turn,
)
from synap_integrations_common import set_auto_listen
from synap_integrations_common.testing import RecordingSDK, make_ladder

PATH = {"customer": "acme", "team": "payments", "user": "dana"}


def _record(rid="r1"):
    return MemoryRecord(id=rid, content="likes tea", scope="/x", categories=["preference"],
                        importance=0.5)


# ── SynapStorageBackend ────────────────────────────────────────────────────


async def test_storage_default_calls_are_unchanged():
    sdk = RecordingSDK()
    backend = SynapStorageBackend(sdk, "dana", "acme", conversation_id="c1")
    await backend.asave([_record()])
    await backend.asearch([0.0], metadata_filter={"_query_text": "tea"})
    create = sdk.last_kwargs("memories.create")
    assert set(create) == {"document", "user_id", "customer_id", "metadata"}
    assert create["customer_id"] == "acme"
    assert sdk.last_kwargs("fetch") == {
        "conversation_id": coerce_conversation_id("c1"), "user_id": "dana", "customer_id": "acme",
        "search_query": ["tea"], "max_results": 10, "types": None, "mode": "fast",
        "include_conversation_context": False,
    }


async def test_storage_default_annotation_is_customer_slash_user():
    sdk = RecordingSDK()
    backend = SynapStorageBackend(sdk, "dana", "acme")
    results = await backend.asearch([0.0], metadata_filter={"_query_text": "tea"})
    assert results and {r.scope for r, _ in results} == {"/acme/dana"}
    assert SynapStorageBackend(sdk, "dana")._default_scope == "/dana"


async def test_storage_path_reaches_memories_create_as_scope():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    backend = SynapStorageBackend(sdk, "dana", "acme", scope_path=PATH)
    await backend.asave([_record("a"), _record("b")])
    creates = sdk.kwargs_of("memories.create")
    assert [k["scope"] for k in creates] == [PATH, PATH]
    assert all("scope_path" not in k for k in creates)


async def test_storage_path_reaches_fetch():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    backend = SynapStorageBackend(sdk, "dana", "acme", scope_path=PATH)
    await backend.asearch([0.0], metadata_filter={"_query_text": "tea"})
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


def test_storage_sync_save_and_search_carry_the_path():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    backend = SynapStorageBackend(sdk, "dana", "acme", scope_path=PATH)
    backend.save([_record()])
    backend.search([0.0], metadata_filter={"_query_text": "tea"})
    assert sdk.last_kwargs("memories.create")["scope"] == PATH
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


async def test_storage_annotation_follows_the_path_values_in_the_order_given():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    backend = SynapStorageBackend(sdk, "dana", "acme", scope_path=PATH)
    assert backend._default_scope == "/acme/payments/dana"
    results = await backend.asearch([0.0], metadata_filter={"_query_text": "tea"})
    assert results and {r.scope for r, _ in results} == {"/acme/payments/dana"}

    reordered = {"user": "dana", "team": "payments", "customer": "acme"}
    other = SynapStorageBackend(sdk, "dana", "acme", scope_path=reordered)
    assert other._default_scope == "/dana/payments/acme"
    # ... but the path sent to the server is the same mapping: the server orders by ladder position.
    await other.asearch([0.0], metadata_filter={"_query_text": "tea"})
    assert sdk.last_kwargs("fetch")["scope_path"] == reordered


async def test_storage_explicit_scope_prefix_still_wins_over_the_annotation():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    backend = SynapStorageBackend(sdk, "dana", "acme", scope_path=PATH)
    results = await backend.asearch([0.0], scope_prefix="/custom",
                                    metadata_filter={"_query_text": "tea"})
    assert {r.scope for r, _ in results} == {"/custom"}


def test_storage_empty_path_counts_as_unset():
    sdk = RecordingSDK()
    backend = SynapStorageBackend(sdk, "dana", "acme", scope_path={})
    assert backend.scope_path is None and backend._default_scope == "/acme/dana"


# ── listener and report_user_turn ──────────────────────────────────────────


async def _settle(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(0.01)
        if predicate():
            return
    await asyncio.sleep(0.05)


async def _run_listener(sdk, **kw):
    listener = SynapCrewAIListener(sdk, user_id="dana", conversation_id="conv-1",
                                   customer_id="acme", **kw)
    try:
        crewai_event_bus.emit(object(), CrewKickoffStartedEvent(
            crew_name="c", inputs={"q": "where is my order"}))
        crewai_event_bus.emit(object(), CrewKickoffCompletedEvent(
            crew_name="c", output="it ships tomorrow"))
        await _settle(lambda: len(sdk.calls_to("instance.send_message")) >= 2)
    finally:
        listener.stop()


async def test_listener_default_turns_are_unchanged():
    sdk = RecordingSDK()
    await _run_listener(sdk)
    sent = sdk.kwargs_of("instance.send_message")
    assert sorted(k["event_type"] for k in sent) == ["assistant_message", "user_message"]
    assert all("scope_path" not in k for k in sent)


async def test_listener_path_reaches_user_and_assistant_turns():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    await _run_listener(sdk, scope_path=PATH)
    sent = sdk.kwargs_of("instance.send_message")
    assert sorted(k["event_type"] for k in sent) == ["assistant_message", "user_message"]
    assert [k["scope_path"] for k in sent] == [PATH, PATH]


async def test_listener_path_reaches_the_rest_fallback():
    sdk = RecordingSDK(ladder=make_ladder("deep"), listening=False)
    set_auto_listen(sdk, False)
    listener = SynapCrewAIListener(sdk, user_id="dana", conversation_id="conv-1",
                                   customer_id="acme", scope_path=PATH)
    try:
        crewai_event_bus.emit(object(), CrewKickoffStartedEvent(crew_name="c", inputs={"q": "hi"}))
        crewai_event_bus.emit(object(), CrewKickoffCompletedEvent(crew_name="c", output="done"))
        await _settle(lambda: len(sdk.calls_to("conversation.record_message")) >= 2)
    finally:
        listener.stop()
    recorded = sdk.kwargs_of("conversation.record_message")
    assert [k["scope_path"] for k in recorded] == [PATH, PATH]


async def test_report_user_turn_default_and_path():
    plain = RecordingSDK()
    await report_user_turn(plain, "hi", conversation_id="c", user_id="dana", customer_id="acme")
    assert "scope_path" not in plain.last_kwargs("instance.send_message")

    sdk = RecordingSDK(ladder=make_ladder("deep"))
    await report_user_turn(sdk, "hi", conversation_id="c", user_id="dana",
                           customer_id="acme", scope_path=PATH)
    assert sdk.last_kwargs("instance.send_message")["scope_path"] == PATH


# ── conversation-only read ─────────────────────────────────────────────────


async def test_short_term_backstory_takes_no_path_and_still_works():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    SynapStorageBackend(sdk, "dana", "acme", scope_path=PATH)  # a path-configured setup
    await build_synap_st_backstory(sdk, "conv-1", base_backstory="You help.")
    calls = sdk.kwargs_of("conversation.context.get_context_for_prompt")
    assert calls and all("scope_path" not in k and "user_id" not in k for k in calls)


# ── ids that disagree with the path are refused ─────────────────────────────

from synap_integrations_common import SynapIntegrationError  # noqa: E402


async def test_construction_refuses_ids_that_disagree_with_the_path():
    sdk = RecordingSDK()
    builders = (
        lambda u, c: SynapStorageBackend(sdk, u, c, scope_path=PATH),
        lambda u, c: SynapCrewAIListener(sdk, user_id=u, conversation_id="conv-1",
                                         customer_id=c, scope_path=PATH),
    )
    for build in builders:
        with pytest.raises(SynapIntegrationError):
            build("bob", "acme")      # alice's path, bob's id
        with pytest.raises(SynapIntegrationError):
            build("dana", "other")
    ok = SynapStorageBackend(sdk, "dana", "acme", scope_path=PATH)
    SynapStorageBackend(sdk, "dana", "", scope_path=PATH)
    SynapStorageBackend(sdk, "bob", "other")  # no path: unchanged
    listener = SynapCrewAIListener(sdk, user_id="dana", conversation_id="conv-1",
                                   customer_id="acme", scope_path=PATH)
    listener.stop()
    assert ok.scope_path == PATH
    assert sdk.calls == []


async def test_report_user_turn_refuses_another_user_and_sends_nothing():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    with pytest.raises(SynapIntegrationError):
        await report_user_turn(sdk, "hi", conversation_id="c", user_id="bob",
                               customer_id="acme", scope_path=PATH)
    assert sdk.calls == []


def test_a_path_value_with_a_slash_is_refused():
    """The annotation joins values with '/', so a value containing one is ambiguous."""
    with pytest.raises(SynapIntegrationError):
        SynapStorageBackend(RecordingSDK(), "dana", "acme",
                            scope_path={"customer": "acme", "user": "da/na"})


# ── the default REST fallback is exactly what it was ────────────────────────


async def test_default_rest_fallback_kwargs_are_unchanged():
    sdk = RecordingSDK(listening=False)
    set_auto_listen(sdk, False)
    await report_user_turn(sdk, "where is my order", conversation_id="conv-1",
                           user_id="dana", customer_id="acme")
    listener = SynapCrewAIListener(sdk, user_id="dana", conversation_id="conv-1", customer_id="acme")
    try:
        crewai_event_bus.emit(object(), CrewKickoffCompletedEvent(crew_name="c", output="tomorrow"))
        await _settle(lambda: len(sdk.calls_to("conversation.record_message")) >= 2)
    finally:
        listener.stop()
    assert sdk.kwargs_of("conversation.record_message") == [
        {"conversation_id": coerce_conversation_id("conv-1"), "content": "where is my order", "role": "user",
         "user_id": "dana", "customer_id": "acme"},
        {"conversation_id": coerce_conversation_id("conv-1"), "content": "tomorrow", "role": "assistant",
         "user_id": "dana", "customer_id": "acme"},
    ]
