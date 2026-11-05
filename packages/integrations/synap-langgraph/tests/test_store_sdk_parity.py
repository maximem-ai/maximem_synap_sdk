"""0.6.0: every Python SDK capability on SynapStore, and the BaseStore methods
Synap does not have raise SynapStoreUnsupportedError.

The SDK is the RecordingSDK stand-in (real signatures: a keyword the SDK does
not take raises TypeError) with the remaining SDK methods recorded the same way.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.store.base import GetOp, ListNamespacesOp, PutOp, SearchOp

from maximem_synap import MaximemSynapSDK
from maximem_synap.memories.interface import MemoriesInterface
from maximem_synap.sdk import ConversationContextInterface, ConversationInterface, UserInterface
from synap_integrations_common import SynapIntegrationError
from synap_integrations_common.testing import RecordedCall
from synap_langgraph import SynapStore, SynapStoreUnsupportedError

from ._support import UUIDS, Sdk, response_of, ranked

NS = ("acme", "alice")
MID = UUIDS[1]


def run(coro):
    return asyncio.run(coro)


class ParitySdk(Sdk):
    """Sdk plus every other SDK method the store calls, recorded with its real signature."""

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.memories.delete = self._rec("memories.delete", MemoriesInterface, "delete")
        for attr in ("status", "wait_for_completion", "get", "update", "create_from_file"):
            setattr(self.memories, attr, self._rec(f"memories.{attr}", MemoriesInterface, attr))
        self.user = SimpleNamespace(get_profile=self._rec("user.get_profile", UserInterface, "get_profile"))
        self.conversation.ingest_transcript = self._rec(
            "conversation.ingest_transcript", ConversationInterface, "ingest_transcript")
        from maximem_synap.data.interface import DataInterface
        self.data = SimpleNamespace(**{a: self._rec(f"data.{a}", DataInterface, a)
                                       for a in ("delete", "restore", "status")})
        for attr in ("compact", "get_compacted", "get_compaction_status", "get_context_for_prompt"):
            setattr(self.conversation.context, attr,
                    self._rec(f"conversation.context.{attr}", ConversationContextInterface, attr))

    def _rec(self, name: str, owner: Any, attr: str):
        sig = inspect.signature(getattr(owner, attr))
        sig = sig.replace(parameters=[p for n, p in sig.parameters.items() if n != "self"])

        async def call(*args: Any, **kwargs: Any) -> Any:
            sig.bind(*args, **kwargs)
            self.calls.append(RecordedCall(name, args, dict(kwargs)))
            if name in self.errors:
                raise self.errors[name]
            return self.results.get(name, SimpleNamespace(name=name))

        return call

    def last(self, name: str) -> RecordedCall:
        return self.calls_to(name)[-1]


# ── decision 2: no key lookup ────────────────────────────────────────────────

@pytest.mark.parametrize("call", [
    lambda s: s.aget(NS, "k"),
    lambda s: s.adelete(NS, "k"),
    lambda s: s.alist_namespaces(),
])
def test_async_unsupported_methods_raise_and_touch_nothing(call):
    sdk = ParitySdk()
    with pytest.raises(SynapStoreUnsupportedError) as ei:
        run(call(SynapStore(sdk)))
    assert isinstance(ei.value, NotImplementedError)
    assert "search-based" in str(ei.value)
    assert sdk.calls == []


def test_sync_unsupported_methods_raise_and_name_the_alternative():
    store = SynapStore(ParitySdk())
    with pytest.raises(SynapStoreUnsupportedError, match=r"search\(\)"):
        store.get(NS, "k")
    with pytest.raises(SynapStoreUnsupportedError, match="adelete_memory"):
        store.delete(NS, "k")
    with pytest.raises(SynapStoreUnsupportedError, match="namespaces"):
        store.list_namespaces()


@pytest.mark.parametrize("op", [GetOp(NS, "k"), PutOp(NS, "k", None), ListNamespacesOp()])
def test_a_batch_with_an_unsupported_op_runs_nothing(op):
    sdk = ParitySdk()
    store = SynapStore(sdk)
    with pytest.raises(SynapStoreUnsupportedError):
        run(store.abatch([PutOp(NS, "x", {"v": 1}), op]))
    with pytest.raises(SynapStoreUnsupportedError):
        store.batch([PutOp(NS, "x", {"v": 1}), op])
    assert sdk.calls_to("memories.create") == []


def test_put_and_search_still_work():
    sdk = ParitySdk(response_of(ranked(MID, 1, "facts", "Alice likes tea", 0.9, "user")))
    store = SynapStore(sdk)
    assert run(store.aput(NS + ("prefs",), "drink", {"v": "tea"})) is None
    hits = run(store.asearch(NS, query="drink"))
    assert [h.value["_synap"]["id"] for h in hits] == [MID]
    kw = sdk.last_kwargs("memories.create")
    assert kw["user_id"] == "alice" and kw["customer_id"] == "acme"
    assert kw["metadata"]["lg_store_key"] == "drink"      # still written: search reads it


# ── put with the SDK's create options ───────────────────────────────────────

def test_put_passes_create_options_and_keeps_its_tag():
    sdk = ParitySdk()
    when = datetime(2023, 3, 1, tzinfo=timezone.utc)
    out = run(SynapStore(sdk).aput(
        NS + ("notes",), "n1", {"t": "x"}, document_type="email", document_id="d-1",
        document_created_at=when, mode="fast", objective="remember", metadata={"src": "crm"}))
    assert out is None
    kw = sdk.last_kwargs("memories.create")
    assert (kw["document_type"], kw["document_id"], kw["document_created_at"], kw["mode"],
            kw["objective"]) == ("email", "d-1", when, "fast", "remember")
    assert kw["metadata"]["src"] == "crm" and kw["metadata"]["lg_store_ns"] == "notes"
    assert kw["user_id"] == "alice"


def test_put_without_options_sends_what_it_always_sent():
    sdk = ParitySdk()
    run(SynapStore(sdk).aput(NS, "k", {"a": 1}))
    assert set(sdk.last_kwargs("memories.create")) == {"document", "metadata", "user_id", "customer_id"}


def test_put_wait_returns_the_ingestion_status():
    sdk = ParitySdk()
    sdk.results["memories.wait_for_completion"] = "done-status"
    store = SynapStore(sdk)
    assert run(store.aput(NS, "k", {"a": 1}, wait=True, timeout_seconds=5,
                          poll_interval_seconds=1)) == "done-status"
    call = sdk.last("memories.wait_for_completion")
    assert call.args == ("ing-001",) and call.kwargs == {"timeout_seconds": 5, "poll_interval_seconds": 1}
    assert store.put(NS, "k", {"a": 1}, wait=True) == "done-status"


def test_sync_put_passes_options():
    sdk = ParitySdk()
    SynapStore(sdk).put(NS, "k", {"a": 1}, document_type="note")
    assert sdk.last_kwargs("memories.create")["document_type"] == "note"


def test_put_metadata_may_not_overwrite_the_store_tag():
    with pytest.raises(ValueError, match="lg_store_key"):
        run(SynapStore(ParitySdk()).aput(NS, "k", {"a": 1}, metadata={"lg_store_key": "x"}))


def test_put_with_options_still_refuses_ttl_and_none():
    store = SynapStore(ParitySdk())
    with pytest.raises(NotImplementedError, match="TTL"):
        run(store.aput(NS, "k", {"a": 1}, ttl=5, mode="fast"))
    with pytest.raises(SynapStoreUnsupportedError):
        run(store.aput(NS, "k", None, mode="fast"))  # type: ignore[arg-type]


# ── search with per-call fetch options ──────────────────────────────────────

def test_search_takes_fetch_options_for_one_call():
    sdk = ParitySdk()
    store = SynapStore(sdk, fetch_options={"types": ["facts"]})
    run(store.asearch(NS, query="q", mode="fast", precision_level="medium",
                      query_date="2023-03-11", max_results_per_query=3))
    kw = sdk.last_kwargs("fetch")
    assert (kw["mode"], kw["precision_level"], kw["query_date"], kw["max_results_per_query"],
            kw["types"]) == ("fast", "medium", "2023-03-11", 3, ["facts"])
    run(store.asearch(NS, query="q"))
    assert sdk.last_kwargs("fetch")["mode"] == "accurate"   # one call only
    store.search(NS, query="q", precision_level="low")
    assert sdk.last_kwargs("fetch")["precision_level"] == "low"


@pytest.mark.parametrize("bad", ["user_id", "response_shape", "max_results", "search_query"])
def test_search_refuses_the_stores_own_arguments(bad):
    with pytest.raises(ValueError):
        run(SynapStore(ParitySdk()).asearch(NS, query="q", **{bad: "x"}))


# ── memories ───────────────────────────────────────────────────────────────

def test_create_memory_sends_the_document_as_given():
    sdk = ParitySdk()
    parts = ["a note", {"k": "v"}]
    out = run(SynapStore(sdk).acreate_memory(NS, parts, document_type="ticket", mode="fast"))
    kw = sdk.last_kwargs("memories.create")
    assert kw["document"] is parts and kw["document_type"] == "ticket" and kw["mode"] == "fast"
    assert "metadata" not in kw and kw["user_id"] == "alice" and kw["customer_id"] == "acme"
    assert out.ingestion_id == "ing-001"
    assert SynapStore(sdk).create_memory(NS, "x").ingestion_id == "ing-001"


def test_create_memory_wait():
    sdk = ParitySdk()
    sdk.results["memories.wait_for_completion"] = "completed"
    assert run(SynapStore(sdk).acreate_memory(NS, "x", wait=True)) == "completed"


def test_create_memory_on_a_deeper_ladder_writes_the_scope():
    sdk = ParitySdk(ladder="deep")
    run(SynapStore(sdk).acreate_memory(("acme", "payments", "dana"), "x"))
    assert sdk.last_kwargs("memories.create")["scope"] == {
        "customer": "acme", "team": "payments", "user": "dana"}


def test_create_memory_from_file():
    sdk = ParitySdk()
    run(SynapStore(sdk).acreate_memory_from_file(NS, text="hello", document_type="doc"))
    kw = sdk.last("memories.create_from_file").kwargs
    assert kw == {"user_id": "alice", "customer_id": "acme", "relationship_type": "b2b",
                  "text": "hello", "document_type": "doc"}


def test_create_memory_from_file_needs_a_user():
    sdk = ParitySdk()
    with pytest.raises(SynapIntegrationError, match="needs a user"):
        run(SynapStore(sdk, level="customer").acreate_memory_from_file(("acme",), text="x"))
    assert sdk.calls_to("memories.create_from_file") == []


def test_ingestion_status_and_wait():
    sdk = ParitySdk()
    store = SynapStore(sdk)
    run(store.aget_ingestion_status("ing-9"))
    assert sdk.last("memories.status").args == ("ing-9",)
    run(store.await_ingestion("ing-9", timeout_seconds=7))
    assert sdk.last("memories.wait_for_completion").kwargs["timeout_seconds"] == 7
    store.get_ingestion_status("ing-8")
    store.wait_for_ingestion("ing-8")
    assert len(sdk.calls_to("memories.status")) == 2


def test_memory_by_id_get_update_delete():
    sdk = ParitySdk()
    store = SynapStore(sdk)
    run(store.aget_memory(NS, MID))
    assert sdk.last("memories.get").args == (MID,)
    run(store.aupdate_memory(NS, MID, "new text", merge_strategy="replace"))
    call = sdk.last("memories.update")
    assert call.args == (MID, "new text") and call.kwargs["merge_strategy"] == "replace"
    run(store.adelete_memory(NS, MID))
    assert sdk.last("memories.delete").args == (MID,)
    store.delete_memory(NS, MID)
    assert len(sdk.calls_to("memories.delete")) == 2


def test_delete_memory_failure_is_an_integration_error():
    sdk = ParitySdk()
    sdk.errors["memories.delete"] = RuntimeError("404")
    with pytest.raises(SynapIntegrationError, match="404"):
        run(SynapStore(sdk).adelete_memory(NS, MID))


def test_forget_uses_the_namespace_identity():
    sdk = ParitySdk()
    run(SynapStore(sdk).aforget(NS + ("prefs",), "my old address", mode="direct"))
    call = sdk.calls_to("memories.forget")[-1]
    assert call.args == ("my old address",)
    assert call.kwargs == {"mode": "direct", "user_id": "alice", "customer_id": "acme"}


def test_single_user_store_methods_use_the_construction_identity():
    sdk = ParitySdk()
    store = SynapStore(sdk, user_id="bob", customer_id="beta")
    run(store.aforget(("anything",), "x"))
    assert sdk.calls_to("memories.forget")[-1].kwargs["user_id"] == "bob"


# ── raw fetch, profile, ladder ──────────────────────────────────────────────

def test_fetch_raw_uses_the_sdks_own_defaults_and_returns_its_response():
    resp = response_of(ranked(MID, 1, "facts", "x", 0.9, "user"))
    sdk = ParitySdk(resp)
    out = run(SynapStore(sdk, mode="accurate").afetch_raw(NS, "tea"))
    assert out is resp
    assert sdk.last_kwargs("fetch") == {
        "search_query": ["tea"], "user_id": "alice", "customer_id": "acme"}


def test_fetch_raw_passes_any_fetch_option():
    sdk = ParitySdk()
    SynapStore(sdk).fetch_raw(NS, ["a", "b"], response_shape="by_type", max_results=50,
                              include_profile=False, conversation_id="thread-1")
    kw = sdk.last_kwargs("fetch")
    assert kw["search_query"] == ["a", "b"] and kw["max_results"] == 50
    assert kw["response_shape"] == "by_type" and kw["include_profile"] is False
    assert kw["conversation_id"] != "thread-1"     # any string becomes the UUID the SDK needs


@pytest.mark.parametrize("bad", ["user_id", "customer_id", "scope_path", "search_query"])
def test_fetch_raw_refuses_identity(bad):
    with pytest.raises(ValueError):
        run(SynapStore(ParitySdk()).afetch_raw(NS, "q", **{bad: "x"}))


def test_fetch_raw_failure_raises_whatever_on_error_is():
    sdk = ParitySdk().fails(RuntimeError("down"))
    with pytest.raises(SynapIntegrationError, match="down"):
        run(SynapStore(sdk, on_error="fallback").afetch_raw(NS, "q"))


def test_profile_and_ladder():
    sdk = ParitySdk()
    store = SynapStore(sdk)
    run(store.aget_profile(NS))
    assert sdk.last("user.get_profile").kwargs == {"user_id": "alice", "customer_id": "acme"}
    assert run(store.aget_scope_ladder()) is sdk.ladder
    assert store.get_scope_ladder(refresh=True) is sdk.ladder


# ── conversations ──────────────────────────────────────────────────────────

def test_ingest_transcript():
    sdk = ParitySdk()
    run(SynapStore(sdk).aingest_transcript(NS, "c1", "user: hi", conversation_type="call"))
    kw = sdk.last("conversation.ingest_transcript").kwargs
    assert (kw["conversation_id"], kw["user_id"], kw["customer_id"], kw["transcript"],
            kw["conversation_type"]) == ("c1", "alice", "acme", "user: hi", "call")


def test_conversation_context_methods():
    sdk = ParitySdk()
    store = SynapStore(sdk)
    run(store.aget_context_for_prompt("c1", style="narrative", query="tea"))
    assert sdk.last("conversation.context.get_context_for_prompt").kwargs == {
        "conversation_id": "c1", "style": "narrative", "search_history": True, "query": "tea"}
    run(store.acompact("c1", strategy="aggressive", force=True))
    assert sdk.last("conversation.context.compact").kwargs["strategy"] == "aggressive"
    run(store.aget_compacted("c1", version=2))
    assert sdk.last("conversation.context.get_compacted").kwargs["version"] == 2
    store.get_compaction_status("c1")
    assert sdk.last("conversation.context.get_compaction_status").kwargs == {"conversation_id": "c1"}


# ── sdk_provider ──────────────────────────────────────────────────────────

def test_new_methods_resolve_the_sdk_from_the_provider():
    sdk = ParitySdk()
    store = SynapStore(sdk_provider=lambda: sdk)
    run(store.adelete_memory(NS, MID))
    store.forget(NS, "x")
    run(store.aput(NS, "k", {"a": 1}, mode="fast"))
    assert sdk.calls_to("memories.delete") and sdk.calls_to("memories.forget")
    assert sdk.last_kwargs("memories.create")["mode"] == "fast"


def test_every_async_method_has_a_sync_twin():
    names = ["create_memory", "create_memory_from_file", "get_ingestion_status", "get_memory",
             "update_memory", "delete_memory", "forget", "fetch_raw", "get_profile",
             "ingest_transcript", "get_context_for_prompt", "compact", "get_compacted",
             "get_compaction_status", "get_scope_ladder", "delete_data", "restore",
             "deletion_status"]
    for n in names:
        assert callable(getattr(SynapStore, n)) and callable(getattr(SynapStore, "a" + n))
    assert callable(SynapStore.await_ingestion) and callable(SynapStore.wait_for_ingestion)


# ── V2 fixes: compaction_level, record_message extras ───────────────────────

def test_compact_passes_compaction_level():
    sdk = ParitySdk()
    run(SynapStore(sdk).acompact("c1", compaction_level="balanced"))
    kw = sdk.last("conversation.context.compact").kwargs
    assert kw["compaction_level"] == "balanced" and kw["strategy"] is None


def test_record_message_plain_call_is_unchanged_and_streams():
    sdk = ParitySdk()
    run(SynapStore(sdk).arecord_message("c1", "user", "hi", namespace=NS))
    assert sdk.calls_to("instance.send_message") and not sdk.calls_to("conversation.record_message")


def test_record_message_model_context_window_and_parts_go_to_the_sdks_record_message():
    from maximem_synap.memories.parts import TextPart
    sdk = ParitySdk()
    store = SynapStore(sdk)
    conv = "00000000-0000-0000-0000-0000000000aa"
    parts = ["look at this", TextPart(text="more")]
    run(store.arecord_message(conv, "user", parts, namespace=NS, model="gpt-4o",
                              context_window=128000, session_id="s1"))
    kw = sdk.last_kwargs("conversation.record_message")
    assert kw == {"conversation_id": conv, "role": "user", "content": parts,
                  "user_id": "alice", "customer_id": "acme", "session_id": "s1",
                  "model": "gpt-4o", "context_window": 128000}
    assert not sdk.calls_to("instance.send_message")      # one transport, never both
    store.record_message(conv, "assistant", "ok", namespace=NS, model="gpt-4o")
    assert sdk.last_kwargs("conversation.record_message")["model"] == "gpt-4o"


def test_record_message_rest_failure_is_logged_not_raised():
    sdk = ParitySdk()
    sdk.errors["conversation.record_message"] = RuntimeError("down")
    run(SynapStore(sdk).arecord_message("c1", "user", "hi", namespace=NS, model="m"))


# ── permanent delete and sdk.data ───────────────────────────────────────────

def test_delete_memory_forwards_permanent():
    sdk = ParitySdk()
    store = SynapStore(sdk)
    run(store.adelete_memory(NS, MID))
    assert sdk.last("memories.delete").kwargs == {}            # default: soft, as before
    run(store.adelete_memory(NS, MID, permanent=True))
    assert sdk.last("memories.delete").kwargs == {"permanent": True}
    store.delete_memory(NS, MID, permanent=True)
    assert sdk.last("memories.delete").kwargs == {"permanent": True}


def test_delete_data_user_scope_takes_identity_from_the_namespace():
    sdk = ParitySdk()
    run(SynapStore(sdk).adelete_data(NS + ("prefs",), "user", confirm="alice"))
    assert sdk.last("data.delete").kwargs == {
        "scope": "user", "dry_run": False, "confirm": "alice", "wait": True,
        "customer_id": "acme", "user_id": "alice"}


def test_delete_data_conversation_and_sdk_options():
    sdk = ParitySdk()
    SynapStore(sdk).delete_data(NS, "conversation", conversation_id="c1", dry_run=True,
                                wait=False, instances="all", permanent=True)
    kw = sdk.last("data.delete").kwargs
    assert (kw["conversation_id"], kw["dry_run"], kw["wait"], kw["instances"], kw["permanent"],
            kw["user_id"]) == ("c1", True, False, "all", True, "alice")


def test_delete_data_customer_scope_needs_a_customer_level_namespace():
    sdk = ParitySdk()
    with pytest.raises(SynapIntegrationError, match="customer-level"):
        run(SynapStore(sdk).adelete_data(NS, "customer", confirm="acme"))
    assert sdk.calls_to("data.delete") == []
    run(SynapStore(sdk, level="customer").adelete_data(("acme",), "customer", confirm="acme"))
    kw = sdk.last("data.delete").kwargs
    assert kw["customer_id"] == "acme" and "user_id" not in kw


def test_delete_data_refuses_bad_scope_and_identity_arguments():
    store = SynapStore(ParitySdk())
    with pytest.raises(ValueError):
        run(store.adelete_data(NS, "client"))
    with pytest.raises(TypeError):
        run(store.adelete_data(NS, "user", user_id="bob"))   # type: ignore[call-arg]
    with pytest.raises(SynapIntegrationError, match="conversation_id"):
        run(store.adelete_data(NS, "conversation", confirm="x"))
    with pytest.raises(SynapIntegrationError, match="needs a user"):
        run(SynapStore(ParitySdk(), level="customer").adelete_data(("acme",), "user"))


def test_restore_and_deletion_status():
    sdk = ParitySdk()
    store = SynapStore(sdk)
    run(store.arestore(NS, memory_ids=[MID]))
    assert sdk.last("data.restore").kwargs == {"memory_ids": [MID], "deletion_id": None}
    store.restore(NS, deletion_id="del-1")
    assert sdk.last("data.restore").kwargs == {"memory_ids": None, "deletion_id": "del-1"}
    run(store.adeletion_status(NS, "del-1"))
    assert sdk.last("data.status").args == ("del-1",)
    store.deletion_status(NS, "del-2")
    assert sdk.last("data.status").args == ("del-2",)
