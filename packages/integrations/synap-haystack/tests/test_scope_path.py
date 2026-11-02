"""``scope_path`` (a deeper scope ladder) reaches every SDK call in this package."""

from __future__ import annotations

import pytest
from haystack import Document
from haystack.dataclasses import ChatMessage

from synap_integrations_common.testing import RecordingSDK, make_ladder

from synap_haystack import (
    SynapMemoryRetriever,
    SynapMemoryStore,
    SynapMemoryWriter,
    SynapRetriever,
    SynapShortTermContext,
)

PATH = {"customer": "acme", "team": "payments", "user": "dana"}
OTHER = {"customer": "evil", "team": "x", "user": "y"}


@pytest.fixture
def sdk():
    return RecordingSDK(ladder=make_ladder("deep"))


def store_of(sdk, **kw):
    kw.setdefault("user_id", "dana")
    kw.setdefault("customer_id", "acme")
    return SynapMemoryStore(sdk, **kw)


def MSG():
    # fresh each time: the store stamps a message it has recorded
    return [ChatMessage.from_user("I like window seats")]


# -- reads ------------------------------------------------------------------


async def test_search_default_kwargs_unchanged(sdk):
    await store_of(sdk).asearch_memories(query="seat")
    assert sdk.last_kwargs("fetch") == {
        "user_id": "dana", "customer_id": "acme", "search_query": ["seat"],
        "max_results": 20, "mode": "accurate", "include_conversation_context": False,
    }


async def test_search_memories_carries_path(sdk):
    await store_of(sdk, scope_path=PATH).asearch_memories(query="seat")
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


async def test_search_documents_carries_path(sdk):
    await store_of(sdk, scope_path=PATH).asearch_documents(query="seat")
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


async def test_search_as_single_message_carries_path(sdk):
    await store_of(sdk, scope_path=PATH)._asearch_single(query="seat")
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


def test_sync_search_carries_path(sdk):
    store_of(sdk, scope_path=PATH).search_memories(query="seat")
    store_of(sdk, scope_path=PATH).search_memories_as_single_message(query="seat")
    store_of(sdk, scope_path=PATH).search_documents(query="seat")
    assert [k["scope_path"] for k in sdk.kwargs_of("fetch")] == [PATH] * 3


async def test_per_call_path_wins(sdk):
    await store_of(sdk, scope_path=PATH).asearch_memories(query="q", scope_path=OTHER)
    assert sdk.last_kwargs("fetch")["scope_path"] == OTHER


async def test_per_call_id_override_with_store_path_is_refused(sdk):
    store = store_of(sdk, scope_path=PATH)
    with pytest.raises(ValueError, match="scope_path"):
        await store.aadd_memories(messages=MSG(), conversation_id="c", user_id="mallory")
    # the search read path degrades to [] rather than raising, but never sends the id
    # alone: the ValueError is raised before the try block.
    with pytest.raises(ValueError, match="scope_path"):
        await store.asearch_memories(query="q", customer_id="other")
    assert sdk.calls == []


def test_path_alone_is_a_scope(sdk):
    assert SynapMemoryStore(sdk, scope_path=PATH).scope_path == PATH
    with pytest.raises(ValueError):
        SynapMemoryStore(sdk)


# -- writes -----------------------------------------------------------------


async def test_turn_on_the_stream_default_unchanged(sdk):
    await store_of(sdk).aadd_memories(messages=MSG(), conversation_id="c1")
    kw = sdk.last_kwargs("instance.send_message")
    assert "scope_path" not in kw and kw["user_id"] == "dana"


async def test_turn_on_the_stream_carries_path(sdk):
    await store_of(sdk, scope_path=PATH).aadd_memories(messages=MSG(), conversation_id="c1")
    assert sdk.last_kwargs("instance.send_message")["scope_path"] == PATH


async def test_rest_fallback_turn_carries_path_and_default_does_not():
    for path, expect in ((PATH, True), (None, False)):
        sdk = RecordingSDK(ladder=make_ladder("deep"), listening=False)

        async def no_stream():
            raise RuntimeError("no stream")

        sdk.instance.listen = no_stream
        await store_of(sdk, scope_path=path).aadd_memories(
            messages=[ChatMessage.from_user(f"hello {expect}")], conversation_id="c1")
        kw = sdk.last_kwargs("conversation.record_message")
        assert ("scope_path" in kw) is expect
        if expect:
            assert kw["scope_path"] == PATH


async def test_write_messages_carries_path(sdk):
    n = await store_of(sdk, scope_path=PATH).awrite_messages("c1", MSG())
    assert n == 1 and sdk.last_kwargs("instance.send_message")["scope_path"] == PATH


async def test_customer_scoped_path_only_store_uses_rest_with_path():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    store = SynapMemoryStore(sdk, scope_path={"customer": "acme"})
    await store.aadd_memories(messages=MSG(), conversation_id="c1")
    assert sdk.calls_to("instance.send_message") == []  # no user_id: REST, as before
    assert sdk.last_kwargs("conversation.record_message")["scope_path"] == {"customer": "acme"}


async def test_serialisation_round_trip_keeps_the_path_and_default_omits_it(sdk):
    d = store_of(sdk, scope_path=PATH).to_dict()["init_parameters"]
    assert d["scope_path"] == PATH
    assert "scope_path" not in store_of(sdk).to_dict()["init_parameters"]


# -- components -------------------------------------------------------------


def test_retrievers_build_a_store_with_the_path(sdk):
    r1 = SynapRetriever(sdk=sdk, user_id="dana", customer_id="acme", scope_path=PATH)
    r2 = SynapMemoryRetriever(sdk=sdk, user_id="dana", customer_id="acme", scope_path=PATH)
    r1.run("q")
    r2.run("q")
    assert [k["scope_path"] for k in sdk.kwargs_of("fetch")] == [PATH, PATH]


def test_retrievers_without_path_unchanged(sdk):
    SynapRetriever(sdk=sdk, user_id="dana", customer_id="acme").run("q")
    assert "scope_path" not in sdk.last_kwargs("fetch")


def test_retriever_path_alone_and_path_with_store(sdk):
    SynapRetriever(sdk=sdk, scope_path=PATH)
    with pytest.raises(ValueError, match="ambiguous"):
        SynapRetriever(store=store_of(sdk), scope_path=PATH)
    with pytest.raises(ValueError, match="ambiguous"):
        SynapMemoryRetriever(store=store_of(sdk), scope_path=PATH)


def test_writer_carries_path_on_every_turn(sdk):
    w = SynapMemoryWriter(sdk=sdk, conversation_id="c1", user_id="dana",
                          customer_id="acme", scope_path=PATH)
    out = w.run(documents=[Document(content="hi", meta={"role": "user"})],
                messages=[ChatMessage.from_assistant("hello")])
    assert out["written_count"] == 2
    assert [k["scope_path"] for k in sdk.kwargs_of("instance.send_message")] == [PATH, PATH]


def test_writer_without_path_unchanged(sdk):
    SynapMemoryWriter(sdk=sdk, conversation_id="c1", user_id="dana").run(
        messages=[ChatMessage.from_user("hi")])
    assert "scope_path" not in sdk.last_kwargs("instance.send_message")


def test_writer_with_store_and_path_is_refused(sdk):
    with pytest.raises(ValueError, match="ambiguous"):
        SynapMemoryWriter(store=store_of(sdk), conversation_id="c1", scope_path=PATH)
    SynapMemoryWriter(sdk=sdk, conversation_id="c1", scope_path=PATH)  # path alone is a scope


# -- hooks and conversation-only reads --------------------------------------


class _State(dict):
    def set(self, key, value, handler_override=None):
        self[key] = value


def test_record_hook_carries_the_stores_path(sdk):
    pytest.importorskip("haystack.hooks", exc_type=ImportError)
    from synap_haystack import SynapRecordHook

    hook = SynapRecordHook(store_of(sdk, scope_path=PATH), "c1")
    hook.run(_State(messages=[ChatMessage.from_user("hello there")]))
    assert sdk.last_kwargs("instance.send_message")["scope_path"] == PATH


async def test_history_hook_reads_without_a_path():
    pytest.importorskip("haystack.hooks", exc_type=ImportError)
    from maximem_synap.models.context import ContextForPromptResponse
    from synap_haystack import SynapHistoryHook

    sdk = RecordingSDK(ladder=make_ladder("deep"))
    sdk.results["conversation.context.get_context_for_prompt"] = ContextForPromptResponse(
        formatted_context="x", context_block="## Conversation so far (compacted 1x)\nS",
        available=True)
    hook = SynapHistoryHook(store_of(sdk, scope_path=PATH), "c1")
    state = _State(messages=[ChatMessage.from_user("a"), ChatMessage.from_assistant("b"),
                             ChatMessage.from_user("c")])
    await hook.run_async(state)
    assert any("compacted" in (m.text or "") for m in state["messages"])
    kw = sdk.last_kwargs("conversation.context.get_context_for_prompt")
    assert "scope_path" not in kw and "scope" not in kw


async def test_retrieve_messages_and_short_term_work_with_a_path(sdk):
    from maximem_synap.models.context import ContextForPromptResponse

    sdk.results["conversation.context.get_context_for_prompt"] = ContextForPromptResponse(
        formatted_context="x", context_block="## Conversation so far (compacted 1x)\nS",
        available=True)
    msgs = await store_of(sdk, scope_path=PATH).aretrieve_messages("c1")
    assert msgs and "compacted" in msgs[0].text
    out = SynapShortTermContext(sdk, "c1", system="sys").run()["synap_st"]
    assert "compacted" in out
    for kw in sdk.kwargs_of("conversation.context.get_context_for_prompt"):
        assert "scope_path" not in kw and "scope" not in kw


async def test_older_sdk_not_sent_the_keyword_when_unset():
    sdk = RecordingSDK(accept_scope_path=False)
    store = store_of(sdk)
    await store.asearch_memories(query="q")
    await store.aadd_memories(messages=MSG(), conversation_id="c")
    assert len(sdk.calls_to("fetch")) == 1 and len(sdk.calls_to("instance.send_message")) == 1


# -- serialisation: a dict without a path is what it was before ------------

BASELINE_COMPONENT_KEYS = {
    "SynapRetriever": ["sdk", "user_id", "customer_id", "conversation_id", "mode",
                       "max_results", "store"],
    "SynapMemoryRetriever": ["store", "sdk", "user_id", "customer_id", "conversation_id",
                             "mode", "max_results"],
    "SynapMemoryWriter": ["sdk", "conversation_id", "user_id", "customer_id", "store"],
}
BASELINE_STORE_KEYS = ["instance_id", "user_id", "customer_id", "conversation_id", "mode",
                       "max_results", "include_conversation_context"]


def _components(sdk, **kw):
    return [
        SynapRetriever(sdk=sdk, user_id="u", **kw),
        SynapMemoryRetriever(sdk=sdk, user_id="u", **kw),
        SynapMemoryWriter(sdk=sdk, conversation_id="c", user_id="u", **kw),
    ]


def test_to_dict_without_a_path_has_the_baseline_shape(sdk):
    for comp in _components(sdk):
        d = comp.to_dict()["init_parameters"]
        assert list(d) == BASELINE_COMPONENT_KEYS[type(comp).__name__]
        assert "scope_path" not in d
        assert list(d["store"]["init_parameters"]) == BASELINE_STORE_KEYS


def test_to_dict_with_a_path_carries_it_on_the_store_only(sdk):
    for comp in _components(sdk, scope_path=PATH):
        d = comp.to_dict()["init_parameters"]
        assert "scope_path" not in d  # no second copy: from_dict would see both
        assert d["store"]["init_parameters"]["scope_path"] == PATH


def test_pipeline_to_dict_has_no_null_scope_path(sdk):
    from haystack import Pipeline

    p = Pipeline()
    for name, comp in zip("rmw", _components(sdk)):
        p.add_component(name, comp)
    assert '"scope_path"' not in __import__("json").dumps(p.to_dict(), default=str)


@pytest.mark.parametrize("path", [None, PATH])
def test_store_from_dict_round_trips_with_and_without_a_path(sdk, monkeypatch, path):
    import synap_haystack.store as store_mod

    monkeypatch.setattr(store_mod, "MaximemSynapSDK", lambda instance_id="": sdk)
    original = store_of(sdk, scope_path=path)
    restored = SynapMemoryStore.from_dict(original.to_dict())
    assert restored.scope_path == (PATH if path else None)
    assert restored.to_dict() == original.to_dict()
