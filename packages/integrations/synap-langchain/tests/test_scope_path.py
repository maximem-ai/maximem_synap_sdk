"""``scope_path`` reaches every SDK call these classes make, and only when set.

Built on ``RecordingSDK``, which keeps each call's exact keyword arguments and
refuses a keyword the installed SDK does not take. A call WITHOUT ``scope_path``
must be exactly the call made before the keyword existed.
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from maximem_synap import MaximemSynapSDK
from synap_integrations_common.testing import RecordingSDK as _RecordingSDK, make_ladder

from synap_langchain import (
    SynapCallbackHandler,
    SynapChatMessageHistory,
    SynapRetriever,
    SynapSearchTool,
    SynapStoreTool,
    create_synap_node,
)



class RecordingSDK(_RecordingSDK):
    """RecordingSDK that passes pydantic's ``isinstance(sdk, MaximemSynapSDK)``
    (the retriever and the tools declare that field type)."""

    @property
    def __class__(self):  # noqa: D105
        return MaximemSynapSDK


PATH = {"customer": "acme", "team": "payments", "user": "dana"}


def run(coro):
    return asyncio.run(coro)


# ── retriever ────────────────────────────────────────────────────────────────

RETRIEVER_DEFAULT = {
    "conversation_id": None, "user_id": "dana", "customer_id": None,
    "search_query": ["q"], "max_results": 20, "types": None, "mode": "accurate",
    "include_conversation_context": False,
}


def test_retriever_default_call_is_unchanged():
    sdk = RecordingSDK()
    run(SynapRetriever(sdk=sdk, user_id="dana")._aget_relevant_documents("q", run_manager=None))
    assert sdk.last_kwargs("fetch") == RETRIEVER_DEFAULT


def test_retriever_default_call_works_on_an_sdk_without_scope_path():
    sdk = RecordingSDK(accept_scope_path=False)
    run(SynapRetriever(sdk=sdk, user_id="dana")._aget_relevant_documents("q", run_manager=None))
    assert "scope_path" not in sdk.last_kwargs("fetch")


def test_retriever_forwards_scope_path():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    r = SynapRetriever(sdk=sdk, user_id="dana", customer_id="acme", scope_path=PATH)
    run(r._aget_relevant_documents("q", run_manager=None))
    kw = sdk.last_kwargs("fetch")
    assert kw["scope_path"] == PATH
    assert kw["user_id"] == "dana" and kw["customer_id"] == "acme"
    assert {k: v for k, v in kw.items() if k not in ("scope_path", "customer_id")} == {
        k: v for k, v in RETRIEVER_DEFAULT.items() if k != "customer_id"}


# ── tools ────────────────────────────────────────────────────────────────────

def test_search_tool_default_and_path():
    sdk = RecordingSDK()
    run(SynapSearchTool(sdk=sdk, user_id="dana")._arun("q"))
    assert sdk.last_kwargs("fetch") == {
        "conversation_id": None, "user_id": "dana", "customer_id": None,
        "search_query": ["q"], "mode": "accurate", "max_results": 10,
        "include_conversation_context": False,
    }
    sdk.reset()
    run(SynapSearchTool(sdk=sdk, user_id="dana", customer_id="acme", scope_path=PATH)._arun("q"))
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


def test_store_tool_default_and_path():
    sdk = RecordingSDK()
    run(SynapStoreTool(sdk=sdk, user_id="dana")._arun("remember"))
    assert sdk.last_kwargs("memories.create") == {
        "document": "remember", "user_id": "dana", "customer_id": ""}
    sdk.reset()
    run(SynapStoreTool(sdk=sdk, user_id="dana", customer_id="acme", scope_path=PATH)._arun("remember"))
    assert sdk.last_kwargs("memories.create") == {
        "document": "remember", "user_id": "dana", "customer_id": "acme", "scope": PATH}
    assert "scope_path" not in sdk.last_kwargs("memories.create")


def test_store_tool_default_works_on_an_sdk_without_scope():
    sdk = RecordingSDK(accept_scope_path=False)
    run(SynapStoreTool(sdk=sdk, user_id="dana")._arun("remember"))
    assert "scope" not in sdk.last_kwargs("memories.create")


# ── graph node ───────────────────────────────────────────────────────────────

STATE = {"messages": [{"role": "user", "content": "hello"}]}
NODE_DEFAULT = {
    "conversation_id": None, "user_id": "dana", "customer_id": None,
    "search_query": ["hello"], "max_results": 20, "mode": "fast",
    "include_conversation_context": False, "include_scope_labels": False,
}


def test_graph_node_default_call_is_unchanged():
    sdk = RecordingSDK()
    run(create_synap_node(sdk, user_id="dana")(STATE))
    assert sdk.last_kwargs("fetch") == NODE_DEFAULT


def test_graph_node_forwards_scope_path():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    run(create_synap_node(sdk, user_id="dana", customer_id="acme", scope_path=PATH)(STATE))
    kw = sdk.last_kwargs("fetch")
    assert kw["scope_path"] == PATH and kw["customer_id"] == "acme"


# ── chat history (turns) ─────────────────────────────────────────────────────

def test_history_default_turn_is_unchanged():
    sdk = RecordingSDK()
    h = SynapChatMessageHistory(sdk, "c1", "dana")
    run(h.aadd_messages([HumanMessage("hi")]))
    kw = sdk.last_kwargs("instance.send_message")
    assert kw["content"] == "hi" and kw["user_id"] == "dana" and kw["conversation_id"] == "c1"
    assert "scope_path" not in kw


def test_history_turn_carries_scope_path_on_the_stream():
    sdk = RecordingSDK()
    h = SynapChatMessageHistory(sdk, "c1", "dana", "acme", scope_path=PATH)
    run(h.aadd_messages([HumanMessage("hi"), AIMessage("hello")]))
    calls = sdk.kwargs_of("instance.send_message")
    assert [c["role"] for c in calls] == ["user", "assistant"]
    assert all(c["scope_path"] == PATH for c in calls)


def test_history_turn_carries_scope_path_over_rest():
    sdk = RecordingSDK(listening=False)
    sdk.errors["instance.send_message"] = RuntimeError("stream down")
    h = SynapChatMessageHistory(sdk, "c1", "dana", scope_path=PATH)
    run(h.aadd_messages([HumanMessage("hi")]))
    assert sdk.last_kwargs("conversation.record_message")["scope_path"] == PATH


def test_history_from_instance_accepts_scope_path(monkeypatch):
    import synap_langchain.memory as mem
    monkeypatch.setattr(mem, "MaximemSynapSDK", lambda instance_id: RecordingSDK())
    h = SynapChatMessageHistory.from_instance("inst", "c1", "dana", scope_path=PATH)
    assert h.scope_path == PATH


def test_history_reads_are_conversation_only():
    """get_context_for_prompt takes a conversation id and no path (design F8)."""
    sdk = RecordingSDK()
    run(SynapChatMessageHistory(sdk, "c1", "dana", scope_path=PATH).aget_messages())
    assert sdk.last_kwargs("conversation.context.get_context_for_prompt") == {
        "conversation_id": "c1"}


# ── callback handler (turns; base of synap_langgraph's handler) ─────────────

async def _chat_run(handler):
    from uuid import uuid4
    rid = uuid4()
    await handler.on_chain_start({}, {"messages": [HumanMessage("question")]}, run_id=rid)
    await handler.on_chain_end({"messages": [AIMessage("answer")]}, run_id=rid)


def test_handler_default_turns_are_unchanged():
    sdk = RecordingSDK()
    run(_chat_run(SynapCallbackHandler(sdk, "c1", "dana")))
    calls = sdk.kwargs_of("instance.send_message")
    assert [c["role"] for c in calls] == ["user", "assistant"]
    assert all("scope_path" not in c for c in calls)


def test_handler_turns_carry_scope_path():
    sdk = RecordingSDK()
    run(_chat_run(SynapCallbackHandler(sdk, "c1", "dana", "acme", scope_path=PATH)))
    calls = sdk.kwargs_of("instance.send_message")
    assert [c["role"] for c in calls] == ["user", "assistant"]
    assert all(c["scope_path"] == PATH for c in calls)


def test_handler_tool_events_never_carry_a_path():
    """The SDK's record_tool_* take no scope_path, so none is sent."""
    sdk = RecordingSDK()
    h = SynapCallbackHandler(sdk, "c1", "dana", scope_path=PATH)
    from uuid import uuid4
    rid = uuid4()
    run(h.on_tool_start({"name": "t"}, "x", run_id=rid, inputs={"a": 1}))
    run(h.on_tool_end("out", run_id=rid))
    for name in ("instance.record_tool_call", "instance.record_tool_result"):
        assert "scope_path" not in sdk.last_kwargs(name)


def test_handler_default_turns_work_on_an_sdk_without_scope_path():
    sdk = RecordingSDK(accept_scope_path=False)
    run(_chat_run(SynapCallbackHandler(sdk, "c1", "dana")))
    assert len(sdk.kwargs_of("instance.send_message")) == 2


# ── conversation-only files: no identity, no path ───────────────────────────

def test_short_term_and_middleware_take_no_identity():
    """short_term.py and middleware.py read only through a conversation id."""
    import inspect
    from synap_langchain import short_term, middleware
    for mod in (short_term, middleware):
        src = inspect.getsource(mod)
        assert "user_id" not in src and "customer_id" not in src


def test_short_term_reads_only_the_conversation():
    from synap_langchain import synap_st_runnable
    sdk = RecordingSDK()
    run(synap_st_runnable(sdk, "c1").ainvoke(None))
    assert sdk.last_kwargs("conversation.context.get_context_for_prompt") == {
        "conversation_id": "c1"} or sdk.last_kwargs(
        "conversation.context.get_context_for_prompt")["conversation_id"] == "c1"
