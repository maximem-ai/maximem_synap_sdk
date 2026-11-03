"""Framework-neutral reporting nodes, and the callback handler's scope_path.

C-1 inverted: a graph whose LLM is an in-house client (``await llm.complete``)
now reports the user and assistant turns.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, TypedDict

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState

from synap_integrations_common import SynapIntegrationError
from synap_langgraph import (
    SynapLangGraphCallbackHandler,
    create_synap_reasoning_node,
    create_synap_tool_call_node,
    create_synap_tool_result_node,
    create_synap_turn_node,
)

from ._support import Sdk

PATH = {"customer": "acme", "team": "payments", "user": "dana"}
CONF = {"configurable": {"user_id": "alice", "customer_id": "acme", "conversation_id": "c1"}}


def run(coro):
    return asyncio.run(coro)


class S(TypedDict, total=False):
    q: str
    a: str
    reasoning: str
    args: dict
    out: Any
    call_id: str


class InHouseLLM:                      # the playground's shape: llm.complete()
    async def complete(self, messages): return "an answer"


def in_house_graph(sdk, **turn_kw):
    async def answer(state: S):
        return {"a": await InHouseLLM().complete([{"role": "user", "content": state["q"]}])}

    g = StateGraph(S)
    g.add_node("emit_user", create_synap_turn_node(sdk, role="user", content="q", **turn_kw))
    g.add_node("answer", answer)
    g.add_node("emit_assistant", create_synap_turn_node(
        sdk, role="assistant", content=lambda s: s["a"], **turn_kw))
    g.add_edge(START, "emit_user")
    g.add_edge("emit_user", "answer")
    g.add_edge("answer", "emit_assistant")
    g.add_edge("emit_assistant", END)
    return g.compile()


# ── C-1 inverted ────────────────────────────────────────────────────────────

def test_C1_inverted_in_house_llm_client_now_reports_user_and_assistant_turns():
    sdk = Sdk()
    out = run(in_house_graph(sdk).ainvoke({"q": "hi"}, config=CONF))
    assert out["a"] == "an answer"
    calls = sdk.kwargs_of("instance.send_message")
    assert [(c["role"], c["content"]) for c in calls] == [("user", "hi"), ("assistant", "an answer")]
    for c in calls:
        assert c["user_id"] == "alice" and c["customer_id"] == "acme" and c["conversation_id"] == "c1"
        assert "scope_path" not in c


def test_C1_inverted_one_compiled_graph_reports_for_two_users():
    sdk = Sdk()
    app = in_house_graph(sdk)
    for who, conv in (("alice", "c1"), ("bob", "c2")):
        run(app.ainvoke({"q": "hi"}, config={"configurable": {
            "user_id": who, "customer_id": "acme", "conversation_id": conv}}))
    assert [(c["user_id"], c["conversation_id"]) for c in sdk.kwargs_of("instance.send_message")] == [
        ("alice", "c1"), ("alice", "c1"), ("bob", "c2"), ("bob", "c2")]


def test_C1_control_langchain_chat_model_is_still_reported_by_the_handler():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    sdk = Sdk()
    model = GenericFakeChatModel(messages=iter([AIMessage(content="an answer")]))

    async def answer(state: MessagesState):
        return {"messages": [await model.ainvoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("answer", answer)
    g.add_edge(START, "answer")
    g.add_edge("answer", END)
    handler = SynapLangGraphCallbackHandler(sdk=sdk, conversation_id="c1", user_id="alice")
    run(g.compile().ainvoke({"messages": [HumanMessage("hi")]}, config={"callbacks": [handler]}))
    assert [c["role"] for c in sdk.kwargs_of("instance.send_message")] == ["user", "assistant"]


def test_the_handler_carries_scope_path_on_its_turns():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    sdk = Sdk(ladder="deep")
    model = GenericFakeChatModel(messages=iter([AIMessage(content="an answer")]))

    async def answer(state: MessagesState):
        return {"messages": [await model.ainvoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("answer", answer)
    g.add_edge(START, "answer")
    g.add_edge("answer", END)
    handler = SynapLangGraphCallbackHandler(
        sdk=sdk, conversation_id="c1", user_id="dana", customer_id="acme", scope_path=PATH)
    run(g.compile().ainvoke({"messages": [HumanMessage("hi")]}, config={"callbacks": [handler]}))
    calls = sdk.kwargs_of("instance.send_message")
    assert len(calls) == 2 and all(c["scope_path"] == PATH for c in calls)


def test_the_handler_without_scope_path_sends_none_and_builds_the_base_as_before(monkeypatch):
    handler = SynapLangGraphCallbackHandler(sdk=Sdk(), conversation_id="c1", user_id="a")
    assert handler.scope_path is None
    import synap_langchain.callbacks as base
    seen = {}
    orig = base.SynapCallbackHandler.__init__

    def spy(self, *a, **kw):
        seen.update(kw)
        return orig(self, *a, **kw)

    monkeypatch.setattr(base.SynapCallbackHandler, "__init__", spy)
    SynapLangGraphCallbackHandler(sdk=Sdk(), conversation_id="c1", user_id="a")
    assert "scope_path" not in seen


# ── identity ────────────────────────────────────────────────────────────────

def test_scope_path_comes_from_the_config():
    sdk = Sdk(ladder="deep")
    conf = {"configurable": {**CONF["configurable"], "user_id": "dana", "scope_path": PATH}}
    run(in_house_graph(sdk).ainvoke({"q": "hi"}, config=conf))
    assert all(c["scope_path"] == PATH for c in sdk.kwargs_of("instance.send_message"))


def test_an_argument_wins_over_the_config():
    sdk = Sdk()
    run(in_house_graph(sdk, user_id="carol", conversation_id="cX").ainvoke({"q": "hi"}, config=CONF))
    assert {(c["user_id"], c["conversation_id"]) for c in sdk.kwargs_of("instance.send_message")} == {
        ("carol", "cX")}


def test_missing_identity_is_logged_by_default_and_nothing_is_sent(caplog):
    sdk = Sdk()
    with caplog.at_level("ERROR"):
        run(in_house_graph(sdk).ainvoke({"q": "hi"}, config={"configurable": {"conversation_id": "c1"}}))
    assert sdk.calls == []
    assert any("missing user_id" in r.message for r in caplog.records)


def test_missing_identity_raises_with_on_error_raise():
    sdk = Sdk()
    with pytest.raises(SynapIntegrationError, match="missing user_id and conversation_id"):
        run(create_synap_turn_node(sdk, role="user", content="q", on_error="raise")({"q": "hi"}, {"configurable": {}}))
    assert sdk.calls == []


def test_a_turn_nobody_accepted_raises_with_on_error_raise():
    sdk = Sdk(listening=False)
    sdk.errors["instance.send_message"] = RuntimeError("stream down")
    sdk.errors["conversation.record_message"] = RuntimeError("rest down")
    node = create_synap_turn_node(sdk, role="user", content="q", on_error="raise")
    with pytest.raises(SynapIntegrationError, match="not recorded"):
        run(node({"q": "hi"}, CONF))
    run(create_synap_turn_node(sdk, role="user", content="q")({"q": "hi"}, CONF))   # default: quiet


def test_turn_falls_back_to_rest_with_the_path_when_the_stream_cannot_carry_it():
    sdk = Sdk(listening=False, ladder="deep")
    sdk.errors["instance.send_message"] = RuntimeError("stream down")
    conf = {"configurable": {**CONF["configurable"], "user_id": "dana", "scope_path": PATH}}
    run(create_synap_turn_node(sdk, role="user", content="q")({"q": "hi"}, conf))
    assert sdk.last_kwargs("conversation.record_message")["scope_path"] == PATH


def test_bad_scope_path_in_the_config_is_an_error_not_a_guess():
    sdk = Sdk()
    conf = {"configurable": {**CONF["configurable"], "scope_path": {"user": ""}}}
    with pytest.raises(SynapIntegrationError, match="scope_path"):
        run(create_synap_turn_node(sdk, role="user", content="q", on_error="raise")({"q": "x"}, conf))
    assert sdk.calls == []


def test_empty_content_reports_nothing():
    sdk = Sdk()
    run(create_synap_turn_node(sdk, role="user", content="q", on_error="raise")({"q": "   "}, CONF))
    run(create_synap_turn_node(sdk, role="user", content="missing", on_error="raise")({"q": "x"}, CONF))
    assert sdk.calls == []


def test_content_may_be_a_message_a_dict_or_a_list_of_them():
    sdk = Sdk()
    node = create_synap_turn_node(sdk, role="user", content="m")
    for value in ("plain", HumanMessage("a message"), {"content": "a dict"},
                  [HumanMessage("first"), HumanMessage("the last one")]):
        run(node({"m": value}, CONF))
    assert [c["content"] for c in sdk.kwargs_of("instance.send_message")] == [
        "plain", "a message", "a dict", "the last one"]


def test_an_async_callable_is_awaited():
    sdk = Sdk()

    async def content(state):
        return state["q"].upper()

    run(create_synap_turn_node(sdk, role="user", content=content)({"q": "hi"}, CONF))
    assert sdk.last_kwargs("instance.send_message")["content"] == "HI"


def test_role_and_arguments_are_validated():
    with pytest.raises(ValueError, match="role"):
        create_synap_turn_node(Sdk(), role="system", content="q")
    with pytest.raises(ValueError, match="non-None sdk"):
        create_synap_turn_node(None, role="user", content="q")
    with pytest.raises(ValueError, match="on_error"):
        create_synap_turn_node(Sdk(), role="user", content="q", on_error="x")


# ── tool call / result / reasoning ──────────────────────────────────────────

def test_tool_call_result_and_reasoning_nodes():
    sdk = Sdk()
    state = {"args": {"pnr": "QX41RT"}, "call_id": "tc-1", "reasoning": "need the booking",
             "out": ToolMessage(content="CONFIRMED", tool_call_id="tc-1", name="lookup")}
    run(create_synap_reasoning_node(sdk, content="reasoning")(state, CONF))
    run(create_synap_tool_call_node(sdk, tool_name="lookup", tool_args="args", tool_call_id="call_id")(state, CONF))
    run(create_synap_tool_result_node(sdk, result="out", tool_name="lookup", tool_call_id="call_id")(state, CONF))
    thinking = sdk.last_kwargs("instance.record_thinking")
    assert thinking["conversation_id"] == "c1" and thinking["user_id"] == "alice"
    call = sdk.calls_to("instance.record_tool_call")[0]
    assert call.args == ("lookup", {"pnr": "QX41RT"}) and call.kwargs["tool_call_id"] == "tc-1"
    result = sdk.calls_to("instance.record_tool_result")[0]
    assert result.args == ("CONFIRMED",)                       # the ToolMessage is unwrapped
    assert result.kwargs["tool_call_id"] == "tc-1" and result.kwargs["tool_name"] == "lookup"
    for name in ("instance.record_thinking", "instance.record_tool_call", "instance.record_tool_result"):
        assert "scope_path" not in sdk.last_kwargs(name)


def test_tool_nodes_are_silent_without_a_stream():
    sdk = Sdk(listening=False)
    sdk.instance.listen = None  # nothing can open one
    sdk.errors["instance.listen"] = RuntimeError("no stream")
    run(create_synap_tool_call_node(sdk, tool_name="t", on_error="raise")({}, CONF))
    assert sdk.calls_to("instance.record_tool_call") == []
