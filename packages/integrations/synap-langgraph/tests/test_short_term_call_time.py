"""Short-term helpers read the conversation id when they run (issue 7, ST-1).

Verified against langgraph 1.2.19: ``RunnableCallable`` injects ``config`` into
a node, and into an async ``prompt`` callable of ``create_react_agent``, when
the function declares a ``config`` parameter (annotated ``RunnableConfig``,
``Optional[RunnableConfig]`` or not at all).
"""

from __future__ import annotations

import asyncio
from typing import Any, List, TypedDict

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from synap_integrations_common import SynapIntegrationError
from synap_langgraph import create_synap_st_node, synap_st_prompt

from ._support import Sdk, prompt_ctx
from synap_integrations_common import coerce_conversation_id  # REST calls get the derived UUID

CTX = "conversation.context.get_context_for_prompt"
COMPACTION = "## Conversation so far (compacted 1x)\nThe user is planning a trip."


def run(coro):
    return asyncio.run(coro)


_NAMES = {coerce_conversation_id(n): n for n in
          ("conv-A", "conv-B", "pinned", "from-config", "c1", "c9")}


def per_conversation(sdk: Sdk) -> Sdk:
    """get_context_for_prompt answers 'SUMMARY OF <name>' for whichever id it is asked.

    The SDK is asked for the UUID derived from the caller's id (it refuses any other
    string); ``sdk.asked`` records that UUID and the summary names the caller's id."""
    async def get_ctx(conversation_id: str, style: str = "structured", **kw: Any):
        sdk.asked.append(conversation_id)
        name = _NAMES.get(conversation_id, conversation_id)
        return prompt_ctx(f"SUMMARY OF {name}", turns=[("user", f"hello from {name}")])
    sdk.asked = []
    sdk.conversation.context.get_context_for_prompt = get_ctx
    return sdk


class Recorder(BaseChatModel):
    seen: List[Any] = []

    @property
    def _llm_type(self) -> str:
        return "recorder"

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


# ── ST-1 inverted ───────────────────────────────────────────────────────────

def test_ST1_inverted_a_node_compiled_once_serves_two_conversations():
    class S(TypedDict, total=False):
        synap_st: str

    sdk = per_conversation(Sdk())
    g = StateGraph(S)
    g.add_node("st", create_synap_st_node(sdk))            # no conversation id
    g.add_edge(START, "st")
    g.add_edge("st", END)
    app = g.compile()

    a = run(app.ainvoke({}, config={"configurable": {"conversation_id": "conv-A"}}))
    b = run(app.ainvoke({}, config={"configurable": {"conversation_id": "conv-B"}}))
    assert sdk.asked == [coerce_conversation_id("conv-A"), coerce_conversation_id("conv-B")]
    assert a["synap_st"].endswith("SUMMARY OF conv-A")
    assert b["synap_st"].endswith("SUMMARY OF conv-B")


def test_ST1_inverted_the_prompt_callable_of_one_agent_serves_two_conversations():
    sdk = per_conversation(Sdk())
    model = Recorder()
    agent = create_react_agent(model, tools=[], prompt=synap_st_prompt(sdk, system="Be brief."))
    for cid in ("conv-A", "conv-B"):
        run(agent.ainvoke({"messages": [HumanMessage("hi")]},
                          config={"configurable": {"conversation_id": cid}}))
    assert sdk.asked == [coerce_conversation_id("conv-A"), coerce_conversation_id("conv-B")]
    system_a, system_b = (m[0].content for m in model.seen)
    assert "SUMMARY OF conv-A" in system_a and "SUMMARY OF conv-B" not in system_a
    assert "SUMMARY OF conv-B" in system_b and "SUMMARY OF conv-A" not in system_b
    assert system_a.endswith("Be brief.")


def test_an_explicit_conversation_id_wins_over_the_config():
    sdk = per_conversation(Sdk())
    run(create_synap_st_node(sdk, "pinned")({}, {"configurable": {"conversation_id": "from-config"}}))
    assert sdk.asked == [coerce_conversation_id("pinned")]

    class S(TypedDict, total=False):
        synap_st: str

    g = StateGraph(S)
    g.add_node("st", create_synap_st_node(sdk, "pinned"))
    g.add_edge(START, "st")
    g.add_edge("st", END)
    run(g.compile().ainvoke({}, config={"configurable": {"conversation_id": "from-config"}}))
    assert sdk.asked == [coerce_conversation_id("pinned")] * 2


def test_a_node_called_directly_with_a_config_argument():
    sdk = per_conversation(Sdk())
    out = run(create_synap_st_node(sdk)({}, {"configurable": {"conversation_id": "c9"}}))
    assert out["synap_st"].endswith("SUMMARY OF c9")


def test_a_node_called_directly_with_an_explicit_id_needs_no_config():
    sdk = per_conversation(Sdk())
    assert run(create_synap_st_node(sdk, "c1")({}))["synap_st"].endswith("SUMMARY OF c1")


# ── no conversation id anywhere ─────────────────────────────────────────────

def test_no_conversation_id_with_on_error_raise_raises_a_clear_error():
    sdk = per_conversation(Sdk())
    with pytest.raises(SynapIntegrationError, match="no conversation id"):
        run(create_synap_st_node(sdk, on_error="raise")({}, {"configurable": {}}))
    with pytest.raises(SynapIntegrationError, match="no conversation id"):
        run(synap_st_prompt(sdk, on_error="raise")({"messages": []}, {"configurable": {}}))
    assert sdk.asked == []


def test_no_conversation_id_with_the_default_degrades_and_logs(caplog):
    sdk = per_conversation(Sdk())
    with caplog.at_level("ERROR"):
        out = run(create_synap_st_node(sdk, include_recent_turns=True)({}, {"configurable": {}}))
        msgs = run(synap_st_prompt(sdk, system="S")({"messages": [HumanMessage("q")]}, {"configurable": {}}))
    assert out == {"synap_st": "", "synap_recent_turns": []}
    assert isinstance(msgs[0], SystemMessage) and msgs[0].content == "S" and msgs[1].content == "q"
    assert any("no conversation id" in r.message for r in caplog.records)
    assert sdk.asked == []


def test_a_blank_explicit_id_is_still_a_construction_error():
    with pytest.raises(ValueError, match="non-empty conversation_id"):
        create_synap_st_node(Sdk(), "  ")
    with pytest.raises(ValueError, match="non-empty conversation_id"):
        synap_st_prompt(Sdk(), "")


def test_style_is_still_validated_and_is_forwarded_unchanged():
    with pytest.raises(ValueError, match="unsupported style"):
        create_synap_st_node(Sdk(), style="poetic")
    sdk = Sdk()
    sdk.results[CTX] = prompt_ctx(COMPACTION)
    run(create_synap_st_node(sdk, "c1", style="bullet_points")({}))
    assert sdk.last_kwargs(CTX) == {"conversation_id": coerce_conversation_id("c1"), "style": "bullet_points"}


def test_default_call_to_the_sdk_is_unchanged():
    sdk = Sdk()
    run(create_synap_st_node(sdk, "c1")({}))
    assert sdk.last_kwargs(CTX) == {"conversation_id": coerce_conversation_id("c1"), "style": "narrative"}


# ── include_recent_turns ────────────────────────────────────────────────────

def test_node_returns_the_recent_turns_when_asked():
    sdk = Sdk()
    sdk.results[CTX] = prompt_ctx(None, turns=[
        ("user", "my address is in Texas"), ("assistant", "Got it, Texas."),
        ("user", "actually, I moved, I'm in California now")])
    out = run(create_synap_st_node(sdk, "c1", include_recent_turns=True)({}))
    assert out["synap_st"] == ""                       # not compacted yet
    assert out["synap_recent_turns"] == [
        {"role": "user", "content": "my address is in Texas"},
        {"role": "assistant", "content": "Got it, Texas."},
        {"role": "user", "content": "actually, I moved, I'm in California now"},
    ]


def test_node_does_not_write_the_recent_key_by_default():
    sdk = Sdk()
    sdk.results[CTX] = prompt_ctx(COMPACTION, turns=[("user", "x")])
    assert set(run(create_synap_st_node(sdk, "c1")({}))) == {"synap_st"}
    out = run(create_synap_st_node(sdk, "c1", include_recent_turns=True, recent_key="rt")({}))
    assert set(out) == {"synap_st", "rt"}


def test_node_recent_turns_empty_on_failure_unless_raise():
    sdk = Sdk()
    sdk.errors[CTX] = RuntimeError("down")
    assert run(create_synap_st_node(sdk, "c1", include_recent_turns=True)({})) == {
        "synap_st": "", "synap_recent_turns": []}
    with pytest.raises(SynapIntegrationError):
        run(create_synap_st_node(sdk, "c1", include_recent_turns=True, on_error="raise")({}))


def test_prompt_with_recent_turns_puts_california_in_front_of_the_model():
    """The playground-shaped graph keeps no message list: Synap's recent turns
    are the history. The user's latest correction must reach the model."""
    sdk = Sdk()
    sdk.results[CTX] = prompt_ctx(None, turns=[
        ("user", "my address is in Texas"), ("assistant", "Got it, Texas."),
        ("user", "actually, I moved, I'm in California now")])
    prompt = synap_st_prompt(sdk, "c1", system="Be brief.", include_recent_turns=True)
    out = run(prompt({"messages": []}))
    text = out[0].content
    assert "<synap_recent_turns>" in text and "user: actually, I moved, I'm in California now" in text
    assert text.endswith("Be brief.")


def test_prompt_with_recent_turns_does_not_repeat_the_current_message():
    sdk = Sdk()
    sdk.results[CTX] = prompt_ctx(COMPACTION, turns=[
        ("assistant", "Got it."), ("user", "what is my plan?")])
    prompt = synap_st_prompt(sdk, "c1", include_recent_turns=True)
    out = run(prompt({"messages": [HumanMessage("what is my plan?")]}))
    assert "assistant: Got it." in out[0].content
    assert "user: what is my plan?" not in out[0].content
    assert COMPACTION.splitlines()[0] in out[0].content
    assert [m.content for m in out[1:]] == ["what is my plan?"]     # messages untouched


def test_prompt_default_still_splices_and_ignores_recent_turns_block():
    sdk = Sdk()
    sdk.results[CTX] = prompt_ctx(COMPACTION, turns=[("assistant", "Got it.")])
    out = run(synap_st_prompt(sdk, "c1")({"messages": [HumanMessage("q")]}))
    assert "<synap_recent_turns>" not in out[0].content
