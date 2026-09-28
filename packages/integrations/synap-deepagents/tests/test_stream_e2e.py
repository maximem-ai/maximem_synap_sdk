"""SynapStreamMiddleware inside a real deepagents graph.

``tests/test_stream.py`` calls the hooks directly, which is how the three bugs
this module's docstring describes survived a green suite in the LangChain
handler it is modelled on: a hook called by hand fires once, and the bug is that
the framework calls it more than once.

So these run ``create_deep_agent`` for real — a compiled graph, a tool node, a
tool loop — against a scripted model, and read what came out the other end.
Sync and async both, because ``invoke()`` and ``ainvoke()`` take different
halves of every hook and a middleware that implements one of them raises
``NotImplementedError`` on the other.
"""

import asyncio
from typing import Any, List, Optional

import pytest
from unittest.mock import AsyncMock, MagicMock

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from synap_deepagents import SynapStreamMiddleware


# ── a model that says what we tell it to ─────────────────────────────────────


class ScriptedModel(BaseChatModel):
    """Replays one queued ``AIMessage`` per model call."""

    replies: List[AIMessage] = []
    seen: List[int] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(
        self, messages: Any, stop: Optional[List[str]] = None,
        run_manager: Any = None, **kwargs: Any,
    ) -> ChatResult:
        index = min(len(self.seen), len(self.replies) - 1)
        self.seen.append(index)
        return ChatResult(generations=[ChatGeneration(message=self.replies[index])])


@tool
def lookup_order(pnr: str) -> str:
    """Look up an order by its PNR."""
    return "shipped"


@tool
def check_stock(sku: str) -> str:
    """Check whether a SKU is in stock."""
    return "in stock"


def _sdk(is_listening: bool = True):
    sdk = MagicMock()
    sdk.instance.is_listening = is_listening
    sdk.instance.send_message = AsyncMock()
    sdk.instance.record_thinking = AsyncMock()
    sdk.instance.record_tool_call = AsyncMock()
    sdk.instance.record_tool_result = AsyncMock()
    sdk.conversation.record_message = AsyncMock()
    return sdk


def _agent(sdk, replies):
    model = ScriptedModel(replies=replies, seen=[])
    return create_deep_agent(
        model=model,
        tools=[lookup_order, check_stock],
        middleware=[
            SynapStreamMiddleware(
                sdk=sdk, conversation_id="conv-1", user_id="alice"
            )
        ],
    )


def _one_tool_turn():
    return [
        AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "the PNR is a booking"},
                {"type": "text", "text": "let me look"},
            ],
            tool_calls=[
                {"name": "lookup_order", "args": {"pnr": "QX41RT"}, "id": "call-1"}
            ],
        ),
        AIMessage(content="it ships tomorrow"),
    ]


def _three_tool_turn():
    """Three laps, each of which says something on its way to a tool.

    The text matters. A model that narrates while it works ("let me look") is
    the normal case, and it is what makes a turn reported from the model hook
    show up as four assistant messages instead of one.
    """
    return [
        AIMessage(
            content="let me look up the booking",
            tool_calls=[
                {"name": "lookup_order", "args": {"pnr": "QX41RT"}, "id": "call-1"}
            ],
        ),
        AIMessage(
            content="now checking stock on the first item",
            tool_calls=[{"name": "check_stock", "args": {"sku": "A1"}, "id": "call-2"}],
        ),
        AIMessage(
            content="and the second",
            tool_calls=[{"name": "check_stock", "args": {"sku": "B2"}, "id": "call-3"}],
        ),
        AIMessage(content="it ships tomorrow"),
    ]


def _turns(sdk):
    return [
        (call.kwargs.get("event_type"), call.kwargs.get("content"))
        for call in sdk.instance.send_message.await_args_list
    ]


def _run(agent, sync):
    payload = {"messages": [HumanMessage("where is my order", id="h1")]}
    if sync:
        agent.invoke(payload)
    else:
        asyncio.run(agent.ainvoke(payload))


SYNC_AND_ASYNC = pytest.mark.parametrize(
    "sync", [True, False], ids=["invoke", "ainvoke"]
)


# ── the whole turn, through the graph ────────────────────────────────────────


@SYNC_AND_ASYNC
def test_all_five_events_reach_the_stream(sync):
    sdk = _sdk()
    _run(_agent(sdk, _one_tool_turn()), sync)

    assert _turns(sdk) == [
        ("user_message", "where is my order"),
        ("assistant_message", "it ships tomorrow"),
    ]
    assert sdk.instance.record_tool_call.await_args.args[0] == "lookup_order"
    assert sdk.instance.record_tool_call.await_args.args[1] == {"pnr": "QX41RT"}
    assert sdk.instance.record_tool_result.await_args.args[0] == "shipped"
    assert sdk.instance.record_thinking.await_args.args[0] == "the PNR is a booking"


@SYNC_AND_ASYNC
def test_the_call_and_its_result_share_the_frameworks_own_id(sync):
    sdk = _sdk()
    _run(_agent(sdk, _one_tool_turn()), sync)
    assert (
        sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
        == sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"]
        == "call-1"
    )


@SYNC_AND_ASYNC
def test_the_tool_result_is_not_a_raw_ToolMessage(sync):
    """``ToolNode`` hands back a ``ToolMessage``, and the SDK JSON-encodes a
    result. Unwrapped, the encode raises, the helper swallows it, and every tool
    result in the run is silently never reported."""
    import json

    sdk = _sdk()
    _run(_agent(sdk, _one_tool_turn()), sync)
    json.dumps(sdk.instance.record_tool_result.await_args.args[0])


@SYNC_AND_ASYNC
def test_a_streamed_turn_is_not_also_written_over_rest(sync):
    sdk = _sdk()
    _run(_agent(sdk, _one_tool_turn()), sync)
    sdk.conversation.record_message.assert_not_awaited()


@SYNC_AND_ASYNC
def test_with_no_stream_the_turn_falls_back_to_rest(sync):
    sdk = _sdk(is_listening=False)
    _run(_agent(sdk, _one_tool_turn()), sync)
    assert [
        (call.kwargs["role"], call.kwargs["content"])
        for call in sdk.conversation.record_message.await_args_list
    ] == [("user", "where is my order"), ("assistant", "it ships tomorrow")]
    sdk.instance.send_message.assert_not_awaited()


# ── the bug a single-shot run cannot see ─────────────────────────────────────


@SYNC_AND_ASYNC
def test_a_three_tool_turn_still_reports_the_question_once(sync):
    """⚠ The bug this hook choice exists to avoid.

    A ReAct loop calls the model again after every tool result with the same
    human message still in the list. Hung off a per-model-call hook, this run
    would report the question four times and Synap would extract it four times.
    Invisible in a single-shot run, which is why it shipped.
    """
    sdk = _sdk()
    _run(_agent(sdk, _three_tool_turn()), sync)
    assert _turns(sdk).count(("user_message", "where is my order")) == 1


@SYNC_AND_ASYNC
def test_a_three_tool_turn_reports_exactly_one_answer(sync):
    """The model speaks on every lap; only the last one is the answer."""
    sdk = _sdk()
    _run(_agent(sdk, _three_tool_turn()), sync)
    assert [t for t in _turns(sdk) if t[0] == "assistant_message"] == [
        ("assistant_message", "it ships tomorrow")
    ]


@SYNC_AND_ASYNC
def test_every_tool_in_the_loop_is_reported_and_paired(sync):
    sdk = _sdk()
    _run(_agent(sdk, _three_tool_turn()), sync)
    called = [
        (c.args[0], c.kwargs["tool_call_id"])
        for c in sdk.instance.record_tool_call.await_args_list
    ]
    returned = [
        (c.kwargs["tool_name"], c.kwargs["tool_call_id"])
        for c in sdk.instance.record_tool_result.await_args_list
    ]
    assert called == [
        ("lookup_order", "call-1"),
        ("check_stock", "call-2"),
        ("check_stock", "call-3"),
    ]
    assert returned == called


@SYNC_AND_ASYNC
def test_the_question_is_reported_before_the_answer(sync):
    sdk = _sdk()
    _run(_agent(sdk, _three_tool_turn()), sync)
    events = [event for event, _content in _turns(sdk)]
    assert events == ["user_message", "assistant_message"]


# ── it never changes what the agent does ─────────────────────────────────────


@SYNC_AND_ASYNC
def test_a_stream_that_fails_on_every_call_still_answers(sync):
    sdk = _sdk()
    for name in (
        "send_message", "record_thinking", "record_tool_call", "record_tool_result",
    ):
        getattr(sdk.instance, name).side_effect = RuntimeError("stream died")
    agent = _agent(sdk, _one_tool_turn())
    payload = {"messages": [HumanMessage("where is my order", id="h1")]}
    result = agent.invoke(payload) if sync else asyncio.run(agent.ainvoke(payload))
    assert result["messages"][-1].content == "it ships tomorrow"


@SYNC_AND_ASYNC
def test_an_sdk_with_no_instance_namespace_still_answers(sync):
    sdk = MagicMock()
    del sdk.instance
    sdk.conversation.record_message = AsyncMock()
    agent = _agent(sdk, _one_tool_turn())
    payload = {"messages": [HumanMessage("where is my order", id="h1")]}
    result = agent.invoke(payload) if sync else asyncio.run(agent.ainvoke(payload))
    assert result["messages"][-1].content == "it ships tomorrow"
