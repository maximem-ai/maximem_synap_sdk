"""The defaults: the stream opens on the first turn, one turn per run, and
Synap's compaction reaches the model of a real ``create_agent`` agent.

Nothing here calls ``listen()`` or opens a stream by hand. The SDK is a fake
whose ``instance.is_listening`` starts False, the way a fresh SDK's does.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from synap_integrations_common import set_auto_listen
from synap_langchain import SynapCallbackHandler, SynapChatMessageHistory
from synap_langchain.middleware import SynapHistoryMiddleware

COMPACTION = "## Conversation so far (compacted 1x)\nThe user is planning a trip to Lisbon."


class FakeInstance:
    def __init__(self):
        self.is_listening = False
        self.listen_calls = 0
        self.send_message = AsyncMock()
        self.record_tool_call = AsyncMock()
        self.record_tool_result = AsyncMock()
        self.record_thinking = AsyncMock()

    async def listen(self):
        self.listen_calls += 1
        self.is_listening = True


def make_sdk(context=None):
    return SimpleNamespace(
        instance=FakeInstance(),
        conversation=SimpleNamespace(
            record_message=AsyncMock(),
            context=SimpleNamespace(get_context_for_prompt=AsyncMock(return_value=context)),
        ),
        cache=SimpleNamespace(clear=lambda: None),
        archive_tool_result=lambda *a, **k: None,
    )


def compacted(turns=()):
    return SimpleNamespace(
        context_block=COMPACTION, history_block=None, compaction_id="c1", available=True,
        formatted_context="", recent_messages=[SimpleNamespace(role=r, content=t) for r, t in turns],
    )


class ScriptedModel(BaseChatModel):
    """Plays a script of replies and keeps what each call was sent."""

    script: List[AIMessage]
    seen: List[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


@tool
async def lookup(city: str) -> str:
    """Look a city up."""
    return f"{city}: sunny"


def sent(sdk, event_type):
    return [c.kwargs["content"] for c in sdk.instance.send_message.await_args_list
            if c.kwargs["event_type"] == event_type]


@pytest.mark.asyncio
async def test_a_tool_loop_is_one_user_turn_and_one_assistant_turn_on_an_auto_opened_stream():
    sdk = make_sdk()
    model = ScriptedModel(script=[
        AIMessage(content="Let me check the weather.",
                  tool_calls=[{"name": "lookup", "args": {"city": "Lisbon"}, "id": "t1"}]),
        AIMessage(content="It is sunny in Lisbon."),
    ], seen=[])
    handler = SynapCallbackHandler(sdk=sdk, conversation_id="c1", user_id="u1")
    agent = create_agent(model, tools=[lookup])

    await agent.ainvoke({"messages": [HumanMessage("weather in Lisbon?")]}, config={"callbacks": [handler]})

    assert sdk.instance.listen_calls == 1  # opened by the first turn, never by the caller
    assert sent(sdk, "user_message") == ["weather in Lisbon?"]
    assert sent(sdk, "assistant_message") == ["It is sunny in Lisbon."]  # not the lap with the tool call
    sdk.conversation.record_message.assert_not_awaited()  # stream, not also REST
    sdk.instance.record_tool_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_model_a_tool_calls_does_not_speak_for_the_assistant():
    sdk = make_sdk()
    h = SynapCallbackHandler(sdk=sdk, conversation_id="c1", user_id="u1")
    root, tool_run, inner_llm, main_llm = uuid4(), uuid4(), uuid4(), uuid4()
    await h.on_chain_start({}, {"messages": [HumanMessage("q", id="m1")]}, run_id=root)
    await h.on_tool_start({"name": "t"}, "x", run_id=tool_run, parent_run_id=root)
    # the tool calls a model of its own, and a summarizer-style call under the root
    await h.on_chat_model_start({}, [[HumanMessage("Summarize this: ...")]], run_id=inner_llm, parent_run_id=tool_run)
    await h.on_llm_end(_result("tool-internal summary"), run_id=inner_llm, parent_run_id=tool_run)
    await h.on_tool_end("ok", run_id=tool_run, parent_run_id=root)
    await h.on_chat_model_start({}, [[HumanMessage("q", id="m1")]], run_id=main_llm, parent_run_id=root)
    await h.on_llm_end(_result("the answer"), run_id=main_llm, parent_run_id=root)
    await h.on_chain_end({"messages": [AIMessage("the answer")]}, run_id=root)

    assert sent(sdk, "user_message") == ["q"]
    assert sent(sdk, "assistant_message") == ["the answer"]


@pytest.mark.asyncio
async def test_a_bare_model_call_with_no_chain_around_it_is_still_recorded():
    sdk = make_sdk()
    h = SynapCallbackHandler(sdk=sdk, conversation_id="c1", user_id="u1")
    rid = uuid4()
    await h.on_chat_model_start({}, [[HumanMessage("hi")]], run_id=rid)
    await h.on_llm_end(_result("hello"), run_id=rid)
    assert sent(sdk, "user_message") == ["hi"]
    assert sent(sdk, "assistant_message") == ["hello"]


@pytest.mark.asyncio
async def test_a_second_turn_that_repeats_the_first_is_not_swallowed():
    sdk = make_sdk()
    h = SynapCallbackHandler(sdk=sdk, conversation_id="c1", user_id="u1")
    for _ in range(2):
        root = uuid4()
        await h.on_chain_start({}, {"input": "ok"}, run_id=root)
        await h.on_chain_end(AIMessage("sure"), run_id=root)
    assert sent(sdk, "user_message") == ["ok", "ok"]


def _result(text):
    from langchain_core.outputs import LLMResult
    return LLMResult(generations=[[ChatGeneration(message=AIMessage(content=text))]])


@pytest.mark.asyncio
async def test_history_writes_ride_the_auto_opened_stream_and_skip_system_messages():
    sdk = make_sdk()
    history = SynapChatMessageHistory(sdk, "c1", "u1")
    await history.aadd_messages([
        SystemMessage("You are a travel agent."),
        HumanMessage("hi"),
        AIMessage("hello"),
    ])
    assert sdk.instance.listen_calls == 1
    assert sent(sdk, "user_message") == ["hi"]
    assert sent(sdk, "assistant_message") == ["hello"]
    sdk.conversation.record_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_history_writes_fall_back_to_rest_when_the_stream_is_switched_off():
    sdk = make_sdk()
    set_auto_listen(sdk, False)
    await SynapChatMessageHistory(sdk, "c1", "u1").aadd_messages([HumanMessage("hi")])
    assert sdk.instance.listen_calls == 0
    sdk.conversation.record_message.assert_awaited_once()


OLD_TURNS = [HumanMessage("old question one"), AIMessage("old answer one"),
             HumanMessage("old question two"), AIMessage("old answer two")]


def texts(messages):
    return [m.content for m in messages]


@pytest.mark.asyncio
async def test_create_agent_sends_the_compaction_instead_of_the_earlier_turns():
    sdk = make_sdk(compacted(turns=[("user", "synap kept this"), ("assistant", "and this")]))
    model = ScriptedModel(script=[AIMessage("Book the 9am flight.")], seen=[])
    agent = create_agent(model, tools=[lookup], system_prompt="Be brief.",
                         middleware=[SynapHistoryMiddleware(sdk, "c1")])

    await agent.ainvoke({"messages": [*OLD_TURNS, HumanMessage("what now?")]})

    sent_to_model = model.seen[0]
    assert isinstance(sent_to_model[0], SystemMessage)
    assert sent_to_model[0].content.startswith("Be brief.")
    assert "The user is planning a trip to Lisbon." in sent_to_model[0].content
    assert texts(sent_to_model[1:]) == ["synap kept this", "and this", "what now?"]
    assert not any("old question" in str(m.content) for m in sent_to_model)


@pytest.mark.asyncio
async def test_create_agent_is_untouched_until_synap_has_compacted():
    sdk = make_sdk(SimpleNamespace(context_block=None, history_block=None, recent_messages=[]))
    model = ScriptedModel(script=[AIMessage("fine")], seen=[])
    agent = create_agent(model, tools=[lookup], middleware=[SynapHistoryMiddleware(sdk, "c1")])

    await agent.ainvoke({"messages": [*OLD_TURNS, HumanMessage("what now?")]})

    assert texts(model.seen[0]) == ["old question one", "old answer one", "old question two",
                                    "old answer two", "what now?"]


@pytest.mark.asyncio
async def test_create_agent_survives_a_synap_outage():
    sdk = make_sdk()
    sdk.conversation.context.get_context_for_prompt.side_effect = RuntimeError("down")
    model = ScriptedModel(script=[AIMessage("fine")], seen=[])
    agent = create_agent(model, tools=[lookup], middleware=[SynapHistoryMiddleware(sdk, "c1")])

    await agent.ainvoke({"messages": [HumanMessage("hi")]})

    assert texts(model.seen[0]) == ["hi"]


@pytest.mark.asyncio
async def test_the_current_turns_tool_loop_stays_paired_when_spliced():
    sdk = make_sdk(compacted())
    model = ScriptedModel(script=[
        AIMessage("", tool_calls=[{"name": "lookup", "args": {"city": "Lisbon"}, "id": "t1"}]),
        AIMessage("sunny"),
    ], seen=[])
    agent = create_agent(model, tools=[lookup], middleware=[SynapHistoryMiddleware(sdk, "c1")])

    await agent.ainvoke({"messages": [*OLD_TURNS, HumanMessage("weather?")]})

    second = model.seen[1]
    assert any(isinstance(m, ToolMessage) for m in second)
    assert not any("old question" in str(m.content) for m in second)
    assert COMPACTION.splitlines()[0] in second[0].content


def test_the_sync_path_splices_too():
    sdk = make_sdk(compacted())
    model = ScriptedModel(script=[AIMessage("fine")], seen=[])
    agent = create_agent(model, tools=[lookup], middleware=[SynapHistoryMiddleware(sdk, "c1")])

    agent.invoke({"messages": [*OLD_TURNS, HumanMessage("what now?")]})

    assert not any("old question" in str(m.content) for m in model.seen[0])
    assert "Lisbon" in model.seen[0][0].content


@pytest.mark.asyncio
async def test_runnable_with_message_history_reads_the_compaction_and_writes_on_the_stream():
    """The real RunnableWithMessageHistory: it loads the store, sends the model
    the compaction and the turns after it once each, then writes the new turn."""
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory

    sdk = make_sdk(compacted(turns=[("user", "synap turn q"), ("assistant", "synap turn a")]))
    model = ScriptedModel(script=[AIMessage("reply")], seen=[])
    prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder("history"), ("human", "{input}")])
    chain = RunnableWithMessageHistory(
        prompt | model,
        lambda sid: SynapChatMessageHistory(sdk, sid, "u1"),
        input_messages_key="input",
        history_messages_key="history",
    )

    await chain.ainvoke({"input": "next question"}, config={"configurable": {"session_id": "c1"}})

    seen = model.seen[0]
    assert isinstance(seen[0], SystemMessage) and "Lisbon" in seen[0].content
    assert texts(seen[1:]) == ["synap turn q", "synap turn a", "next question"]
    assert sdk.instance.listen_calls == 1
    assert sent(sdk, "user_message") == ["next question"]
    assert sent(sdk, "assistant_message") == ["reply"]
    sdk.conversation.record_message.assert_not_awaited()
