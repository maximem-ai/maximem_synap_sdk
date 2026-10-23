"""The defaults, against a real LangGraph ``create_react_agent``.

Nothing here calls ``listen()``. The SDK is a fake whose ``instance.is_listening``
starts False, the way a fresh SDK's does, and the model is a scripted fake that
keeps what it was sent.
"""

import warnings
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from synap_langgraph import SynapCheckpointSaver, SynapLangGraphCallbackHandler, synap_st_prompt

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langgraph.prebuilt import create_react_agent

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
    )


def compacted(turns=()):
    return SimpleNamespace(
        context_block=COMPACTION, history_block=None, available=True, formatted_context="",
        recent_messages=[SimpleNamespace(role=r, content=t) for r, t in turns],
    )


class ScriptedModel(BaseChatModel):
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


OLD_TURNS = [HumanMessage("old question one"), AIMessage("old answer one"),
             HumanMessage("old question two"), AIMessage("old answer two")]


def texts(messages):
    return [m.content for m in messages]


def agent_with(sdk, model, system="Be brief."):
    return create_react_agent(
        model, tools=[lookup],
        prompt=synap_st_prompt(sdk, "c1", system=system),
    )


@pytest.mark.asyncio
async def test_react_agent_sends_the_compaction_in_place_of_the_earlier_turns():
    sdk = make_sdk(compacted(turns=[("user", "synap kept this"), ("assistant", "and this")]))
    model = ScriptedModel(script=[AIMessage("Book the 9am flight.")], seen=[])

    await agent_with(sdk, model).ainvoke({"messages": [*OLD_TURNS, HumanMessage("what now?")]})

    sent = model.seen[0]
    assert isinstance(sent[0], SystemMessage)
    assert "The user is planning a trip to Lisbon." in sent[0].content
    assert sent[0].content.rstrip().endswith("Be brief.")
    assert texts(sent[1:]) == ["synap kept this", "and this", "what now?"]
    assert not any("old question" in str(m.content) for m in sent)


@pytest.mark.asyncio
async def test_react_agent_is_untouched_until_synap_has_compacted():
    sdk = make_sdk(SimpleNamespace(context_block=None, history_block=None, recent_messages=[],
                                   available=True, formatted_context="recent turns"))
    model = ScriptedModel(script=[AIMessage("fine")], seen=[])

    await agent_with(sdk, model).ainvoke({"messages": [*OLD_TURNS, HumanMessage("what now?")]})

    sent = model.seen[0]
    assert sent[0].content == "Be brief."
    assert texts(sent[1:]) == ["old question one", "old answer one", "old question two",
                               "old answer two", "what now?"]


@pytest.mark.asyncio
async def test_react_agent_survives_a_synap_outage():
    sdk = make_sdk()
    sdk.conversation.context.get_context_for_prompt.side_effect = RuntimeError("down")
    model = ScriptedModel(script=[AIMessage("fine")], seen=[])

    await agent_with(sdk, model).ainvoke({"messages": [HumanMessage("hi")]})

    assert texts(model.seen[0]) == ["Be brief.", "hi"]


@pytest.mark.asyncio
async def test_the_current_turns_tool_loop_stays_paired():
    sdk = make_sdk(compacted())
    model = ScriptedModel(script=[
        AIMessage("", tool_calls=[{"name": "lookup", "args": {"city": "Lisbon"}, "id": "t1"}]),
        AIMessage("sunny"),
    ], seen=[])

    await agent_with(sdk, model).ainvoke({"messages": [*OLD_TURNS, HumanMessage("weather?")]})

    second = model.seen[1]
    assert any(isinstance(m, ToolMessage) for m in second)
    assert not any("old question" in str(m.content) for m in second)
    assert "Lisbon" in second[0].content


@pytest.mark.asyncio
async def test_the_handler_opens_the_stream_and_records_one_turn_each_way():
    sdk = make_sdk()
    model = ScriptedModel(script=[
        AIMessage("Let me check.", tool_calls=[{"name": "lookup", "args": {"city": "Lisbon"}, "id": "t1"}]),
        AIMessage("It is sunny."),
    ], seen=[])
    handler = SynapLangGraphCallbackHandler(sdk=sdk, conversation_id="c1", user_id="u1")

    await create_react_agent(model, tools=[lookup]).ainvoke(
        {"messages": [HumanMessage("weather?")]}, config={"callbacks": [handler]})

    assert sdk.instance.listen_calls == 1
    sent = lambda t: [c.kwargs["content"] for c in sdk.instance.send_message.await_args_list
                      if c.kwargs["event_type"] == t]
    assert sent("user_message") == ["weather?"]
    assert sent("assistant_message") == ["It is sunny."]  # not "Let me check."
    sdk.conversation.record_message.assert_not_awaited()
    sdk.instance.record_tool_call.assert_awaited_once()
    sdk.instance.record_tool_result.assert_awaited_once()


def test_the_checkpointer_says_it_cannot_restore_anything():
    with pytest.warns(FutureWarning, match="cannot restore checkpoints"):
        SynapCheckpointSaver(make_sdk(), user_id="u1")


@pytest.mark.asyncio
async def test_a_thread_it_wrote_and_cannot_read_back_is_an_error_not_a_silent_miss(caplog):
    """Synap returns the facts extracted from the checkpoint blob, never the blob."""
    import logging
    sdk = make_sdk()
    sdk.memories = SimpleNamespace(create=AsyncMock())
    sdk.fetch = AsyncMock(return_value=SimpleNamespace(facts=[SimpleNamespace(
        content="The user is planning a trip", metadata={}, extracted_at=0)]))
    saver = SynapCheckpointSaver(sdk, user_id="u1")
    config = {"configurable": {"thread_id": "t1"}}
    await saver.aput(config, {"id": "cp1"}, {}, {})

    with caplog.at_level(logging.ERROR, logger="synap_langgraph.checkpointer"):
        assert await saver.aget_tuple(config) is None
        assert await saver.aget_tuple(config) is None

    errors = [r for r in caplog.records if "cannot read one back" in r.getMessage()]
    assert len(errors) == 1  # once per thread, not once per read
