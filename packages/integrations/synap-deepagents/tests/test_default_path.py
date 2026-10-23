"""The defaults, against a real ``create_deep_agent``.

Nothing here calls ``listen()``. The SDK is a fake whose ``instance.is_listening``
starts False, the way a fresh SDK's does, and the model is a scripted fake that
keeps what each call was sent.
"""

from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock

import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

from synap_deepagents import SynapMemoryMiddleware, SynapShortTermMiddleware, SynapStreamMiddleware

COMPACTION = "## Conversation so far (compacted 1x)\nThe user is planning a trip to Lisbon."
OLD_TURNS = [HumanMessage("old question one"), AIMessage("old answer one"),
             HumanMessage("old question two"), AIMessage("old answer two")]


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
        fetch=AsyncMock(return_value=SimpleNamespace(formatted_context="")),
        archive_tool_result=lambda *a, **k: None,
    )


def compacted(turns=()):
    return SimpleNamespace(
        context_block=COMPACTION, history_block=None, available=True, formatted_context="",
        recent_messages=[SimpleNamespace(role=r, content=t) for r, t in turns],
    )


class ScriptedModel(BaseChatModel):
    replies: List[AIMessage] = []
    seen: List[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.replies[min(len(self.seen) - 1, len(self.replies) - 1)])])


def _lookup(city: str) -> str:
    return f"{city}: sunny"


async def _alookup(city: str) -> str:
    return f"{city}: sunny"


lookup = StructuredTool.from_function(func=_lookup, coroutine=_alookup, name="lookup",
                                      description="Look a city up.")


def agent_with(middleware, replies):
    model = ScriptedModel(replies=replies, seen=[])
    return model, create_deep_agent(model=model, tools=[lookup], middleware=middleware)


def tool_turn():
    return [
        AIMessage("Let me check.", tool_calls=[{"name": "lookup", "args": {"city": "Lisbon"}, "id": "c1"}]),
        AIMessage("It is sunny in Lisbon."),
    ]


def sent(sdk, event_type):
    return [c.kwargs["content"] for c in sdk.instance.send_message.await_args_list
            if c.kwargs["event_type"] == event_type]


def system_text(messages):
    return messages[0].text if hasattr(messages[0], "text") else str(messages[0].content)


@pytest.mark.parametrize("sync", [False, True])
async def test_memory_middleware_with_a_conversation_id_records_on_an_auto_opened_stream(sync):
    sdk = make_sdk()
    _, agent = agent_with([SynapMemoryMiddleware(sdk=sdk, user_id="u1", conversation_id="c1")], tool_turn())
    payload = {"messages": [HumanMessage("weather in Lisbon?", id="h1")]}

    if sync:
        import asyncio
        await asyncio.to_thread(agent.invoke, payload)
    else:
        await agent.ainvoke(payload)

    assert sdk.instance.listen_calls == 1  # opened by the first turn, never by the caller
    assert sent(sdk, "user_message") == ["weather in Lisbon?"]
    assert sent(sdk, "assistant_message") == ["It is sunny in Lisbon."]  # not "Let me check."
    sdk.conversation.record_message.assert_not_awaited()
    sdk.instance.record_tool_call.assert_awaited_once()
    sdk.instance.record_tool_result.assert_awaited_once()


async def test_memory_middleware_without_a_conversation_id_records_nothing():
    sdk = make_sdk()
    _, agent = agent_with([SynapMemoryMiddleware(sdk=sdk, user_id="u1")], tool_turn())
    await agent.ainvoke({"messages": [HumanMessage("hi", id="h1")]})
    assert sdk.instance.listen_calls == 0
    sdk.instance.send_message.assert_not_awaited()


async def test_the_stream_middleware_alone_opens_the_stream_and_records_on_it():
    sdk = make_sdk()
    _, agent = agent_with([SynapStreamMiddleware(sdk=sdk, conversation_id="c1", user_id="u1")], tool_turn())
    await agent.ainvoke({"messages": [HumanMessage("weather in Lisbon?", id="h1")]})
    assert sdk.instance.listen_calls == 1
    assert sent(sdk, "user_message") == ["weather in Lisbon?"]
    assert sent(sdk, "assistant_message") == ["It is sunny in Lisbon."]
    sdk.conversation.record_message.assert_not_awaited()


@pytest.mark.parametrize("make_middleware", [
    lambda sdk: SynapMemoryMiddleware(sdk=sdk, user_id="u1", conversation_id="c1"),
    lambda sdk: SynapStreamMiddleware(sdk=sdk, conversation_id="c1", user_id="u1"),
    lambda sdk: SynapShortTermMiddleware(sdk=sdk, conversation_id="c1"),
], ids=["memory", "stream", "short_term"])
@pytest.mark.parametrize("sync", [False, True])
async def test_the_model_sees_the_compaction_in_place_of_the_earlier_turns(make_middleware, sync):
    sdk = make_sdk(compacted(turns=[("user", "synap kept this"), ("assistant", "and this")]))
    model, agent = agent_with([make_middleware(sdk)], [AIMessage("Book the 9am flight.")])
    payload = {"messages": [*OLD_TURNS, HumanMessage("what now?", id="now")]}

    if sync:
        import asyncio
        await asyncio.to_thread(agent.invoke, payload)
    else:
        await agent.ainvoke(payload)

    sent_to_model = model.seen[0]
    assert "The user is planning a trip to Lisbon." in system_text(sent_to_model)
    assert [m.content for m in sent_to_model[1:]] == ["synap kept this", "and this", "what now?"]
    assert not any("old question" in str(m.content) for m in sent_to_model)


async def test_the_current_turns_tool_loop_stays_paired_on_every_lap():
    sdk = make_sdk(compacted())
    model, agent = agent_with([SynapStreamMiddleware(sdk=sdk, conversation_id="c1", user_id="u1")],
                              tool_turn())
    await agent.ainvoke({"messages": [*OLD_TURNS, HumanMessage("weather?", id="now")]})

    assert len(model.seen) == 2
    assert sdk.conversation.context.get_context_for_prompt.await_count == 2  # per model call
    second = model.seen[1]
    assert any(isinstance(m, ToolMessage) for m in second)
    assert not any("old question" in str(m.content) for m in second)


async def test_the_model_sees_the_messages_as_they_are_until_synap_has_compacted():
    sdk = make_sdk(SimpleNamespace(context_block=None, history_block=None, recent_messages=[],
                                   available=True, formatted_context="recent"))
    model, agent = agent_with([SynapStreamMiddleware(sdk=sdk, conversation_id="c1", user_id="u1")],
                              [AIMessage("fine")])
    await agent.ainvoke({"messages": [*OLD_TURNS, HumanMessage("what now?", id="now")]})
    assert [m.content for m in model.seen[0][1:]] == [
        "old question one", "old answer one", "old question two", "old answer two", "what now?"]


async def test_a_synap_outage_does_not_stop_the_run():
    sdk = make_sdk()
    sdk.conversation.context.get_context_for_prompt.side_effect = RuntimeError("down")
    model, agent = agent_with([SynapStreamMiddleware(sdk=sdk, conversation_id="c1", user_id="u1")],
                              [AIMessage("fine")])
    await agent.ainvoke({"messages": [HumanMessage("hi", id="h1")]})
    assert model.seen[0][-1].content == "hi"
