"""The defaults, against Haystack's real experimental ``Agent``.

Nothing here calls ``listen()``. The SDK is a fake whose ``instance.is_listening``
starts False, the way a fresh SDK's does, and the generator is a scripted fake
that keeps every message list it was sent.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest
from haystack import Document, component
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool

from synap_haystack import SynapMemoryStore, SynapMemoryWriter

# The experimental Agent (and its chat_message_store slot) needs haystack-ai 2.x;
# haystack-experimental 0.19 does not import against haystack-ai 3.
try:
    from haystack_experimental.components.agents import Agent
except ImportError:
    Agent = None

needs_experimental_agent = pytest.mark.skipif(
    Agent is None, reason="haystack_experimental's Agent needs haystack-ai 2.x")

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
        instance_id="inst",
        conversation=SimpleNamespace(
            record_message=AsyncMock(return_value={"message_id": "m1"}),
            context=SimpleNamespace(get_context_for_prompt=AsyncMock(return_value=context)),
        ),
        fetch=AsyncMock(return_value=SimpleNamespace(
            facts=[], preferences=[], episodes=[], emotions=[], temporal_events=[], scope_map={})),
    )


def ctx(compaction=COMPACTION, turns=(), total=0):
    return SimpleNamespace(
        context_block=compaction, history_block=None, available=True,
        formatted_context="recent", total_message_count=total,
        recent_messages=[SimpleNamespace(role=r, content=t) for r, t in turns],
    )


@component
class ScriptedGenerator:
    def __init__(self, script):
        self.script = script
        self.seen: List[Any] = []

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], tools: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        self.seen.append(list(messages))
        return {"replies": [self.script.pop(0)]}


def lookup(city: str) -> str:
    return f"{city}: sunny"


TOOL = Tool(name="lookup", description="Look a city up.",
            parameters={"type": "object", "properties": {"city": {"type": "string"}},
                        "required": ["city"]},
            function=lookup)


def tool_turn():
    return [
        ChatMessage.from_assistant("Let me check.", tool_calls=[
            ToolCall(tool_name="lookup", arguments={"city": "Lisbon"}, id="c1")]),
        ChatMessage.from_assistant("It is sunny in Lisbon."),
    ]


def agent_for(script, **stores):
    gen = ScriptedGenerator(script)
    agent = Agent(chat_generator=gen, tools=[TOOL], system_prompt="Be brief.", **stores)
    return gen, agent


def sent(sdk, event_type):
    return [c.kwargs["content"] for c in sdk.instance.send_message.await_args_list
            if c.kwargs["event_type"] == event_type]


def texts(messages):
    return [m.text for m in messages]


def test_the_store_is_a_chat_message_store():
    store = SynapMemoryStore(make_sdk(), user_id="u1")
    for name in ("to_dict", "from_dict", "count_messages", "write_messages",
                 "delete_messages", "delete_all_messages", "retrieve_messages"):
        assert callable(getattr(store, name))
    if Agent is not None:
        Agent(chat_generator=ScriptedGenerator([]), chat_message_store=store)


@needs_experimental_agent
def test_an_agent_loads_synaps_compaction_instead_of_a_ten_message_window():
    sdk = make_sdk(ctx(turns=[("user", "synap kept q"), ("assistant", "synap kept a")]))
    store = SynapMemoryStore(sdk, user_id="u1")
    gen, agent = agent_for([ChatMessage.from_assistant("Book the 9am flight.")], chat_message_store=store)

    agent.run(messages=[ChatMessage.from_user("what now?")],
              chat_message_store_kwargs={"chat_history_id": "c1"})

    seen = gen.seen[0]
    assert seen[0].text == "Be brief."
    assert "The user is planning a trip to Lisbon." in seen[1].text and seen[1].is_from("system")
    assert texts(seen[2:]) == ["synap kept q", "synap kept a", "what now?"]


@needs_experimental_agent
def test_an_agent_before_the_first_compaction_gets_the_recent_turns():
    sdk = make_sdk(ctx(compaction=None, turns=[("user", "q1"), ("assistant", "a1")]))
    gen, agent = agent_for([ChatMessage.from_assistant("ok")],
                           chat_message_store=SynapMemoryStore(sdk, user_id="u1"))
    agent.run(messages=[ChatMessage.from_user("q2")], chat_message_store_kwargs={"chat_history_id": "c1"})
    assert texts(gen.seen[0][1:]) == ["q1", "a1", "q2"]


def test_last_k_keeps_the_compaction_and_trims_the_turns():
    sdk = make_sdk(ctx(turns=[("user", "t1"), ("assistant", "t2"), ("user", "t3"), ("assistant", "t4")]))
    msgs = SynapMemoryStore(sdk, user_id="u1").retrieve_messages("c1", last_k=2)
    assert msgs[0].is_from("system") and texts(msgs[1:]) == ["t3", "t4"]


@needs_experimental_agent
def test_an_agent_run_records_each_turn_once_on_an_auto_opened_stream():
    sdk = make_sdk(ctx(compaction=None, turns=[("user", "q1"), ("assistant", "a1")]))
    store = SynapMemoryStore(sdk, user_id="u1", conversation_id="c1")
    gen, agent = agent_for(tool_turn(), chat_message_store=store, memory_store=store)

    agent.run(messages=[ChatMessage.from_user("weather in Lisbon?")],
              chat_message_store_kwargs={"chat_history_id": "c1"})

    assert sdk.instance.listen_calls == 1  # opened by the first turn, never by the caller
    # Both slots got the same messages, history included: each turn goes out once,
    # and the retrieved history (already in Synap) not at all.
    assert sent(sdk, "user_message") == ["weather in Lisbon?"]
    assert sent(sdk, "assistant_message") == ["It is sunny in Lisbon."]  # the tool-call lap has no words to record
    sdk.conversation.record_message.assert_not_awaited()
    sdk.instance.record_tool_call.assert_awaited_once()
    sdk.instance.record_tool_result.assert_awaited_once()


def test_a_message_with_no_words_is_not_an_empty_turn():
    sdk = make_sdk()
    store = SynapMemoryStore(sdk, user_id="u1")
    results = store.add_memories(
        messages=[ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="lookup", arguments={}, id="c1")])],
        conversation_id="c1")
    assert results == [{"role": "assistant", "status": "skipped", "reason": "no text"}]
    sdk.instance.send_message.assert_not_awaited()
    sdk.conversation.record_message.assert_not_awaited()
    sdk.instance.record_tool_call.assert_awaited_once()


def test_count_and_delete():
    store = SynapMemoryStore(make_sdk(ctx(total=7)), user_id="u1")
    assert store.count_messages("c1") == 7
    store.delete_messages("c1")
    store.delete_all_messages()  # warn once, no-op


def test_retrieve_degrades_to_nothing_on_an_outage():
    sdk = make_sdk()
    sdk.conversation.context.get_context_for_prompt.side_effect = RuntimeError("down")
    store = SynapMemoryStore(sdk, user_id="u1")
    assert store.retrieve_messages("c1") == []
    assert store.count_messages("c1") == 0


def test_the_writer_keeps_tool_calls_and_results_when_given_messages():
    sdk = make_sdk()
    writer = SynapMemoryWriter(store=SynapMemoryStore(sdk, user_id="u1"), conversation_id="c1")
    result = writer.run(messages=[
        ChatMessage.from_user("weather?"),
        ChatMessage.from_assistant("checking", tool_calls=[ToolCall(tool_name="lookup", arguments={"city": "X"}, id="c1")]),
        ChatMessage.from_tool("sunny", origin=ToolCall(tool_name="lookup", arguments={"city": "X"}, id="c1")),
    ])
    assert result["written_count"] == 1  # the question; "checking" narrates a tool call
    sdk.instance.record_tool_call.assert_awaited_once()
    sdk.instance.record_tool_result.assert_awaited_once()


def test_the_writer_still_takes_documents():
    sdk = make_sdk()
    writer = SynapMemoryWriter(store=SynapMemoryStore(sdk, user_id="u1"), conversation_id="c1")
    result = writer.run(documents=[Document(content="hi", meta={"role": "user"})])
    assert result["written_count"] == 1
    assert sent(sdk, "user_message") == ["hi"]
