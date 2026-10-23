"""The defaults, against a real LlamaIndex ``FunctionAgent``.

Nothing here calls ``listen()``. The SDK is a fake whose ``instance.is_listening``
starts False, the way a fresh SDK's does, and the LLM is a scripted fake that
keeps every message list it was sent.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Sequence
from unittest.mock import AsyncMock

import pytest
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    LLMMetadata,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from llama_index.core.llms.llm import ToolSelection

from synap_integrations_common import set_auto_listen
from synap_llamaindex import SynapChatMemory

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
        fetch=AsyncMock(return_value=SimpleNamespace(formatted_context="")),
    )


def ctx(compaction=COMPACTION, turns=()):
    return SimpleNamespace(
        context_block=compaction, history_block=None, available=True,
        formatted_context=f"{compaction}\nRECENT-TURNS-DUPLICATE" if compaction else "recent",
        recent_messages=[SimpleNamespace(role=r, content=t) for r, t in turns],
    )


class ScriptedLLM(FunctionCallingLLM):
    """Replays one queued assistant message per call and keeps what it was sent."""

    script: List[Any] = []
    seen: List[Any] = []

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)

    def _prepare_chat_with_tools(self, tools, user_msg=None, chat_history=None, verbose=False,
                                 allow_parallel_tool_calls=False, **kwargs) -> Dict[str, Any]:
        return {"messages": list(chat_history or [])}

    def get_tool_calls_from_response(self, response, error_on_no_tool_call=True, **kwargs):
        return [ToolSelection(tool_id=b.tool_call_id, tool_name=b.tool_name, tool_kwargs=b.tool_kwargs)
                for b in response.message.blocks if isinstance(b, ToolCallBlock)]

    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        self.seen.append(list(messages))
        return ChatResponse(message=self.script.pop(0))

    async def achat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def complete(self, *a, **k): raise NotImplementedError
    def stream_chat(self, *a, **k): raise NotImplementedError
    def stream_complete(self, *a, **k): raise NotImplementedError
    async def acomplete(self, *a, **k): raise NotImplementedError
    async def astream_chat(self, *a, **k): raise NotImplementedError
    async def astream_complete(self, *a, **k): raise NotImplementedError


def lookup(city: str) -> str:
    """Look a city up."""
    return f"{city}: sunny"


def tool_turn():
    return [
        ChatMessage(role=MessageRole.ASSISTANT, blocks=[
            TextBlock(text="Let me check."),
            ToolCallBlock(tool_call_id="c1", tool_name="lookup", tool_kwargs={"city": "Lisbon"}),
        ]),
        ChatMessage(role=MessageRole.ASSISTANT, content="It is sunny in Lisbon."),
    ]


def agent_for(script):
    llm = ScriptedLLM(script=script, seen=[])
    return llm, FunctionAgent(tools=[FunctionTool.from_defaults(lookup)], llm=llm, streaming=False)


def sent(sdk, event_type):
    return [c.kwargs["content"] for c in sdk.instance.send_message.await_args_list
            if c.kwargs["event_type"] == event_type]


def contents(messages):
    return [str(m.content) for m in messages]


OLD = [ChatMessage(role=MessageRole.USER, content="old question one"),
       ChatMessage(role=MessageRole.ASSISTANT, content="old answer one"),
       ChatMessage(role=MessageRole.USER, content="old question two"),
       ChatMessage(role=MessageRole.ASSISTANT, content="old answer two")]


@pytest.mark.asyncio
async def test_a_function_agent_turn_opens_the_stream_and_records_on_it():
    sdk = make_sdk(ctx(compaction=None))
    memory = SynapChatMemory(sdk, "c1", "u1")
    _, agent = agent_for(tool_turn())

    await agent.run(user_msg="weather in Lisbon?", memory=memory)

    assert sdk.instance.listen_calls == 1  # opened by the first turn, never by the caller
    assert sent(sdk, "user_message") == ["weather in Lisbon?"]
    assert "It is sunny in Lisbon." in sent(sdk, "assistant_message")
    sdk.conversation.record_message.assert_not_awaited()
    sdk.instance.record_tool_call.assert_awaited_once()
    sdk.instance.record_tool_result.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_agent_is_sent_the_compaction_in_place_of_the_earlier_turns():
    sdk = make_sdk(ctx(turns=[("user", "synap kept this"), ("assistant", "and this")]))
    memory = SynapChatMemory(sdk, "c1", "u1")
    llm, agent = agent_for([ChatMessage(role=MessageRole.ASSISTANT, content="Book the 9am flight.")])

    await agent.run(user_msg="what now?", chat_history=OLD, memory=memory)

    sent_to_model = llm.seen[0]
    texts = contents(sent_to_model)
    assert "The user is planning a trip to Lisbon." in texts[0]
    assert sent_to_model[0].role == MessageRole.SYSTEM
    assert texts[1:] == ["synap kept this", "and this", "what now?"]
    assert not any("old question" in t for t in texts)
    assert not any("RECENT-TURNS-DUPLICATE" in t for t in texts)


@pytest.mark.asyncio
async def test_each_turn_is_sent_once_not_two_or_three_times():
    """The formatted context, the recent turns and the local buffer all held the
    same turns; the model was sent each of them two or three times."""
    sdk = make_sdk(ctx(compaction=None))
    memory = SynapChatMemory(sdk, "c1", "u1")
    llm, agent = agent_for([ChatMessage(role=MessageRole.ASSISTANT, content="first answer"),
                            ChatMessage(role=MessageRole.ASSISTANT, content="second answer")])

    await agent.run(user_msg="first question", memory=memory)
    sdk.conversation.context.get_context_for_prompt.return_value = ctx(
        compaction=None, turns=[("user", "first question"), ("assistant", "first answer")])
    await agent.run(user_msg="second question", memory=memory)

    assert contents(llm.seen[1]) == ["first question", "first answer", "second question"]


@pytest.mark.asyncio
async def test_the_agent_is_untouched_until_synap_has_compacted():
    sdk = make_sdk(ctx(compaction=None))
    memory = SynapChatMemory(sdk, "c1", "u1")
    llm, agent = agent_for([ChatMessage(role=MessageRole.ASSISTANT, content="fine")])

    await agent.run(user_msg="what now?", chat_history=OLD, memory=memory)

    assert contents(llm.seen[0]) == ["old question one", "old answer one", "old question two",
                                     "old answer two", "what now?"]


@pytest.mark.asyncio
async def test_the_current_turns_tool_loop_is_not_spliced_away():
    sdk = make_sdk(ctx())
    memory = SynapChatMemory(sdk, "c1", "u1")
    llm, agent = agent_for(tool_turn())

    await agent.run(user_msg="weather?", chat_history=OLD, memory=memory)

    second = llm.seen[1]
    assert any(m.role == MessageRole.TOOL for m in second)
    assert not any("old question" in str(m.content) for m in second)


@pytest.mark.asyncio
async def test_a_history_set_on_the_memory_reaches_synap_once():
    """`aset` replaced the local messages and recorded nothing."""
    sdk = make_sdk(ctx(compaction=None, turns=[("user", "old question one")]))
    memory = SynapChatMemory(sdk, "c1", "u1")

    await memory.aset(OLD)

    # The first turn was already in Synap; the other three were not.
    assert sent(sdk, "user_message") == ["old question two"]
    assert sent(sdk, "assistant_message") == ["old answer one", "old answer two"]
    # Setting the same history again records nothing new.
    sdk.instance.send_message.reset_mock()
    await memory.aset(OLD)
    sdk.instance.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_compaction_the_memory_returned_is_never_written_back_as_a_turn():
    sdk = make_sdk(ctx())
    memory = SynapChatMemory(sdk, "c1", "u1")
    got = await memory.aget()
    set_auto_listen(sdk, False)
    await memory.aput(ChatMessage(role=MessageRole.USER, content=got[0].content))
    sdk.conversation.record_message.assert_not_awaited()
    sdk.instance.send_message.assert_not_awaited()
