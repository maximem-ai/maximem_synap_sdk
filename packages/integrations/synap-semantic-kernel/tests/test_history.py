"""Synap's compaction reaches the model through Semantic Kernel's thread.

These run a real ``ChatCompletionAgent`` on a service that records exactly the
chat history it was sent, so they prove the framework reads the history through
the seam, not that a stand-in does.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock

import pytest
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import AuthorRole, ChatHistory, ChatMessageContent
from semantic_kernel.contents.history_reducer.chat_history_truncation_reducer import (
    ChatHistoryTruncationReducer,
)

from synap_integrations_common import set_auto_listen
from synap_semantic_kernel import (
    SynapAgentThread,
    SynapChatHistoryReducer,
    synap_st_chat_message,
)

BLOCK = "## Conversation so far (compacted 2026-10-22)\nThe user is planning a trip to Lisbon."


class RecordingService(ChatCompletionClientBase):
    """Answers "ok" and keeps what it was sent."""

    ai_model_id: str = "fake"
    sent: List[List[ChatMessageContent]] = []

    async def _inner_get_chat_message_contents(self, chat_history, settings):
        self.sent.append(list(chat_history.messages))
        return [ChatMessageContent(role=AuthorRole.ASSISTANT, content="ok")]

    def get_prompt_execution_settings_class(self):
        return PromptExecutionSettings

    @property
    def last(self) -> str:
        return "\n".join(f"{m.role.value}:{m.content}" for m in self.sent[-1])


class FakeInstance:
    """Behaves like sdk.instance: listen() flips is_listening on."""

    def __init__(self):
        self.is_listening = False
        self.listen_calls = 0
        self.send_message = AsyncMock()
        self.record_tool_call = AsyncMock()
        self.record_tool_result = AsyncMock()
        self.record_thinking = AsyncMock()
        self.end_session = AsyncMock()
        self._controller = SimpleNamespace(_transport=object())

    async def listen(self):
        self.listen_calls += 1
        self.is_listening = True

    async def stop_listening(self):
        self.is_listening = False


def make_sdk(compaction=True, recent=(("user", "flights?"), ("assistant", "TAP at 9am"))):
    ctx = SimpleNamespace(context_block=None, history_block=None, recent_messages=[], compaction_id=None)
    if compaction:
        ctx = SimpleNamespace(
            context_block=BLOCK, history_block=None, compaction_id="cmp_1",
            recent_messages=[SimpleNamespace(role=r, content=c) for r, c in recent],
        )
    return SimpleNamespace(
        instance=FakeInstance(),
        _initialized=True,
        conversation=SimpleNamespace(
            record_message=AsyncMock(),
            context=SimpleNamespace(get_context_for_prompt=AsyncMock(return_value=ctx)),
        ),
    )


def earlier_turns() -> ChatHistory:
    return ChatHistory(messages=[
        ChatMessageContent(role=AuthorRole.USER, content="I want to visit Portugal"),
        ChatMessageContent(role=AuthorRole.ASSISTANT, content="Great, when?"),
        ChatMessageContent(role=AuthorRole.USER, content="In May, for a week"),
        ChatMessageContent(role=AuthorRole.ASSISTANT, content="Lisbon is lovely in May."),
    ])


def agent_on(service):
    return ChatCompletionAgent(service=service, name="travel", instructions="You are a travel agent.")


def thread_for(sdk, **kw):
    return SynapAgentThread(sdk, conversation_id="conv-1", user_id="u1", customer_id="c1", **kw)


# ---------------------------------------------------------------------------
# The seam, driven by the real framework
# ---------------------------------------------------------------------------


class TestCompactionReachesTheModel:
    @pytest.mark.asyncio
    async def test_the_default_thread_sends_the_compaction_in_place_of_earlier_turns(self):
        svc = RecordingService(sent=[])
        thread = thread_for(make_sdk(), chat_history=None)
        for m in earlier_turns().messages:
            await thread.on_new_message(m)
        await agent_on(svc).get_response(messages="and the hotel?", thread=thread)
        seen = svc.last
        assert "The user is planning a trip to Lisbon" in seen
        assert "<synap_conversation_history>" in seen
        assert "I want to visit Portugal" not in seen
        assert "In May, for a week" not in seen
        assert "TAP at 9am" in seen
        assert "and the hotel?" in seen
        # the agent's own instructions are still first
        assert svc.sent[-1][0].content == "You are a travel agent."

    @pytest.mark.asyncio
    async def test_the_default_history_is_the_synap_reducer(self):
        assert isinstance(thread_for(make_sdk())._chat_history, SynapChatHistoryReducer)

    @pytest.mark.asyncio
    async def test_messages_are_untouched_when_synap_has_no_compaction(self):
        svc = RecordingService(sent=[])
        thread = thread_for(make_sdk(compaction=False))
        for m in earlier_turns().messages:
            await thread.on_new_message(m)
        await agent_on(svc).get_response(messages="and the hotel?", thread=thread)
        assert "I want to visit Portugal" in svc.last
        assert "In May, for a week" in svc.last
        assert "synap_conversation_history" not in svc.last

    @pytest.mark.asyncio
    async def test_messages_are_untouched_when_the_sdk_fails(self):
        svc = RecordingService(sent=[])
        sdk = make_sdk()
        sdk.conversation.context.get_context_for_prompt = AsyncMock(side_effect=RuntimeError("down"))
        thread = thread_for(sdk)
        for m in earlier_turns().messages:
            await thread.on_new_message(m)
        await agent_on(svc).get_response(messages="and the hotel?", thread=thread)
        assert "In May, for a week" in svc.last

    @pytest.mark.asyncio
    async def test_the_compaction_is_never_recorded_back_as_a_turn(self):
        svc = RecordingService(sent=[])
        sdk = make_sdk()
        thread = thread_for(sdk)
        for m in earlier_turns().messages:
            await thread.on_new_message(m)
        sdk.instance.send_message.reset_mock()
        await agent_on(svc).get_response(messages="and the hotel?", thread=thread)
        await agent_on(svc).get_response(messages="and the car?", thread=thread)
        for call in sdk.instance.send_message.await_args_list:
            assert "synap_conversation_history" not in call.kwargs["content"]
        assert svc.last.count("<synap_conversation_history>") == 1

    @pytest.mark.asyncio
    async def test_a_callers_own_history_is_spliced_for_the_model_but_not_rewritten(self):
        svc = RecordingService(sent=[])
        mine = ChatHistoryTruncationReducer(target_count=50, threshold_count=10)
        for m in earlier_turns().messages:
            mine.add_message(m)
        thread = thread_for(make_sdk(), chat_history=mine)
        await agent_on(svc).get_response(messages="and the hotel?", thread=thread)
        assert "The user is planning a trip to Lisbon" in svc.last
        assert "In May, for a week" not in svc.last
        stored = " ".join(str(m.content) for m in mine.messages)
        assert "In May, for a week" in stored
        assert "synap_conversation_history" not in stored

    @pytest.mark.asyncio
    async def test_the_reducer_on_its_own_replaces_in_place_and_reports_a_change(self):
        history = SynapChatHistoryReducer(sdk=make_sdk(), conversation_id="conv-1")
        for m in earlier_turns().messages:
            history.add_message(m)
        history.add_message(ChatMessageContent(role=AuthorRole.USER, content="and the hotel?"))
        assert await history.reduce() is history
        text = " ".join(str(m.content) for m in history.messages)
        assert "Lisbon" in text and "In May, for a week" not in text and "and the hotel?" in text

    @pytest.mark.asyncio
    async def test_the_reducer_returns_none_when_there_is_nothing_to_replace(self):
        history = SynapChatHistoryReducer(sdk=make_sdk(compaction=False), conversation_id="conv-1")
        history.add_message(ChatMessageContent(role=AuthorRole.USER, content="hi"))
        assert await history.reduce() is None
        assert len(history.messages) == 1


# ---------------------------------------------------------------------------
# The ST message does not repeat turns the framework sends
# ---------------------------------------------------------------------------


class TestSTMessageDoesNotDuplicateTurns:
    def _sdk(self):
        sdk = make_sdk(compaction=False)
        sdk.conversation.context.get_context_for_prompt = AsyncMock(return_value=SimpleNamespace(
            available=True, formatted_context="User asked about Lisbon."))
        return sdk

    @pytest.mark.asyncio
    async def test_with_no_history_the_block_is_sent(self):
        msg = await synap_st_chat_message(self._sdk(), "conv-1", system="Be brief.")()
        assert "User asked about Lisbon." in msg.content

    @pytest.mark.asyncio
    async def test_next_to_a_history_with_turns_only_the_system_text_is_sent(self):
        sdk = self._sdk()
        msg = await synap_st_chat_message(sdk, "conv-1", system="Be brief.")(earlier_turns())
        assert msg.content == "Be brief."
        sdk.conversation.context.get_context_for_prompt.assert_not_awaited()


# ---------------------------------------------------------------------------
# The stream is the default path: nobody called listen()
# ---------------------------------------------------------------------------


class TestStreamOpensByDefault:
    @pytest.mark.asyncio
    async def test_the_first_turn_opens_the_stream_and_both_turns_go_on_it(self):
        svc = RecordingService(sent=[])
        sdk = make_sdk(compaction=False)
        assert sdk.instance.is_listening is False
        await agent_on(svc).get_response(messages="where is my order", thread=thread_for(sdk))
        assert sdk.instance.listen_calls == 1
        sent = [(c.kwargs["event_type"], c.kwargs["content"])
                for c in sdk.instance.send_message.await_args_list]
        assert sent == [("user_message", "where is my order"), ("assistant_message", "ok")]
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_stream_that_will_not_open_falls_back_to_rest(self):
        svc = RecordingService(sent=[])
        sdk = make_sdk(compaction=False)
        sdk.instance.listen = AsyncMock(side_effect=ConnectionError("no route"))
        await agent_on(svc).get_response(messages="hi", thread=thread_for(sdk))
        roles = [c.kwargs["role"] for c in sdk.conversation.record_message.await_args_list]
        assert roles == ["user", "assistant"]
        sdk.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_opting_out_keeps_turns_on_rest_and_never_listens(self):
        svc = RecordingService(sent=[])
        sdk = make_sdk(compaction=False)
        set_auto_listen(sdk, False)
        await agent_on(svc).get_response(messages="hi", thread=thread_for(sdk))
        assert sdk.instance.listen_calls == 0
        assert sdk.conversation.record_message.await_count == 2
