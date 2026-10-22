"""The default path of a real ``ChatAgent`` over ``SynapAgentMemory``.

Nothing here calls ``listen()`` or any Synap-specific hook besides attaching the
memory: the stream must open on the first turn, and once Synap holds a
compaction the model must see it in place of CAMEL's earlier turns.
"""

from datetime import datetime, timezone

import pytest
from unittest.mock import AsyncMock, MagicMock

from camel.agents import ChatAgent
from camel.memories import MemoryRecord
from camel.messages import BaseMessage
from camel.models.stub_model import StubModel
from camel.types import ModelType, OpenAIBackendRole
from maximem_synap.models.context import ContextForPromptResponse, RecentMessage
from maximem_synap.models.context import UnifiedContextResponse

from synap_camel_ai import SynapAgentMemory, synap_st_system_message
from synap_integrations_common import set_auto_listen

BLOCK = "## Conversation so far (compacted 2026-10-22)\nThe user is planning a trip to Lisbon."


class CapturingModel(StubModel):
    """The real model backend interface; remembers what the agent sent."""

    def __init__(self):
        super().__init__(ModelType.STUB)
        self.seen = []

    def _run(self, messages, response_format=None, tools=None):
        self.seen.append(list(messages))
        return super()._run(messages, response_format, tools)


def _sdk(*, compaction=None, listening=False):
    """An SDK whose stream is closed until ``listen()`` is called."""
    sdk = MagicMock()
    sdk._initialized = True
    sdk.instance.is_listening = listening

    async def listen():
        sdk.instance.is_listening = True

    sdk.instance.listen = AsyncMock(side_effect=listen)
    sdk.instance.send_message = AsyncMock()
    sdk.instance.record_tool_call = AsyncMock()
    sdk.instance.record_tool_result = AsyncMock()
    sdk.instance.record_thinking = AsyncMock()
    sdk.conversation.record_message = AsyncMock()
    sdk.memories.create = AsyncMock()
    sdk.fetch = AsyncMock(
        return_value=UnifiedContextResponse(formatted_context="RECALLED-FACT")
    )
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=compaction or ContextForPromptResponse()
    )
    return sdk


def _compaction(turns=()):
    return ContextForPromptResponse(
        available=True,
        formatted_context="unused",
        context_block=BLOCK,
        recent_messages=[
            RecentMessage(
                role=r, content=c, timestamp=datetime.now(timezone.utc), message_id=f"m{i}"
            )
            for i, (r, c) in enumerate(turns)
        ],
    )


def _agent(sdk, model, **kwargs):
    memory = SynapAgentMemory(sdk, "alice", conversation_id="conv-1", **kwargs)
    return ChatAgent(system_message="SYSTEM-PROMPT", model=model, memory=memory)


def _flat(messages):
    return [(m["role"], m["content"]) for m in messages]


# ── gRPC by default ──────────────────────────────────────────────────────────


def test_the_first_turn_opens_the_stream_and_both_turns_go_on_it():
    sdk = _sdk()
    assert sdk.instance.is_listening is False
    agent = _agent(sdk, CapturingModel())
    agent.step("where is my order")
    sdk.instance.listen.assert_awaited_once()
    sent = [
        (c.kwargs["event_type"], c.kwargs["content"])
        for c in sdk.instance.send_message.await_args_list
    ]
    assert sent == [
        ("user_message", "where is my order"),
        ("assistant_message", "Lorem Ipsum"),
    ]
    sdk.conversation.record_message.assert_not_awaited()
    sdk.memories.create.assert_not_awaited()


def test_a_second_turn_reuses_the_stream():
    sdk = _sdk()
    agent = _agent(sdk, CapturingModel())
    agent.step("one")
    agent.step("two")
    sdk.instance.listen.assert_awaited_once()
    assert sdk.instance.send_message.await_count == 4


def test_the_opt_out_keeps_turns_on_rest():
    sdk = _sdk()
    set_auto_listen(sdk, False)
    _agent(sdk, CapturingModel()).step("hello")
    sdk.instance.listen.assert_not_awaited()
    sdk.instance.send_message.assert_not_awaited()
    assert [
        c.kwargs["role"] for c in sdk.conversation.record_message.await_args_list
    ] == ["user", "assistant"]


def test_a_stream_that_will_not_open_falls_back_to_rest_without_raising():
    sdk = _sdk()
    sdk.instance.listen = AsyncMock(side_effect=RuntimeError("no stream"))
    _agent(sdk, CapturingModel()).step("hello")
    assert sdk.conversation.record_message.await_count == 2


# ── Synap's compaction replaces CAMEL's earlier turns ────────────────────────


def _agent_with_history(sdk, model):
    agent = _agent(sdk, model)
    agent.step("OLD-Q1 tell me about Porto")
    agent.step("OLD-Q2 and about Faro")
    sdk.instance.send_message.reset_mock()
    return agent


def test_after_a_compaction_the_model_sees_it_and_not_the_replaced_turns():
    model = CapturingModel()
    sdk = _sdk()
    agent = _agent_with_history(sdk, model)
    sdk.conversation.context.get_context_for_prompt.return_value = _compaction(
        [("user", "SYNAP-Q"), ("assistant", "SYNAP-A")]
    )
    agent.step("NEW-Q what did we decide")
    sent = _flat(model.seen[-1])
    texts = " | ".join(c for _, c in sent)
    assert "Lisbon" in texts
    assert "OLD-Q1" not in texts and "OLD-Q2" not in texts
    assert [c for r, c in sent if r in ("user", "assistant")] == [
        "SYNAP-Q", "SYNAP-A", "NEW-Q what did we decide",
    ]
    # CAMEL's system prompt stays, first.
    assert sent[0] == ("system", "SYSTEM-PROMPT")


def test_without_a_compaction_the_framework_history_is_untouched():
    model = CapturingModel()
    sdk = _sdk()  # get_context_for_prompt answers with no context_block
    agent = _agent_with_history(sdk, model)
    agent.step("NEW-Q")
    texts = [c for _, c in _flat(model.seen[-1])]
    assert any("OLD-Q1" in t for t in texts) and any("OLD-Q2" in t for t in texts)
    assert not any("Lisbon" in t for t in texts)


def test_a_failing_compaction_fetch_leaves_the_framework_history_alone():
    model = CapturingModel()
    sdk = _sdk()
    agent = _agent_with_history(sdk, model)
    sdk.conversation.context.get_context_for_prompt.side_effect = RuntimeError("down")
    agent.step("NEW-Q")
    texts = [c for _, c in _flat(model.seen[-1])]
    assert any("OLD-Q1" in t for t in texts)


def test_the_recall_block_sits_before_the_users_question():
    """ScoreBasedContextCreator sorts non-first system records by timestamp, so a
    block stamped "now" used to land after the question it was fetched for."""
    model = CapturingModel()
    sdk = _sdk()
    _agent(sdk, model).step("THE-QUESTION")
    sent = _flat(model.seen[-1])
    recall = next(i for i, (_, c) in enumerate(sent) if "RECALLED-FACT" in c)
    question = next(i for i, (_, c) in enumerate(sent) if c == "THE-QUESTION")
    assert recall < question
    assert sent[-1] == ("user", "THE-QUESTION")


def test_with_a_compaction_the_order_is_system_compaction_turns_recall_question():
    model = CapturingModel()
    sdk = _sdk()
    agent = _agent_with_history(sdk, model)
    sdk.conversation.context.get_context_for_prompt.return_value = _compaction(
        [("user", "SYNAP-Q"), ("assistant", "SYNAP-A")]
    )
    agent.step("NEW-Q")
    sent = _flat(model.seen[-1])
    assert [r for r, _ in sent] == ["system", "system", "system", "user", "assistant", "user"]
    assert "Lisbon" in sent[1][1] and "RECALLED-FACT" in sent[2][1]


def test_the_short_term_snapshot_in_the_system_message_is_replaced_not_repeated():
    model = CapturingModel()
    sdk = _sdk(compaction=_compaction([("user", "SYNAP-Q"), ("assistant", "SYNAP-A")]))
    system = synap_st_system_message(sdk, "conv-1", system_message="SYSTEM-PROMPT")
    assert system.count("Lisbon") == 1
    memory = SynapAgentMemory(sdk, "alice", conversation_id="conv-1")
    agent = ChatAgent(system_message=system, model=model, memory=memory)
    agent.step("NEW-Q")
    sent = _flat(model.seen[-1])
    assert sum("Lisbon" in c for _, c in sent) == 1
    assert sent[0] == ("system", "SYSTEM-PROMPT")


# ── CAMEL's own summarizer is not a conversation turn ────────────────────────


def _summarize(agent):
    agent.update_memory(
        BaseMessage.make_user_message("user", "question one"), OpenAIBackendRole.USER
    )
    agent.update_memory(
        BaseMessage.make_assistant_message("assistant", "answer one"),
        OpenAIBackendRole.ASSISTANT,
    )
    agent._update_memory_with_summary(
        "[CONTEXT_SUMMARY] user asked q1, got a1", include_summaries=False
    )


def _written(sdk):
    return [
        (c.kwargs["role"], c.kwargs["content"])
        for c in sdk.conversation.record_message.await_args_list
    ]


def test_camel_summarization_writes_no_summary_and_no_rewritten_user_turn():
    sdk = _sdk()
    set_auto_listen(sdk, False)
    agent = _agent(sdk, CapturingModel())
    _summarize(agent)
    # question one and answer one, once each; nothing the summarizer wrote.
    assert _written(sdk) == [("user", "question one"), ("assistant", "answer one")]


def test_camel_summarization_on_the_stream_writes_no_summary_either():
    sdk = _sdk()
    agent = _agent(sdk, CapturingModel())
    _summarize(agent)
    sent = [c.kwargs["content"] for c in sdk.instance.send_message.await_args_list]
    assert sent == ["question one", "answer one"]


def test_recalled_memory_never_comes_back_as_an_assistant_turn():
    """summarize() reads get_context(), which holds the recall block; what the
    summarizer writes back must not reach Synap."""
    sdk = _sdk()
    set_auto_listen(sdk, False)
    agent = _agent(sdk, CapturingModel())
    agent.update_memory(
        BaseMessage.make_user_message("user", "q"), OpenAIBackendRole.USER
    )
    agent._update_memory_with_summary(
        "[CONTEXT_SUMMARY] <synap_long_term_memory>RECALLED-FACT</synap_long_term_memory>",
        include_summaries=False,
    )
    assert all("RECALLED-FACT" not in c for _, c in _written(sdk))


def test_after_summarization_the_rewritten_user_message_matches_synaps_copy():
    """The current turn is the rewritten message; Synap holds the original, so
    the splice must not send the question twice."""
    model = CapturingModel()
    sdk = _sdk()
    agent = _agent(sdk, model)
    agent.update_memory(
        BaseMessage.make_user_message("user", "what did we decide"), OpenAIBackendRole.USER
    )
    agent._update_memory_with_summary("[CONTEXT_SUMMARY] earlier stuff", include_summaries=False)
    sdk.conversation.context.get_context_for_prompt.return_value = _compaction(
        [("user", "SYNAP-Q"), ("assistant", "SYNAP-A"), ("user", "what did we decide")]
    )
    messages, _ = agent.memory.get_context()
    users = [m["content"] for m in messages if m["role"] == "user"]
    assert users == [
        "SYNAP-Q",
        "Based on the previous CONTEXT_SUMMARY, continue with my current message: what did we decide",
    ]


def test_token_limit_on_the_memory_does_not_change_the_context():
    """CAMEL's creator no longer filters on it; ChatAgent(token_limit=) decides."""
    model = CapturingModel()
    sdk = _sdk()
    memory = SynapAgentMemory(sdk, "alice", conversation_id="conv-1", token_limit=1)
    agent = ChatAgent(system_message="SYSTEM-PROMPT", model=model, memory=memory)
    agent.step("one two three")
    assert ("user", "one two three") in _flat(model.seen[-1])
