"""Synap's compaction in place of an agent's earlier steps, and the gRPC stream
as the default write path, proven against a real smolagents agent.

The model is a recording double: the assertions are on the messages smolagents
actually hands to ``Model.generate``, which is the only place "what the model
sees" can be read.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from smolagents import CodeAgent
from smolagents.models import ChatMessage, MessageRole, Model

from synap_integrations_common import set_auto_listen
from synap_smolagents import attach_synap_history, synap_step_callbacks

PREAMBLE_MARK = "The budget is 2,400 EUR."
FINAL = "<code>\nfinal_answer('ok')\n</code>"


class RecordingModel(Model):
    """Answers every call with a final_answer and keeps what it was sent."""

    def __init__(self, replies=None):
        super().__init__(model_id="recording")
        self.calls = []
        self.replies = list(replies or [])

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs):
        self.calls.append(list(messages))
        text = self.replies.pop(0) if self.replies else FINAL
        return ChatMessage(role=MessageRole.ASSISTANT, content=text)


def flat(messages):
    out = []
    for m in messages:
        c = m.content
        text = c if isinstance(c, str) else "\n".join(
            p.get("text", "") for p in c if isinstance(p, dict))
        out.append((getattr(m.role, "value", m.role), text))
    return out


def joined(messages):
    return "\n".join(t for _, t in flat(messages))


def compaction(recent=()):
    return SimpleNamespace(
        context_block=f"## Conversation so far (compacted)\n{PREAMBLE_MARK}",
        history_block=None,
        compaction_id="cmp_1",
        recent_messages=[SimpleNamespace(role=r, content=c) for r, c in recent],
        formatted_context="",
        available=True,
    )


class FakeInstance:
    """Behaves like sdk.instance: listen() flips is_listening on."""

    def __init__(self):
        self.is_listening = False
        self.listen = AsyncMock(side_effect=self._listen)
        self.send_message = AsyncMock()
        self.record_thinking = AsyncMock()
        self.record_tool_call = AsyncMock()
        self.record_tool_result = AsyncMock()

    async def _listen(self):
        self.is_listening = True


def make_sdk(context=None, error=None):
    sdk = MagicMock()
    sdk._initialized = True
    sdk.instance = FakeInstance()
    sdk.conversation.record_message = AsyncMock()
    sdk.memories.create = AsyncMock()
    get = AsyncMock(return_value=context, side_effect=error)
    sdk.conversation.context.get_context_for_prompt = get
    return sdk


def make_agent(sdk, model, **cb_kwargs):
    return CodeAgent(
        tools=[], model=model, verbosity_level=-1,
        step_callbacks=synap_step_callbacks(sdk, "alice", "conv1", **cb_kwargs),
    )


# ── gRPC by default ──────────────────────────────────────────────────────────


def test_the_first_turn_opens_the_stream_and_the_run_rides_on_it():
    sdk = make_sdk()
    agent = make_agent(sdk, RecordingModel(["Thought: x\n<code>\nprint(1)\n</code>"]))
    assert sdk.instance.is_listening is False
    agent.run("where is my order")
    sdk.instance.listen.assert_awaited_once()
    sent = [(c.kwargs["event_type"], c.kwargs["content"])
            for c in sdk.instance.send_message.await_args_list]
    assert sent == [("user_message", "where is my order"), ("assistant_message", "ok")]
    sdk.conversation.record_message.assert_not_awaited()
    assert sdk.instance.record_thinking.await_count >= 1
    sdk.memories.create.assert_not_called()


def test_a_second_run_reuses_the_stream():
    sdk = make_sdk()
    agent = make_agent(sdk, RecordingModel())
    agent.run("one")
    agent.run("two")
    sdk.instance.listen.assert_awaited_once()
    assert sdk.instance.send_message.await_count == 4


def test_opting_out_keeps_turns_on_rest():
    sdk = make_sdk()
    set_auto_listen(sdk, False)
    make_agent(sdk, RecordingModel()).run("where is my order")
    sdk.instance.listen.assert_not_awaited()
    assert [(c.kwargs["role"], c.kwargs["content"])
            for c in sdk.conversation.record_message.await_args_list] == [
        ("user", "where is my order"), ("assistant", "ok")]


# ── the splice is on the default path ───────────────────────────────────────


def test_once_synap_has_compacted_the_model_sees_the_compaction_not_its_old_steps():
    sdk = make_sdk(compaction(recent=[("user", "and shipping?"), ("assistant", "free over 50")]))
    model = RecordingModel()
    agent = make_agent(sdk, model)
    agent.run("old question about invoices")          # run 1: attaches after its step
    model.calls.clear()
    agent.run("what is next", reset=False)            # run 2: old steps are in memory
    seen = model.calls[0]
    text = joined(seen)
    assert PREAMBLE_MARK in text
    assert "<synap_conversation_history>" in text
    assert "old question about invoices" not in text  # replaced by the compaction
    assert ("user", "and shipping?") in flat(seen)
    assert ("assistant", "free over 50") in flat(seen)
    assert "New task:\nwhat is next" in text           # the current task is untouched
    assert flat(seen)[0][0] == "system"                 # smolagents' own system prompt first


def test_a_conversation_with_no_compaction_is_left_alone():
    sdk = make_sdk(context=SimpleNamespace(context_block=None, recent_messages=[]))
    model = RecordingModel()
    agent = make_agent(sdk, model)
    agent.run("first question")
    model.calls.clear()
    agent.run("second", reset=False)
    text = joined(model.calls[0])
    assert "first question" in text and "<synap_conversation_history>" not in text


def test_a_synap_failure_leaves_the_agents_messages_alone():
    sdk = make_sdk(error=RuntimeError("down"))
    model = RecordingModel()
    agent = make_agent(sdk, model)
    agent.run("first question")
    model.calls.clear()
    agent.run("second", reset=False)
    assert "first question" in joined(model.calls[0])


def test_attach_before_the_first_run_covers_a_resumed_conversations_first_step():
    """smolagents has no hook before a model call, so the recorder can only
    attach after step one; attaching up front covers step one of a new process."""
    sdk = make_sdk(compaction())
    model = RecordingModel()
    agent = CodeAgent(tools=[], model=model, verbosity_level=-1)
    assert attach_synap_history(agent, sdk, "conv1") is True
    agent.run("continue please")
    assert PREAMBLE_MARK in joined(model.calls[0])
    assert "New task:\ncontinue please" in joined(model.calls[0])


def test_the_conversation_is_read_once_per_run_not_once_per_step():
    sdk = make_sdk(compaction())
    model = RecordingModel(["Thought: a\n<code>\nprint(1)\n</code>",
                            "Thought: b\n<code>\nprint(2)\n</code>"])
    agent = CodeAgent(tools=[], model=model, verbosity_level=-1)
    attach_synap_history(agent, sdk, "conv1")
    agent.run("t")
    assert len(model.calls) == 3
    assert sdk.conversation.context.get_context_for_prompt.await_count == 1
    # the later steps still carry the current run's own steps
    assert "print(2)" in joined(model.calls[2]) and PREAMBLE_MARK in joined(model.calls[2])


def test_attaching_twice_wraps_once():
    sdk = make_sdk(compaction())
    agent = CodeAgent(tools=[], model=RecordingModel(), verbosity_level=-1)
    attach_synap_history(agent, sdk, "conv1")
    wrapped = agent.write_memory_to_messages
    attach_synap_history(agent, sdk, "conv1")
    assert agent.write_memory_to_messages is wrapped


def test_compact_history_false_does_not_touch_the_agent():
    sdk = make_sdk(compaction())
    model = RecordingModel()
    agent = make_agent(sdk, model, compact_history=False)
    agent.run("one")
    agent.run("two", reset=False)
    assert "New task:\none" in joined(model.calls[-1])
    sdk.conversation.context.get_context_for_prompt.assert_not_awaited()


# ── audit: a repeated identical task and answer ─────────────────────────────


def test_a_repeated_identical_task_and_answer_are_both_recorded_again():
    sdk = make_sdk()
    agent = make_agent(sdk, RecordingModel())
    agent.run("what is the status?")
    agent.run("what is the status?")
    sent = [(c.kwargs["event_type"], c.kwargs["content"])
            for c in sdk.instance.send_message.await_args_list]
    assert sent == [("user_message", "what is the status?"), ("assistant_message", "ok")] * 2


def test_one_run_still_reports_its_task_and_answer_once():
    sdk = make_sdk()
    model = RecordingModel(["Thought: a\n<code>\nprint(1)\n</code>"])
    make_agent(sdk, model).run("task")
    sent = [c.kwargs["event_type"] for c in sdk.instance.send_message.await_args_list]
    assert sent == ["user_message", "assistant_message"]
