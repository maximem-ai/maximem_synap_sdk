"""Synap's compaction reaches the model through Strands' model-call middleware.

These run a real ``Agent`` on a model that records exactly the messages it was
sent, so they prove the framework calls the middleware on its default path, not
that a stand-in does.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock

import pytest
from strands import Agent, tool
from strands.models import Model

from synap_integrations_common import set_auto_listen
from synap_strands_agents import SynapShortTermHook, SynapStreamHook

BLOCK = "## Conversation so far (compacted 2026-10-22)\nThe user is planning a trip to Lisbon."


class RecordingModel(Model):
    """Answers "ok" (or calls a tool first) and keeps what it was sent."""

    def __init__(self, tool_first: str = ""):
        self.calls: List[List[dict]] = []
        self.tool_first = tool_first

    def update_config(self, **kw): ...

    def get_config(self):
        return {}

    def structured_output(self, *a, **kw):  # pragma: no cover
        raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kw):
        self.calls.append(json.loads(json.dumps(messages)))
        yield {"messageStart": {"role": "assistant"}}
        if self.tool_first and len(self.calls) == 1:
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": self.tool_first}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": "ok"}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                            "metrics": {"latencyMs": 1}}}

    def seen(self, call: int = -1) -> str:
        out = []
        for m in self.calls[call]:
            for b in m["content"]:
                out.append(f"{m['role']}:{b.get('text', b)}")
        return "\n".join(out)


class FakeInstance:
    """Behaves like sdk.instance: listen() flips is_listening on."""

    def __init__(self):
        self.is_listening = False
        self.listen_calls = 0
        self.send_message = AsyncMock()
        self.record_tool_call = AsyncMock()
        self.record_tool_result = AsyncMock()
        self._controller = SimpleNamespace(_transport=object())

    async def listen(self):
        self.listen_calls += 1
        self.is_listening = True

    async def stop_listening(self):
        self.is_listening = False


def make_sdk(compaction=True, recent=(("user", "flights?"), ("assistant", "TAP at 9am"))):
    ctx = SimpleNamespace(context_block=None, history_block=None, recent_messages=[],
                          compaction_id=None, available=False, formatted_context="")
    if compaction:
        ctx = SimpleNamespace(
            context_block=BLOCK, history_block=None, compaction_id="cmp_1",
            available=True, formatted_context=BLOCK,
            recent_messages=[SimpleNamespace(role=r, content=c) for r, c in recent],
        )
    return SimpleNamespace(
        instance=FakeInstance(),
        _initialized=True,
        conversation=SimpleNamespace(
            record_message=AsyncMock(),
            context=SimpleNamespace(get_context_for_prompt=AsyncMock(return_value=ctx)),
        ),
        memories=SimpleNamespace(create=AsyncMock()),
    )


def earlier_turns() -> List[dict]:
    return [
        {"role": "user", "content": [{"text": "I want to visit Portugal"}]},
        {"role": "assistant", "content": [{"text": "Great, when?"}]},
        {"role": "user", "content": [{"text": "In May, for a week"}]},
        {"role": "assistant", "content": [{"text": "Lisbon is lovely in May."}]},
    ]


def make_agent(model, sdk, *, hooks=("short_term",), messages=None, **kw):
    built = []
    if "short_term" in hooks:
        built.append(SynapShortTermHook(sdk, "conv-1"))
    if "stream" in hooks:
        built.append(SynapStreamHook(sdk, "conv-1", "alice", customer_id="acme"))
    return Agent(model=model, hooks=built, messages=messages if messages is not None else earlier_turns(),
                 callback_handler=None, **kw)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("SYNAP_AUTO_LISTEN", raising=False)


# ---------------------------------------------------------------------------
# The seam, driven by the real framework
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hook", ["short_term", "stream"])
async def test_either_hook_makes_the_model_see_the_compaction(hook):
    model = RecordingModel()
    agent = make_agent(model, make_sdk(), hooks=(hook,))
    await agent.invoke_async("and the hotel?")
    seen = model.seen()
    assert "The user is planning a trip to Lisbon" in seen
    assert "<synap_conversation_history>" in seen
    assert "I want to visit Portugal" not in seen
    assert "In May, for a week" not in seen
    assert "TAP at 9am" in seen
    assert "and the hotel?" in seen
    # the roles still alternate, starting with the user
    roles = [m["role"] for m in model.calls[0]]
    assert roles[0] == "user" and all(a != b for a, b in zip(roles, roles[1:]))


async def test_agent_messages_are_not_rewritten():
    model = RecordingModel()
    agent = make_agent(model, make_sdk())
    await agent.invoke_async("and the hotel?")
    stored = " ".join(b.get("text", "") for m in agent.messages for b in m["content"])
    assert "I want to visit Portugal" in stored
    assert "synap_conversation_history" not in stored


async def test_messages_are_untouched_when_synap_has_no_compaction():
    model = RecordingModel()
    agent = make_agent(model, make_sdk(compaction=False))
    await agent.invoke_async("and the hotel?")
    assert "I want to visit Portugal" in model.seen()
    assert "In May, for a week" in model.seen()
    assert "synap_conversation_history" not in model.seen()


async def test_messages_are_untouched_when_the_sdk_fails():
    model = RecordingModel()
    sdk = make_sdk()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(side_effect=RuntimeError("down"))
    agent = make_agent(model, sdk, hooks=("stream",))
    await agent.invoke_async("and the hotel?")
    assert "In May, for a week" in model.seen()


async def test_both_hooks_install_it_once_and_a_tool_loop_fetches_once():
    @tool
    def ping() -> str:
        """ping"""
        return "pong"

    model = RecordingModel(tool_first="ping")
    sdk = make_sdk()
    agent = make_agent(model, sdk, hooks=("short_term", "stream"), tools=[ping])
    await agent.invoke_async("go")
    assert len(model.calls) == 2
    assert sdk.conversation.context.get_context_for_prompt.await_count == 1
    second = model.seen(1)
    assert "Lisbon" in second and "pong" in second
    assert "In May, for a week" not in second


async def test_the_compaction_is_never_recorded_back_as_a_turn():
    model = RecordingModel()
    sdk = make_sdk()
    agent = make_agent(model, sdk, hooks=("short_term", "stream"))
    await agent.invoke_async("and the hotel?")
    await agent.invoke_async("and the car?")
    for call in sdk.instance.send_message.await_args_list:
        assert "synap_conversation_history" not in call.kwargs["content"]
    assert model.seen().count("<synap_conversation_history>") == 1


# ---------------------------------------------------------------------------
# The short-term block no longer sits next to a history that has the turns
# ---------------------------------------------------------------------------


async def test_the_st_block_is_not_folded_next_to_a_framework_history():
    model = RecordingModel()
    sdk = make_sdk(compaction=False)
    sdk.conversation.context.get_context_for_prompt = AsyncMock(return_value=SimpleNamespace(
        available=True, formatted_context="User asked about Lisbon.", context_block=None,
        history_block=None, recent_messages=[], compaction_id=None))
    agent = make_agent(model, sdk)
    await agent.invoke_async("and the hotel?")
    assert "User asked about Lisbon." not in model.seen()
    await agent.invoke_async("and the car?")
    assert "User asked about Lisbon." not in model.seen()


async def test_the_st_block_is_folded_once_when_the_framework_has_no_history():
    model = RecordingModel()
    sdk = make_sdk(compaction=False)
    sdk.conversation.context.get_context_for_prompt = AsyncMock(return_value=SimpleNamespace(
        available=True, formatted_context="User asked about Lisbon.", context_block=None,
        history_block=None, recent_messages=[], compaction_id=None))
    agent = make_agent(model, sdk, hooks=("short_term", "stream"), messages=[])
    await agent.invoke_async("hello again")
    assert "User asked about Lisbon." in model.seen()
    await agent.invoke_async("and the car?")
    assert model.seen().count("User asked about Lisbon.") == 1  # the first message's, not a new one


async def test_the_folded_block_is_not_recorded_as_something_the_user_said():
    model = RecordingModel()
    sdk = make_sdk(compaction=False)
    sdk.conversation.context.get_context_for_prompt = AsyncMock(return_value=SimpleNamespace(
        available=True, formatted_context="User asked about Lisbon.", context_block=None,
        history_block=None, recent_messages=[], compaction_id=None))
    agent = make_agent(model, sdk, hooks=("short_term", "stream"), messages=[])
    await agent.invoke_async("hello again")
    sent = [c.kwargs["content"] for c in sdk.instance.send_message.await_args_list]
    assert sent == ["hello again", "ok"]


async def test_no_st_fold_when_a_compaction_exists_and_the_history_is_empty():
    model = RecordingModel()
    agent = make_agent(model, make_sdk(), messages=[])
    await agent.invoke_async("hello again")
    seen = model.seen()
    assert seen.count("The user is planning a trip to Lisbon") == 1
    assert "<synap_short_term_context>" not in seen


# ---------------------------------------------------------------------------
# The stream is the default path: nobody called listen()
# ---------------------------------------------------------------------------


async def test_the_first_turn_opens_the_stream_and_everything_goes_on_it():
    @tool
    def ping() -> str:
        """ping"""
        return "pong"

    model = RecordingModel(tool_first="ping")
    sdk = make_sdk(compaction=False)
    assert sdk.instance.is_listening is False
    agent = make_agent(model, sdk, hooks=("stream",), tools=[ping])
    await agent.invoke_async("go")
    assert sdk.instance.listen_calls == 1
    kinds = [(c.kwargs["event_type"], c.kwargs["content"]) for c in sdk.instance.send_message.await_args_list]
    assert kinds == [("user_message", "go"), ("assistant_message", "ok")]
    sdk.instance.record_tool_call.assert_awaited_once()
    sdk.instance.record_tool_result.assert_awaited_once()
    sdk.conversation.record_message.assert_not_awaited()


async def test_a_stream_that_will_not_open_falls_back_to_rest():
    model = RecordingModel()
    sdk = make_sdk(compaction=False)
    sdk.instance.listen = AsyncMock(side_effect=ConnectionError("no route"))
    agent = make_agent(model, sdk, hooks=("stream",), messages=[])
    await agent.invoke_async("hi")
    roles = [c.kwargs["role"] for c in sdk.conversation.record_message.await_args_list]
    assert roles == ["user", "assistant"]


async def test_opting_out_keeps_turns_on_rest_and_never_listens():
    model = RecordingModel()
    sdk = make_sdk(compaction=False)
    set_auto_listen(sdk, False)
    agent = make_agent(model, sdk, hooks=("stream",), messages=[])
    await agent.invoke_async("hi")
    assert sdk.instance.listen_calls == 0
    assert sdk.conversation.record_message.await_count == 2
