"""The hooks, against haystack-ai 3's real ``Agent``.

Nothing here calls ``listen()``: the SDK is a fake whose ``instance.is_listening``
starts False.
"""

from typing import Any, Dict, List, Optional

import pytest
from haystack import component
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool

from synap_haystack import SynapMemoryStore

from .test_default_path import COMPACTION, ctx, make_sdk, sent, texts, TOOL, tool_turn, ScriptedGenerator

Agent = pytest.importorskip("haystack.components.agents", exc_type=ImportError).Agent
hooks_mod = pytest.importorskip("haystack.hooks", exc_type=ImportError)
from synap_haystack import SynapHistoryHook, SynapRecordHook  # noqa: E402

OLD = [ChatMessage.from_user("old q1"), ChatMessage.from_assistant("old a1"),
       ChatMessage.from_user("old q2"), ChatMessage.from_assistant("old a2")]


def agent_for(script, store, with_record=True):
    gen = ScriptedGenerator(script)
    hooks = {"before_llm": [SynapHistoryHook(store, "c1")]}
    if with_record:
        hooks["after_run"] = [SynapRecordHook(store, "c1")]
    return gen, Agent(chat_generator=gen, tools=[TOOL], system_prompt="Be brief.", hooks=hooks)


def test_the_model_sees_synaps_compaction_in_place_of_the_earlier_turns():
    sdk = make_sdk(ctx(turns=[("user", "synap kept q"), ("assistant", "synap kept a")]))
    gen, agent = agent_for([ChatMessage.from_assistant("Book the 9am flight.")],
                           SynapMemoryStore(sdk, user_id="u1"))

    agent.run(messages=[*OLD, ChatMessage.from_user("what now?")])

    seen = gen.seen[0]
    assert seen[0].text == "Be brief."
    assert seen[1].is_from("system") and "The user is planning a trip to Lisbon." in seen[1].text
    assert texts(seen[2:]) == ["synap kept q", "synap kept a", "what now?"]
    assert not any("old q" in (m.text or "") for m in seen)


def test_the_messages_are_untouched_until_synap_has_compacted():
    sdk = make_sdk(ctx(compaction=None))
    gen, agent = agent_for([ChatMessage.from_assistant("fine")], SynapMemoryStore(sdk, user_id="u1"))
    agent.run(messages=[*OLD, ChatMessage.from_user("what now?")])
    assert texts(gen.seen[0][1:]) == ["old q1", "old a1", "old q2", "old a2", "what now?"]


def test_a_synap_outage_does_not_stop_the_run():
    sdk = make_sdk()
    sdk.conversation.context.get_context_for_prompt.side_effect = RuntimeError("down")
    gen, agent = agent_for([ChatMessage.from_assistant("fine")], SynapMemoryStore(sdk, user_id="u1"))
    agent.run(messages=[ChatMessage.from_user("hi")])
    assert gen.seen[0][-1].text == "hi"


def test_the_tool_loop_stays_paired_on_every_lap():
    sdk = make_sdk(ctx())
    gen, agent = agent_for(tool_turn(), SynapMemoryStore(sdk, user_id="u1"), with_record=False)
    agent.run(messages=[*OLD, ChatMessage.from_user("weather?")])
    second = gen.seen[1]
    assert any(m.tool_call_results for m in second)
    assert not any("old q" in (m.text or "") for m in second)
    assert sdk.conversation.context.get_context_for_prompt.await_count == 2  # per model call


def test_a_run_is_recorded_once_on_an_auto_opened_stream():
    sdk = make_sdk(ctx(compaction=None))
    gen, agent = agent_for(tool_turn(), SynapMemoryStore(sdk, user_id="u1"))

    agent.run(messages=[ChatMessage.from_user("weather in Lisbon?")])

    assert sdk.instance.listen_calls == 1  # opened by the first turn, never by the caller
    assert sent(sdk, "user_message") == ["weather in Lisbon?"]
    assert sent(sdk, "assistant_message") == ["It is sunny in Lisbon."]  # not the tool-call lap
    sdk.conversation.record_message.assert_not_awaited()
    sdk.instance.record_tool_call.assert_awaited_once()
    sdk.instance.record_tool_result.assert_awaited_once()


def test_history_the_hook_loaded_is_not_recorded_again():
    sdk = make_sdk(ctx(turns=[("user", "synap kept q"), ("assistant", "synap kept a")]))
    gen, agent = agent_for([ChatMessage.from_assistant("Book it.")], SynapMemoryStore(sdk, user_id="u1"))
    agent.run(messages=[ChatMessage.from_user("what now?")])
    assert sent(sdk, "user_message") == ["what now?"]
    assert sent(sdk, "assistant_message") == ["Book it."]


def test_the_hooks_serialize():
    store = SynapMemoryStore(make_sdk(), user_id="u1")
    data = SynapHistoryHook(store, "c1").to_dict()
    assert data["init_parameters"]["chat_history_id"] == "c1"


def test_the_hooks_need_a_conversation():
    with pytest.raises(ValueError, match="chat_history_id"):
        SynapHistoryHook(SynapMemoryStore(make_sdk(), user_id="u1"), "")


async def _app_turns(users, store, sdk):
    """An app that feeds the agent's returned messages back as the next turn's history."""
    gen = ScriptedGenerator([ChatMessage.from_assistant(f"reply {i}") for i in range(len(users))])
    agent = Agent(chat_generator=gen, system_prompt="Be brief.",
                  hooks={"before_llm": [SynapHistoryHook(store, "c1")],
                         "after_run": [SynapRecordHook(store, "c1")]})
    history: List[ChatMessage] = []
    for text in users:
        out = await agent.run_async(messages=[*history, ChatMessage.from_user(text)])
        history = [m for m in out["messages"] if not m.is_from("system")]


async def test_each_turn_of_an_app_that_feeds_history_back_is_written_once():
    # State.get hands the hook deep copies, so the stamp never reaches the app's history.
    sdk = make_sdk(ctx(compaction=None))
    await _app_turns([f"question {i}" for i in range(6)], SynapMemoryStore(sdk, user_id="u1"), sdk)
    assert sent(sdk, "user_message") == [f"question {i}" for i in range(6)]
    assert sent(sdk, "assistant_message") == [f"reply {i}" for i in range(6)]


async def test_a_message_the_user_repeats_is_still_written_each_time():
    sdk = make_sdk(ctx(compaction=None))
    await _app_turns(["yes", "tell me more", "yes", "yes"], SynapMemoryStore(sdk, user_id="u1"), sdk)
    assert sent(sdk, "user_message") == ["yes", "tell me more", "yes", "yes"]
    assert len(sent(sdk, "assistant_message")) == 4
