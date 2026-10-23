"""Synap's compaction in place of CrewAI's history, through CrewAI itself.

These run a real ``Agent`` and a real ``Crew`` against a scripted model, so
what is asserted is what the model was handed, not what our hook was asked to
do. The two things that could be wrong without our code being wrong are
whether CrewAI calls a globally registered ``before_llm_call`` hook on its
default path, and whether it sends the list we edited in place.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from maximem_synap.models.context import ContextForPromptResponse, RecentMessage

from synap_integrations_common import set_auto_listen

from synap_crewai import SynapCrewAIListener, build_synap_st_backstory
from synap_crewai.stream import _drop_static_short_term


class ScriptedLLM(BaseLLM):
    """Answers with a fixed string and keeps every message list it was sent."""

    seen: List[List[dict]] = []

    def __init__(self, answer: str = "done") -> None:
        super().__init__(model="scripted")
        self._answer = answer
        self.seen = []

    def call(self, messages, *args: Any, **kwargs: Any):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        self.seen.append([dict(m) for m in messages])
        return self._answer

    def supports_function_calling(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8000


def _rm(role: str, content: str) -> RecentMessage:
    return RecentMessage(role=role, content=content, message_id=content,
                         timestamp=datetime.now(timezone.utc))


def _sdk(compacted: bool, *, recent=None):
    sdk = MagicMock()
    sdk._initialized = True
    sdk.instance.is_listening = True
    for name in ("send_message", "record_thinking", "record_tool_call", "record_tool_result"):
        setattr(sdk.instance, name, AsyncMock())
    sdk.conversation.record_message = AsyncMock()
    if compacted:
        resp = ContextForPromptResponse(
            available=True,
            formatted_context="ignored",
            context_block="## Conversation so far (compacted)\nThe user is Isha and is vegetarian.",
            recent_messages=recent if recent is not None else [
                _rm("user", "what is on the menu"), _rm("assistant", "paneer and dal")],
        )
    else:
        resp = ContextForPromptResponse(available=True, formatted_context="x")
    sdk.conversation.context.get_context_for_prompt = AsyncMock(return_value=resp)
    return sdk


def _joined(messages) -> str:
    return "\n".join(str(m.get("content")) for m in messages)


def _history(extra=()):
    return [
        {"role": "user", "content": "OLD-TURN-ONE my name is Isha"},
        {"role": "assistant", "content": "OLD-TURN-TWO nice to meet you"},
        *extra,
        {"role": "user", "content": "what should I eat"},
    ]


@pytest.fixture
def listener_for():
    made = []

    def build(sdk):
        set_auto_listen(sdk, False)
        listener = SynapCrewAIListener(sdk, user_id="u", conversation_id="conv-1")
        made.append(listener)
        return listener

    yield build
    for listener in made:
        listener.stop()


class TestAgentKickoff:
    def test_once_compacted_the_model_sees_synaps_history_not_the_frameworks(self, listener_for):
        sdk = _sdk(True)
        listener_for(sdk)
        llm = ScriptedLLM()
        agent = Agent(role="r", goal="g", backstory="b", llm=llm)
        agent.kickoff(_history())
        sent = llm.seen[0]
        text = _joined(sent)
        assert "The user is Isha and is vegetarian." in text
        assert "<synap_conversation_history>" in text
        assert "OLD-TURN-ONE" not in text and "OLD-TURN-TWO" not in text
        # Synap's post-compaction turns, then the current request, in order.
        roles = [m["role"] for m in sent]
        assert roles[-1] == "user" and "what should I eat" in sent[-1]["content"]
        assert "paneer and dal" in text
        assert text.index("The user is Isha") < text.index("paneer and dal") < text.index("what should I eat")
        sdk.conversation.context.get_context_for_prompt.assert_awaited()

    def test_with_no_compaction_the_framework_messages_are_untouched(self, listener_for):
        listener_for(_sdk(False))
        llm = ScriptedLLM()
        Agent(role="r", goal="g", backstory="b", llm=llm).kickoff(_history())
        text = _joined(llm.seen[0])
        assert "OLD-TURN-ONE" in text and "OLD-TURN-TWO" in text
        assert "<synap_conversation_history>" not in text

    def test_when_synap_fails_the_framework_messages_are_untouched(self, listener_for):
        sdk = _sdk(True)
        sdk.conversation.context.get_context_for_prompt = AsyncMock(side_effect=RuntimeError("down"))
        listener_for(sdk)
        llm = ScriptedLLM()
        Agent(role="r", goal="g", backstory="b", llm=llm).kickoff(_history())
        assert "OLD-TURN-ONE" in _joined(llm.seen[0])

    def test_a_stopped_listener_leaves_no_hook_behind(self, listener_for):
        sdk = _sdk(True)
        listener_for(sdk).stop()
        llm = ScriptedLLM()
        Agent(role="r", goal="g", backstory="b", llm=llm).kickoff(_history())
        assert "OLD-TURN-ONE" in _joined(llm.seen[0])
        sdk.conversation.context.get_context_for_prompt.assert_not_awaited()

    def test_the_agent_still_answers(self, listener_for):
        listener_for(_sdk(True))
        out = Agent(role="r", goal="g", backstory="b", llm=ScriptedLLM("the answer")).kickoff(_history())
        assert "the answer" in out.raw


class TestCrewKickoff:
    def test_a_crew_task_gets_the_compaction_and_the_backstory_snapshot_is_dropped(self, listener_for):
        sdk = _sdk(True)
        listener_for(sdk)
        llm = ScriptedLLM()
        backstory = ("<synap_short_term_context>\nSTALE-SNAPSHOT\n</synap_short_term_context>\n\n"
                     "You are a helpful cook.")
        agent = Agent(role="cook", goal="cook", backstory=backstory, llm=llm)
        task = Task(description="Answer: {q}", expected_output="an answer", agent=agent)
        Crew(agents=[agent], tasks=[task]).kickoff(inputs={"q": "what should I eat"})
        sent = llm.seen[0]
        text = _joined(sent)
        assert "The user is Isha and is vegetarian." in text
        assert "STALE-SNAPSHOT" not in text and "You are a helpful cook." in text
        assert "Answer: what should I eat" in sent[-1]["content"]

    def test_the_recorded_copy_of_this_kickoffs_inputs_is_not_sent_twice(self, listener_for):
        """Synap holds the crew's inputs as the newest user turn; the task
        prompt is the same request, rendered."""
        sdk = _sdk(True, recent=[_rm("user", "what is on the menu"),
                                 _rm("assistant", "paneer and dal"),
                                 _rm("user", "what should I eat")])
        listener_for(sdk)
        llm = ScriptedLLM()
        agent = Agent(role="cook", goal="cook", backstory="b", llm=llm)
        task = Task(description="Answer: {q}", expected_output="an answer", agent=agent)
        Crew(agents=[agent], tasks=[task]).kickoff(inputs={"q": "what should I eat"})
        contents = [m["content"] for m in llm.seen[0] if m["role"] == "user"]
        assert contents.count("what should I eat") == 0
        assert sum("what should I eat" in c for c in contents) == 1  # inside the task prompt


def test_drop_static_short_term_only_touches_system_messages():
    msgs = [
        {"role": "system", "content": "<synap_short_term_context>a</synap_short_term_context>\n\nkeep"},
        {"role": "user", "content": "<synap_short_term_context>a</synap_short_term_context>"},
    ]
    _drop_static_short_term(msgs)
    assert msgs[0]["content"] == "keep"
    assert "synap_short_term_context" in msgs[1]["content"]


def test_the_backstory_helper_tags_are_the_ones_the_hook_removes():
    sdk = _sdk(True)
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=ContextForPromptResponse(available=True, formatted_context="SNAP"))
    backstory = asyncio.run(build_synap_st_backstory(sdk, "conv-1", base_backstory="base"))
    msgs = [{"role": "system", "content": backstory}]
    _drop_static_short_term(msgs)
    assert msgs[0]["content"] == "base"
