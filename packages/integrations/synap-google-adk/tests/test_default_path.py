"""The defaults, proved on the real ADK Runner with a scripted model.

Four things were wrong: the callbacks named their parameters `context`, which
ADK (which calls callbacks by keyword) rejects on versions that do not adapt
them; the user's turn was never written; a model response that called a tool
was stored as an assistant turn; and ADK's own history decided what the model
saw even after Synap had compacted the conversation. A run here uses
`create_synap_callbacks` the documented way, with no `listen()` and no manual
`report_user_turn`.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, AsyncGenerator, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import Field

from synap_integrations_common import set_auto_listen
from synap_google_adk import create_synap_callbacks, report_user_turn, synap_st_instruction

BLOCK = "## Conversation so far (compacted 2026-10-22)\nThe user is planning a trip to Lisbon."


class ScriptedLlm(BaseLlm):
    """Plays the replies it was given, in order, and keeps every request."""

    model: str = "scripted"
    script: List[LlmResponse] = Field(default_factory=list)
    requests: List[Any] = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: Any, stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request.model_copy(deep=True))
        yield self.script.pop(0) if self.script else text_reply("Booked.")


def text_reply(text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def make_sdk(*, listening: bool = False):
    s = MagicMock()
    s.instance = MagicMock()
    s.instance.is_listening = listening

    async def listen():
        s.instance.is_listening = True

    s.instance.listen = AsyncMock(side_effect=listen)
    s.instance.send_message = AsyncMock()
    s.instance.record_tool_call = AsyncMock()
    s.instance.record_tool_result = AsyncMock()
    s.conversation = MagicMock()
    s.conversation.record_message = AsyncMock()
    s.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=SimpleNamespace(context_block=None, history_block=None, recent_messages=[]))
    return s


def compact(s):
    s.conversation.context.get_context_for_prompt.return_value = SimpleNamespace(
        context_block=BLOCK, history_block=None, compaction_id="cmp_1",
        recent_messages=[SimpleNamespace(role="user", content="and hotels?"),
                         SimpleNamespace(role="assistant", content="Three near the river.")])


def sent(s):
    return [(c.kwargs["role"], c.kwargs["content"]) for c in s.instance.send_message.await_args_list]


async def run(agent, session_id: str, text: str) -> None:
    runner = Runner(app_name="t", agent=agent, session_service=SERVICE, auto_create_session=True)
    async for _ in runner.run_async(
        user_id="u1", session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
    ):
        pass


SERVICE = InMemorySessionService()


def make_agent(s, llm, **callback_kw):
    return LlmAgent(
        name="a", model=llm, instruction="Be brief.",
        **create_synap_callbacks(s, user_id="u1", conversation_id="c1", **callback_kw),
    )


def request_texts(request) -> List[str]:
    return ["".join(p.text or "" for p in (c.parts or [])) for c in request.contents]


class TestCallbacksTakeAdksParameterNames:
    def test_the_names_are_the_ones_adk_calls_by(self):
        cb = create_synap_callbacks(make_sdk(), user_id="u1")
        assert list(inspect.signature(cb["before_model_callback"]).parameters) == [
            "callback_context", "llm_request"]
        assert list(inspect.signature(cb["after_model_callback"]).parameters) == [
            "callback_context", "llm_response"]
        assert list(inspect.signature(cb["before_tool_callback"]).parameters) == [
            "tool", "args", "tool_context"]
        assert list(inspect.signature(cb["after_tool_callback"]).parameters) == [
            "tool", "args", "tool_context", "tool_response"]

    @pytest.mark.asyncio
    async def test_they_accept_the_keyword_call_adk_makes(self):
        s = make_sdk()
        cb = create_synap_callbacks(s, user_id="u1", conversation_id="c1")
        ctx = SimpleNamespace(invocation_id="i1", user_content=None)
        assert await cb["before_model_callback"](callback_context=ctx, llm_request=SimpleNamespace(contents=[])) is None
        assert await cb["after_model_callback"](callback_context=ctx, llm_response=SimpleNamespace()) is None
        tool = SimpleNamespace(name="t")
        assert await cb["before_tool_callback"](tool=tool, args={}, tool_context=ctx) is None
        assert await cb["after_tool_callback"](tool=tool, args={}, tool_context=ctx, tool_response={}) is None


class TestTheStreamIsTheDefault:
    @pytest.mark.asyncio
    async def test_a_run_with_no_listen_opens_it_and_sends_both_turns_on_it(self):
        s = make_sdk()
        await run(make_agent(s, ScriptedLlm()), "s-default", "book the first one")
        s.instance.listen.assert_awaited_once()
        assert sent(s) == [("user", "book the first one"), ("assistant", "Booked.")]
        s.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_with_auto_listen_off_the_turns_go_over_rest(self):
        s = make_sdk()
        set_auto_listen(s, False)
        await run(make_agent(s, ScriptedLlm()), "s-rest", "book it")
        s.instance.listen.assert_not_awaited()
        calls = s.conversation.record_message.await_args_list
        assert [(c.kwargs["role"], c.kwargs["content"]) for c in calls] == [
            ("user", "book it"), ("assistant", "Booked.")]

    @pytest.mark.asyncio
    async def test_a_manual_report_user_turn_plus_the_callback_writes_the_turn_once(self):
        s = make_sdk()
        await report_user_turn(s, "book it", conversation_id="c1", user_id="u1")
        await run(make_agent(s, ScriptedLlm()), "s-manual", "book it")
        assert sent(s) == [("user", "book it"), ("assistant", "Booked.")]


class TestOnlyTheFinalReplyIsATurn:
    @pytest.mark.asyncio
    async def test_text_that_comes_with_a_tool_call_is_not_stored(self):
        """"Let me check." then a tool call is the model on its way to the answer."""
        s = make_sdk()

        def lookup(order: str) -> str:
            """Look an order up."""
            return "shipped"

        llm = ScriptedLlm(script=[
            LlmResponse(content=types.Content(role="model", parts=[
                types.Part(text="Let me check."),
                types.Part(function_call=types.FunctionCall(name="lookup", args={"order": "A1"})),
            ])),
            text_reply("It shipped."),
        ])
        agent = LlmAgent(
            name="a", model=llm, instruction="x", tools=[lookup],
            **create_synap_callbacks(s, user_id="u1", conversation_id="c1"))
        await run(agent, "s-tool", "where is A1")
        assert len(llm.requests) == 2
        assert sent(s) == [("user", "where is A1"), ("assistant", "It shipped.")]
        s.instance.record_tool_call.assert_awaited_once()
        s.instance.record_tool_result.assert_awaited_once()


class TestSynapsCompactionReplacesAdksHistory:
    @pytest.mark.asyncio
    async def test_the_model_sees_the_compaction_and_not_the_earlier_turns(self):
        s = make_sdk()
        llm = ScriptedLlm()
        agent = make_agent(s, llm)
        await run(agent, "s-splice", "flights to Lisbon?")
        await run(agent, "s-splice", "and hotels?")
        compact(s)
        await run(agent, "s-splice", "book the first one")

        before, after = llm.requests[1], llm.requests[2]
        assert "flights to Lisbon?" in request_texts(before)  # ADK's own history, untouched
        assert "flights to Lisbon?" not in request_texts(after)
        assert request_texts(after) == ["and hotels?", "Three near the river.", "book the first one"]
        assert "planning a trip to Lisbon" in after.config.system_instruction
        assert "Be brief." in after.config.system_instruction
        assert [c.role for c in after.contents] == ["user", "model", "user"]

    @pytest.mark.asyncio
    async def test_the_contents_are_untouched_when_synap_has_no_compaction(self):
        s = make_sdk()
        llm = ScriptedLlm()
        agent = make_agent(s, llm)
        await run(agent, "s-none", "flights to Lisbon?")
        await run(agent, "s-none", "and hotels?")
        assert request_texts(llm.requests[1]) == ["flights to Lisbon?", "Booked.", "and hotels?"]
        assert COMPACTION not in (llm.requests[1].config.system_instruction or "")

    @pytest.mark.asyncio
    async def test_the_contents_are_untouched_when_synap_errors(self):
        s = make_sdk()
        s.conversation.context.get_context_for_prompt.side_effect = RuntimeError("down")
        llm = ScriptedLlm()
        agent = make_agent(s, llm)
        await run(agent, "s-err", "flights to Lisbon?")
        await run(agent, "s-err", "and hotels?")
        assert request_texts(llm.requests[1]) == ["flights to Lisbon?", "Booked.", "and hotels?"]

    @pytest.mark.asyncio
    async def test_a_tool_exchange_in_the_current_turn_stays_paired(self):
        s = make_sdk()
        compact(s)

        def lookup(order: str) -> str:
            """Look an order up."""
            return "shipped"

        llm = ScriptedLlm(script=[
            LlmResponse(content=types.Content(role="model", parts=[
                types.Part(function_call=types.FunctionCall(name="lookup", args={"order": "A1"}))])),
            text_reply("It shipped."),
        ])
        agent = LlmAgent(
            name="a", model=llm, instruction="x", tools=[lookup],
            **create_synap_callbacks(s, user_id="u1", conversation_id="c1"))
        await run(agent, "s-pair", "where is A1")
        second = llm.requests[1]
        kinds = [("call" if any(p.function_call for p in c.parts)
                  else "result" if any(p.function_response for p in c.parts) else "text")
                 for c in second.contents]
        assert kinds[-2:] == ["call", "result"]
        assert "planning a trip to Lisbon" in second.config.system_instruction

    @pytest.mark.asyncio
    async def test_the_compaction_is_not_sent_twice_next_to_synap_st_instruction(self):
        s = make_sdk()
        compact(s)
        s.conversation.context.get_context_for_prompt.return_value.available = True
        llm = ScriptedLlm()
        agent = LlmAgent(
            name="a", model=llm,
            instruction=synap_st_instruction(s, "c1", instruction="Be brief."),
            **create_synap_callbacks(s, user_id="u1", conversation_id="c1"))
        await run(agent, "s-twice", "book it")
        assert llm.requests[0].config.system_instruction.count("planning a trip to Lisbon") == 1


COMPACTION = "## Conversation so far (compacted"
