"""The defaults, proved on a real MAF agent with a recording chat client.

Three things were wrong. The stream was used only if the caller had opened it.
The agent's own history provider decided what the model saw even after Synap had
compacted the conversation. And wiring a Synap history provider next to
`SynapContextProvider` wrote every turn twice, while a `system` message made
`SynapHistoryProvider.save_messages` raise out of `after_run` and skip the rest.
Every run here is `Agent.run` with the documented providers and no `listen()`.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_framework import (
    Agent,
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    Content,
    FunctionInvocationLayer,
    InMemoryHistoryProvider,
    Message,
)

from synap_integrations_common import SynapIntegrationError, set_auto_listen
from synap_microsoft_agent import (
    SynapContextProvider,
    SynapHistoryProvider,
    SynapShortTermContextProvider,
)
from tests._harness import requires_harness

BLOCK = "## Conversation so far (compacted 2026-10-22)\nThe user is planning a trip to Lisbon."


class RecordingClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):
    """Answers "ok" and keeps the messages and options of every model call."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls: List[List[tuple]] = []
        self.options: List[Any] = []

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.calls.append([(str(getattr(m.role, "value", m.role)), m.text) for m in messages])
        self.options.append(options)

        async def _go():
            return ChatResponse(
                messages=[Message(role="assistant", contents=[Content.from_text("ok")])],
                response_id="r",
            )

        return _go()


def make_sdk(*, compaction: bool = False, listening: bool = False):
    s = MagicMock()
    s.instance = MagicMock()
    s.instance.is_listening = listening

    async def listen():
        s.instance.is_listening = True

    s.instance.listen = AsyncMock(side_effect=listen)
    for name in ("send_message", "record_tool_call", "record_tool_result", "record_thinking"):
        setattr(s.instance, name, AsyncMock())
    s.conversation = MagicMock()
    s.conversation.record_message = AsyncMock()
    s.fetch = AsyncMock(return_value=SimpleNamespace(formatted_context=""))
    set_compaction(s, compaction)
    return s


def set_compaction(s, compaction: bool) -> None:
    recent = [SimpleNamespace(role="user", content="and hotels?"),
              SimpleNamespace(role="assistant", content="Three near the river.")]
    s.conversation.context.get_context_for_prompt = AsyncMock(return_value=SimpleNamespace(
        available=True, compaction_id="cmp_1" if compaction else None,
        context_block=BLOCK if compaction else None, history_block=None,
        recent_messages=recent if compaction else [], formatted_context=BLOCK,
    ))


def sent(s):
    return [(c.kwargs["role"], c.kwargs["content"]) for c in s.instance.send_message.await_args_list]


def agent_with(client, *providers):
    return Agent(client, instructions="Be brief.", context_providers=list(providers))


def synap(s, **kw):
    return SynapContextProvider(s, user_id="u1", conversation_id="c1", **kw)


class TestTheStreamIsTheDefault:
    @pytest.mark.asyncio
    async def test_a_run_with_no_listen_opens_it_and_sends_both_turns_on_it(self):
        s = make_sdk()
        agent = agent_with(RecordingClient(), synap(s), InMemoryHistoryProvider())
        await agent.run("book the first one", session=agent.create_session())
        s.instance.listen.assert_awaited_once()
        assert sent(s) == [("user", "book the first one"), ("assistant", "ok")]
        s.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_with_auto_listen_off_both_turns_go_over_rest_once(self):
        s = make_sdk()
        set_auto_listen(s, False)
        agent = agent_with(RecordingClient(), synap(s), InMemoryHistoryProvider())
        await agent.run("book it", session=agent.create_session())
        calls = s.conversation.record_message.await_args_list
        assert [(c.kwargs["role"], c.kwargs["content"]) for c in calls] == [
            ("user", "book it"), ("assistant", "ok")]
        s.instance.listen.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_history_provider_opens_it_too(self):
        s = make_sdk()
        agent = agent_with(RecordingClient(), SynapHistoryProvider(s, user_id="u1", conversation_id="c1"))
        await agent.run("hi", session=agent.create_session())
        s.instance.listen.assert_awaited_once()
        assert sent(s) == [("user", "hi"), ("assistant", "ok")]


class TestSynapsCompactionReplacesTheHistoryProvidersTurns:
    async def _three_turns(self, providers_order: str = "synap_first"):
        s = make_sdk()
        client = RecordingClient()
        history = InMemoryHistoryProvider()
        providers = [synap(s), history] if providers_order == "synap_first" else [history, synap(s)]
        agent = agent_with(client, *providers)
        session = agent.create_session()
        await agent.run("flights to Lisbon?", session=session)
        await agent.run("and hotels?", session=session)
        set_compaction(s, True)
        await agent.run("book the first one", session=session)
        return client

    @pytest.mark.asyncio
    @pytest.mark.parametrize("order", ["synap_first", "history_first"])
    async def test_the_model_sees_the_compaction_and_not_the_earlier_turns(self, order):
        """The README lists the Synap provider first, the history provider second;
        either order must work, which is why this is a chat middleware."""
        client = await self._three_turns(order)
        before, after = client.calls[1], client.calls[2]
        assert ("user", "flights to Lisbon?") in before  # no compaction yet: MAF's own history
        texts = [t for _, t in after]
        assert "flights to Lisbon?" not in texts
        assert any("planning a trip to Lisbon" in t for t in texts)
        assert [t for r, t in after if r != "system"] == [
            "and hotels?", "Three near the river.", "book the first one"]

    @pytest.mark.asyncio
    async def test_messages_are_untouched_when_synap_has_no_compaction(self):
        s = make_sdk()
        client = RecordingClient()
        agent = agent_with(client, synap(s), InMemoryHistoryProvider())
        session = agent.create_session()
        await agent.run("flights to Lisbon?", session=session)
        await agent.run("and hotels?", session=session)
        assert [t for _, t in client.calls[1]] == ["flights to Lisbon?", "ok", "and hotels?"]

    @pytest.mark.asyncio
    async def test_messages_are_untouched_when_synap_errors(self):
        s = make_sdk(compaction=True)
        s.conversation.context.get_context_for_prompt.side_effect = RuntimeError("down")
        client = RecordingClient()
        agent = agent_with(client, synap(s), InMemoryHistoryProvider())
        session = agent.create_session()
        await agent.run("flights to Lisbon?", session=session)
        await agent.run("and hotels?", session=session)
        assert [t for _, t in client.calls[1]] == ["flights to Lisbon?", "ok", "and hotels?"]

    @pytest.mark.asyncio
    async def test_a_tool_exchange_in_the_current_turn_stays_paired(self):
        s = make_sdk(compaction=True)

        def lookup(pnr: str) -> str:
            """Look a booking up."""
            return "confirmed"

        class ToolThenAnswer(RecordingClient):
            def _inner_get_response(self, *, messages, stream, options, **kwargs):
                self.calls.append([(str(getattr(m.role, "value", m.role)), m.text,
                                    [c.type for c in m.contents]) for m in messages])

                async def _go():
                    if len(self.calls) == 1:
                        msg = Message(role="assistant", contents=[
                            Content.from_function_call("k1", "lookup", arguments={"pnr": "X"})])
                    else:
                        msg = Message(role="assistant", contents=[Content.from_text("done")])
                    return ChatResponse(messages=[msg], response_id="r")

                return _go()

        client = ToolThenAnswer()
        agent = Agent(client, instructions="x", tools=[lookup], context_providers=[synap(s)])
        await agent.run("where is X", session=agent.create_session())
        second = client.calls[1]
        kinds = [c for _, _, c in second]
        assert ["function_call"] in kinds and ["function_result"] in kinds
        assert kinds.index(["function_call"]) + 1 == kinds.index(["function_result"])
        assert any("planning a trip to Lisbon" in t for _, t, _ in second)

    @pytest.mark.asyncio
    async def test_the_compaction_is_not_sent_twice_next_to_the_short_term_provider(self):
        s = make_sdk(compaction=True)
        s.conversation.context.get_context_for_prompt.return_value.available = True
        client = RecordingClient()
        agent = agent_with(client, synap(s), SynapShortTermContextProvider(s, "c1"),
                           InMemoryHistoryProvider())
        await agent.run("book it", session=agent.create_session())
        assert client.options[0]["instructions"].count("planning a trip to Lisbon") == 1
        assert not any("planning a trip" in t for _, t in client.calls[0])


class TestASynapHistoryProviderNextToTheContextProvider:
    @pytest.mark.asyncio
    async def test_each_turn_is_written_once(self):
        """Wiring both used to write every turn twice: once from each `after_run`."""
        s = make_sdk()
        agent = agent_with(
            RecordingClient(), synap(s),
            SynapHistoryProvider(s, user_id="u1", conversation_id="c1"))
        await agent.run("book it", session=agent.create_session())
        assert sent(s) == [("user", "book it"), ("assistant", "ok")]
        s.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_over_rest_too(self):
        s = make_sdk()
        set_auto_listen(s, False)
        agent = agent_with(
            RecordingClient(), synap(s),
            SynapHistoryProvider(s, user_id="u1", conversation_id="c1"))
        await agent.run("book it", session=agent.create_session())
        calls = s.conversation.record_message.await_args_list
        assert [(c.kwargs["role"], c.kwargs["content"]) for c in calls] == [
            ("user", "book it"), ("assistant", "ok")]

    @pytest.mark.asyncio
    async def test_the_history_comes_from_synap_without_a_second_fetch_per_call(self):
        s = make_sdk(compaction=True)
        client = RecordingClient()
        agent = agent_with(
            client, synap(s), SynapHistoryProvider(s, user_id="u1", conversation_id="c1"))
        await agent.run("book the first one", session=agent.create_session())
        texts = [t for _, t in client.calls[0]]
        assert sum("planning a trip to Lisbon" in t for t in texts) == 1
        # The provider's get_messages is the one fetch; the middleware stays off.
        assert s.conversation.context.get_context_for_prompt.await_count == 1


class TestSystemMessagesAreNotTurns:
    @pytest.mark.asyncio
    async def test_a_system_message_does_not_stop_the_turns_after_it(self):
        s = make_sdk()
        set_auto_listen(s, False)
        provider = SynapHistoryProvider(s, user_id="u1", conversation_id="c1")
        await provider.save_messages("sess", [
            Message(role="system", contents=["be helpful"]),
            Message(role="user", contents=["hi"]),
            Message(role="assistant", contents=["hello"]),
        ])
        calls = s.conversation.record_message.await_args_list
        assert [(c.kwargs["role"], c.kwargs["content"]) for c in calls] == [
            ("user", "hi"), ("assistant", "hello")]

    @pytest.mark.asyncio
    async def test_a_failed_write_does_not_skip_the_rest_but_still_raises(self):
        s = make_sdk()
        set_auto_listen(s, False)
        s.conversation.record_message.side_effect = [RuntimeError("boom"), None]
        provider = SynapHistoryProvider(s, user_id="u1", conversation_id="c1")
        with pytest.raises(SynapIntegrationError):
            await provider.save_messages("sess", [
                Message(role="user", contents=["one"]),
                Message(role="assistant", contents=["two"]),
            ])
        assert s.conversation.record_message.await_count == 2

    @pytest.mark.asyncio
    async def test_the_context_provider_logs_no_error_for_a_system_input(self, caplog):
        s = make_sdk()
        set_auto_listen(s, False)
        agent = agent_with(RecordingClient(), synap(s))
        with caplog.at_level(logging.ERROR):
            await agent.run(
                [Message(role="system", contents=["policy"]),
                 Message(role="user", contents=["hi"])],
                session=agent.create_session())
        assert [c.kwargs["role"] for c in s.conversation.record_message.await_args_list] == [
            "user", "assistant"]
        assert not [r for r in caplog.records if "record_message failed" in r.getMessage()]


@requires_harness
class TestTheHarnessKeepsTheCompaction:
    def _provider(self, s, **kw):
        from synap_microsoft_agent import create_synap_harness_memory

        return create_synap_harness_memory(s, user_id="u1", **kw)

    @pytest.mark.asyncio
    async def test_get_messages_leads_with_the_compaction(self):
        """It returned only `recent_messages`, so compacted turns vanished."""
        s = make_sdk(compaction=True)
        msgs = await self._provider(s).get_messages("sess", state={})
        assert [(str(getattr(m.role, "value", m.role)), m.text) for m in msgs] == [
            ("system", BLOCK), ("user", "and hotels?"), ("assistant", "Three near the river.")]

    @pytest.mark.asyncio
    async def test_the_model_is_given_the_compaction_when_it_is_given_a_transcript(self):
        from agent_framework import AgentSession, SessionContext

        s = make_sdk(compaction=True)
        provider = self._provider(s, recent_turns=4)
        session = AgentSession(session_id="sess")
        context = SessionContext(session_id="sess", input_messages=[
            Message(role="user", contents=["book it"])])
        await provider.before_run(agent=MagicMock(), session=session, context=context,
                                  state=session.state.setdefault(provider.source_id, {}))
        assert any("planning a trip to Lisbon" in i for i in context.instructions)
        roles_texts = [t for t in (m.text for m in context.get_messages())]
        assert "and hotels?" in roles_texts

    @pytest.mark.asyncio
    async def test_with_no_transcript_requested_none_is_added(self):
        from agent_framework import AgentSession, SessionContext

        s = make_sdk(compaction=True)
        provider = self._provider(s)  # recent_turns defaults to 0
        session = AgentSession(session_id="sess")
        context = SessionContext(session_id="sess", input_messages=[
            Message(role="user", contents=["book it"])])
        await provider.before_run(agent=MagicMock(), session=session, context=context,
                                  state=session.state.setdefault(provider.source_id, {}))
        assert not any("planning a trip" in i for i in context.instructions)
