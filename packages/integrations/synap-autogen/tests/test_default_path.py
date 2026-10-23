"""The defaults, driven through a real ``AssistantAgent``.

- The stream opens by itself: the first message the context sees calls
  ``listen()`` and every later event rides on it.
- Synap's compaction replaces the framework's earlier turns, by default, and the
  framework's messages are untouched when Synap has none.
- The audit's findings: a tool-call reply is not lost, a team does not write the
  same text twice, and the short-term context no longer sends turns twice.
"""

from __future__ import annotations

import pytest
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import AssistantMessage, SystemMessage, UserMessage
from synap_integrations_common import set_auto_listen

from synap_autogen import SynapShortTermChatContext, SynapStreamingChatContext
from tests._fakes import RecordingClient, lookup, sdk_with_closed_stream, text, tool_call

AssistantAgent = pytest.importorskip("autogen_agentchat.agents").AssistantAgent

COMPACTION = "## Conversation so far (compacted)\nThe user is planning a trip to Lisbon."


def _ctx(sdk, conv="conv-1", **kw):
    return SynapStreamingChatContext(sdk, conversation_id=conv, user_id="u1", **kw)


def _events(sdk):
    return [(c.kwargs["role"], c.kwargs["content"]) for c in sdk.instance.send_message.await_args_list]


class TestTheStreamOpensByDefault:
    @pytest.mark.asyncio
    async def test_the_first_run_opens_listen_and_everything_rides_on_it(self):
        sdk = sdk_with_closed_stream()
        assert sdk.instance.is_listening is False
        agent = AssistantAgent(
            "a", model_client=RecordingClient([tool_call("lookup", {"pnr": "X"}), text("done")]),
            tools=[lookup], model_context=_ctx(sdk), reflect_on_tool_use=True,
        )
        await agent.run(task="where is my order")
        sdk.instance.listen.assert_awaited_once()
        assert _events(sdk) == [("user", "where is my order"), ("assistant", "done")]
        sdk.instance.record_tool_call.assert_awaited_once()
        sdk.instance.record_tool_result.assert_awaited_once()
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_tool_event_as_the_first_event_still_opens_it(self):
        sdk = sdk_with_closed_stream()
        ctx = _ctx(sdk)
        await ctx.add_message(AssistantMessage(content=[__import__("autogen_core").FunctionCall(
            id="c", name="t", arguments="{}")], source="a"))
        sdk.instance.listen.assert_awaited_once()
        sdk.instance.record_tool_call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_opting_out_keeps_turns_on_rest(self):
        sdk = sdk_with_closed_stream()
        set_auto_listen(sdk, False)
        await _ctx(sdk).add_message(UserMessage(content="hi", source="user"))
        sdk.instance.listen.assert_not_awaited()
        sdk.conversation.record_message.assert_awaited_once()


class TestCompactionReplacesTheFrameworksTurns:
    @pytest.mark.asyncio
    async def test_the_model_sees_the_compaction_and_not_the_replaced_turns(self):
        sdk = sdk_with_closed_stream(
            compaction=COMPACTION,
            recent=[("user", "what about hotels"), ("assistant", "I found three")],
        )
        client = RecordingClient([text("ok 1"), text("ok 2"), text("ok 3")])
        agent = AssistantAgent("a", model_client=client, model_context=_ctx(sdk), system_message="SYS")
        await agent.run(task="OLD-Q1")
        await agent.run(task="what about hotels")
        await agent.run(task="and flights")
        last = client.seen[-1]
        flat = [(type(m).__name__, m.content) for m in last]
        assert flat[0] == ("SystemMessage", "SYS")
        assert flat[1][0] == "SystemMessage" and "Lisbon" in flat[1][1]
        assert flat[2:] == [("UserMessage", "what about hotels"),
                            ("AssistantMessage", "I found three"),
                            ("UserMessage", "and flights")]
        assert "OLD-Q1" not in str(flat)

    @pytest.mark.asyncio
    async def test_without_a_compaction_the_framework_messages_are_untouched(self):
        sdk = sdk_with_closed_stream()
        client = RecordingClient([text("ok 1"), text("ok 2")])
        agent = AssistantAgent("a", model_client=client, model_context=_ctx(sdk))
        await agent.run(task="first")
        await agent.run(task="second")
        assert [m.content for m in client.seen[-1] if not isinstance(m, SystemMessage)] == [
            "first", "ok 1", "second"]

    @pytest.mark.asyncio
    async def test_a_synap_error_leaves_the_framework_messages_untouched(self):
        sdk = sdk_with_closed_stream()
        sdk.conversation.context.get_context_for_prompt.side_effect = RuntimeError("down")
        client = RecordingClient([text("ok 1"), text("ok 2")])
        agent = AssistantAgent("a", model_client=client, model_context=_ctx(sdk))
        await agent.run(task="first")
        await agent.run(task="second")
        assert [m.content for m in client.seen[-1] if not isinstance(m, SystemMessage)] == [
            "first", "ok 1", "second"]

    @pytest.mark.asyncio
    async def test_the_frameworks_own_limit_stays_the_fallback(self):
        sdk = sdk_with_closed_stream()
        client = RecordingClient([text("1"), text("2"), text("3")])
        ctx = _ctx(sdk, inner=BufferedChatCompletionContext(buffer_size=2))
        agent = AssistantAgent("a", model_client=client, model_context=ctx)
        for task in ("q1", "q2", "q3"):
            await agent.run(task=task)
        assert len([m for m in client.seen[-1] if not isinstance(m, SystemMessage)]) == 2

    @pytest.mark.asyncio
    async def test_a_short_term_inner_splices_once(self):
        sdk = sdk_with_closed_stream(compaction=COMPACTION)
        inner = SynapShortTermChatContext(sdk, "conv-1")
        client = RecordingClient([text("ok")])
        agent = AssistantAgent("a", model_client=client, model_context=_ctx(sdk, inner=inner))
        await agent.run(task="hello")
        assert sdk.conversation.context.get_context_for_prompt.await_count == 1
        assert sum("Lisbon" in str(m.content) for m in client.seen[0]) == 1


class TestAuditFindings:
    @pytest.mark.asyncio
    async def test_a_tool_call_reply_is_not_lost_on_the_default_path(self):
        """reflect_on_tool_use=False: the summary never reaches the context, and
        the assistant turn used to be lost unless the caller used report_reply."""
        sdk = sdk_with_closed_stream()
        ctx = _ctx(sdk)
        agent = AssistantAgent(
            "a", model_client=RecordingClient([tool_call("lookup", {"pnr": "X"}), text("again")]),
            tools=[lookup], model_context=ctx, reflect_on_tool_use=False,
        )
        await agent.run(task="where is my order")
        assert _events(sdk) == [("user", "where is my order")]
        await agent.run(task="and now?")  # the next turn closes the last one
        assert _events(sdk) == [("user", "where is my order"), ("assistant", "shipped"),
                                ("user", "and now?"), ("assistant", "again")]

    @pytest.mark.asyncio
    async def test_save_state_closes_the_turn_too(self):
        sdk = sdk_with_closed_stream()
        ctx = _ctx(sdk)
        agent = AssistantAgent(
            "a", model_client=RecordingClient([tool_call("lookup", {"pnr": "X"})]),
            tools=[lookup], model_context=ctx,
        )
        await agent.run(task="where is my order")
        await agent.save_state()
        assert _events(sdk)[-1] == ("assistant", "shipped")

    @pytest.mark.asyncio
    async def test_report_reply_sends_the_exact_reply_and_the_stand_in_is_not_added(self):
        sdk = sdk_with_closed_stream()
        ctx = _ctx(sdk)
        agent = AssistantAgent(
            "a", model_client=RecordingClient([tool_call("lookup", {"pnr": "X"}), text("next")]),
            tools=[lookup], model_context=ctx, tool_call_summary_format="Order status: {result}",
        )
        result = await agent.run(task="where is my order")
        assert await ctx.report_reply(result) is True
        await agent.run(task="next")
        assistants = [t for r, t in _events(sdk) if r == "assistant"]
        assert assistants == ["Order status: shipped", "next"]

    @pytest.mark.asyncio
    async def test_no_stand_in_when_tool_results_are_not_reported(self):
        sdk = sdk_with_closed_stream()
        ctx = _ctx(sdk, report_tool_results=False)
        agent = AssistantAgent(
            "a", model_client=RecordingClient([tool_call("lookup", {"pnr": "X"}), text("next")]),
            tools=[lookup], model_context=ctx,
        )
        await agent.run(task="where is my order")
        await agent.run(task="next")
        assert [t for r, t in _events(sdk) if r == "assistant"] == ["next"]

    @pytest.mark.asyncio
    async def test_a_reflecting_run_writes_no_stand_in(self):
        sdk = sdk_with_closed_stream()
        agent = AssistantAgent(
            "a", model_client=RecordingClient([tool_call("lookup", {"pnr": "X"}), text("final"), text("2")]),
            tools=[lookup], model_context=_ctx(sdk), reflect_on_tool_use=True,
        )
        await agent.run(task="q")
        await agent.run(task="q2")
        assert [t for r, t in _events(sdk) if r == "assistant"] == ["final", "2"]

    @pytest.mark.asyncio
    async def test_two_agents_on_one_conversation_write_each_text_once(self):
        """In a team each agent receives the other's reply, and the task, as a UserMessage."""
        sdk = sdk_with_closed_stream()
        a, b = _ctx(sdk), _ctx(sdk)
        task = UserMessage(content="plan a trip", source="user")
        await a.add_message(task)
        await b.add_message(task)
        await a.add_message(AssistantMessage(content="lisbon?", source="planner"))
        await b.add_message(UserMessage(content="lisbon?", source="planner"))
        await b.add_message(AssistantMessage(content="book it", source="booker"))
        await a.add_message(UserMessage(content="book it", source="booker"))
        assert _events(sdk) == [("user", "plan a trip"), ("assistant", "lisbon?"),
                                ("assistant", "book it")]

    @pytest.mark.asyncio
    async def test_the_same_words_twice_from_one_user_are_both_written(self):
        sdk = sdk_with_closed_stream()
        ctx = _ctx(sdk)
        for _ in range(2):
            await ctx.add_message(UserMessage(content="yes", source="user"))
        assert _events(sdk) == [("user", "yes"), ("user", "yes")]

    @pytest.mark.asyncio
    async def test_a_new_context_on_the_same_conversation_is_not_a_copy_after_the_window(self, monkeypatch):
        from synap_autogen import stream

        sdk = sdk_with_closed_stream()
        await _ctx(sdk).add_message(UserMessage(content="yes", source="user"))
        monkeypatch.setattr(stream, "_WINDOW_S", -1.0)
        await _ctx(sdk).add_message(UserMessage(content="yes", source="user"))
        assert _events(sdk) == [("user", "yes"), ("user", "yes")]

    @pytest.mark.asyncio
    async def test_short_term_context_no_longer_sends_turns_the_framework_already_sends(self):
        """It used to prepend formatted_context (which carries the recent turns)
        to the full inner history."""
        sdk = sdk_with_closed_stream(compaction=COMPACTION, recent=[("user", "RECENT-TURN")])
        client = RecordingClient([text("ok")])
        ctx = SynapShortTermChatContext(sdk, "conv-1")
        await ctx.add_message(UserMessage(content="RECENT-TURN", source="user"))
        await ctx.add_message(AssistantMessage(content="reply", source="a"))
        await ctx.add_message(UserMessage(content="now", source="user"))
        agent = AssistantAgent("a", model_client=client, model_context=ctx)
        await agent.run(task="now")  # adds a second "now"; the earlier turns stay unsent
        sent = " ".join(str(m.content) for m in client.seen[0])
        assert sent.count("RECENT-TURN") == 1
