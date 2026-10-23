"""What a run reports to Synap.

This package gave an agent two tools and reported nothing about the run
itself: not the reply, not the tools, not the reasoning. Anticipation had
nothing to work with, so every fetch stayed a cold retrieval.

Two rules hold in every test here. The stream is the default: with no
`listen()` running the first report opens it, and only with auto-listen off do
the user and assistant turns go over REST (`sdk.conversation.record_message`)
and everything else become a no-op. And nothing raises: these run inside the
agent's loop, where an exception from telemetry ends the run.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_integrations_common import set_auto_listen
from synap_openai_agents import SynapRunHooks, report_user_turn


def sdk(listening: bool = True, auto_listen: bool = False):
    """A fake SDK. `listening=False` starts with no stream; `listen()` opens it
    when `auto_listen` lets the integration call it."""
    s = MagicMock()
    s.instance = MagicMock()
    s.instance.is_listening = listening

    async def listen():
        s.instance.is_listening = True

    s.instance.listen = AsyncMock(side_effect=listen)
    set_auto_listen(s, auto_listen)
    s.instance.send_message = AsyncMock()
    s.instance.record_tool_call = AsyncMock()
    s.instance.record_tool_result = AsyncMock()
    s.instance.record_thinking = AsyncMock()
    s.conversation = MagicMock()
    s.conversation.record_message = AsyncMock()
    return s


def hooks(s, **kw):
    return SynapRunHooks(s, user_id="u1", customer_id="cus1",
                         conversation_id="c1", **kw)


def tool(name: str = "lookup_order"):
    return SimpleNamespace(name=name)


def reasoning_response(*texts: str):
    """A ModelResponse carrying a reasoning summary, in its documented shape."""
    return SimpleNamespace(output=[
        SimpleNamespace(
            type="reasoning",
            summary=[SimpleNamespace(text=t) for t in texts],
        )
    ])


class TestItRefusesToBeBuiltWrong:
    def test_no_sdk(self):
        with pytest.raises(ValueError):
            SynapRunHooks(None, user_id="u1")

    def test_no_user_id(self):
        with pytest.raises(ValueError):
            SynapRunHooks(sdk(), user_id="")


class TestTheReplyIsReported:
    @pytest.mark.asyncio
    async def test_on_agent_end_sends_the_assistant_turn(self):
        """The event anticipation acts on. Without it, nothing is predicted."""
        s = sdk()
        await hooks(s).on_agent_end(MagicMock(), MagicMock(), "Your order shipped.")
        kwargs = s.instance.send_message.await_args.kwargs
        assert kwargs["event_type"] == "assistant_message"
        assert kwargs["content"] == "Your order shipped."
        assert kwargs["conversation_id"] == "c1"

    @pytest.mark.asyncio
    async def test_a_structured_output_is_read_for_its_text(self):
        s = sdk()
        await hooks(s).on_agent_end(
            MagicMock(), MagicMock(), SimpleNamespace(final_output="shipped"),
        )
        assert s.instance.send_message.await_args.kwargs["content"] == "shipped"

    @pytest.mark.asyncio
    async def test_an_empty_output_sends_nothing(self):
        s = sdk()
        await hooks(s).on_agent_end(MagicMock(), MagicMock(), "")
        s.instance.send_message.assert_not_awaited()


class TestToolsAreReported:
    @pytest.mark.asyncio
    async def test_a_call_and_its_result_share_an_id(self):
        """The two hooks arrive separately with nothing tying them together,
        so the pairing is kept by the hooks themselves."""
        s = sdk()
        h = hooks(s)
        await h.on_tool_start(MagicMock(), MagicMock(), tool())
        await h.on_tool_end(MagicMock(), MagicMock(), tool(), {"status": "shipped"})

        call_id = s.instance.record_tool_call.await_args.kwargs["tool_call_id"]
        result_id = s.instance.record_tool_result.await_args.kwargs["tool_call_id"]
        assert call_id and call_id == result_id

    @pytest.mark.asyncio
    async def test_the_result_travels(self):
        s = sdk()
        h = hooks(s)
        await h.on_tool_start(MagicMock(), MagicMock(), tool())
        await h.on_tool_end(MagicMock(), MagicMock(), tool(), {"status": "shipped"})
        assert s.instance.record_tool_result.await_args.args[0] == {"status": "shipped"}

    @pytest.mark.asyncio
    async def test_results_can_be_turned_off(self):
        """A tool result is usually the caller's own customer data."""
        s = sdk()
        h = hooks(s, report_tool_results=False)
        await h.on_tool_start(MagicMock(), MagicMock(), tool())
        await h.on_tool_end(MagicMock(), MagicMock(), tool(), {"card": "4111"})
        s.instance.record_tool_call.assert_awaited()
        s.instance.record_tool_result.assert_not_awaited()


class TestReasoningIsReportedWhenThereIsAny:
    @pytest.mark.asyncio
    async def test_a_summary_is_sent(self):
        s = sdk()
        await hooks(s).on_llm_end(
            MagicMock(), MagicMock(), reasoning_response("Check the order first."),
        )
        assert s.instance.record_thinking.await_args.args[0] == "Check the order first."

    @pytest.mark.asyncio
    async def test_no_reasoning_sends_nothing(self):
        """Most providers return none. That is an ordinary turn, not a gap."""
        s = sdk()
        await hooks(s).on_llm_end(
            MagicMock(), MagicMock(), SimpleNamespace(output=[]),
        )
        s.instance.record_thinking.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_response_of_an_unexpected_shape_sends_nothing(self):
        """Guessing at a field and mislabelling it as the customer's reasoning
        is worse than reporting nothing."""
        s = sdk()
        await hooks(s).on_llm_end(MagicMock(), MagicMock(), SimpleNamespace())
        s.instance.record_thinking.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_it_can_be_turned_off(self):
        s = sdk()
        h = hooks(s, report_model_reasoning=False)
        await h.on_llm_end(MagicMock(), MagicMock(), reasoning_response("x"))
        s.instance.record_thinking.assert_not_awaited()


class TestTheUserTurn:
    @pytest.mark.asyncio
    async def test_the_helper_reports_it(self):
        """No hook carries the user's input: on_agent_start gets the agent,
        and the run's input is the caller's, not the SDK's."""
        s = sdk()
        assert await report_user_turn(
            s, "where is my order", conversation_id="c1", user_id="u1",
        ) is True
        assert s.instance.send_message.await_args.kwargs["event_type"] == "user_message"


class TestWithoutAStream:
    @pytest.mark.asyncio
    async def test_tool_and_reasoning_hooks_are_a_no_op(self):
        s = sdk(listening=False)
        h = hooks(s)
        await h.on_tool_start(MagicMock(), MagicMock(), tool())
        await h.on_tool_end(MagicMock(), MagicMock(), tool(), "x")
        await h.on_llm_end(MagicMock(), MagicMock(), reasoning_response("x"))
        s.instance.send_message.assert_not_awaited()
        s.instance.record_tool_call.assert_not_awaited()
        s.instance.record_tool_result.assert_not_awaited()
        s.instance.record_thinking.assert_not_awaited()
        s.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_user_and_assistant_turns_are_written_over_rest_once_each(self):
        """A turn is the conversation itself. Without it nothing is kept for
        compaction or history search, so it must not depend on listen()."""
        s = sdk(listening=False)
        assert await report_user_turn(
            s, "where is my order", conversation_id="c1", user_id="u1",
        ) is True
        await hooks(s).on_agent_end(MagicMock(), MagicMock(), "Your order shipped.")
        calls = s.conversation.record_message.await_args_list
        assert [(c.kwargs["role"], c.kwargs["content"]) for c in calls] == [
            ("user", "where is my order"), ("assistant", "Your order shipped."),
        ]
        assert all(c.kwargs["conversation_id"] == "c1" for c in calls)
        assert calls[1].kwargs["customer_id"] == "cus1"
        s.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_conversation_id_writes_nothing(self):
        s = sdk(listening=False)
        h = SynapRunHooks(s, user_id="u1")
        await h.on_agent_end(MagicMock(), MagicMock(), "done")
        s.conversation.record_message.assert_not_awaited()


class TestWithAStream:
    @pytest.mark.asyncio
    async def test_turns_go_on_the_stream_and_not_over_rest(self):
        s = sdk(listening=True)
        await report_user_turn(s, "hi", conversation_id="c1", user_id="u1")
        await hooks(s).on_agent_end(MagicMock(), MagicMock(), "hello")
        events = [c.kwargs["event_type"] for c in s.instance.send_message.await_args_list]
        assert events == ["user_message", "assistant_message"]
        s.conversation.record_message.assert_not_awaited()


class TestAFailingReportNeverEndsTheRun:
    @pytest.mark.asyncio
    async def test_a_raising_stream_is_swallowed(self):
        s = sdk()
        s.instance.send_message.side_effect = RuntimeError("stream is gone")
        await hooks(s).on_agent_end(MagicMock(), MagicMock(), "done")

    @pytest.mark.asyncio
    async def test_a_tool_with_no_name_does_not_raise(self):
        s = sdk()
        await hooks(s).on_tool_start(MagicMock(), MagicMock(), SimpleNamespace())
