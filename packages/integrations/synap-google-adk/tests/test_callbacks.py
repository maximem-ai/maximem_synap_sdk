"""What an ADK run reports to Synap.

The package gave an agent tools and reported nothing about the run itself, so
anticipation had nothing to work with and every fetch stayed a cold retrieval.

Two rules hold in every test. The stream is the default: with no `listen()`
running the first report opens it, and only with auto-listen off do the reply
and the user turn go over REST (`sdk.conversation.record_message`) and the tool
callbacks become a no-op. And every callback returns None, always: ADK reads a
returned value as an override of the model or tool result, so telemetry that
returns anything else silently changes what the agent does.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_integrations_common import set_auto_listen
from synap_google_adk import create_synap_callbacks, report_user_turn


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
    s.conversation = MagicMock()
    s.conversation.record_message = AsyncMock()
    return s


def callbacks(s, **kw):
    return create_synap_callbacks(
        s, user_id="u1", customer_id="cus1", conversation_id="c1", **kw,
    )


def tool(name: str = "lookup_order"):
    return SimpleNamespace(name=name)


def llm_response(text: str):
    return SimpleNamespace(
        content=SimpleNamespace(parts=[SimpleNamespace(text=text)])
    )


class TestItRefusesToBeBuiltWrong:
    def test_no_sdk(self):
        with pytest.raises(ValueError):
            create_synap_callbacks(None, user_id="u1")

    def test_no_user_id(self):
        with pytest.raises(ValueError):
            create_synap_callbacks(sdk(), user_id="")


class TestItReturnsTheCallbacksAdkExpects:
    def test_the_four_keys(self):
        assert set(callbacks(sdk())) == {
            "before_model_callback", "before_tool_callback", "after_tool_callback",
            "after_model_callback",
        }


class TestEveryCallbackReturnsNone:
    """ADK treats a returned value as an override. Telemetry must not be one."""

    @pytest.mark.asyncio
    async def test_before_tool(self):
        cb = callbacks(sdk())
        assert await cb["before_tool_callback"](tool(), {}, MagicMock()) is None

    @pytest.mark.asyncio
    async def test_after_tool(self):
        cb = callbacks(sdk())
        assert await cb["after_tool_callback"](tool(), {}, MagicMock(), {"a": 1}) is None

    @pytest.mark.asyncio
    async def test_after_model(self):
        cb = callbacks(sdk())
        assert await cb["after_model_callback"](MagicMock(), llm_response("hi")) is None


class TestWhatReachesTheStream:
    @pytest.mark.asyncio
    async def test_the_reply(self):
        s = sdk()
        await callbacks(s)["after_model_callback"](
            MagicMock(), llm_response("Your order shipped."),
        )
        kwargs = s.instance.send_message.await_args.kwargs
        assert kwargs["event_type"] == "assistant_message"
        assert kwargs["content"] == "Your order shipped."

    @pytest.mark.asyncio
    async def test_an_empty_reply_sends_nothing(self):
        s = sdk()
        await callbacks(s)["after_model_callback"](MagicMock(), llm_response(""))
        s.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_response_of_an_unexpected_shape_sends_nothing(self):
        s = sdk()
        await callbacks(s)["after_model_callback"](MagicMock(), SimpleNamespace())
        s.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_tool_call_and_its_result_share_an_id(self):
        s = sdk()
        cb = callbacks(s)
        await cb["before_tool_callback"](tool(), {"order_id": "A-1"}, MagicMock())
        await cb["after_tool_callback"](tool(), {}, MagicMock(), {"status": "shipped"})
        call_id = s.instance.record_tool_call.await_args.kwargs["tool_call_id"]
        result_id = s.instance.record_tool_result.await_args.kwargs["tool_call_id"]
        assert call_id and call_id == result_id

    @pytest.mark.asyncio
    async def test_the_tool_args_travel(self):
        s = sdk()
        await callbacks(s)["before_tool_callback"](
            tool(), {"order_id": "A-1"}, MagicMock(),
        )
        assert s.instance.record_tool_call.await_args.args[1] == {"order_id": "A-1"}

    @pytest.mark.asyncio
    async def test_results_can_be_turned_off(self):
        s = sdk()
        cb = callbacks(s, report_tool_results=False)
        await cb["before_tool_callback"](tool(), {}, MagicMock())
        await cb["after_tool_callback"](tool(), {}, MagicMock(), {"card": "4111"})
        s.instance.record_tool_call.assert_awaited()
        s.instance.record_tool_result.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_user_turn_has_its_own_helper(self):
        s = sdk()
        assert await report_user_turn(
            s, "where is my order", conversation_id="c1", user_id="u1",
        ) is True
        assert s.instance.send_message.await_args.kwargs["event_type"] == "user_message"


class TestWithoutAStream:
    @pytest.mark.asyncio
    async def test_tool_callbacks_send_nothing(self):
        s = sdk(listening=False)
        cb = callbacks(s)
        await cb["before_tool_callback"](tool(), {}, MagicMock())
        await cb["after_tool_callback"](tool(), {}, MagicMock(), {})
        s.instance.send_message.assert_not_awaited()
        s.instance.record_tool_call.assert_not_awaited()
        s.instance.record_tool_result.assert_not_awaited()
        s.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_user_and_assistant_turns_are_written_over_rest_once_each(self):
        """A turn is the conversation itself; it must not depend on listen()."""
        s = sdk(listening=False)
        assert await report_user_turn(
            s, "where is my order", conversation_id="c1", user_id="u1",
        ) is True
        out = await callbacks(s)["after_model_callback"](MagicMock(), llm_response("shipped"))
        assert out is None
        calls = s.conversation.record_message.await_args_list
        assert [(c.kwargs["role"], c.kwargs["content"]) for c in calls] == [
            ("user", "where is my order"), ("assistant", "shipped"),
        ]
        assert all(c.kwargs["conversation_id"] == "c1" for c in calls)
        s.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_partial_chunks_and_thoughts_are_not_turns(self):
        s = sdk(listening=False)
        cb = callbacks(s)
        partial = SimpleNamespace(partial=True, content=SimpleNamespace(
            parts=[SimpleNamespace(text="ship")]))
        await cb["after_model_callback"](MagicMock(), partial)
        whole = SimpleNamespace(partial=False, content=SimpleNamespace(parts=[
            SimpleNamespace(text="thinking about it", thought=True),
            SimpleNamespace(text="shipped"),
        ]))
        await cb["after_model_callback"](MagicMock(), whole)
        calls = s.conversation.record_message.await_args_list
        assert [c.kwargs["content"] for c in calls] == ["shipped"]


class TestWithAStream:
    @pytest.mark.asyncio
    async def test_turns_go_on_the_stream_and_not_over_rest(self):
        s = sdk(listening=True)
        await report_user_turn(s, "hi", conversation_id="c1", user_id="u1")
        await callbacks(s)["after_model_callback"](MagicMock(), llm_response("hello"))
        events = [c.kwargs["event_type"] for c in s.instance.send_message.await_args_list]
        assert events == ["user_message", "assistant_message"]
        s.conversation.record_message.assert_not_awaited()


class TestAFailingReportNeverBreaksTheRun:
    @pytest.mark.asyncio
    async def test_a_raising_stream_is_swallowed_and_still_returns_none(self):
        s = sdk()
        s.instance.send_message.side_effect = RuntimeError("stream is gone")
        out = await callbacks(s)["after_model_callback"](MagicMock(), llm_response("hi"))
        assert out is None
