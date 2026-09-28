"""What an ADK run reports to Synap.

The package gave an agent tools and reported nothing about the run itself, so
anticipation had nothing to work with and every fetch stayed a cold retrieval.

Two rules hold in every test. Nothing requires a stream: with no `listen()`
running the callbacks are a no-op. And every callback returns None, always —
ADK reads a returned value as an override of the model or tool result, so
telemetry that returns anything else silently changes what the agent does.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_google_adk import create_synap_callbacks, report_user_turn


def sdk(listening: bool = True):
    s = MagicMock()
    s.instance = MagicMock()
    s.instance.is_listening = listening
    s.instance.send_message = AsyncMock()
    s.instance.record_tool_call = AsyncMock()
    s.instance.record_tool_result = AsyncMock()
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
    def test_the_three_keys(self):
        assert set(callbacks(sdk())) == {
            "before_tool_callback", "after_tool_callback", "after_model_callback",
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
    async def test_nothing_is_sent(self):
        s = sdk(listening=False)
        cb = callbacks(s)
        await cb["before_tool_callback"](tool(), {}, MagicMock())
        await cb["after_tool_callback"](tool(), {}, MagicMock(), {})
        await cb["after_model_callback"](MagicMock(), llm_response("hi"))
        s.instance.send_message.assert_not_awaited()
        s.instance.record_tool_call.assert_not_awaited()


class TestAFailingReportNeverBreaksTheRun:
    @pytest.mark.asyncio
    async def test_a_raising_stream_is_swallowed_and_still_returns_none(self):
        s = sdk()
        s.instance.send_message.side_effect = RuntimeError("stream is gone")
        out = await callbacks(s)["after_model_callback"](MagicMock(), llm_response("hi"))
        assert out is None
