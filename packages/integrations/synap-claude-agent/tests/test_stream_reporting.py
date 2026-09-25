"""What these hooks put on the stream, and what they cannot.

The hooks already fetched context and recorded the user's prompt. What they
never did was report the run to anticipation, which is what makes the NEXT
turn's fetch a cache hit instead of a cold retrieval.

Two rules hold everywhere here. Nothing requires a stream: with no `listen()`
running, every report is a no-op and the hooks behave exactly as before. And
nothing raises: these run inside the agent's own loop, where a telemetry
exception would take the run down.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_claude_agent.hooks import create_synap_hooks, report_assistant_turn


def listening_sdk(mock_sdk, listening: bool = True):
    mock_sdk.instance = MagicMock()
    mock_sdk.instance.is_listening = listening
    mock_sdk.instance.send_message = AsyncMock()
    mock_sdk.instance.record_tool_call = AsyncMock()
    mock_sdk.instance.record_tool_result = AsyncMock()
    mock_sdk.instance.end_session = AsyncMock()
    return mock_sdk


def hook(sdk, name: str):
    hooks = create_synap_hooks(sdk, user_id="u1", customer_id="cus1")
    assert name in hooks, f"{name} is not registered"
    return hooks[name][0].hooks[0]


class TestTheHooksAreRegistered:
    def test_all_four_are_there(self, mock_sdk):
        hooks = create_synap_hooks(listening_sdk(mock_sdk), user_id="u1")
        assert set(hooks) == {"UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop"}


class TestWhatReachesTheStream:
    @pytest.mark.asyncio
    async def test_the_user_turn_does(self, mock_sdk):
        sdk = listening_sdk(mock_sdk)
        await hook(sdk, "UserPromptSubmit")(
            {"prompt": "where is my order", "session_id": "c1"}, None, MagicMock(),
        )
        kwargs = sdk.instance.send_message.await_args.kwargs
        assert kwargs["event_type"] == "user_message"
        assert kwargs["content"] == "where is my order"
        assert kwargs["conversation_id"] == "c1"

    @pytest.mark.asyncio
    async def test_a_tool_call_does(self, mock_sdk):
        sdk = listening_sdk(mock_sdk)
        await hook(sdk, "PreToolUse")(
            {"session_id": "c1", "tool_name": "Read",
             "tool_input": {"path": "/tmp/x"}, "tool_use_id": "call_1"},
            None, MagicMock(),
        )
        args, kwargs = sdk.instance.record_tool_call.await_args
        assert args[0] == "Read"
        assert kwargs["tool_call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_a_tool_result_does(self, mock_sdk):
        sdk = listening_sdk(mock_sdk)
        await hook(sdk, "PostToolUse")(
            {"session_id": "c1", "tool_name": "Read",
             "tool_response": {"content": "file body"}, "tool_use_id": "call_1"},
            None, MagicMock(),
        )
        args, kwargs = sdk.instance.record_tool_result.await_args
        assert args[0] == {"content": "file body"}
        assert kwargs["tool_call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_the_end_of_the_run_closes_the_session(self, mock_sdk):
        sdk = listening_sdk(mock_sdk)
        await hook(sdk, "Stop")({"session_id": "c1"}, None, MagicMock())
        sdk.instance.end_session.assert_awaited_once_with("c1")


class TestTheOneEventAHookCannotProduce:
    """`assistant_message` is what anticipation acts on, and no hook input
    carries the final assistant text — StopHookInput has a session id and a
    flag. So it is a one-line call at the end of a run, deliberately."""

    @pytest.mark.asyncio
    async def test_the_helper_reports_it(self, mock_sdk):
        sdk = listening_sdk(mock_sdk)
        sent = await report_assistant_turn(
            sdk, "Your order shipped.", conversation_id="c1",
            user_id="u1", customer_id="cus1",
        )
        assert sent is True
        assert sdk.instance.send_message.await_args.kwargs["event_type"] == "assistant_message"

    @pytest.mark.asyncio
    async def test_it_says_so_when_there_is_no_stream(self, mock_sdk):
        sdk = listening_sdk(mock_sdk, listening=False)
        assert await report_assistant_turn(
            sdk, "hi", conversation_id="c1", user_id="u1",
        ) is False


class TestWithoutAStreamNothingChanges:
    @pytest.mark.asyncio
    async def test_no_events_are_sent(self, mock_sdk):
        sdk = listening_sdk(mock_sdk, listening=False)
        await hook(sdk, "UserPromptSubmit")(
            {"prompt": "hi", "session_id": "c1"}, None, MagicMock(),
        )
        await hook(sdk, "PreToolUse")(
            {"session_id": "c1", "tool_name": "Read"}, None, MagicMock(),
        )
        sdk.instance.send_message.assert_not_awaited()
        sdk.instance.record_tool_call.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_prompt_is_still_recorded_and_context_still_injected(self, mock_sdk):
        """The stream is additive. The behaviour that existed before it stays."""
        sdk = listening_sdk(mock_sdk, listening=False)
        out = await hook(sdk, "UserPromptSubmit")(
            {"prompt": "hi", "session_id": "c1"}, None, MagicMock(),
        )
        sdk.conversation.record_message.assert_awaited()
        assert "hookSpecificOutput" in out


class TestAFailingReportNeverBreaksTheRun:
    @pytest.mark.asyncio
    async def test_a_raising_stream_is_swallowed(self, mock_sdk):
        sdk = listening_sdk(mock_sdk)
        sdk.instance.record_tool_call.side_effect = RuntimeError("stream is gone")
        out = await hook(sdk, "PreToolUse")(
            {"session_id": "c1", "tool_name": "Read"}, None, MagicMock(),
        )
        assert out == {}

    @pytest.mark.asyncio
    async def test_an_sdk_with_no_instance_namespace_is_fine(self, mock_sdk):
        """A half-built or older SDK answers "not listening", not an error."""
        del mock_sdk.instance
        out = await hook(mock_sdk, "Stop")({"session_id": "c1"}, None, MagicMock())
        assert out == {}
