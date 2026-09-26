"""Tests for synap_strands_agents.stream — SynapStreamHook.

Contract:
- feeds conversation turns (MessageAddedEvent) and tool intent
  (BeforeToolCallEvent) onto Synap's Listen stream via sdk.instance.send_message.
- gates every send on sdk.instance.is_listening (silent no-op when not listening).
- logs, never raises (must not abort the agent turn).
- skips messages with no text (tool-result / tool-use-only turns).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from strands.hooks import BeforeToolCallEvent, MessageAddedEvent

from synap_strands_agents.stream import SynapStreamHook


def _streaming(mock_sdk, *, listening=True, send=None):
    """Attach a Listen-stream instance controller to the shared mock."""
    inst = MagicMock()
    inst.is_listening = listening
    inst.send_message = send or AsyncMock()
    mock_sdk.instance = inst
    return mock_sdk


def _msg_event(role, text):
    return MessageAddedEvent(agent=MagicMock(), message={"role": role, "content": [{"text": text}]})


def _tool_event(name="search_memory", tool_input=None):
    return BeforeToolCallEvent(
        agent=MagicMock(),
        selected_tool=None,
        tool_use={"name": name, "input": tool_input or {"query": "x"}, "toolUseId": "t1"},
        invocation_state={},
    )


def _hook(sdk):
    return SynapStreamHook(sdk, conversation_id="conv_1", user_id="alice", customer_id="acme")


# ── construction / registration ──────────────────────────────────────────────


def test_requires_sdk():
    with pytest.raises(ValueError):
        SynapStreamHook(None, "conv_1", "alice")


def test_requires_conversation_id(mock_sdk):
    with pytest.raises(ValueError):
        SynapStreamHook(mock_sdk, "", "alice")


def test_requires_user_id(mock_sdk):
    with pytest.raises(ValueError):
        SynapStreamHook(mock_sdk, "conv_1", "")


def test_register_hooks_subscribes_all_three_events(mock_sdk):
    """AfterToolCallEvent joined the set. Strands was already firing it and
    nothing listened, so every tool call went out with no result behind it."""
    hook = _hook(mock_sdk)
    registry = MagicMock()
    hook.register_hooks(registry)
    subscribed = {c.args[0] for c in registry.add_callback.call_args_list}
    assert subscribed == {
        MessageAddedEvent, BeforeToolCallEvent, AfterToolCallEvent}


# ── message feed ─────────────────────────────────────────────────────────────


async def test_user_message_feeds_stream(mock_sdk):
    sdk = _streaming(mock_sdk)
    await _hook(sdk)._on_message(_msg_event("user", "hello there"))

    sdk.instance.send_message.assert_awaited_once()
    kwargs = sdk.instance.send_message.await_args.kwargs
    assert kwargs["content"] == "hello there"
    assert kwargs["event_type"] == "user_message"
    assert kwargs["conversation_id"] == "conv_1"
    assert kwargs["user_id"] == "alice"


async def test_assistant_message_event_type(mock_sdk):
    sdk = _streaming(mock_sdk)
    await _hook(sdk)._on_message(_msg_event("assistant", "sure, done"))
    assert sdk.instance.send_message.await_args.kwargs["event_type"] == "assistant_message"


async def test_no_send_when_not_listening(mock_sdk):
    sdk = _streaming(mock_sdk, listening=False)
    await _hook(sdk)._on_message(_msg_event("user", "hello"))
    sdk.instance.send_message.assert_not_awaited()


async def test_no_send_when_message_has_no_text(mock_sdk):
    sdk = _streaming(mock_sdk)
    event = MessageAddedEvent(
        agent=MagicMock(),
        message={"role": "user", "content": [{"toolResult": {"x": 1}}]},
    )
    await _hook(sdk)._on_message(event)
    sdk.instance.send_message.assert_not_awaited()


async def test_message_feed_logs_not_raises(mock_sdk):
    sdk = _streaming(mock_sdk, send=AsyncMock(side_effect=RuntimeError("boom")))
    await _hook(sdk)._on_message(_msg_event("user", "hello"))  # must not raise


# ── tool-call feed ───────────────────────────────────────────────────────────


async def test_tool_call_feeds_stream(mock_sdk):
    """Through the typed `record_tool_call`, not a hand-rolled
    `send_message(event_type="tool_call", role="assistant")`.

    The role on a tool event is a thing the server reads, and setting it by
    hand is how it gets read as the assistant's reply instead. The typed
    method also carries `tool_call_id`, which the hand-rolled call had no
    field for, so the result had nothing to pair with.
    """
    sdk = _streaming_sdk(mock_sdk)
    await _hook(sdk)._on_tool_call(_tool_event("search_memory", {"query": "budget"}))

    call = sdk.instance.record_tool_call.await_args
    # tool_name and tool_args are positional on the typed method.
    assert call.args[0] == "search_memory"
    assert call.args[1] == {"query": "budget"}
    assert call.kwargs["tool_call_id"] == "t1"


async def test_tool_call_no_send_when_not_listening(mock_sdk):
    sdk = _streaming(mock_sdk, listening=False)
    await _hook(sdk)._on_tool_call(_tool_event())
    sdk.instance.send_message.assert_not_awaited()


async def test_tool_call_logs_not_raises(mock_sdk):
    sdk = _streaming(mock_sdk, send=AsyncMock(side_effect=RuntimeError("boom")))
    await _hook(sdk)._on_tool_call(_tool_event())  # must not raise


# ---------------------------------------------------------------------------
# The tool RESULT, which was never reported
#
# This hook fed MessageAddedEvent and BeforeToolCallEvent and stopped there.
# Strands fires AfterToolCallEvent and nothing listened, so anticipation saw
# every tool call go out and no answer come back: it knew the agent had asked
# something and never what it learned, which is the half that says what to
# prefetch next.
#
# `toolUseId` is Strands' own id for the invocation and arrives on both
# events, so the call and the result tie themselves together.
# ---------------------------------------------------------------------------

from strands.hooks import AfterToolCallEvent  # noqa: E402
from synap_strands_agents.stream import _tool_use_id  # noqa: E402


def _streaming_sdk(mock_sdk, listening=True):
    sdk = _streaming(mock_sdk, listening=listening)
    sdk.instance.record_tool_call = AsyncMock()
    sdk.instance.record_tool_result = AsyncMock()
    sdk.instance.record_thinking = AsyncMock()
    return sdk


def _after_tool(name="lookup", result=None, exception=None, tool_use_id="t1"):
    return AfterToolCallEvent(
        agent=MagicMock(),
        selected_tool=MagicMock(),
        tool_use={"name": name, "input": {}, "toolUseId": tool_use_id},
        invocation_state={},
        result=result,
        exception=exception,
    )


class TestTheToolResultIsReported:
    @pytest.mark.asyncio
    async def test_a_result_reaches_the_stream(self, mock_sdk):
        sdk = _streaming_sdk(mock_sdk)
        hook = SynapStreamHook(sdk, conversation_id="c1", user_id="u1")
        await hook._on_tool_result(_after_tool(result={"status": "ok"}))
        sdk.instance.record_tool_result.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_it_carries_the_id_the_call_used(self, mock_sdk):
        """Without this the anticipation agent cannot pair them."""
        sdk = _streaming_sdk(mock_sdk)
        hook = SynapStreamHook(sdk, conversation_id="c1", user_id="u1")
        await hook._on_tool_result(_after_tool(result="ok", tool_use_id="shared-7"))
        assert sdk.instance.record_tool_result.await_args.kwargs[
            "tool_call_id"] == "shared-7"

    @pytest.mark.asyncio
    async def test_a_failed_tool_is_reported_as_a_result(self, mock_sdk):
        """A tool that raised is a fact about the turn: the agent is about to
        apologise or retry, and either is worth anticipating."""
        sdk = _streaming_sdk(mock_sdk)
        hook = SynapStreamHook(sdk, conversation_id="c1", user_id="u1")
        await hook._on_tool_result(_after_tool(exception=RuntimeError("no route")))
        sdk.instance.record_tool_result.assert_awaited_once()
        assert "no route" in str(
            sdk.instance.record_tool_result.await_args.args[0])

    @pytest.mark.asyncio
    async def test_the_exception_wins_over_a_partial_result(self, mock_sdk):
        sdk = _streaming_sdk(mock_sdk)
        hook = SynapStreamHook(sdk, conversation_id="c1", user_id="u1")
        await hook._on_tool_result(
            _after_tool(result={"partial": True}, exception=RuntimeError("boom")))
        assert "boom" in str(
            sdk.instance.record_tool_result.await_args.args[0])

    @pytest.mark.asyncio
    async def test_nothing_is_sent_with_no_stream(self, mock_sdk):
        sdk = _streaming_sdk(mock_sdk, listening=False)
        hook = SynapStreamHook(sdk, conversation_id="c1", user_id="u1")
        await hook._on_tool_result(_after_tool(result="ok"))
        sdk.instance.record_tool_result.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_dead_sdk_does_not_abort_the_turn(self, mock_sdk):
        """These run inside Strands' event loop; a raising hook kills the run."""
        sdk = _streaming_sdk(mock_sdk)
        sdk.instance.record_tool_result = AsyncMock(side_effect=RuntimeError("boom"))
        hook = SynapStreamHook(sdk, conversation_id="c1", user_id="u1")
        await hook._on_tool_result(_after_tool(result="ok"))

    def test_the_hook_is_registered(self, mock_sdk):
        sdk = _streaming_sdk(mock_sdk)
        hook = SynapStreamHook(sdk, conversation_id="c1", user_id="u1")
        subscribed = set()
        registry = MagicMock()
        registry.add_callback = lambda ev, cb: subscribed.add(ev)
        hook.register_hooks(registry)
        assert AfterToolCallEvent in subscribed, (
            "AfterToolCallEvent is fired by Strands and was not listened for")


class TestTheToolUseId:
    def test_bedrock_spelling(self):
        assert _tool_use_id({"toolUseId": "a"}) == "a"

    def test_snake_case_spelling(self):
        assert _tool_use_id({"tool_use_id": "b"}) == "b"

    def test_plain_id(self):
        assert _tool_use_id({"id": "c"}) == "c"

    def test_missing_is_empty_not_an_error(self):
        # An empty id silently unlinks a call from its result, so this must
        # degrade rather than raise, and be visible in the data.
        assert _tool_use_id({}) == ""
        assert _tool_use_id(None) == ""
