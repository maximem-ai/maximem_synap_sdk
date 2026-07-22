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


def test_register_hooks_subscribes_both_events(mock_sdk):
    hook = _hook(mock_sdk)
    registry = MagicMock()
    hook.register_hooks(registry)
    subscribed = {c.args[0] for c in registry.add_callback.call_args_list}
    assert subscribed == {MessageAddedEvent, BeforeToolCallEvent}


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
    sdk = _streaming(mock_sdk)
    await _hook(sdk)._on_tool_call(_tool_event("search_memory", {"query": "budget"}))

    kwargs = sdk.instance.send_message.await_args.kwargs
    assert kwargs["event_type"] == "tool_call"
    assert kwargs["tool_name"] == "search_memory"
    assert kwargs["tool_args"] == {"query": "budget"}


async def test_tool_call_no_send_when_not_listening(mock_sdk):
    sdk = _streaming(mock_sdk, listening=False)
    await _hook(sdk)._on_tool_call(_tool_event())
    sdk.instance.send_message.assert_not_awaited()


async def test_tool_call_logs_not_raises(mock_sdk):
    sdk = _streaming(mock_sdk, send=AsyncMock(side_effect=RuntimeError("boom")))
    await _hook(sdk)._on_tool_call(_tool_event())  # must not raise
