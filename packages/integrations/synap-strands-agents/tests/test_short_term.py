"""Tests for synap_strands_agents.short_term — SynapShortTermHook.

Contract:
- folds Synap short-term context into the current turn's first user message, but
  only where the framework sends no history of its own (see test_history.py for
  the rest: the compaction reaches the model through the model-call middleware).
- empty / unavailable ST is a no-op (input untouched).
- SDK failure: on_error="fallback" -> no-op; on_error="raise" -> SynapIntegrationError.
- injection is idempotent within a re-entered invocation (no double-fold).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from strands.hooks import BeforeInvocationEvent

from synap_integrations_common import SynapIntegrationError

from synap_strands_agents.short_term import (
    _DEFAULT_OPEN,
    SynapShortTermHook,
)


def _event(messages, history=()):
    """An invocation of an agent that holds `history` (none: a fresh process)."""
    return BeforeInvocationEvent(agent=MagicMock(messages=list(history)), messages=messages)


def _user(text):
    return {"role": "user", "content": [{"text": text}]}


def _first_user_text(event):
    for m in event.messages:
        if m.get("role") == "user":
            return m["content"][0]["text"]
    raise AssertionError("no user message")


# ── construction / validation ───────────────────────────────────────────────


def test_requires_sdk():
    with pytest.raises(ValueError):
        SynapShortTermHook(None, "conv_1")


def test_requires_conversation_id(mock_sdk):
    with pytest.raises(ValueError):
        SynapShortTermHook(mock_sdk, "")


def test_rejects_bad_style(mock_sdk):
    with pytest.raises(ValueError):
        SynapShortTermHook(mock_sdk, "conv_1", style="poem")


def test_rejects_bad_on_error(mock_sdk):
    with pytest.raises(ValueError):
        SynapShortTermHook(mock_sdk, "conv_1", on_error="explode")


def test_register_hooks_subscribes_before_invocation(mock_sdk):
    hook = SynapShortTermHook(mock_sdk, "conv_1")
    registry = MagicMock()
    hook.register_hooks(registry)
    registry.add_callback.assert_called_once()
    args = registry.add_callback.call_args.args
    assert args[0] is BeforeInvocationEvent


# ── injection ────────────────────────────────────────────────────────────────


async def test_folds_context_into_first_user_message(mock_sdk):
    hook = SynapShortTermHook(mock_sdk, "conv_1")
    event = _event([_user("what's my status?")])

    await hook._inject(event)

    text = _first_user_text(event)
    assert text.startswith(_DEFAULT_OPEN)
    assert "Recent conversation summary" in text
    # original user text preserved as a separate content block
    assert event.messages[0]["content"][1] == {"text": "what's my status?"}


async def test_no_fold_next_to_a_framework_history(mock_sdk):
    """The block carries the recent turns; the history already has them."""
    hook = SynapShortTermHook(mock_sdk, "conv_1")
    event = _event([_user("what's my status?")], history=[_user("earlier")])

    await hook._inject(event)

    assert event.messages[0]["content"] == [{"text": "what's my status?"}]
    mock_sdk.conversation.context.get_context_for_prompt.assert_not_awaited()


async def test_no_fold_when_synap_has_a_compaction(mock_sdk):
    """The middleware sends the compaction; folding it too would send it twice."""
    mock_sdk.conversation.context.get_context_for_prompt.return_value = MagicMock(
        available=True, formatted_context="block", context_block="## Conversation so far (compacted x)")
    hook = SynapShortTermHook(mock_sdk, "conv_1")
    event = _event([_user("hi")])

    await hook._inject(event)

    assert event.messages[0]["content"] == [{"text": "hi"}]


async def test_no_op_when_unavailable(mock_sdk):
    mock_sdk.conversation.context.get_context_for_prompt.return_value = MagicMock(
        available=False, formatted_context=""
    )
    hook = SynapShortTermHook(mock_sdk, "conv_1")
    event = _event([_user("hi")])

    await hook._inject(event)

    assert event.messages[0]["content"] == [{"text": "hi"}]


async def test_no_op_when_empty_context(mock_sdk):
    mock_sdk.conversation.context.get_context_for_prompt.return_value = MagicMock(
        available=True, formatted_context="   "
    )
    hook = SynapShortTermHook(mock_sdk, "conv_1")
    event = _event([_user("hi")])

    await hook._inject(event)

    assert event.messages[0]["content"] == [{"text": "hi"}]


async def test_idempotent_no_double_fold(mock_sdk):
    hook = SynapShortTermHook(mock_sdk, "conv_1")
    event = _event([_user("hi")])

    await hook._inject(event)
    await hook._inject(event)

    texts = [b["text"] for b in event.messages[0]["content"]]
    assert sum(t.lstrip().startswith(_DEFAULT_OPEN) for t in texts) == 1


async def test_no_user_message_is_safe(mock_sdk):
    hook = SynapShortTermHook(mock_sdk, "conv_1")
    event = _event([{"role": "assistant", "content": [{"text": "prior"}]}])

    await hook._inject(event)  # must not raise

    assert event.messages[0]["content"] == [{"text": "prior"}]


# ── error policy ─────────────────────────────────────────────────────────────


async def test_fallback_swallows_sdk_failure(failing_sdk):
    hook = SynapShortTermHook(failing_sdk, "conv_1", on_error="fallback")
    event = _event([_user("hi")])

    await hook._inject(event)  # must not raise

    assert event.messages[0]["content"] == [{"text": "hi"}]


async def test_raise_propagates_sdk_failure(failing_sdk):
    hook = SynapShortTermHook(failing_sdk, "conv_1", on_error="raise")
    event = _event([_user("hi")])

    with pytest.raises(SynapIntegrationError):
        await hook._inject(event)
