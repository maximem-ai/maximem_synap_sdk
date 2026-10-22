"""Synap's compaction replaces the framework's earlier turns, never the current one."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from synap_integrations_common import (
    CompactedHistory,
    fetch_compacted_history,
    is_synap_history,
    splice_history,
)

BLOCK = "## Conversation so far (compacted 2026-10-22)\nThe user is planning a trip to Lisbon."


def rm(role, content):
    return SimpleNamespace(role=role, content=content)


def sdk_returning(ctx=None, error=None):
    get = AsyncMock(side_effect=error) if error else AsyncMock(return_value=ctx)
    return SimpleNamespace(conversation=SimpleNamespace(context=SimpleNamespace(get_context_for_prompt=get)))


# The framework's messages, as plain tuples, so the splice is tested on its own.
def role_of(m):
    return m[0]


def text_of(m):
    return m[1]


def make(role, text):
    return (role, text)


def splice(messages, history):
    return splice_history(messages, history, role_of=role_of, text_of=text_of, make_message=make)


@pytest.mark.asyncio
async def test_fetch_wraps_the_block_and_keeps_only_user_and_assistant_turns():
    ctx = SimpleNamespace(
        context_block=BLOCK, history_block="Earlier: the budget was 2,400 EUR.",
        recent_messages=[rm("user", "flights?"), rm("assistant", "TAP at 9am"), rm("tool", "x"), rm("user", " ")],
        compaction_id="cmp_1",
    )
    h = await fetch_compacted_history(sdk_returning(ctx), "c1")
    assert h.preamble.startswith("<synap_conversation_history>\n## Conversation so far")
    assert "2,400 EUR" in h.preamble and h.preamble.endswith("</synap_conversation_history>")
    assert h.turns == (("user", "flights?"), ("assistant", "TAP at 9am"))
    assert h.compaction_id == "cmp_1"
    assert is_synap_history(h.preamble)


@pytest.mark.asyncio
async def test_no_compaction_yet_means_nothing_to_replace():
    ctx = SimpleNamespace(context_block=None, recent_messages=[rm("user", "hi")])
    assert await fetch_compacted_history(sdk_returning(ctx), "c1") is None


@pytest.mark.asyncio
async def test_a_failure_leaves_the_framework_history_alone():
    assert await fetch_compacted_history(sdk_returning(error=RuntimeError("down")), "c1") is None
    assert await fetch_compacted_history(sdk_returning(), "") is None


def test_earlier_turns_are_replaced_and_the_current_turn_kept_whole():
    history = CompactedHistory(preamble="<synap_conversation_history>\nB\n</synap_conversation_history>",
                               turns=(("user", "q3"), ("assistant", "a3"), ("user", "q4 now")))
    messages = [
        ("system", "You are a travel agent."),
        ("user", "q1"), ("assistant", "a1"), ("tool", "old result"),
        ("user", "q2"), ("assistant", "a2"), ("user", "q3"), ("assistant", "a3"),
        ("user", "q4  now"), ("assistant", ""), ("tool", "fresh result"),
    ]
    out = splice(messages, history)
    assert out == [
        ("system", "You are a travel agent."),
        ("system", history.preamble),
        ("user", "q3"), ("assistant", "a3"),
        ("user", "q4  now"), ("assistant", ""), ("tool", "fresh result"),
    ]


def test_turns_the_framework_has_and_synap_does_not_yet_are_kept():
    history = CompactedHistory(preamble="P", turns=(("user", "q1"), ("assistant", "a1")))
    messages = [("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2"), ("user", "q3")]
    out = splice(messages, history)
    assert out == [("system", "P"), ("user", "q1"), ("assistant", "a1"),
                   ("user", "q2"), ("assistant", "a2"), ("user", "q3")]


def test_a_preamble_from_a_previous_splice_is_not_kept_twice():
    history = CompactedHistory(preamble="<synap_conversation_history>\nnew\n</synap_conversation_history>", turns=())
    messages = [("system", "<synap_conversation_history>\nold\n</synap_conversation_history>"), ("user", "q")]
    out = splice(messages, history)
    assert out == [("system", history.preamble), ("user", "q")]


def test_a_list_with_no_user_message_replaces_everything():
    history = CompactedHistory(preamble="P", turns=(("assistant", "hello"),))
    assert splice([("assistant", "hello")], history) == [("system", "P"), ("assistant", "hello")]
