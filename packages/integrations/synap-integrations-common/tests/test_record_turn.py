"""record_turn: stream first, REST when there is no stream, never both, never raises."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from synap_integrations_common import record_turn


def sdk(listening: bool, record=None):
    return SimpleNamespace(
        instance=SimpleNamespace(is_listening=listening, send_message=AsyncMock()),
        conversation=SimpleNamespace(record_message=record or AsyncMock()),
    )


@pytest.mark.asyncio
async def test_a_stream_is_used_when_open_and_rest_is_not():
    s = sdk(True)
    assert await record_turn(s, role="user", content="hi", conversation_id="c1", user_id="u1")
    s.instance.send_message.assert_awaited_once()
    s.conversation.record_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_without_a_stream_the_turn_goes_over_rest():
    s = sdk(False)
    assert await record_turn(s, role="assistant", content="hello", conversation_id="c1", user_id="u1")
    s.conversation.record_message.assert_awaited_once_with(
        conversation_id="c1", role="assistant", content="hello", user_id="u1")


@pytest.mark.asyncio
async def test_empty_turns_and_missing_conversations_write_nothing():
    s = sdk(False)
    assert not await record_turn(s, role="user", content="  ", conversation_id="c1")
    assert not await record_turn(s, role="user", content="hi", conversation_id="")
    s.conversation.record_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_write_never_raises():
    s = sdk(False, record=AsyncMock(side_effect=RuntimeError("down")))
    assert not await record_turn(s, role="user", content="hi", conversation_id="c1", customer_id="acme")
