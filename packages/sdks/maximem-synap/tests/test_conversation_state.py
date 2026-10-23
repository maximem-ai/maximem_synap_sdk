"""The server's message count for a conversation, arriving on the Listen stream.

After the server has written a streamed message it sends ``ConversationState``:
how many messages the conversation now holds. The SDK's short-term store uses
that to tell whether it holds the whole conversation. This checks the message
gets from the wire to the callback, through the real listen loop and the real
generated stubs.
"""
from __future__ import annotations

import asyncio
import tempfile

import pytest

from maximem_synap.transport.grpc_client import GRPCTransport, StreamState
from maximem_synap.transport.proto import synap_service_pb2


class OneShotStream:
    """Yields the given responses, then ends the loop that is reading it."""

    def __init__(self, transport, responses):
        self._transport = transport
        self._responses = list(responses)

    async def read(self):
        if self._responses:
            return self._responses.pop(0)
        self._transport._shutdown_event.set()
        raise asyncio.CancelledError()


def _state(conversation_id, count):
    return synap_service_pb2.StreamResponse(
        conversation_state=synap_service_pb2.ConversationState(
            conversation_id=conversation_id, message_count=count,
        )
    )


async def _listen(responses, **kwargs):
    t = GRPCTransport(instance_id="inst_test", host="localhost", port=1,
                      use_tls=False, storage_path=tempfile.mkdtemp(), **kwargs)
    t._stream = OneShotStream(t, responses)
    t._state = StreamState.CONNECTED
    await t._listen_loop()
    return t


@pytest.mark.asyncio
async def test_the_count_reaches_the_callback():
    seen = []
    await _listen(
        [_state("conv-1", 7), _state("conv-2", 3)],
        on_conversation_state=lambda conversation_id, count: seen.append((conversation_id, count)),
    )
    assert seen == [("conv-1", 7), ("conv-2", 3)]


@pytest.mark.asyncio
async def test_it_is_not_handed_to_the_bundle_callback():
    """A count is not a context bundle, and a caller's on_message must not see one."""
    bundles = []
    await _listen([_state("conv-1", 7)], on_message=bundles.append)
    assert bundles == []


@pytest.mark.asyncio
async def test_a_callback_that_raises_does_not_stop_the_loop():
    seen = []

    def handler(conversation_id, count):
        seen.append(count)
        if count == 1:
            raise RuntimeError("boom")

    await _listen([_state("conv-1", 1), _state("conv-1", 2)], on_conversation_state=handler)
    assert seen == [1, 2]


@pytest.mark.asyncio
async def test_with_no_callback_it_is_skipped():
    await _listen([_state("conv-1", 7)])
