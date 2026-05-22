"""Unit tests for GRPCTransport's outbound retry queue."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from maximem_synap.transport.grpc_client import GRPCTransport, StreamState


@pytest.fixture
def transport() -> GRPCTransport:
    return GRPCTransport(instance_id="inst-test", host="localhost", port=12345)


@pytest.mark.asyncio
async def test_send_when_disconnected_queues_message(transport: GRPCTransport):
    msg = {"event_type": "user_message", "conversation_id": "c1", "content": "hi"}
    await transport.send(msg)
    assert len(transport._send_queue) == 1
    _, queued = transport._send_queue[0]
    assert queued["content"] == "hi"


@pytest.mark.asyncio
async def test_send_queue_respects_max_depth(transport: GRPCTransport):
    transport.SEND_QUEUE_MAX_DEPTH = 3
    for i in range(5):
        await transport.send({"event_type": "user_message", "content": f"msg-{i}"})
    # Oldest two dropped; queue holds the most recent three
    assert len(transport._send_queue) == 3
    contents = [m["content"] for _, m in transport._send_queue]
    assert contents == ["msg-2", "msg-3", "msg-4"]


@pytest.mark.asyncio
async def test_prune_drops_aged_entries(transport: GRPCTransport):
    transport.SEND_QUEUE_MAX_AGE = timedelta(seconds=10)
    # Inject an old entry manually
    stale = datetime.now(timezone.utc) - timedelta(minutes=1)
    transport._send_queue.append((stale, {"event_type": "user_message", "content": "old"}))
    fresh = datetime.now(timezone.utc)
    transport._send_queue.append((fresh, {"event_type": "user_message", "content": "new"}))
    transport._prune_send_queue_locked()
    contents = [m["content"] for _, m in transport._send_queue]
    assert contents == ["new"]


@pytest.mark.asyncio
async def test_drain_writes_in_order(transport: GRPCTransport):
    written = []

    async def fake_write(payload):
        written.append(payload["content"])

    transport._write_conversation_event = fake_write
    transport._state = StreamState.CONNECTED

    for i in range(3):
        transport._send_queue.append(
            (datetime.now(timezone.utc), {"event_type": "user_message", "content": f"q-{i}"})
        )

    drained = await transport._drain_send_queue()
    assert drained == 3
    assert written == ["q-0", "q-1", "q-2"]
    assert len(transport._send_queue) == 0


@pytest.mark.asyncio
async def test_drain_requeues_remaining_on_disconnect_mid_drain(
    transport: GRPCTransport,
):
    written = []

    async def fake_write(payload):
        written.append(payload["content"])
        if len(written) == 2:
            # Simulate the stream dropping mid-drain
            transport._state = StreamState.DISCONNECTED

    transport._write_conversation_event = fake_write
    transport._state = StreamState.CONNECTED

    for i in range(4):
        transport._send_queue.append(
            (datetime.now(timezone.utc), {"event_type": "user_message", "content": f"q-{i}"})
        )

    drained = await transport._drain_send_queue()
    assert drained == 2
    # Remaining items q-2, q-3 must still be queued (order preserved)
    remaining = [m["content"] for _, m in transport._send_queue]
    assert remaining == ["q-2", "q-3"]


@pytest.mark.asyncio
async def test_drain_requeues_on_write_failure(transport: GRPCTransport):
    async def boom(payload):
        raise RuntimeError("write failed")

    transport._write_conversation_event = boom
    transport._state = StreamState.CONNECTED
    transport._send_queue.append(
        (datetime.now(timezone.utc), {"event_type": "user_message", "content": "x"})
    )
    drained = await transport._drain_send_queue()
    assert drained == 0
    assert len(transport._send_queue) == 1


@pytest.mark.asyncio
async def test_send_context_assembled_noop_when_disconnected(transport: GRPCTransport):
    # Should not raise; just no-op.
    await transport.send_context_assembled(correlation_id="abc")
    # And it should NOT have been queued (audit events are fire-and-forget)
    assert len(transport._send_queue) == 0
