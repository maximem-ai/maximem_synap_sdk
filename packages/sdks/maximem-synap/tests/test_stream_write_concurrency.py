"""Every write to the Listen stream must go through one lock.

grpc.aio raises ``GRPC_CALL_ERROR_TOO_MANY_OPERATIONS`` (surfaced as
``AioRpcError`` INTERNAL "Internal error from Core") when a second write is
started on a call while an earlier write has not finished. Production saw it
in the Playground voice bot: it fires right after a context fetch, because the
fetch schedules ``send_context_assembled`` as a background task and that method
wrote WITHOUT ``_write_lock`` while conversation events, heartbeats and
``context_used`` were writing on the same call.

These tests use a REAL local grpc.aio server and the real transport, with a
server that reads slowly and messages large enough that a write stays pending
(flow control), so two unserialised writers collide.
"""
from __future__ import annotations

import asyncio
import contextlib

import grpc
import pytest
from grpc import aio

from maximem_synap.auth.models import AuthContext
from maximem_synap.transport.grpc_client import GRPCTransport, StreamState
from maximem_synap.transport.proto import synap_service_pb2, synap_service_pb2_grpc


class _SlowServicer(synap_service_pb2_grpc.SynapServiceServicer):
    def __init__(self):
        self.received = []

    async def Listen(self, request_iterator, context):
        async for ev in request_iterator:
            self.received.append(ev)
            kind = ev.WhichOneof("payload")
            if kind == "heartbeat_ping":
                yield synap_service_pb2.StreamResponse(
                    heartbeat_pong=synap_service_pb2.HeartbeatPong(
                        timestamp_ms=ev.heartbeat_ping.timestamp_ms))
            elif kind == "conversation_event":
                yield synap_service_pb2.StreamResponse(
                    event_ack=synap_service_pb2.EventAck(event_ids=[ev.conversation_event.event_id]))
            await asyncio.sleep(0.002)  # slow reader: keeps client writes pending


@contextlib.asynccontextmanager
async def _server():
    servicer = _SlowServicer()
    server = aio.server(options=[("grpc.so_reuseport", 0)])
    synap_service_pb2_grpc.add_SynapServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        yield servicer, port
    finally:
        await server.stop(0)


async def _transport(port, tmp_path):
    t = GRPCTransport(instance_id="inst_x", host="127.0.0.1", port=port,
                      use_tls=False, storage_path=str(tmp_path))
    await t.connect(AuthContext(client_id="c", instance_id="inst_x", api_key="k"))
    assert t.state == StreamState.CONNECTED
    return t


BIG = "x" * 300_000


@pytest.mark.asyncio
async def test_many_concurrent_producers_never_collide(tmp_path):
    async with _server() as (servicer, port):
        t = await _transport(port, tmp_path)
        try:
            async def producer(i):
                for j in range(4):
                    await t.send({"event_type": "user_message", "conversation_id": f"c{i}",
                                  "content": BIG, "role": "user"})
                    await t.send_context_assembled(correlation_id=f"k{i}-{j}")
                    await t.send_context_used(bundle_id=f"b{i}-{j}")

            await asyncio.gather(*(producer(i) for i in range(50)))
            await asyncio.sleep(1.0)
            assert t.state == StreamState.CONNECTED, "a write collision tore the stream down"
            assert t._reconnect_attempts == 0
            kinds = [e.WhichOneof("payload") for e in servicer.received]
            assert kinds.count("conversation_event") == 200
            assert kinds.count("context_assembled") == 200
            assert kinds.count("context_used") == 200
        finally:
            await t.close()


@pytest.mark.asyncio
async def test_cancelled_writers_do_not_poison_the_next_write(tmp_path):
    """A caller that times out mid-write must not leave the call with an
    operation still in flight when the next writer takes the lock."""
    async with _server() as (servicer, port):
        t = await _transport(port, tmp_path)
        try:
            async def one(i):
                try:
                    await asyncio.wait_for(
                        t.send({"event_type": "user_message", "conversation_id": f"c{i}",
                                "content": BIG, "role": "user"}),
                        timeout=0.005 * (i % 5 + 1))
                except asyncio.TimeoutError:
                    pass

            await asyncio.gather(*(one(i) for i in range(80)))
            for i in range(10):
                await t.send({"event_type": "user_message", "conversation_id": f"z{i}",
                              "content": BIG, "role": "user"})
            await asyncio.sleep(1.0)
            assert t.state == StreamState.CONNECTED
            assert t._reconnect_attempts == 0
        finally:
            await t.close()
