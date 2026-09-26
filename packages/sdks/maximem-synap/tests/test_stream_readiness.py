"""An event sent right after `listen()` must not be silently lost.

This is the bug that survived every other test in this suite, because every
other test asserts on what the CLIENT did. The client was never wrong: it
wrote the event, the write returned, and `send()` returned None. The server
got nothing.

What actually happened, measured against deployed staging: `stub.Listen(...)`
hands back a call object immediately and does not wait for the server to
accept the RPC. The transport set `CONNECTED` on that object, so `send()`
believed it had a live stream and wrote straight into grpc's outgoing buffer
instead of the retry queue that exists for exactly this moment. Close the SDK
before the buffer drained and the events were gone.

Five events sent immediately after `listen()` returned gave the server
`conversation_events=0`. The same five with a three second pause gave 5 of 5,
twice. Nothing in the client's behaviour distinguished the two.

So these tests pin the boundary rather than the write: while the server has
not accepted the stream, an event belongs in the queue, and it goes out when
the stream is genuinely up. A test that asserts `write()` was called would
pass on the broken code.
"""
from __future__ import annotations

import asyncio

import pytest

from maximem_synap.transport.grpc_client import GRPCTransport, StreamState


class SlowToAcceptStream:
    """A stream the server has not accepted yet.

    `wait_for_connection()` blocks until `accept()` is called, which is what
    the real call object does while the server is still getting to the RPC.
    """

    def __init__(self):
        self.written = []
        self._accepted = asyncio.Event()

    async def wait_for_connection(self):
        await self._accepted.wait()

    def accept(self):
        self._accepted.set()

    async def write(self, event):
        self.written.append(event)

    async def read(self):
        await asyncio.sleep(3600)


def transport() -> GRPCTransport:
    return GRPCTransport(
        instance_id="inst_test", host="localhost", port=1, use_tls=False,
    )


def turn(**over):
    payload = {"event_type": "user_message", "conversation_id": "c1",
               "user_id": "u1", "content": "hello", "role": "user"}
    payload.update(over)
    return payload


def written_ids(stream):
    return [e.conversation_event.event_id for e in stream.written]


class TestAnUnacceptedStreamIsNotAConnectedOne:
    @pytest.mark.asyncio
    async def test_an_event_sent_before_the_server_accepts_is_queued(self):
        """The whole bug in one assertion. CONNECTING, not CONNECTED, so the
        event takes the queue branch and survives."""
        t = transport()
        t._stream = SlowToAcceptStream()
        t._state = StreamState.CONNECTING

        await t.send(turn(event_id="ev-1"))

        assert t._stream.written == [], "it went out on a stream nobody accepted"
        assert [p["event_id"] for _ts, p in t._send_queue] == ["ev-1"]

    @pytest.mark.asyncio
    async def test_it_goes_out_once_the_stream_is_really_up(self):
        t = transport()
        t._stream = SlowToAcceptStream()
        t._state = StreamState.CONNECTING
        await t.send(turn(event_id="ev-1"))

        t._stream.accept()
        t._state = StreamState.CONNECTED
        await t._drain_send_queue()

        assert written_ids(t._stream) == ["ev-1"]

    @pytest.mark.asyncio
    async def test_a_whole_turn_survives_the_gap_in_order(self):
        """Five events, the shape of a real turn, all sent before the server
        accepts. Losing four of five is what the live run did."""
        t = transport()
        t._stream = SlowToAcceptStream()
        t._state = StreamState.CONNECTING

        for i, kind in enumerate(["user_message", "agent_thinking", "tool_call",
                                  "tool_result", "assistant_message"]):
            await t.send(turn(event_id=f"ev-{i}", event_type=kind))

        t._stream.accept()
        t._state = StreamState.CONNECTED
        await t._drain_send_queue()

        assert written_ids(t._stream) == [f"ev-{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_connect_waits_for_the_server_to_accept(self, monkeypatch):
        """`_establish_connection` must not report CONNECTED off a call object
        the server has not taken. It used to, and that is the bug.

        Driven end to end rather than by setting `_state` by hand, because the
        defect was in that function deciding when to set it. A test that sets
        the state itself cannot see this.
        """
        from maximem_synap.transport import grpc_client as gc

        class Chan:
            async def channel_ready(self): return None
            async def close(self, grace=None): return None

        monkeypatch.setattr(gc.aio, "insecure_channel", lambda *a, **k: Chan())
        monkeypatch.setattr(gc.aio, "secure_channel", lambda *a, **k: Chan())

        t = transport()
        stream = SlowToAcceptStream()

        async def fake_open():
            return stream

        t._open_stream = fake_open
        t._create_stub = lambda ch: object()

        task = asyncio.create_task(t._establish_connection())
        await asyncio.sleep(0.05)
        assert t._state is not StreamState.CONNECTED, (
            "reported CONNECTED before the server accepted the stream")

        stream.accept()
        await asyncio.wait_for(task, timeout=2)
        assert t._state == StreamState.CONNECTED


class TestTheCallerCanSeeWhatWasLost:
    @pytest.mark.asyncio
    async def test_undelivered_reports_what_never_went_out(self):
        """Every send path returns None, so without this a caller cannot tell
        a delivered turn from a lost one. Eleven green ticks and one arrived
        event looked identical."""
        t = transport()
        t._stream = SlowToAcceptStream()
        t._state = StreamState.CONNECTING
        for i in range(3):
            await t.send(turn(event_id=f"ev-{i}"))

        assert t.undelivered() == {"queued": 3, "unacknowledged": 0}

    @pytest.mark.asyncio
    async def test_nothing_outstanding_reports_zero(self):
        t = transport()
        assert t.undelivered() == {"queued": 0, "unacknowledged": 0}


class TestCloseDoesNotCutOffWritesItAccepted:
    @pytest.mark.asyncio
    async def test_the_channel_is_closed_with_a_grace_period(self):
        """`close()` with no grace cancels in-flight RPCs, including writes
        grpc has taken but not yet put on the wire. That is the second half of
        the same lost turn."""
        seen = {}

        class Chan:
            async def close(self, grace=None):
                seen["grace"] = grace

        t = transport()
        t._channel = Chan()
        t._state = StreamState.CLOSED

        await t.close()

        assert seen["grace"] == GRPCTransport.CLOSE_GRACE
        assert seen["grace"] > 0
