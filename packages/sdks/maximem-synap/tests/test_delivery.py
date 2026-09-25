"""Delivery: an event survives a dropped stream, and arrives once.

Before this, "sent" meant "written to a socket". There was no id on the event
and no acknowledgement coming back, so the SDK could not tell a write the
server processed from one that died with the stream, and could not retry
without risking a doubled turn. A reconnect in the middle of a turn lost it.

Two buffers, because there are two ways to lose an event and only one of them
was covered:

- queued: there was no stream when we tried. Python already had this.
- unacknowledged: the write went out and the stream broke before the server
  said it had it. Nothing covered this, and it is the one that loses a turn
  mid-conversation rather than during a visible outage.

Every test here is about the SAME id surviving a replay, because that id is the
only thing standing between "retry" and "the turn is in there twice".
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from maximem_synap.transport.grpc_client import GRPCTransport, StreamState


class FakeStream:
    def __init__(self):
        self.written = []
        self.fail_next = False

    async def write(self, event):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("stream broke mid-write")
        self.written.append(event)


def transport(connected: bool = True) -> GRPCTransport:
    t = GRPCTransport(instance_id="inst_test", host="localhost", port=1, use_tls=False)
    t._stream = FakeStream()
    t._state = StreamState.CONNECTED if connected else StreamState.RECONNECTING
    return t


def turn(**overrides):
    payload = {
        "event_type": "user_message",
        "conversation_id": "c1",
        "user_id": "u1",
        "content": "where is my order",
        "role": "user",
    }
    payload.update(overrides)
    return payload


class TestEveryEventGetsAnId:
    @pytest.mark.asyncio
    async def test_an_id_is_minted_when_the_caller_sends_none(self):
        t = transport()
        await t.send(turn())
        assert t._stream.written[0].conversation_event.event_id

    @pytest.mark.asyncio
    async def test_the_caller_can_supply_one(self):
        t = transport()
        await t.send(turn(event_id="mine"))
        assert t._stream.written[0].conversation_event.event_id == "mine"

    @pytest.mark.asyncio
    async def test_two_events_get_two_ids(self):
        t = transport()
        await t.send(turn())
        await t.send(turn())
        ids = {e.conversation_event.event_id for e in t._stream.written}
        assert len(ids) == 2

    @pytest.mark.asyncio
    async def test_sent_at_is_when_we_first_tried_not_when_it_happened(self):
        """A queued event replayed ten minutes later still happened when it
        happened. Two clocks, two fields."""
        t = transport(connected=False)
        payload = turn(timestamp_ms=1_000)
        await t.send(payload)
        assert payload["sent_at_ms"] > 1_000


class TestTheUnacknowledgedBuffer:
    @pytest.mark.asyncio
    async def test_a_written_event_is_held_until_the_server_says_it_has_it(self):
        t = transport()
        await t.send(turn(event_id="ev-1"))
        assert "ev-1" in t._unacked

    @pytest.mark.asyncio
    async def test_an_ack_releases_it(self):
        t = transport()
        await t.send(turn(event_id="ev-1"))
        t._handle_event_ack(_ack("ev-1"))
        assert t._unacked == {}

    @pytest.mark.asyncio
    async def test_an_ack_for_something_else_releases_nothing(self):
        t = transport()
        await t.send(turn(event_id="ev-1"))
        t._handle_event_ack(_ack("ev-other"))
        assert "ev-1" in t._unacked

    @pytest.mark.asyncio
    async def test_it_is_bounded_by_depth(self):
        t = transport()
        for i in range(GRPCTransport.SEND_QUEUE_MAX_DEPTH + 10):
            await t.send(turn(event_id=f"ev-{i}"))
        assert len(t._unacked) <= GRPCTransport.SEND_QUEUE_MAX_DEPTH

    @pytest.mark.asyncio
    async def test_it_is_bounded_by_age(self):
        t = transport()
        await t.send(turn(event_id="old"))
        stale = datetime.now(timezone.utc) - GRPCTransport.SEND_QUEUE_MAX_AGE - timedelta(
            minutes=1,
        )
        t._unacked["old"] = (stale, t._unacked["old"][1])
        await t.send(turn(event_id="new"))
        assert "old" not in t._unacked and "new" in t._unacked


class TestAReconnectReplaysWhatWasLost:
    @pytest.mark.asyncio
    async def test_an_unacknowledged_event_is_sent_again(self):
        t = transport()
        await t.send(turn(event_id="ev-1"))
        t._stream.written.clear()

        await t._replay_unacked()

        assert [e.conversation_event.event_id for e in t._stream.written] == ["ev-1"]

    @pytest.mark.asyncio
    async def test_the_replay_carries_the_same_id(self):
        """The whole point. A new id on the replay means the server's dedupe
        has nothing to match and the turn is recorded twice."""
        t = transport()
        await t.send(turn(event_id="ev-1"))
        first = t._stream.written[0].conversation_event
        t._stream.written.clear()

        await t._replay_unacked()

        assert t._stream.written[0].conversation_event.event_id == first.event_id

    @pytest.mark.asyncio
    async def test_an_acknowledged_event_is_not_replayed(self):
        t = transport()
        await t.send(turn(event_id="ev-1"))
        t._handle_event_ack(_ack("ev-1"))
        t._stream.written.clear()

        await t._replay_unacked()

        assert t._stream.written == []

    @pytest.mark.asyncio
    async def test_a_failed_replay_keeps_the_event_for_next_time(self):
        t = transport()
        await t.send(turn(event_id="ev-1"))
        t._stream.fail_next = True

        await t._replay_unacked()

        assert "ev-1" in t._unacked, "a failed replay must not forget the event"


class TestAnEventSentWithNoStream:
    @pytest.mark.asyncio
    async def test_it_is_queued_rather_than_dropped(self):
        t = transport(connected=False)
        await t.send(turn(event_id="ev-1"))
        assert len(t._send_queue) == 1
        assert t._stream.written == []

    @pytest.mark.asyncio
    async def test_it_keeps_its_id_through_the_queue(self):
        t = transport(connected=False)
        await t.send(turn(event_id="ev-1"))
        t._state = StreamState.CONNECTED

        await t._drain_send_queue()

        assert t._stream.written[0].conversation_event.event_id == "ev-1"

    @pytest.mark.asyncio
    async def test_a_queued_event_with_no_caller_id_still_has_one(self):
        t = transport(connected=False)
        await t.send(turn())
        _queued_at, payload = t._send_queue[0]
        assert payload["event_id"]


class TestClosing:
    @pytest.mark.asyncio
    async def test_the_queue_goes_out_before_the_stream_does(self):
        t = transport(connected=False)
        await t.send(turn(event_id="ev-1"))
        t._state = StreamState.CONNECTED

        await t._flush_before_close()

        assert [e.conversation_event.event_id for e in t._stream.written] == ["ev-1"]

    @pytest.mark.asyncio
    async def test_closing_with_no_stream_does_not_raise(self):
        """A teardown path that throws is worse than useless."""
        t = transport(connected=False)
        await t.send(turn(event_id="ev-1"))
        await t._flush_before_close()

    @pytest.mark.asyncio
    async def test_a_hung_flush_does_not_hang_the_close(self):
        t = transport()

        async def _never():
            await asyncio.sleep(3600)
        t._drain_send_queue = _never  # type: ignore[assignment]
        t._send_queue.append((datetime.now(timezone.utc), turn()))
        t.CLOSE_FLUSH_TIMEOUT = 0.05

        await asyncio.wait_for(t._flush_before_close(), timeout=2)


def _ack(*event_ids: str):
    from maximem_synap.transport.proto import synap_service_pb2
    return synap_service_pb2.EventAck(event_ids=list(event_ids))
