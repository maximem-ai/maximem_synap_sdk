"""The server refuses work while an instance's memory architecture is built.

Two surfaces, one behaviour in both SDKs (the JS twin is
js/src/__tests__/instance-not-ready.test.ts):

* HTTP 409 {"error": "instance_not_ready", ...} -> ``InstanceNotReadyError``,
  a ``ConflictError`` subclass, raised immediately and never retried.
* A ``throttle`` stream signal with metadata code ``instance_not_ready`` -> the
  events are held (never dropped, never acknowledged), ONE warning is logged,
  and they are replayed after the server's retry_after_seconds. No reconnect.
"""
from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

from maximem_synap import ConflictError, InstanceNotReadyError, TranscriptConflictError
from maximem_synap.transport.grpc_client import GRPCTransport, StreamState
from maximem_synap.transport.http_client import HTTPTransport  # noqa: F401


def _body(**over):
    payload = {
        "error": "instance_not_ready",
        "message": "This instance's memory architecture is still being built.",
        "state": "building",
        "retry_after_seconds": 420,
        "instance_id": "inst_1",
    }
    payload.update(over)
    return {**payload, "detail": {"code": "instance_not_ready", **payload}}


def _handle(status, json_body=None, headers=None, text=None):
    from maximem_synap.transport import http_client

    resp = httpx.Response(status, json=json_body, headers=headers or {}) if text is None \
        else httpx.Response(status, text=text, headers=headers or {})
    handler = http_client.HTTPTransport._handle_response
    return handler(object.__new__(http_client.HTTPTransport), resp, "corr-1")


class TestHttp409:
    def test_parsed_into_the_typed_error(self):
        with pytest.raises(InstanceNotReadyError) as ei:
            _handle(409, _body(), {"Retry-After": "420"})
        e = ei.value
        assert e.retry_after_seconds == 420
        assert e.instance_id == "inst_1"
        assert e.state == "building"
        assert "still being built" in str(e)
        assert e.correlation_id == "corr-1"

    def test_is_a_conflict_error_so_existing_handlers_keep_working(self):
        with pytest.raises(ConflictError):
            _handle(409, _body())
        assert issubclass(InstanceNotReadyError, ConflictError)
        assert not issubclass(InstanceNotReadyError, TranscriptConflictError)

    def test_detail_copy_alone_is_enough(self):
        b = _body()
        for k in ("error", "message", "state", "retry_after_seconds", "instance_id"):
            b.pop(k)
        with pytest.raises(InstanceNotReadyError) as ei:
            _handle(409, b, {"Retry-After": "90"})
        assert ei.value.retry_after_seconds == 420  # the detail copy wins over the header
        assert ei.value.instance_id == "inst_1"

    def test_header_used_when_body_has_no_number(self):
        b = _body()
        b.pop("retry_after_seconds")
        b["detail"].pop("retry_after_seconds")
        with pytest.raises(InstanceNotReadyError) as ei:
            _handle(409, b, {"Retry-After": "30"})
        assert ei.value.retry_after_seconds == 30

    def test_a_real_conflict_stays_a_plain_conflict(self):
        with pytest.raises(ConflictError) as ei:
            _handle(409, {"detail": {"code": "other", "message": "busy"}})
        assert not isinstance(ei.value, InstanceNotReadyError)

    def test_never_retried_by_the_transport(self):
        # 409 is raised from _handle_response, a SynapPermanentError subclass:
        # the retry loop only retries transient errors.
        from maximem_synap.models.errors import SynapPermanentError

        assert issubclass(InstanceNotReadyError, SynapPermanentError)


# ---------------------------------------------------------------- the stream

class FakeStream:
    def __init__(self):
        self.written = []

    async def write(self, event):
        self.written.append(event)


class _Signal:
    def __init__(self, metadata, reason="instance_not_ready: building"):
        self.signal_type = "throttle"
        self.reason = reason
        self.metadata = metadata


def _signal(retry=60):
    return _Signal({
        "code": "instance_not_ready", "state": "building",
        "retry_after_seconds": str(retry), "instance_id": "inst_1",
    })


def _ack(*ids):
    class _A:
        event_ids = list(ids)
    return _A()


@pytest.fixture
def t(tmp_path, monkeypatch):
    monkeypatch.setattr(GRPCTransport, "NOT_READY_MIN_WAIT", 0.02)
    monkeypatch.setattr(GRPCTransport, "NOT_READY_MAX_WAIT", 0.05)
    monkeypatch.setattr(GRPCTransport, "NOT_READY_PROBE_GRACE", 0.05)
    tr = GRPCTransport(
        instance_id="inst_1", host="localhost", port=1, use_tls=False,
        storage_path=str(tmp_path),
    )
    tr._stream = FakeStream()
    tr._state = StreamState.CONNECTED
    return tr


def turn(i):
    return {"event_type": "user_message", "conversation_id": "c1", "user_id": "u1",
            "content": f"m{i}", "role": "user", "event_id": f"ev-{i}"}


def ids(tr):
    return [e.conversation_event.event_id for e in tr._stream.written]


def not_ready_warnings(caplog):
    return [r for r in caplog.records
            if r.levelno == logging.WARNING and "still building" in r.getMessage()]


class TestThrottleInstanceNotReady:
    @pytest.mark.asyncio
    async def test_event_is_kept_unacked_and_one_warning_is_logged(self, t, caplog):
        caplog.set_level(logging.DEBUG, logger="synap.sdk.transport.grpc")
        await t.send(turn(1))
        await t.send(turn(2))
        for _ in range(5):  # one refusal per event, plus extras
            t._handle_signal(_signal(300))
        assert set(t._unacked) == {"ev-1", "ev-2"}  # kept, not acknowledged
        assert len(not_ready_warnings(caplog)) == 1
        assert t.instance_not_ready["retry_after_seconds"] == 300
        assert t.instance_not_ready["instance_id"] == "inst_1"
        assert t.state == StreamState.CONNECTED  # no reconnect
        t._shutdown_event.set()
        t._not_ready_task.cancel()

    @pytest.mark.asyncio
    async def test_new_sends_are_held_not_written_while_refused(self, t):
        await t.send(turn(1))
        t._handle_signal(_signal())
        before = len(t._stream.written)
        await t.send(turn(2))
        assert len(t._stream.written) == before
        assert "ev-2" in t._unacked
        t._not_ready_task.cancel()

    @pytest.mark.asyncio
    async def test_held_events_replay_then_stream_resumes(self, t, caplog):
        caplog.set_level(logging.INFO, logger="synap.sdk.transport.grpc")
        await t.send(turn(1))
        t._handle_signal(_signal())
        await t.send(turn(2))
        task = t._not_ready_task
        await asyncio.wait_for(task, 2)  # server accepts: no new refusal
        assert ids(t).count("ev-1") == 2  # first write + the replay
        assert ids(t).count("ev-2") == 1  # held, then sent exactly once
        assert t.instance_not_ready is None
        # normal again: written straight out
        await t.send(turn(3))
        assert ids(t)[-1] == "ev-3"
        t._handle_event_ack(_ack("ev-1", "ev-2", "ev-3"))
        assert not t._unacked
        assert any("is ready" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_still_refused_keeps_waiting_without_a_second_warning(self, t, caplog):
        caplog.set_level(logging.DEBUG, logger="synap.sdk.transport.grpc")
        await t.send(turn(1))
        t._handle_signal(_signal())
        # the server refuses the first replay too
        orig = t._replay_unacked
        calls = {"n": 0}

        async def refusing_replay():
            calls["n"] += 1
            n = await orig()
            if calls["n"] == 1:
                t._handle_signal(_signal())
            return n

        t._replay_unacked = refusing_replay
        await asyncio.wait_for(t._not_ready_task, 3)
        assert calls["n"] == 2  # retried once more, then accepted
        assert len(not_ready_warnings(caplog)) == 1
        assert "ev-1" in t._unacked  # still not acknowledged

    @pytest.mark.asyncio
    async def test_wait_is_bounded_and_jittered(self, t, monkeypatch):
        monkeypatch.setattr(GRPCTransport, "NOT_READY_MIN_WAIT", 5.0)
        monkeypatch.setattr(GRPCTransport, "NOT_READY_MAX_WAIT", 120.0)
        t._not_ready = {"retry_after_seconds": 840}
        waits = {t._not_ready_wait() for _ in range(30)}
        assert all(96.0 <= w <= 144.0 for w in waits)
        assert len(waits) > 1  # jittered
        t._not_ready = {"retry_after_seconds": 0}
        assert all(t._not_ready_wait() >= 5.0 for _ in range(30))  # never a tight loop

    @pytest.mark.asyncio
    async def test_age_limit_does_not_evict_held_events(self, t):
        from datetime import datetime, timedelta, timezone

        await t.send(turn(1))
        t._handle_signal(_signal(300))
        ts, payload = t._unacked["ev-1"]
        t._unacked["ev-1"] = (datetime.now(timezone.utc) - timedelta(minutes=30), payload)
        await t.send(turn(2))  # would evict ev-1 on age if not held
        assert "ev-1" in t._unacked
        t._not_ready_task.cancel()

    @pytest.mark.asyncio
    async def test_other_throttle_still_just_warns(self, t, caplog):
        caplog.set_level(logging.WARNING, logger="synap.sdk.transport.grpc")
        t._handle_signal(_Signal({}, reason="slow down"))
        assert t.instance_not_ready is None
        assert t._not_ready_task is None


class TestLongHold:
    @pytest.mark.asyncio
    async def test_250_events_during_a_hold_are_all_delivered_in_order_once(self, t, caplog):
        caplog.set_level(logging.DEBUG, logger="synap.sdk.transport.grpc")
        await t.send(turn(0))
        t._handle_signal(_signal())
        for i in range(1, 250):
            await t.send(turn(i))
        assert len(t._unacked) == 250  # nothing evicted by the 100 cap
        await asyncio.wait_for(t._not_ready_task, 3)
        sent = ids(t)
        assert [e for e in sent if e != "ev-0"][:249] == [f"ev-{i}" for i in range(1, 250)]
        assert set(sent) == {f"ev-{i}" for i in range(250)}
        # ev-0 went out once before the refusal and once in the replay; the
        # 249 held ones exactly once.
        assert all(sent.count(f"ev-{i}") == 1 for i in range(1, 250))
        # acks arrive: all gone, and a later send does not evict anything
        t._handle_event_ack(_ack(*[f"ev-{i}" for i in range(250)]))
        assert not t._unacked
        assert not any("DROPPED" in r.getMessage() for r in caplog.records)
        assert not any("Nothing is lost" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_backlog_is_not_evicted_in_the_gap_after_resume(self, t):
        await t.send(turn(0))
        t._handle_signal(_signal())
        for i in range(1, 250):
            await t.send(turn(i))
        await asyncio.wait_for(t._not_ready_task, 3)
        await t.send(turn(999))  # before any ack
        assert len(t._unacked) == 251

    @pytest.mark.asyncio
    async def test_at_the_cap_the_oldest_are_dropped_and_counted_truthfully(self, t, caplog, monkeypatch):
        monkeypatch.setattr(GRPCTransport, "NOT_READY_MAX_HELD", 10)
        caplog.set_level(logging.DEBUG, logger="synap.sdk.transport.grpc")
        await t.send(turn(0))
        t._handle_signal(_signal())
        for i in range(1, 25):
            await t.send(turn(i))
        assert len(t._unacked) == 10
        assert list(t._unacked) == [f"ev-{i}" for i in range(15, 25)]
        await asyncio.wait_for(t._not_ready_task, 3)
        drops = [r.getMessage() for r in caplog.records if "DROPPED" in r.getMessage()]
        assert len(drops) == 1 and "15 held event(s)" in drops[0]
        assert not any("Nothing is lost" in r.getMessage() for r in caplog.records)
        assert set(ids(t)) >= {f"ev-{i}" for i in range(15, 25)}
        assert not any(f"ev-{i}" in ids(t)[2:] for i in range(1, 15))
