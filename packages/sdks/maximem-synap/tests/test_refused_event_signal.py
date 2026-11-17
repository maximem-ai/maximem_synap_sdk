"""A6-01 on the SDK side: an error signal that names an event drops that event.

The server acknowledges a refused event as handled and sends an error signal with
`metadata["event_id"]`. The SDK stops holding it (so it is not replayed on every
reconnect), tells the caller in the log, and keeps the stream open. A restored
event keeps the time it was first sent. JS twin:
js/src/__tests__/refused-event-signal.test.ts.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace

from maximem_synap.transport.grpc_client import GRPCTransport, StreamState


def transport(tmp_path):
    tr = GRPCTransport(instance_id="inst_1", host="localhost", port=1, use_tls=False,
                       storage_path=str(tmp_path))
    tr._state = StreamState.CONNECTED
    return tr


def event(i, **extra):
    return {"event_type": "user_message", "conversation_id": "c1", "user_id": "u1",
            "content": f"m{i}", "role": "user", "event_id": f"ev-{i}", **extra}


def signal(reason="customer_id is required", **metadata):
    return SimpleNamespace(signal_type="error", reason=reason, metadata=metadata)


def test_a_signal_naming_an_event_drops_it_from_the_unacknowledged_set(tmp_path, caplog):
    tr = transport(tmp_path)
    tr._track_unacked(event(1))
    tr._track_unacked(event(2))
    with caplog.at_level(logging.WARNING):
        tr._handle_signal(signal(event_id="ev-1"))
    assert list(tr._unacked) == ["ev-2"]
    assert "ev-1" in caplog.text and "will not be retried" in caplog.text
    assert tr._state == StreamState.CONNECTED   # the stream stays open


def test_a_signal_naming_nothing_keeps_the_old_behaviour(tmp_path, caplog):
    tr = transport(tmp_path)
    tr._track_unacked(event(1))
    with caplog.at_level(logging.WARNING):
        tr._handle_signal(signal())
    assert list(tr._unacked) == ["ev-1"]
    assert "stream stays open" in caplog.text


def test_a_signal_naming_an_unknown_event_is_harmless(tmp_path):
    tr = transport(tmp_path)
    tr._handle_signal(signal(event_id="never-sent"))
    assert tr._unacked == {}


def test_events_journaled_ten_minutes_ago_are_still_sent_after_a_restart(tmp_path):
    """A crash and a slow restart must not make the journal's events age out: the
    age limit is for an outage in this process."""
    import asyncio

    first = transport(tmp_path)
    old_ms = int((datetime.now(timezone.utc).timestamp() - 600) * 1000)   # 10 minutes ago
    first._journal.record_unacked(event(1, sent_at_ms=old_ms))
    first._journal.record_queued(event(2, sent_at_ms=old_ms))

    second = transport(tmp_path)
    written = []

    class Stream:
        async def write(self, message):
            written.append(message)

    second._stream = Stream()
    second._restore_from_journal()
    sent = asyncio.run(second._drain_send_queue())
    assert sent == 2, "restored events were pruned as too old"
