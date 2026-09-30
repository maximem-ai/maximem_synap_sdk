"""D12: a process killed mid-turn loses nothing after restart.

`test_delivery.py` covers the two in-memory buffers, which survive a dropped
connection. Neither survives the process. `kill -9` between recording a turn
and the server acknowledging it took that turn with it, and nothing anywhere
showed that it had happened: for a voice agent that is the last thing the
caller said, for a worker it is whatever was in flight when the pod was
evicted.

A killed process is simulated the only honest way available in a test: build a
transport, put events through it, then throw the object away WITHOUT calling
close(), and build a second transport on the same storage root. No flush, no
teardown, no chance to tidy up, which is exactly what a SIGKILL gives you.
Every test that claims something survives has to cross that boundary; a test
that calls close() first is testing the flush, not the crash.
"""
from __future__ import annotations

import json
import tempfile

import pytest

from maximem_synap.transport.grpc_client import GRPCTransport, StreamState
from maximem_synap.transport.outbox_journal import OutboxJournal


class FakeStream:
    def __init__(self):
        self.written = []

    async def write(self, event):
        self.written.append(event)


def transport(root: str, connected: bool = True) -> GRPCTransport:
    t = GRPCTransport(
        instance_id="inst_test", host="localhost", port=1, use_tls=False,
        storage_path=root,
    )
    t._stream = FakeStream()
    t._state = StreamState.CONNECTED if connected else StreamState.RECONNECTING
    return t


def turn(**overrides):
    payload = {
        "event_type": "user_message",
        "conversation_id": "c1",
        "user_id": "u1",
        "content": "my card is not working",
        "role": "user",
    }
    payload.update(overrides)
    return payload


def _ack(*ids):
    class _A:
        event_ids = list(ids)
    return _A()


def written_ids(t: GRPCTransport) -> list[str]:
    return [e.conversation_event.event_id for e in t._stream.written]


@pytest.fixture
def root(tmp_path):
    return str(tmp_path)


class TestAKilledProcessLosesNothing:
    @pytest.mark.asyncio
    async def test_a_queued_event_comes_back_on_the_next_start(self, root):
        """No stream, so the event sits in the send queue. The process dies
        there. The next start has it."""
        dead = transport(root, connected=False)
        await dead.send(turn(event_id="ev-1"))
        assert dead._send_queue  # it never left

        reborn = transport(root, connected=False)
        reborn._restore_from_journal()
        assert [p["event_id"] for _ts, p in reborn._send_queue] == ["ev-1"]

    @pytest.mark.asyncio
    async def test_an_unacknowledged_event_comes_back_too(self, root):
        """The write went out and the server never confirmed it. That is the
        case that loses a turn mid-conversation rather than during a visible
        outage, and it is the one the in-memory buffer could never survive."""
        dead = transport(root)
        await dead.send(turn(event_id="ev-1"))
        assert "ev-1" in dead._unacked

        reborn = transport(root, connected=False)
        reborn._restore_from_journal()
        assert [p["event_id"] for _ts, p in reborn._send_queue] == ["ev-1"]

    @pytest.mark.asyncio
    async def test_an_acknowledged_event_does_not_come_back(self, root):
        """The server has it. Replaying it would be the server's dedupe doing
        work for nothing, and a restart that always resends everything it ever
        sent is not durability, it is a leak."""
        dead = transport(root)
        await dead.send(turn(event_id="ev-1"))
        dead._handle_event_ack(_ack("ev-1"))

        reborn = transport(root, connected=False)
        reborn._restore_from_journal()
        assert list(reborn._send_queue) == []

    @pytest.mark.asyncio
    async def test_the_id_survives_so_the_server_can_deduplicate(self, root):
        """The whole retry story rests on the id being the SAME one. A restart
        that mints a fresh id turns every recovered event into a duplicate
        turn, which is worse than the loss it was fixing."""
        dead = transport(root, connected=False)
        await dead.send(turn(event_id="ev-1"))

        reborn = transport(root)
        reborn._restore_from_journal()
        await reborn._drain_send_queue()
        assert written_ids(reborn) == ["ev-1"]

    @pytest.mark.asyncio
    async def test_order_is_kept_across_the_restart(self, root):
        dead = transport(root, connected=False)
        for i in range(5):
            await dead.send(turn(event_id=f"ev-{i}", content=f"turn {i}"))

        reborn = transport(root)
        reborn._restore_from_journal()
        await reborn._drain_send_queue()
        assert written_ids(reborn) == [f"ev-{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_unacknowledged_goes_out_before_queued(self, root):
        """An event written to the stream that died is older than anything
        that piled up after it died, and the server stores turns in the order
        it receives them."""
        dead = transport(root)
        await dead.send(turn(event_id="written"))      # -> unacked
        dead._state = StreamState.RECONNECTING
        await dead.send(turn(event_id="piled-up"))     # -> queued

        reborn = transport(root)
        reborn._restore_from_journal()
        await reborn._drain_send_queue()
        assert written_ids(reborn) == ["written", "piled-up"]

    @pytest.mark.asyncio
    async def test_a_clean_close_leaves_nothing_to_replay(self, root):
        """The flush-on-close path already got these out. A restart that
        replayed them anyway would resend the tail of every clean shutdown."""
        t = transport(root)
        await t.send(turn(event_id="ev-1"))
        t._handle_event_ack(_ack("ev-1"))
        await t.close()

        reborn = transport(root, connected=False)
        reborn._restore_from_journal()
        assert not reborn._send_queue

    @pytest.mark.asyncio
    async def test_restored_events_go_out_without_waiting_for_a_disconnect(self, root):
        """The drain used to be reachable only from the reconnect path, so a
        restored buffer would sit there until the stream happened to break."""
        dead = transport(root, connected=False)
        await dead.send(turn(event_id="ev-1"))

        reborn = transport(root)
        reborn._restore_from_journal()
        await reborn._drain_send_queue()
        assert written_ids(reborn) == ["ev-1"]
        # And it is now held as unacknowledged rather than forgotten: the
        # drain used to write and move on, so a stream that broke again before
        # the acks arrived lost everything it had just replayed.
        assert "ev-1" in reborn._unacked


class TestTheJournalItself:
    def test_a_torn_last_line_keeps_everything_before_it(self, root):
        """A kill mid-append leaves a partial line. Losing the whole file
        because its last byte is missing would turn a one-event loss into a
        hundred-event one."""
        j = OutboxJournal(root, "inst_test")
        j.record_queued({"event_id": "ev-1", "content": "first"})
        j.record_queued({"event_id": "ev-2", "content": "second"})
        with j.path.open("a", encoding="utf-8") as fh:
            fh.write('{"op": "q", "e": {"event_id": "ev-3", "cont')

        queued, unacked = OutboxJournal(root, "inst_test").load()
        assert [e["event_id"] for e in queued] == ["ev-1", "ev-2"]
        assert unacked == []

    def test_a_dropped_event_is_not_resurrected(self, root):
        """The depth and age bounds exist to stop an outage growing the buffer
        without limit. Replaying on restart what those bounds deliberately
        evicted would quietly undo them."""
        j = OutboxJournal(root, "inst_test")
        j.record_queued({"event_id": "ev-1"})
        j.record_dropped("ev-1")
        assert OutboxJournal(root, "inst_test").load() == ([], [])

    def test_compaction_keeps_what_is_outstanding_and_forgets_the_rest(self, root):
        j = OutboxJournal(root, "inst_test")
        for i in range(20):
            j.record_unacked({"event_id": f"ev-{i}"})
            j.record_acked(f"ev-{i}")
        j.record_queued({"event_id": "still-here"})

        j.compact([{"event_id": "still-here"}], [])
        queued, unacked = OutboxJournal(root, "inst_test").load()
        assert [e["event_id"] for e in queued] == ["still-here"]
        assert unacked == []
        assert j.path.read_text().count("\n") == 1

    def test_an_empty_compaction_removes_the_file(self, root):
        j = OutboxJournal(root, "inst_test")
        j.record_queued({"event_id": "ev-1"})
        assert j.path.exists()
        j.compact([], [])
        assert not j.path.exists()

    def test_an_unwritable_root_degrades_instead_of_raising(self, tmp_path):
        """Persistence is an improvement on losing the events, never a reason
        to fail a turn. A read-only filesystem must cost a warning, not a
        crash in the middle of somebody's conversation."""
        blocker = tmp_path / "outbox"
        blocker.write_text("I am a file where a directory should be")
        j = OutboxJournal(str(tmp_path), "inst_test")
        j.record_queued({"event_id": "ev-1"})   # must not raise
        assert j.load() == ([], [])

    @pytest.mark.asyncio
    async def test_a_write_failure_does_not_break_send(self, tmp_path):
        blocker = tmp_path / "outbox"
        blocker.write_text("not a directory")
        t = transport(str(tmp_path), connected=False)
        await t.send(turn(event_id="ev-1"))     # must not raise
        assert [p["event_id"] for _ts, p in t._send_queue] == ["ev-1"]

    def test_two_instances_do_not_share_a_file(self, root):
        """One file per instance id. Two instances in one process sharing a
        journal would replay each other's events onto the wrong stream."""
        a = OutboxJournal(root, "inst_aaa")
        b = OutboxJournal(root, "inst_bbb")
        a.record_queued({"event_id": "from-a"})
        b.record_queued({"event_id": "from-b"})
        assert [e["event_id"] for e in OutboxJournal(root, "inst_aaa").load()[0]] == ["from-a"]
        assert [e["event_id"] for e in OutboxJournal(root, "inst_bbb").load()[0]] == ["from-b"]

    def test_a_queued_event_later_written_is_not_replayed_twice(self, root):
        """It was queued, then the stream came up and it was written. One
        event, two lines. Restoring it from both lists would double the turn,
        and the server's dedupe is the last line of defence, not the first."""
        j = OutboxJournal(root, "inst_test")
        j.record_queued({"event_id": "ev-1"})
        j.record_unacked({"event_id": "ev-1"})
        queued, unacked = OutboxJournal(root, "inst_test").load()
        assert queued == []
        assert [e["event_id"] for e in unacked] == ["ev-1"]

    def test_the_file_is_a_line_per_record(self, root):
        """An append has to be one line, or a kill mid-write corrupts the
        record before it as well as the one being written."""
        j = OutboxJournal(root, "inst_test")
        j.record_queued({"event_id": "ev-1", "content": "a\nb"})
        lines = [ln for ln in j.path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["e"]["content"] == "a\nb"


class TestTheStorageRootMatchesJS:
    """⚠ Found by running the published 0.5.2 against staging, not by a test.

    The JS SDK honours `SYNAP_STORAGE_PATH` for its journal and Python did
    not, so the same environment produced a journal in one language and none
    in the other. Python is the reference implementation, so the gap was ours.

    Scoped to the journal on purpose: `get_default_storage_path` also roots
    the cache, and making that env-driven would move an existing user's cache
    the first time they set the variable for the outbox.
    """

    def test_the_env_var_is_honoured(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SYNAP_STORAGE_PATH", str(tmp_path))
        t = GRPCTransport(instance_id="inst_env", host="localhost", port=1,
                          use_tls=False)
        assert str(tmp_path) in str(t._journal.path)

    def test_an_explicit_path_still_wins_over_the_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SYNAP_STORAGE_PATH", str(tmp_path / "from_env"))
        explicit = tmp_path / "explicit"
        t = GRPCTransport(instance_id="inst_env", host="localhost", port=1,
                          use_tls=False, storage_path=str(explicit))
        assert "explicit" in str(t._journal.path)
        assert "from_env" not in str(t._journal.path)

    def test_with_neither_it_falls_back_to_the_default_root(self, monkeypatch):
        monkeypatch.delenv("SYNAP_STORAGE_PATH", raising=False)
        t = GRPCTransport(instance_id="inst_env", host="localhost", port=1,
                          use_tls=False)
        assert ".synap" in str(t._journal.path)

    def test_a_blank_env_var_is_not_a_path(self, monkeypatch):
        """An exported-but-empty variable is how shells leak state. Treating
        it as a root would write the journal to `/outbox/...`."""
        monkeypatch.setenv("SYNAP_STORAGE_PATH", "   ")
        t = GRPCTransport(instance_id="inst_env", host="localhost", port=1,
                          use_tls=False)
        assert ".synap" in str(t._journal.path)
