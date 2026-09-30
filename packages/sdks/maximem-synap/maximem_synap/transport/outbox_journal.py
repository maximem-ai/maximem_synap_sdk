"""Disk backing for the stream client's outbound buffers.

The send queue and the unacknowledged map already survive a dropped
connection. They do not survive the process: both live in memory, so `kill -9`
between a turn being recorded and the server acknowledging it loses that turn
with no trace anywhere. For a voice agent that is the last thing the caller
said; for a worker it is whatever was in flight when the pod was evicted.

This is the other half. Every change to either buffer appends one line here,
and the next start reads the file back before the first event goes out.

**Why a log and not a snapshot.** Rewriting both buffers on every change is
simpler to read, but it is O(depth) per event and it has a window: the process
can die between the change and the write. Debouncing the write widens that
window on purpose, which is the opposite of the point. An append is one small
line, it is O(1), and the write happens before the caller is told the event was
accepted, so there is no window to lose. The file is compacted when it grows
past `COMPACT_AT_LINES` and again on close, so the log does not outgrow the
buffers it describes.

**Why a torn line is fine.** A process killed mid-append leaves a partial last
line. Load stops at the first line it cannot parse and keeps everything before
it, which is exactly the prefix that was durably written. A torn line is an
event that was never confirmed to the caller either.

**Nothing here may raise.** Persistence is an improvement on losing the
events, never a reason to fail a turn or block a shutdown. Every public method
swallows its errors and reports through the log, and a journal that cannot
write degrades to the in-memory behaviour this replaced.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Ops, one character each because they are written once per event.
_OP_QUEUED = "q"      # entered the send queue (no stream to write to)
_OP_UNACKED = "u"     # written to the stream, waiting for the server
_OP_ACKED = "a"       # the server confirmed it; forget it
_OP_DROPPED = "d"     # evicted by the depth or age bound; forget it


class OutboxJournal:
    """Append-only record of what the outbound buffers hold.

    One file per instance. Two SDK clients on one instance id share a stream
    in the registry, so they share this too, which is what we want: the events
    are the instance's, not the object's.
    """

    # Rewrite the file once it holds this many lines. The buffers are bounded
    # at 100 entries each, so anything past a few hundred lines is mostly
    # tombstones for events the server already has.
    COMPACT_AT_LINES = 500

    def __init__(self, root: Path, instance_id: str) -> None:
        self.path = Path(root) / "outbox" / f"{instance_id}.jsonl"
        self._lines = 0
        self._disabled = False

    # ---------------------------------------------------------------- writes

    def record_queued(self, message: Dict[str, Any]) -> None:
        self._append({"op": _OP_QUEUED, "e": message})

    def record_unacked(self, message: Dict[str, Any]) -> None:
        self._append({"op": _OP_UNACKED, "e": message})

    def record_acked(self, event_id: str) -> None:
        self._append({"op": _OP_ACKED, "id": event_id})

    def record_dropped(self, event_id: str) -> None:
        """An event the buffers gave up on. It is not coming back on restart
        either: replaying something the in-memory path deliberately evicted
        would quietly undo the bound that evicted it."""
        self._append({"op": _OP_DROPPED, "id": event_id})

    # ----------------------------------------------------------------- reads

    def load(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Replay the file into (queued, unacked), oldest first.

        Both lists come back as plain payloads. The caller decides what to do
        with them, because "unacknowledged from a previous process" and
        "unacknowledged on this stream" are not the same thing: the old stream
        is gone, so everything here has to be written again regardless of which
        list it was in.
        """
        if self._disabled or not self.path.exists():
            return [], []
        queued: "dict[str, Dict[str, Any]]" = {}
        unacked: "dict[str, Dict[str, Any]]" = {}
        lines = 0
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except ValueError:
                        # Torn last line from a kill mid-append. Everything
                        # before it is durable; stop here rather than guess.
                        logger.debug("Outbox journal ends in a partial line; "
                                     "keeping the %d event(s) before it", lines)
                        break
                    lines += 1
                    op = rec.get("op")
                    if op == _OP_QUEUED:
                        event = rec.get("e") or {}
                        eid = event.get("event_id")
                        if eid:
                            queued[eid] = event
                    elif op == _OP_UNACKED:
                        event = rec.get("e") or {}
                        eid = event.get("event_id")
                        if eid:
                            # It left the queue to be written, so it is no
                            # longer queued. Without this a reconnect-then-kill
                            # would replay it from both lists.
                            queued.pop(eid, None)
                            unacked[eid] = event
                    elif op in (_OP_ACKED, _OP_DROPPED):
                        eid = rec.get("id")
                        queued.pop(eid, None)
                        unacked.pop(eid, None)
        except OSError as e:
            logger.warning("Outbox journal unreadable (%s); starting empty", e)
            return [], []
        self._lines = lines
        return list(queued.values()), list(unacked.values())

    # ----------------------------------------------------------- maintenance

    def compact(self, queued: List[Dict[str, Any]],
                unacked: List[Dict[str, Any]]) -> None:
        """Rewrite the file to hold exactly what the buffers hold now.

        Written to a temporary file in the same directory and renamed over the
        old one, because `os.replace` is atomic: a kill during a compaction
        leaves either the whole old file or the whole new one, never a
        half-written mix of the two.
        """
        if self._disabled:
            return
        if not queued and not unacked:
            self.clear()
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    for event in unacked:
                        fh.write(json.dumps({"op": _OP_UNACKED, "e": event}) + "\n")
                    for event in queued:
                        fh.write(json.dumps({"op": _OP_QUEUED, "e": event}) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, self.path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            self._lines = len(queued) + len(unacked)
        except Exception as e:  # noqa: BLE001 — never break a turn or a close
            logger.warning("Outbox journal compaction failed: %s", e)

    def clear(self) -> None:
        """Nothing is outstanding, so there is nothing to replay."""
        if self._disabled:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning("Outbox journal could not be removed: %s", e)
        self._lines = 0

    # -------------------------------------------------------------- internal

    def _append(self, record: Dict[str, Any]) -> None:
        if self._disabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                # The point of the file is to survive a kill, and a kill takes
                # the OS page cache with it only if the machine goes too -- but
                # a container OOM-kill does not flush for us either. fsync is
                # the difference between "probably written" and "written".
                os.fsync(fh.fileno())
            self._lines += 1
        except Exception as e:  # noqa: BLE001
            # One warning, then stay quiet: a read-only or full filesystem
            # would otherwise log once per event for the life of the process.
            if not self._disabled:
                logger.warning(
                    "Outbox journal write failed (%s); buffered events will "
                    "not survive a restart from here on", e)
            self._disabled = True

    @property
    def needs_compaction(self) -> bool:
        return self._lines >= self.COMPACT_AT_LINES
