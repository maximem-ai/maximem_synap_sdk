"""Local store of a conversation's raw turns, for search after compaction.

The short-term store keeps only the turns a compaction has not covered yet;
once a compaction lands it drops the rest. This store keeps every turn this
process saw or fetched, so the details a compaction leaves out can be found
again by a local search (history_search.py) instead of a server round trip.

One SQLite file per SDK instance, next to the response cache
(``~/.synap/{client}/{instance}/turns.db``), standard library only. When the
file cannot be opened (read-only disk, sandbox), it falls back to an
in-memory database with the same limits, so callers never branch on it.

Limits (behavior contract ``history_search.store_limits``, shared with JS):
turns and bytes per conversation (oldest turns go first), conversations held
(least recently used go first), idle time, and the file's size.

The server stays the source of truth for turns. A turn this process did not
see (written by another replica or device) arrives through ``add_many`` from
the turns route, with its position.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..behavior import history_search_params

logger = logging.getLogger(__name__)

SWEEP_EVERY_WRITES = 50


@dataclass(frozen=True)
class StoreLimits:
    max_conversations: int
    max_turns_per_conversation: int
    max_bytes_per_conversation: int
    idle_ttl_seconds: int
    max_file_bytes: int

    @classmethod
    def from_contract(cls) -> "StoreLimits":
        return cls(**{k: int(v) for k, v in history_search_params()["store_limits"].items()})


@dataclass
class StoredTurn:
    seq: int  # insertion order in this store
    role: str
    content: str
    timestamp: str
    message_id: Optional[str]
    position: Optional[int]  # 1-based turn number on the server, when known


_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    last_activity REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    position INTEGER,
    message_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT '',
    chash TEXT NOT NULL,
    bytes INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_by_conversation ON turns (conversation_id, seq);
CREATE UNIQUE INDEX IF NOT EXISTS turns_by_message ON turns (conversation_id, message_id)
    WHERE message_id IS NOT NULL;
CREATE TABLE IF NOT EXISTS vectors (
    seq INTEGER NOT NULL,
    model TEXT NOT NULL,
    vec BLOB NOT NULL,
    PRIMARY KEY (seq, model)
);
"""


def _chash(role: str, content: str) -> str:
    return hashlib.sha1(f"{role}\x00{content}".encode("utf-8")).hexdigest()


class TurnStore:
    """Raw turns per conversation, bounded, on disk when it can be."""

    def __init__(self, path: Optional[Path] = None, limits: Optional[StoreLimits] = None,
                 now: Callable[[], float] = time.time):
        self.limits = limits or StoreLimits.from_contract()
        self._now = now
        self._lock = threading.RLock()
        self._writes = 0
        self.path: Optional[Path] = None
        self._db = self._open(path)

    def _open(self, path: Optional[Path]) -> sqlite3.Connection:
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
                db.execute("PRAGMA journal_mode=WAL")
                # Deleted turns are overwritten, not left in free pages where the
                # text survived a drop() or clear() until a VACUUM.
                db.execute("PRAGMA secure_delete=ON")
                db.executescript(_SCHEMA)
                self.path = path
                return db
            except (sqlite3.Error, OSError) as exc:
                logger.warning("Turn store at %s unavailable (%s); keeping turns in memory", path, exc)
        db = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        db.executescript(_SCHEMA)
        return db

    @property
    def on_disk(self) -> bool:
        return self.path is not None

    # ------------------------------------------------------------------ writes
    def add(self, conversation_id: str, role: str, content: str, timestamp: Optional[str] = None,
            message_id: Optional[str] = None, position: Optional[int] = None) -> None:
        self.add_many(conversation_id, [{"role": role, "content": content, "timestamp": timestamp or "",
                                         "message_id": message_id, "position": position}])

    def add_many(self, conversation_id: str, turns: Iterable[Dict[str, Any]]) -> int:
        """Insert turns not held yet; returns how many were new. A turn held
        without an id or position (a streamed write) gets them when the same
        role and text come back from the server."""
        if not conversation_id:
            return 0
        added = 0
        with self._lock:
            db = self._db
            db.execute("BEGIN")
            try:
                for turn in turns:
                    role = str(turn.get("role") or "user")
                    content = str(turn.get("content") or "")
                    if not content:
                        continue
                    message_id = turn.get("message_id") or None
                    position = turn.get("position")
                    position = int(position) if isinstance(position, int) or str(position or "").isdigit() else None
                    timestamp = turn.get("timestamp") or ""
                    if not isinstance(timestamp, str):
                        timestamp = timestamp.isoformat()
                    chash = _chash(role, content)
                    if message_id and db.execute(
                            "SELECT 1 FROM turns WHERE conversation_id=? AND message_id=?",
                            (conversation_id, message_id)).fetchone():
                        if position is not None:
                            db.execute("UPDATE turns SET position=? WHERE conversation_id=? AND message_id=?",
                                       (position, conversation_id, message_id))
                        continue
                    same = db.execute(
                        "SELECT seq FROM turns WHERE conversation_id=? AND chash=? AND "
                        "(message_id IS NULL OR position IS NULL) ORDER BY seq LIMIT 1",
                        (conversation_id, chash)).fetchone()
                    if same:
                        db.execute("UPDATE turns SET message_id=COALESCE(message_id, ?), "
                                   "position=COALESCE(?, position) WHERE seq=?", (message_id, position, same[0]))
                        continue
                    db.execute(
                        "INSERT INTO turns (conversation_id, position, message_id, role, content, timestamp, "
                        "chash, bytes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (conversation_id, position, message_id, role, content, timestamp, chash,
                         len(content.encode("utf-8"))))
                    added += 1
                db.execute("INSERT INTO conversations (conversation_id, last_activity) VALUES (?, ?) "
                           "ON CONFLICT(conversation_id) DO UPDATE SET last_activity=excluded.last_activity",
                           (conversation_id, self._now()))
                self._trim_conversation(conversation_id)
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
            self._writes += 1
            if self._writes % SWEEP_EVERY_WRITES == 0:
                self.sweep()
        return added

    def _trim_conversation(self, conversation_id: str) -> None:
        db, limits = self._db, self.limits
        count, size = db.execute("SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM turns WHERE conversation_id=?",
                                 (conversation_id,)).fetchone()
        while count > limits.max_turns_per_conversation or size > limits.max_bytes_per_conversation:
            excess = max(count - limits.max_turns_per_conversation, 1)
            rows = db.execute("SELECT seq, bytes FROM turns WHERE conversation_id=? ORDER BY "
                              "COALESCE(position, 1e18), seq LIMIT ?", (conversation_id, excess)).fetchall()
            if not rows:
                break
            db.executemany("DELETE FROM vectors WHERE seq=?", [(r[0],) for r in rows])
            db.executemany("DELETE FROM turns WHERE seq=?", [(r[0],) for r in rows])
            count -= len(rows)
            size -= sum(r[1] for r in rows)

    def sweep(self) -> None:
        """Idle conversations, then the least recently used beyond the
        count, then more of them until the file fits."""
        with self._lock:
            limits, now = self.limits, self._now()
            stale = [r[0] for r in self._db.execute(
                "SELECT conversation_id FROM conversations WHERE last_activity < ?",
                (now - limits.idle_ttl_seconds,))]
            extra = [r[0] for r in self._db.execute(
                "SELECT conversation_id FROM conversations ORDER BY last_activity DESC LIMIT -1 OFFSET ?",
                (limits.max_conversations,))]
            for conversation_id in dict.fromkeys(stale + extra):
                self.drop(conversation_id)
            while self.size_bytes() > limits.max_file_bytes:
                oldest = self._db.execute(
                    "SELECT conversation_id FROM conversations ORDER BY last_activity LIMIT 1").fetchone()
                if not oldest:
                    break
                self.drop(oldest[0])
                if self.on_disk:
                    self._db.execute("VACUUM")

    def drop(self, conversation_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM vectors WHERE seq IN (SELECT seq FROM turns WHERE conversation_id=?)",
                             (conversation_id,))
            self._db.execute("DELETE FROM turns WHERE conversation_id=?", (conversation_id,))
            self._db.execute("DELETE FROM conversations WHERE conversation_id=?", (conversation_id,))
            self._truncate_wal()

    def clear(self) -> None:
        with self._lock:
            self._db.executescript("DELETE FROM vectors; DELETE FROM turns; DELETE FROM conversations;")
            self._truncate_wal()

    def _truncate_wal(self) -> None:
        """Move the deletes into the main file and empty the write-ahead log,
        which otherwise keeps the deleted text in turns.db-wal."""
        if self.on_disk:
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------- reads
    def turns(self, conversation_id: str) -> List[StoredTurn]:
        """In conversation order: by server position where known, else as
        they were written here."""
        with self._lock:
            rows = self._db.execute(
                "SELECT seq, role, content, timestamp, message_id, position FROM turns "
                "WHERE conversation_id=? ORDER BY seq", (conversation_id,)).fetchall()
        turns = [StoredTurn(*row) for row in rows]
        # Positions fix the order of backfilled turns; a turn without one
        # goes after every positioned turn written before it.
        key, top, step = {}, 0.0, 0
        for t in turns:
            if t.position is not None:
                key[t.seq] = float(t.position)
                top, step = max(top, float(t.position)), 0
            else:
                step += 1
                key[t.seq] = top + step * 1e-6
        return sorted(turns, key=lambda t: (key[t.seq], t.seq))

    def count(self, conversation_id: str) -> int:
        """Turns the server also holds: archived tool results are local only."""
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM turns WHERE conversation_id=? AND role != 'tool'",
                                    (conversation_id,)).fetchone()[0]

    def size_bytes(self) -> int:
        with self._lock:
            pages = self._db.execute("PRAGMA page_count").fetchone()[0]
            page = self._db.execute("PRAGMA page_size").fetchone()[0]
        return int(pages) * int(page)

    def conversations(self) -> List[str]:
        with self._lock:
            return [r[0] for r in self._db.execute("SELECT conversation_id FROM conversations ORDER BY last_activity")]

    # ----------------------------------------------------------------- vectors
    def vectors(self, seqs: List[int], model: str) -> Dict[int, bytes]:
        if not seqs:
            return {}
        with self._lock:
            marks = ",".join("?" * len(seqs))
            rows = self._db.execute(f"SELECT seq, vec FROM vectors WHERE model=? AND seq IN ({marks})",
                                    [model, *seqs]).fetchall()
        return {seq: vec for seq, vec in rows}

    def put_vectors(self, model: str, items: Dict[int, bytes]) -> None:
        if not items:
            return
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO vectors (seq, model, vec) VALUES (?, ?, ?)",
                                 [(seq, model, vec) for seq, vec in items.items()])
