"""In-memory short-term context store, keyed by conversation_id.

The SDK becomes the authoritative live store of per-conversation
short-term (ST) context once the ``sdk_st_authoritative`` feature flag
is on for an instance.

Lifecycle is event-driven, not TTL-driven:
  * A ``compaction_update`` bundle from the server REPLACES the
    summary fields and replaces ``recent_turns`` with the messages the
    summary does not cover, by count (see ``apply_compaction``).
  * ``record_message`` / ``send_message`` APPENDS a raw turn to
    ``recent_turns`` so the SDK can serve the live tail without
    round-tripping the server between compactions.
  * Eviction is LRU (max conversations) + age-since-last-activity. There
    is no per-entry TTL — ST stays valid until the next compaction
    arrives or the conversation goes idle for ``max_age``.

Holding an entry is not the same as holding the conversation. A process
that restarted, evicted the entry, or shares the conversation with another
process has only the turns it recorded itself. So the server reports how
many messages it holds, on every write and on every compaction bundle, and
``CachedShortTermContext.is_complete()`` says whether the entry can answer
on its own. When it cannot, the caller asks the server and hands the answer
to ``reseed``.

The class is thread-safe (RLock) so callers can mix sync and async
contexts without external coordination.

See ``docs/internal/sdk_authoritative_short_term_context_plan.md`` in
the cloud repo for the full design.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger("synap.sdk.cache.short_term_store")

DEFAULT_MAX_CONVERSATIONS = 100
DEFAULT_MAX_AGE = timedelta(hours=12)


# How every v3 context_block begins (server: compaction/v3/render.py). A
# summary that begins this way IS the block, which is how a REST-fed entry, or
# a push decoded by stubs older than the context_block field, is recognised.
CONTEXT_BLOCK_PREFIX = "## Conversation so far (compacted"


def context_block_of(cc: Dict[str, Any]) -> Optional[str]:
    """The v3 block in a conversation_context payload, or None (v2 server)."""
    block = cc.get("context_block")
    if block:
        return block
    summary = cc.get("summary")
    if isinstance(summary, str) and summary.startswith(CONTEXT_BLOCK_PREFIX):
        return summary
    return None


@dataclass
class CachedShortTermContext:
    """Per-conversation cached short-term context.

    Mirrors ``ConversationContextProto`` plus a few SDK-side bookkeeping
    fields (``last_activity_at`` for LRU/age eviction).
    """

    conversation_id: str
    summary: Optional[str] = None
    # Compaction v3: the server's one framed block, printed as received and
    # once. When set, the paragraphs are not used (the server sends them empty).
    context_block: Optional[str] = None
    factual_paragraph: Optional[str] = None
    conversational_paragraph: Optional[str] = None
    current_state: Dict[str, Any] = field(default_factory=dict)
    key_extractions: Dict[str, Any] = field(default_factory=dict)
    compaction_id: Optional[str] = None
    compacted_at: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    recent_turns: List[Dict[str, Any]] = field(default_factory=list)
    last_activity_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # --- Completeness bookkeeping. See ``is_complete``. ---
    # Messages the cached summary covers, from the start of the conversation.
    compacted_message_count: int = 0
    # The highest message count the server has reported. None = never heard.
    server_message_count: Optional[int] = None
    # The server has confirmed at least one write made by this process. A
    # process that only reads never hears a count again after it is seeded, so
    # it could not notice the conversation moving on without it.
    participant: bool = False
    # At some point this entry held everything the server held.
    synced: bool = False
    # The server's count ran ahead of an entry that was in sync, so something
    # else is writing to this conversation. Latched: this process cannot see
    # those writes when they happen, only at its own next write.
    shared: bool = False

    def has_compaction(self) -> bool:
        return self.compaction_id is not None

    def local_message_count(self) -> int:
        return self.compacted_message_count + len(self.recent_turns)

    def is_complete(self) -> bool:
        """Whether this entry can stand in for the server's short-term context.

        True when the server holds nothing this entry lacks. Holding MORE than
        the server has confirmed is fine: those are this process's own turns,
        sent and not yet counted.
        """
        return (
            self.participant
            and self.synced
            and not self.shared
            and self.server_message_count is not None
            and self.server_message_count <= self.local_message_count()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializable view, suitable for emitting through telemetry or
        for handing to a local formatter."""
        return {
            "conversation_id": self.conversation_id,
            "summary": self.summary,
            "context_block": self.context_block,
            "factual_paragraph": self.factual_paragraph,
            "conversational_paragraph": self.conversational_paragraph,
            "current_state": dict(self.current_state),
            "key_extractions": dict(self.key_extractions),
            "compaction_id": self.compaction_id,
            "compacted_at": self.compacted_at.isoformat() if self.compacted_at else None,
            "end_timestamp": self.end_timestamp.isoformat() if self.end_timestamp else None,
            "recent_turns": list(self.recent_turns),
        }


class ShortTermContextStore:
    """LRU cache of per-conversation ST blocks. Thread-safe."""

    def __init__(
        self,
        max_conversations: int = DEFAULT_MAX_CONVERSATIONS,
        max_age: timedelta = DEFAULT_MAX_AGE,
        now: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._cache: "OrderedDict[str, CachedShortTermContext]" = OrderedDict()
        self._max = max_conversations
        self._max_age = max_age
        # Injected in tests, as the JS store's ``now`` option is. An entry's
        # age is measured against the timestamps of its turns, so a test that
        # writes fixed timestamps needs a clock that sits next to them.
        self._now = now or (lambda: datetime.now(timezone.utc))
        # Every turn appended here is also kept in the archive (a TurnStore),
        # which a compaction does not prune: the turns it covers stay
        # searchable locally (cache/history_search.py).
        self._archive = None

    def attach_archive(self, archive) -> None:
        self._archive = archive

    @property
    def archive(self):
        return self._archive

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, conversation_id: str) -> Optional[CachedShortTermContext]:
        """Return the cached entry or ``None``. Age-evicts on read."""
        if not conversation_id:
            return None
        with self._lock:
            entry = self._cache.get(conversation_id)
            if entry is None:
                return None
            if self._now() - entry.last_activity_at > self._max_age:
                self._cache.pop(conversation_id, None)
                return None
            self._cache.move_to_end(conversation_id)
            return entry

    def has(self, conversation_id: str) -> bool:
        return self.get(conversation_id) is not None

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def append_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        message_id: Optional[str] = None,
        server_message_count: Optional[int] = None,
        in_flight: int = 0,
    ) -> None:
        """Append a raw turn to ``recent_turns``.

        Called from ``record_message`` / ``send_message``. If no cache
        entry exists yet (first turn before any compaction has arrived),
        a fresh entry is created with an empty summary; the next
        ``compaction_update`` bundle will fill in the summary fields and
        prune turns covered by it.

        ``message_id`` is the server's id for the turn when the caller has it
        (the REST write returns one). A turn already held under that id is not
        appended twice, which is what happens when ``reseed`` brought it in
        from the server before the write's own response was handled.

        ``server_message_count`` is the conversation's message count as the
        server reported it on this write, and ``in_flight`` is how many other
        writes this process still has outstanding for the conversation. See
        ``note_server_count``.
        """
        if not conversation_id:
            return
        ts = timestamp or self._now()
        if self._archive is not None:
            try:
                self._archive.add(conversation_id, role, content, ts.isoformat(), message_id=message_id)
            except Exception as e:  # the archive never blocks the prompt path
                logger.warning("Turn archive write failed: %s", e)
        with self._lock:
            entry = self._cache.get(conversation_id)
            if entry is None:
                entry = CachedShortTermContext(conversation_id=conversation_id)
                self._cache[conversation_id] = entry
            already_held = bool(message_id) and any(
                t.get("message_id") == message_id for t in entry.recent_turns
            )
            if not already_held:
                turn = {
                    "role": role,
                    "content": content,
                    "timestamp": ts.isoformat(),
                }
                if message_id:
                    turn["message_id"] = message_id
                entry.recent_turns.append(turn)
            if server_message_count is not None:
                self._note_count(entry, int(server_message_count), in_flight)
            entry.last_activity_at = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            self._cache.move_to_end(conversation_id)
            self._evict_if_needed()

    def note_server_count(self, conversation_id: str, count: int) -> None:
        """Record the server's message count for a conversation.

        The streamed write path's half of ``append_turn(server_message_count=)``:
        a streamed turn is appended when it is sent, and the count arrives
        later, once the server has written it. A count for a conversation this
        process holds nothing for is ignored.
        """
        if not conversation_id:
            return
        with self._lock:
            entry = self._cache.get(conversation_id)
            if entry is not None:
                self._note_count(entry, int(count), 0)

    @staticmethod
    def _note_count(entry: CachedShortTermContext, count: int, in_flight: int) -> None:
        """Compare the server's count with what the entry holds.

        The count is taken after one of this process's own writes, so it
        proves the process takes part in the conversation.

        Server ahead of an entry that was in sync: something else wrote, and
        the entry is marked shared for good. Server ahead of an entry that was
        never in sync: this process joined late (a restart, an eviction), and
        ``reseed`` will fix it. ``in_flight`` covers this process's own writes
        that the server has counted and the entry has not appended yet, so two
        concurrent writes do not look like a second writer.
        """
        entry.participant = True
        if count > entry.local_message_count() + in_flight:
            if entry.synced:
                entry.shared = True
        elif entry.server_message_count is None:
            # First word from the server, and nothing is missing.
            entry.synced = True
        entry.server_message_count = max(entry.server_message_count or 0, count)

    def apply_compaction(self, bundle: Dict[str, Any]) -> None:
        """Apply a ``compaction_update`` bundle.

        Replaces the cached summary fields with the bundle's
        ``conversation_context`` payload and drops the ``recent_turns`` the
        new summary covers.

        Which turns those are is decided by count, not by clock. The bundle
        says how many messages the summary covers and how many the
        conversation held when the bundle was built, so the last
        ``total - compacted`` of its ``recent_turns`` are exactly the
        uncovered messages. Turns this process appended after the server took
        that count are kept behind them. Timestamps cannot do this job: the
        REST write returns the time of the response and not of the stored row,
        a caller may supply its own message time, and a streamed turn is
        stamped when a background write runs.

        A bundle from a server that sends no counts falls back to the older
        timestamp prune, and the entry is left out of sync so it is never
        used to answer on its own.

        Bundle shape matches the dict produced by ``_proto_to_bundle_dict``
        in ``transport/grpc_client.py``.
        """
        if not isinstance(bundle, dict):
            return
        cc = bundle.get("conversation_context") or {}
        conv_id = (
            cc.get("conversation_id")
            or bundle.get("_anticipation_conversation_id")
            or bundle.get("conversation_id")
        )
        if not conv_id:
            logger.debug("apply_compaction: bundle has no conversation_id, ignoring")
            return

        end_ts = _parse_iso(cc.get("end_timestamp") or cc.get("compacted_at"))
        compacted_at = _parse_iso(cc.get("compacted_at"))
        compacted_count = int(cc.get("compacted_message_count") or 0)
        total_count = int(cc.get("total_message_count") or 0)

        with self._lock:
            entry = self._cache.get(conv_id) or CachedShortTermContext(
                conversation_id=conv_id
            )
            if (
                compacted_at is not None
                and entry.compacted_at is not None
                and compacted_at < entry.compacted_at
            ):
                logger.debug("apply_compaction: older than the cached compaction, ignoring")
                return
            block = context_block_of(cc)
            if block:
                # A v3 compaction replaces the whole record. Keeping a v2
                # paragraph cached from an earlier push would print it next to
                # the block, so the paragraphs are cleared, not kept.
                entry.context_block = block
                entry.summary = cc.get("summary") or block
                entry.factual_paragraph = None
                entry.conversational_paragraph = None
            else:
                entry.summary = cc.get("summary") or entry.summary
                entry.factual_paragraph = (
                    cc.get("factual_paragraph") or entry.factual_paragraph
                )
                entry.conversational_paragraph = (
                    cc.get("conversational_paragraph") or entry.conversational_paragraph
                )
            entry.current_state = dict(cc.get("current_state") or {})
            entry.key_extractions = dict(cc.get("key_extractions") or {})
            entry.compaction_id = cc.get("compaction_id") or entry.compaction_id
            entry.compacted_at = compacted_at or entry.compacted_at
            entry.end_timestamp = end_ts or entry.end_timestamp

            if 0 < compacted_count <= total_count:
                uncovered = total_count - compacted_count
                server_turns = list(cc.get("recent_turns") or [])
                tail = [dict(t) for t in server_turns[-uncovered:]] if uncovered else []
                # Turns appended here after the server counted. Only an entry
                # that was in sync knows how many of its own turns those are.
                ahead = entry.local_message_count() - total_count
                newer = (
                    _not_in(entry.recent_turns[-ahead:], tail)
                    if entry.synced and ahead > 0
                    else []
                )
                entry.recent_turns = tail + newer
                entry.compacted_message_count = compacted_count
                entry.server_message_count = max(
                    entry.server_message_count or 0, total_count
                )
                # A short tail means the bundle did not carry every uncovered
                # message, so the entry cannot claim to hold them.
                entry.synced = len(tail) == uncovered
            else:
                if end_ts is not None and entry.recent_turns:
                    entry.recent_turns = [
                        t
                        for t in entry.recent_turns
                        if _is_after(t.get("timestamp"), end_ts)
                    ]
                entry.synced = False
            entry.last_activity_at = self._now()
            self._cache[conv_id] = entry
            self._cache.move_to_end(conv_id)
            self._evict_if_needed()

    def reseed(
        self,
        conversation_id: str,
        *,
        compaction_id: Optional[str],
        recent_messages: List[Dict[str, Any]],
        compacted_message_count: int,
        total_message_count: int,
    ) -> None:
        """Bring an entry back in step from a ``/context-for-prompt`` response.

        Called after the server was asked because the entry was not complete.
        The response lists every message the server's summary does not cover,
        which is what ``recent_turns`` is meant to hold.

        Only an entry that already exists is reseeded. Creating one here would
        let a process that never writes to the conversation answer from a
        snapshot it has no way to refresh.

        If the server's summary is not the one cached here, the entry cannot
        render it, so it stays out of sync until the next bundle brings the
        summary itself.

        Turns held locally and absent from the response are kept when they
        come after everything the response has: they were written or sent
        after the server read its messages.
        """
        if not conversation_id:
            return
        with self._lock:
            entry = self._cache.get(conversation_id)
            if entry is None or entry.shared:
                return
            total = int(total_message_count or 0)
            entry.server_message_count = max(entry.server_message_count or 0, total)
            if (compaction_id or None) != entry.compaction_id:
                entry.synced = False
                return
            server_turns = [dict(m) for m in (recent_messages or [])]
            entry.recent_turns = server_turns + turns_after(entry.recent_turns, server_turns)
            entry.compacted_message_count = int(compacted_message_count or 0)
            entry.synced = True

    def invalidate(self, conversation_id: str) -> None:
        with self._lock:
            self._cache.pop(conversation_id, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self._max:
            evicted_id, _ = self._cache.popitem(last=False)
            logger.debug("Evicted ST cache entry (LRU) conversation_id=%s", evicted_id)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _parse_iso(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 string into an aware datetime, or return None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        # ``fromisoformat`` accepts both ``…+00:00`` and naive forms; also
        # handle the ``Z`` suffix that the server occasionally emits.
        if isinstance(value, str) and value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        logger.debug("Could not parse ISO timestamp: %r", value)
        return None


def _held_in(turn: Dict[str, Any], turns: List[Dict[str, Any]]) -> bool:
    """Whether ``turns`` already holds ``turn``.

    By message id where both sides have one. Otherwise by role and text: a
    streamed turn has no id (the server assigns it in a background write), and
    the turns in a ``fetch`` response carry none either.
    """
    message_id = turn.get("message_id")
    for held in turns:
        held_id = held.get("message_id")
        if message_id and held_id:
            if held_id == message_id:
                return True
        elif held.get("role") == turn.get("role") and held.get("content") == turn.get("content"):
            return True
    return False


def _not_in(candidates: List[Dict[str, Any]], turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [t for t in candidates if not _held_in(t, turns)]


def turns_after(
    local_turns: List[Dict[str, Any]], server_turns: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """The local turns that come after everything in ``server_turns``.

    Walks back from the newest local turn and stops at the first one the
    server's list already holds. What is left is what this process wrote or
    sent after the server's list was built.

    A turn the server rewrote before storing it (a privacy policy) does not
    match its local copy by text, so it can come back from here as well as
    from the server. That shows one turn twice; it never loses one.
    """
    newer: List[Dict[str, Any]] = []
    for turn in reversed(local_turns or []):
        if _held_in(turn, server_turns):
            break
        newer.insert(0, turn)
    return newer


def _is_after(turn_timestamp: Any, cutoff: datetime) -> bool:
    """True if the turn's timestamp is strictly after the cutoff."""
    parsed = _parse_iso(turn_timestamp)
    if parsed is None:
        # Defensive: when we can't parse the timestamp, keep the turn
        # rather than silently drop it.
        return True
    return parsed > cutoff
