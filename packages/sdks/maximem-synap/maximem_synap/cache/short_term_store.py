"""In-memory short-term context store, keyed by conversation_id.

The SDK becomes the authoritative live store of per-conversation
short-term (ST) context once the ``sdk_st_authoritative`` feature flag
is on for an instance.

Lifecycle is event-driven, not TTL-driven:
  * A ``compaction_update`` bundle from the server REPLACES the
    summary fields and prunes ``recent_turns`` whose timestamp falls at
    or before the new compaction's ``end_timestamp``.
  * ``record_message`` / ``send_message`` APPENDS a raw turn to
    ``recent_turns`` so the SDK can serve the live tail without
    round-tripping the server between compactions.
  * Eviction is LRU (max conversations) + age-since-last-activity. There
    is no per-entry TTL — ST stays valid until the next compaction
    arrives or the conversation goes idle for ``max_age``.

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
from typing import Any, Dict, List, Optional


logger = logging.getLogger("synap.sdk.cache.short_term_store")

DEFAULT_MAX_CONVERSATIONS = 100
DEFAULT_MAX_AGE = timedelta(hours=12)


@dataclass
class CachedShortTermContext:
    """Per-conversation cached short-term context.

    Mirrors ``ConversationContextProto`` plus a few SDK-side bookkeeping
    fields (``last_activity_at`` for LRU/age eviction).
    """

    conversation_id: str
    summary: Optional[str] = None
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

    def has_compaction(self) -> bool:
        return self.compaction_id is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serializable view, suitable for emitting through telemetry or
        for handing to a local formatter."""
        return {
            "conversation_id": self.conversation_id,
            "summary": self.summary,
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
    ) -> None:
        self._lock = threading.RLock()
        self._cache: "OrderedDict[str, CachedShortTermContext]" = OrderedDict()
        self._max = max_conversations
        self._max_age = max_age

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
            if datetime.now(timezone.utc) - entry.last_activity_at > self._max_age:
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
    ) -> None:
        """Append a raw turn to ``recent_turns``.

        Called from ``record_message`` / ``send_message``. If no cache
        entry exists yet (first turn before any compaction has arrived),
        a fresh entry is created with an empty summary; the next
        ``compaction_update`` bundle will fill in the summary fields and
        prune turns covered by it.
        """
        if not conversation_id:
            return
        ts = timestamp or datetime.now(timezone.utc)
        with self._lock:
            entry = self._cache.get(conversation_id)
            if entry is None:
                entry = CachedShortTermContext(conversation_id=conversation_id)
                self._cache[conversation_id] = entry
            entry.recent_turns.append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": ts.isoformat(),
                }
            )
            entry.last_activity_at = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            self._cache.move_to_end(conversation_id)
            self._evict_if_needed()

    def apply_compaction(self, bundle: Dict[str, Any]) -> None:
        """Apply a ``compaction_update`` bundle.

        Replaces the cached summary fields with the bundle's
        ``conversation_context`` payload and prunes locally-cached
        ``recent_turns`` whose timestamp falls at or before the
        compaction's ``end_timestamp`` (turns that arrived during the
        compaction window are preserved by design).

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

        with self._lock:
            entry = self._cache.get(conv_id) or CachedShortTermContext(
                conversation_id=conv_id
            )
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

            if end_ts is not None and entry.recent_turns:
                entry.recent_turns = [
                    t
                    for t in entry.recent_turns
                    if _is_after(t.get("timestamp"), end_ts)
                ]
            entry.last_activity_at = datetime.now(timezone.utc)
            self._cache[conv_id] = entry
            self._cache.move_to_end(conv_id)
            self._evict_if_needed()

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


def _is_after(turn_timestamp: Any, cutoff: datetime) -> bool:
    """True if the turn's timestamp is strictly after the cutoff."""
    parsed = _parse_iso(turn_timestamp)
    if parsed is None:
        # Defensive: when we can't parse the timestamp, keep the turn
        # rather than silently drop it.
        return True
    return parsed > cutoff
