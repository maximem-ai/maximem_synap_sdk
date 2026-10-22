"""SynapMemoryEditor — NAT :class:`MemoryEditor` implementation backed by Synap.

NAT's memory contract is three async methods on
:class:`nat.memory.interfaces.MemoryEditor`:

- ``add_items(items)``        → writes
- ``search(query, top_k, **kw)`` → reads
- ``remove_items(**kw)``      → deletes (no-op on Synap; see below)

Error policy mirrors every other Synap integration:

- **Writes** (``add_items``): failures surface as
  :class:`SynapIntegrationError` so ingestion outages are observable.
- **Reads** (``search``): failures degrade gracefully — log at ERROR and
  return ``[]``. A Synap blip should never crash an agent turn.
- **Deletes** (``remove_items``): Synap has no public delete API. We warn
  once and no-op, same contract used by synap-crewai and synap-agno.

``add_items`` takes two shapes of item. One that carries a ``memory`` string is
a fact the framework wants kept, and is written with ``memories.create``. One
that carries only a ``conversation`` (what NAT's ``auto_memory_agent`` sends for
every user and assistant message) is a turn, and is recorded as one: Synap
extracts from the conversation it keeps, so a transcript document on top of the
turns the ``synap_stream`` exporter already recorded would be extracted twice.
Turns the exporter wrote are skipped here, one for one. A turn that cannot be
recorded (no conversation id and no stream to mint one) falls back to the
document, so nothing is lost.

We tag every Synap memory we create with a marker key so that future
NAT searches can distinguish NAT-originated records from other
memories written to the same user's scope.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from maximem_synap import MaximemSynapSDK
from nat.memory.interfaces import MemoryEditor
from nat.memory.models import MemoryItem

from nat.builder.context import ContextState
from synap_integrations_common import record_turn, wrap_sdk_errors_async

from synap_nemo_agent_toolkit._ledger import consume_written

logger = logging.getLogger(__name__)

# Metadata marker — every Synap memory NAT creates carries this, so we
# can filter them back out on search and avoid leaking non-NAT facts
# into NAT's MemoryItem stream.
_MARKER = "nat_memory_item"


class SynapMemoryEditor(MemoryEditor):
    """Synap-backed NAT :class:`MemoryEditor`.

    Args:
        sdk: Configured :class:`MaximemSynapSDK`.
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on a
            B2C instance (user_context_isolation=equals_customer): the server
            rejects a call carrying one with HTTP 400. Leave it unset there. Empty string means
            customer-less (forwarded to the SDK as ``None``).
        mode: Synap fetch mode — ``"accurate"`` (default) or ``"fast"``.
        document_type: ``document_type`` stamped on every ``memories.create``
            write. Defaults to ``"ai-chat-conversation"`` to match the
            other integrations.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        *,
        customer_id: str = "",
        mode: str = "accurate",
        document_type: str = "ai-chat-conversation",
    ) -> None:
        if sdk is None:
            raise ValueError("SynapMemoryEditor requires a non-None sdk")
        self.sdk = sdk
        self.customer_id = customer_id
        self.mode = mode
        self.document_type = document_type
        self._delete_warned = False

    # ── writes ─────────────────────────────────────────────────────────────

    async def add_items(self, items: list[MemoryItem]) -> None:
        """Insert one ``memories.create`` per ``MemoryItem``.

        Each item's ``user_id`` is required (Synap memory is user-scoped).
        We write ``item.memory`` when present, otherwise fall back to a
        joined string of the conversation's ``content`` fields — NAT's
        Mem0 path accepts conversation-only items, so we must too.
        """
        if not items:
            return

        async def _one(item: MemoryItem) -> None:
            if not item.user_id:
                raise ValueError("MemoryItem.user_id is required for Synap writes")
            conversation = item.conversation
            if not item.memory and conversation:
                conversation = await self._record_turns(item.user_id, conversation)
            document = item.memory or _conversation_to_text(conversation)
            if not document:
                # Nothing worth recording — skip rather than writing empty
                # documents that pollute fetch results.
                return
            metadata: dict[str, Any] = {
                _MARKER: True,
                "tags": list(item.tags or []),
                **(item.metadata or {}),
            }
            if conversation:
                metadata["conversation"] = list(conversation)

            async with wrap_sdk_errors_async(
                "nemo_agent_toolkit.add_items",
                logger,
                user_id=item.user_id,
                tags=item.tags,
            ):
                await self.sdk.memories.create(
                    document=document,
                    user_id=item.user_id,
                    customer_id=self.customer_id or "",
                    document_type=self.document_type,
                    metadata=metadata,
                )

        # Fan out concurrently — mirrors Mem0Editor's asyncio.gather shape
        # so throughput on bulk add_items matches the reference plugin.
        await asyncio.gather(*(_one(item) for item in items))

    async def _record_turns(
        self, user_id: str, conversation: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Record a conversation item's turns; return the ones that could not be.

        A turn the exporter already wrote counts as recorded.
        """
        conversation_id = ""
        try:
            conversation_id = ContextState.get().conversation_id.get() or ""
        except Exception:  # noqa: BLE001 — no NAT context: the stream may still mint one
            pass
        left: list[dict[str, str]] = []
        for msg in conversation:
            role, text = msg.get("role"), (msg.get("content") or "").strip()
            if role not in ("user", "assistant") or not text:
                left.append(msg)
            elif consume_written(user_id, role, text):
                continue
            elif not await record_turn(
                self.sdk, role=role, content=text, conversation_id=conversation_id,
                user_id=user_id, customer_id=self.customer_id,
            ):
                left.append(msg)
        return left

    # ── reads ──────────────────────────────────────────────────────────────

    async def search(self, query: str, top_k: int = 5, **kwargs) -> list[MemoryItem]:
        """Query Synap and map hits to :class:`MemoryItem`.

        The ``user_id`` kwarg is required — Synap memory is user-scoped.
        On SDK failure we log and return an empty list so the agent turn
        can continue without long-term memory.
        """
        user_id = kwargs.pop("user_id", None)
        if not user_id:
            # Mem0Editor also requires user_id in kwargs; be consistent.
            raise ValueError(
                "SynapMemoryEditor.search requires user_id in kwargs"
            )
        customer_id = kwargs.pop("customer_id", None) or self.customer_id or None
        include_conversation_context = bool(
            kwargs.pop("include_conversation_context", False)
        )
        tag_filter = kwargs.pop("tag_filter", None)

        try:
            response = await self.sdk.fetch(
                user_id=user_id,
                customer_id=customer_id,
                search_query=[query] if query else None,
                max_results=max(int(top_k), 1),
                mode=self.mode,
                include_conversation_context=include_conversation_context,
            )
        except Exception as exc:  # noqa: BLE001 — read-side graceful degrade
            logger.error(
                "SynapMemoryEditor.search: sdk.fetch failed user_id=%s error=%s",
                user_id, exc, exc_info=True,
            )
            return []

        # Every memory type comes back, not just facts: a stored preference, episode
        # or emotion is just as much a hit. Rank them together by score.
        scored: list[tuple[float, MemoryItem]] = []
        for kind, obj, content, score in _iter_hits(response):
            # Synap does not echo back custom metadata written with
            # memories.create(), so _MARKER-based filtering is not possible.
            # All hits for the user/customer scope are returned as-is.
            md = getattr(obj, "metadata", None) or {}
            if tag_filter:
                hit_tags = md.get("tags") or []
                if not any(t in hit_tags for t in tag_filter):
                    continue
            scored.append(
                (score if score is not None else 0.0,
                 _hit_to_memory_item(kind, obj, content, score, user_id=user_id))
            )
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[: max(int(top_k), 1)]]

    # ── deletes (warn + no-op) ─────────────────────────────────────────────

    async def remove_items(self, **kwargs) -> None:
        """Warn once, then no-op — Synap has no public delete surface."""
        if not self._delete_warned:
            logger.warning(
                "SynapMemoryEditor: Synap has no public delete API. "
                "remove_items is a no-op. This warning fires once."
            )
            self._delete_warned = True


# ── helpers ────────────────────────────────────────────────────────────────


def _conversation_to_text(conversation: Optional[list[dict[str, str]]]) -> str:
    if not conversation:
        return ""
    lines = []
    for msg in conversation:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _iter_hits(response: Any):
    """Yield ``(kind, object, text, score)`` for every memory type in a fetch response."""
    for fact in getattr(response, "facts", None) or []:
        yield "fact", fact, getattr(fact, "content", None), getattr(fact, "confidence", None)
    for pref in getattr(response, "preferences", None) or []:
        yield "preference", pref, getattr(pref, "content", None), getattr(pref, "strength", None)
    for episode in getattr(response, "episodes", None) or []:
        yield "episode", episode, getattr(episode, "summary", None), getattr(episode, "significance", None)
    for emotion in getattr(response, "emotions", None) or []:
        etype = getattr(emotion, "emotion_type", None)
        ctx = getattr(emotion, "context", None)
        text = f"{etype}: {ctx}" if etype and ctx else (ctx or etype)
        yield "emotion", emotion, text, getattr(emotion, "intensity", None)
    for event in getattr(response, "temporal_events", None) or []:
        score = getattr(event, "temporal_confidence", None)
        if score is None:
            score = getattr(event, "confidence", None)
        yield "temporal_event", event, getattr(event, "content", None), score


def _hit_to_memory_item(
    kind: str, obj: Any, content: Optional[str], score: Optional[float], *, user_id: str
) -> MemoryItem:
    md = getattr(obj, "metadata", None) or {}
    tags = md.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    conversation = md.get("conversation")
    if conversation is not None and not isinstance(conversation, list):
        conversation = None
    # Strip our internal marker + conversation copy so downstream consumers
    # don't see implementation detail in MemoryItem.metadata.
    clean_metadata = {k: v for k, v in md.items() if k not in (_MARKER, "tags", "conversation")}
    # Say what kind of memory this is, since all kinds now come back together.
    clean_metadata["synap_type"] = kind
    return MemoryItem(
        conversation=conversation,
        user_id=user_id,
        memory=content or None,
        tags=list(tags),
        metadata=clean_metadata,
        similarity_score=score,
    )


__all__ = ["SynapMemoryEditor"]
