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

from synap_integrations_common import wrap_sdk_errors_async

logger = logging.getLogger(__name__)

# Metadata marker — every Synap memory NAT creates carries this, so we
# can filter them back out on search and avoid leaking non-NAT facts
# into NAT's MemoryItem stream.
_MARKER = "nat_memory_item"


class SynapMemoryEditor(MemoryEditor):
    """Synap-backed NAT :class:`MemoryEditor`.

    Args:
        sdk: Configured :class:`MaximemSynapSDK`.
        customer_id: Optional customer/org scope. Empty string means
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
            document = item.memory or _conversation_to_text(item.conversation)
            if not document:
                # Nothing worth recording — skip rather than writing empty
                # documents that pollute fetch results.
                return
            metadata: dict[str, Any] = {
                _MARKER: True,
                "tags": list(item.tags or []),
                **(item.metadata or {}),
            }
            if item.conversation:
                metadata["conversation"] = list(item.conversation)

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

        facts = list(getattr(response, "facts", None) or [])
        items: list[MemoryItem] = []
        for fact in facts:
            # Synap does not echo back custom metadata written with
            # memories.create(), so _MARKER-based filtering is not possible.
            # All facts for the user/customer scope are returned as-is.
            md = getattr(fact, "metadata", None) or {}
            if tag_filter:
                fact_tags = md.get("tags") or []
                if not any(t in fact_tags for t in tag_filter):
                    continue
            items.append(_fact_to_memory_item(fact, user_id=user_id))
        return items

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


def _fact_to_memory_item(fact: Any, *, user_id: str) -> MemoryItem:
    md = getattr(fact, "metadata", None) or {}
    tags = md.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    conversation = md.get("conversation")
    if conversation is not None and not isinstance(conversation, list):
        conversation = None
    # Strip our internal marker + conversation copy so downstream consumers
    # don't see implementation detail in MemoryItem.metadata.
    clean_metadata = {k: v for k, v in md.items() if k not in (_MARKER, "tags", "conversation")}
    return MemoryItem(
        conversation=conversation,
        user_id=user_id,
        memory=getattr(fact, "content", None) or None,
        tags=list(tags),
        metadata=clean_metadata,
        similarity_score=getattr(fact, "confidence", None),
    )


__all__ = ["SynapMemoryEditor"]
