"""Synap long-term memory backend for Strands Agents.

Implements Strands' ``MemoryStore`` protocol so ``MemoryManager`` uses Synap as
the agent's long-term memory: recall (a ``search_memory`` tool), automatic
prompt injection, and server-side extraction. ``MemoryStore`` is a
``typing.Protocol`` — this class satisfies it structurally (five config
attributes + ``search`` + optional write sinks), it does not subclass an ABC.

Error policy:
- ``search`` (read) degrades gracefully: log at ERROR, return ``[]`` — a Synap
  blip must not crash recall.
- ``add`` / ``add_messages`` (writes) surface :class:`SynapIntegrationError`.

Why ``add_messages`` ingests through ``memories.create`` rather than
``conversation.record_messages_batch``: recorded conversation messages feed
**compaction / short-term context**, not the long-term extraction layer that
``search``/``fetch`` reads (see ``sdk/ingestion.mdx``). Ingesting the assembled
transcript via ``memories.create`` runs Synap's extraction pipeline server-side
(no client model call) and lands memories on the same layer ``search`` queries.
A **stable ``document_id``** makes ``MemoryManager``'s repeated, growing-
transcript extraction submissions *update* one document instead of accumulating
duplicates.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, List, Optional

from strands.memory import MemoryEntry

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import wrap_sdk_errors_async

from synap_strands_agents._util import message_text

logger = logging.getLogger(__name__)

# Each Synap context category -> (response attr, synap_type tag, content attr).
_CATEGORIES = (
    ("facts", "fact", "content"),
    ("preferences", "preference", "content"),
    ("episodes", "episode", "summary"),
    ("emotions", "emotion", "context"),
    ("temporal_events", "temporal_event", "content"),
)


class SynapMemoryStore:
    """Synap-backed implementation of the Strands ``MemoryStore`` protocol.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        user_id: External user id these memories are about. **Required.**
        customer_id: Optional customer/org scope. Empty means customer-less.
        conversation_id: Optional Synap conversation id; also seeds the stable
            ``document_id`` used by ``add_messages`` so repeated ingests of a
            growing transcript update one document rather than duplicating.
        mode: Synap fetch mode — ``"accurate"`` (default) or ``"fast"``.
        max_search_results: Default cap on ``search`` results.

    Example::

        from strands import Agent
        from strands.memory import MemoryManager
        from maximem_synap import MaximemSynapSDK
        from synap_strands_agents import SynapMemoryStore

        sdk = MaximemSynapSDK(api_key="sk-...")
        store = SynapMemoryStore(sdk, user_id="alice", customer_id="acme")
        agent = Agent(memory_manager=MemoryManager(stores=[store]))
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: str,
        *,
        customer_id: str = "",
        conversation_id: Optional[str] = None,
        mode: str = "accurate",
        max_search_results: int = 5,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapMemoryStore requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapMemoryStore requires a non-empty user_id")
        self._sdk = sdk
        self._user_id = user_id
        self._customer_id = customer_id
        self._conversation_id = conversation_id
        self._mode = mode
        # Stable document id so add_messages re-ingests update one document.
        self._document_id = (
            f"strands-{conversation_id}"
            if conversation_id
            else f"strands-{uuid.uuid4().hex}"
        )
        # ── MemoryStore protocol config fields (exposed as attributes) ──
        self.name = "synap"
        self.description = "Maximem Synap long-term memory"
        self.max_search_results = max_search_results
        self.writable = True
        self.extraction = True

    # ── read ─────────────────────────────────────────────────────────────────

    async def search(self, query: str, options: Any = None) -> List[MemoryEntry]:
        """Return Synap memories matching ``query`` (READ — degrades to ``[]``)."""
        limit = (options or {}).get("max_search_results") or self.max_search_results
        try:
            response = await self._sdk.fetch(
                conversation_id=self._conversation_id,
                user_id=self._user_id,
                customer_id=self._customer_id or None,
                search_query=[query],
                mode=self._mode,
                max_results=limit,
                include_conversation_context=False,
            )
        except Exception as exc:  # noqa: BLE001 — read-side graceful degrade
            logger.error(
                "SynapMemoryStore.search failed user_id=%s error=%s",
                self._user_id, exc, exc_info=True,
            )
            return []
        return self._to_entries(response, limit)

    def _to_entries(self, response: Any, limit: int) -> List[MemoryEntry]:
        entries: List[MemoryEntry] = []
        for attr, synap_type, content_attr in _CATEGORIES:
            for item in getattr(response, attr, None) or []:
                content = getattr(item, content_attr, None) or getattr(
                    item, "content", None
                )
                if not content:
                    continue
                entries.append(
                    MemoryEntry(
                        content=str(content),
                        metadata={
                            "synap_type": synap_type,
                            "id": str(getattr(item, "id", "") or ""),
                        },
                    )
                )
                if len(entries) >= limit:
                    return entries
        return entries

    # ── writes ─────────────────────────────────────────────────────────────────

    async def add(self, content: str, metadata: Any = None) -> None:
        """Ingest a single piece of content (WRITE — raises on failure)."""
        async with wrap_sdk_errors_async(
            "strands.store.add", logger, user_id=self._user_id,
        ):
            await self._sdk.memories.create(
                document=content,
                user_id=self._user_id,
                customer_id=self._customer_id or None,
                metadata=dict(metadata) if metadata else None,
            )

    async def add_messages(self, messages: Any, context: Any = None) -> None:
        """Ingest a conversation batch for server-side extraction (WRITE)."""
        transcript = self._render_transcript(messages)
        if not transcript:
            return  # nothing extractable (e.g. all tool-result turns)
        async with wrap_sdk_errors_async(
            "strands.store.add_messages", logger, user_id=self._user_id,
        ):
            await self._sdk.memories.create(
                document=transcript,
                user_id=self._user_id,
                customer_id=self._customer_id or None,
                document_type="ai-chat-conversation",
                document_id=self._document_id,
            )

    @staticmethod
    def _render_transcript(messages: Any) -> str:
        lines: List[str] = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role not in ("user", "assistant"):
                continue
            text = message_text(message)
            if not text:
                continue
            lines.append(f"{role}: {text}")
        return "\n".join(lines)


__all__ = ["SynapMemoryStore"]
