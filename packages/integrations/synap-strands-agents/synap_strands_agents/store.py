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

``add_messages`` records the batch as conversation turns (``record_turn``: the
gRPC stream, opened on first use, else REST) rather than ingesting the
transcript with ``memories.create``. Synap extracts from the conversation it
keeps, and a transcript ingested on top of turns :class:`SynapStreamHook`
already recorded is extracted twice. Turns the hook wrote are skipped here, one
for one, so the two compose; with no hook, this is what gets the conversation
to Synap at all.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, List, Optional

from strands.memory import MemoryEntry

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    SynapIntegrationError,
    record_turn,
    wrap_sdk_errors_async,
)

from synap_strands_agents._util import consume_written, message_text, note_written

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
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on a
            B2C instance (user_context_isolation=equals_customer): the server
            rejects a call carrying one with HTTP 400. Leave it unset there. Empty means customer-less.
        conversation_id: Optional Synap conversation id the ``add_messages``
            turns are recorded under (a fresh one per store when unset; pass the
            stream hook's so both write to one conversation).
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
        self._turn_conversation_id = conversation_id or f"strands-{uuid.uuid4().hex}"
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
        """Record a conversation batch as turns for server-side extraction
        (WRITE). Raises on failure so the manager retries the batch."""
        done: List[tuple] = []
        for role, text in self._turns(messages):
            if consume_written(self._user_id, role, text):
                continue  # SynapStreamHook already wrote it
            if not await record_turn(
                self._sdk,
                role=role,
                content=text,
                conversation_id=self._turn_conversation_id,
                user_id=self._user_id,
                customer_id=self._customer_id,
            ):
                # The retry resends the whole batch: let it skip what went through.
                for written in done:
                    note_written(self._user_id, *written)
                raise SynapIntegrationError(
                    "strands.store.add_messages",
                    "could not record the conversation turn",
                    {"user_id": self._user_id, "conversation_id": self._turn_conversation_id},
                )
            done.append((role, text))

    @staticmethod
    def _turns(messages: Any) -> List[tuple]:
        turns: List[tuple] = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role not in ("user", "assistant"):
                continue
            text = message_text(message)
            if text:
                turns.append((role, text))
        return turns


__all__ = ["SynapMemoryStore"]
