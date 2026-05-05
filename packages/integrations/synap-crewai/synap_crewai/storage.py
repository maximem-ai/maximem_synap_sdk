"""Synap storage backend for CrewAI Memory.

Implements CrewAI's StorageBackend protocol so that CrewAI's unified
Memory class stores and retrieves memories via Synap's cloud platform.

CrewAI handles LLM analysis, categorization, and importance scoring.
This backend handles persistence and retrieval by delegating to the
Synap SDK.

## Protocol compatibility notes

CrewAI's StorageBackend defines mutation and listing methods
(``delete``, ``update``, ``reset``, ``list_records``, ``count``, etc.)
that assume direct control of the storage medium. Synap's public API
currently exposes only ingestion (``sdk.memories.create``) and
retrieval (``sdk.fetch``). Per docs.maximem.ai, there is no public
endpoint to delete, update, or list individual stored memories.

We therefore:

- Execute :meth:`save` and :meth:`asearch` against the Synap SDK.
- Degrade :meth:`delete`, :meth:`update`, :meth:`reset` to **no-ops
  with a structured warning** — attempting to raise would break
  CrewAI's default crew-reset flow.
- Serve :meth:`list_records`, :meth:`count`, :meth:`get_record`,
  :meth:`get_scope_info`, :meth:`list_scopes`, :meth:`list_categories`
  from an in-process cache populated on save. Callers are warned once
  at construction that these views are session-scoped, not
  authoritative.
- Require a non-empty ``query_text`` (either passed via
  ``metadata_filter['_query_text']`` or supplied through the recall
  path). ``query_embedding`` is accepted for protocol compliance and
  intentionally ignored because Synap embeds server-side.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from crewai.memory.types import MemoryRecord, ScopeInfo

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    SynapIntegrationError,
    default_scope,
    run_async,
    wrap_sdk_errors_async,
)

logger = logging.getLogger(__name__)

_SESSION_ONLY_WARNING = (
    "SynapStorageBackend: listing/metadata methods (list_records, count, "
    "get_record, get_scope_info, list_scopes, list_categories) return a "
    "session-local view populated only by saves issued through THIS "
    "backend instance. Synap does not expose public list APIs; restart "
    "resets the view. See docs.maximem.ai for current API surface."
)

_MUTATION_UNSUPPORTED_WARNING = (
    "SynapStorageBackend.%s called, but Synap does not expose a public "
    "%s endpoint. Treated as a no-op; stored memories on the Synap "
    "backend are unchanged. If you need this, raise a request at "
    "docs.maximem.ai."
)


class SynapStorageBackend:
    """CrewAI StorageBackend backed by Synap.

    Delegates memory persistence to Synap's ingestion system
    and retrieval to Synap's search system.

    Example::

        from crewai.memory import Memory
        from synap_crewai import SynapStorageBackend

        backend = SynapStorageBackend(
            sdk=sdk, user_id="user-456", customer_id="cust-789",
        )
        memory = Memory(storage=backend)
        crew = Crew(agents=agents, tasks=tasks, memory=memory)
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: str,
        customer_id: str = "",
        conversation_id: Optional[str] = None,
        mode: str = "fast",
    ):
        if sdk is None:
            raise ValueError("SynapStorageBackend requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapStorageBackend requires a non-empty user_id")

        self.sdk = sdk
        self.user_id = user_id
        self.customer_id = customer_id
        self.conversation_id = conversation_id
        self.mode = mode
        self._default_scope = default_scope(user_id, customer_id or None)
        self._records: Dict[str, MemoryRecord] = {}
        self._session_warning_logged = False

    def _warn_session_only(self) -> None:
        if not self._session_warning_logged:
            logger.warning(_SESSION_ONLY_WARNING)
            self._session_warning_logged = True

    # ------------------------------------------------------------------
    # Core methods (used by CrewAI's Memory.remember / Memory.recall)
    # ------------------------------------------------------------------

    def save(self, records: List[MemoryRecord]) -> None:
        """Persist memory records via Synap ingestion."""
        run_async(self.asave(records))

    async def asave(self, records: List[MemoryRecord]) -> None:
        for record in records:
            async with wrap_sdk_errors_async(
                "crewai.asave",
                logger,
                record_id=record.id,
            ):
                await self.sdk.memories.create(
                    document=record.content,
                    user_id=self.user_id,
                    customer_id=self.customer_id,
                    metadata={
                        "crewai_record_id": record.id,
                        "scope": record.scope,
                        "categories": record.categories,
                        "importance": record.importance,
                        "source": record.source or "",
                    },
                )
            self._records[record.id] = record

    def search(
        self,
        query_embedding: List[float],
        scope_prefix: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Search via Synap's hybrid retrieval.

        ``query_embedding`` is accepted for CrewAI protocol compliance
        but intentionally ignored — Synap embeds queries server-side.
        The caller must pass the query text via
        ``metadata_filter['_query_text']`` (CrewAI's ``Memory.recall``
        sets this automatically).
        """
        return run_async(self.asearch(
            query_embedding, scope_prefix, categories,
            metadata_filter, limit, min_score,
        ))

    async def asearch(
        self,
        query_embedding: List[float],
        scope_prefix: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[MemoryRecord, float]]:
        del query_embedding  # unused: Synap embeds server-side

        query_text: Optional[str] = None
        if metadata_filter and "_query_text" in metadata_filter:
            query_text = metadata_filter.pop("_query_text")

        if not query_text:
            raise SynapIntegrationError(
                "crewai.asearch",
                "query_text missing — pass via "
                "metadata_filter['_query_text']. CrewAI's Memory.recall "
                "populates this automatically; if you are calling the "
                "backend directly, supply the query text yourself.",
                {"scope_prefix": scope_prefix, "limit": limit},
            )

        types = None
        if categories:
            type_map = {
                "fact": "facts", "facts": "facts",
                "preference": "preferences", "preferences": "preferences",
                "episode": "episodes", "episodes": "episodes",
                "emotion": "emotions", "emotions": "emotions",
            }
            types = [type_map.get(c, c) for c in categories if c in type_map]
            types = types or None

        async with wrap_sdk_errors_async(
            "crewai.asearch",
            logger,
            limit=limit,
            mode=self.mode,
        ):
            response = await self.sdk.fetch(
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                customer_id=self.customer_id or None,
                search_query=[query_text],
                max_results=limit,
                types=types,
                mode=self.mode,
                include_conversation_context=False,
            )

        results: List[Tuple[MemoryRecord, float]] = []
        now = datetime.now(timezone.utc)
        scope = scope_prefix or self._default_scope

        for fact in response.facts:
            results.append((MemoryRecord(
                id=fact.id, content=fact.content, scope=scope,
                categories=["fact"], importance=fact.confidence,
                created_at=fact.extracted_at, last_accessed=now,
                metadata={"source": fact.source,
                          "scope_origin": response.scope_map.get(fact.id, "")},
            ), fact.confidence))

        for pref in response.preferences:
            results.append((MemoryRecord(
                id=pref.id, content=pref.content, scope=scope,
                categories=["preference"], importance=pref.strength,
                created_at=pref.extracted_at, last_accessed=now,
                metadata={"category": pref.category,
                          "scope_origin": response.scope_map.get(pref.id, "")},
            ), pref.strength))

        for ep in response.episodes:
            results.append((MemoryRecord(
                id=ep.id, content=ep.summary, scope=scope,
                categories=["episode"], importance=ep.significance,
                created_at=ep.occurred_at, last_accessed=now,
                metadata={"scope_origin": response.scope_map.get(ep.id, "")},
            ), ep.significance))

        for em in response.emotions:
            results.append((MemoryRecord(
                id=em.id, content=f"{em.emotion_type}: {em.context}",
                scope=scope, categories=["emotion"], importance=em.intensity,
                created_at=em.detected_at, last_accessed=now,
                metadata={"emotion_type": em.emotion_type,
                          "scope_origin": response.scope_map.get(em.id, "")},
            ), em.intensity))

        for te in response.temporal_events:
            results.append((MemoryRecord(
                id=te.id, content=te.content, scope=scope,
                categories=["temporal_event"],
                importance=te.temporal_confidence,
                created_at=te.event_date, last_accessed=now,
                metadata={
                    "valid_until": str(te.valid_until) if te.valid_until else None,
                    "scope_origin": response.scope_map.get(te.id, ""),
                },
            ), te.temporal_confidence))

        results = [(r, s) for r, s in results if s >= min_score]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # Unsupported mutations — no-op with warning (see module docstring)
    # ------------------------------------------------------------------

    def delete(
        self,
        scope_prefix: Optional[str] = None,
        categories: Optional[List[str]] = None,
        record_ids: Optional[List[str]] = None,
        older_than: Optional[datetime] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> int:
        logger.warning(_MUTATION_UNSUPPORTED_WARNING, "delete", "delete")
        return 0

    async def adelete(self, **kwargs) -> int:
        return self.delete(**kwargs)

    def update(self, record: MemoryRecord) -> None:
        logger.warning(_MUTATION_UNSUPPORTED_WARNING, "update", "update")

    def reset(self, scope_prefix: Optional[str] = None) -> None:
        logger.warning(
            "SynapStorageBackend.reset called. Server-side memories are "
            "NOT cleared (Synap exposes no public delete API). Clearing "
            "local session view only."
        )
        if scope_prefix:
            keep = {k: v for k, v in self._records.items()
                    if not v.scope.startswith(scope_prefix)}
            self._records = keep
        else:
            self._records.clear()

    # ------------------------------------------------------------------
    # Session-local read views (see one-time warning on first access)
    # ------------------------------------------------------------------

    def get_record(self, record_id: str) -> Optional[MemoryRecord]:
        self._warn_session_only()
        return self._records.get(record_id)

    def list_records(
        self,
        scope_prefix: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        self._warn_session_only()
        records = list(self._records.values())
        if scope_prefix:
            records = [r for r in records if r.scope.startswith(scope_prefix)]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[offset:offset + limit]

    def get_scope_info(self, scope: str) -> ScopeInfo:
        self._warn_session_only()
        records = [r for r in self._records.values() if r.scope.startswith(scope)]
        cats: set = set()
        for r in records:
            cats.update(r.categories)
        return ScopeInfo(
            path=scope,
            record_count=len(records),
            categories=sorted(cats),
            oldest_record=min((r.created_at for r in records), default=None),
            newest_record=max((r.created_at for r in records), default=None),
            child_scopes=[],
        )

    def list_scopes(self, parent: str = "/") -> List[str]:
        self._warn_session_only()
        scopes: set = set()
        for r in self._records.values():
            if r.scope.startswith(parent) and r.scope != parent:
                rest = r.scope[len(parent):]
                next_part = rest.split("/")[0]
                if next_part:
                    scopes.add(parent + next_part)
        return sorted(scopes)

    def list_categories(self, scope_prefix: Optional[str] = None) -> Dict[str, int]:
        self._warn_session_only()
        counts: Dict[str, int] = {}
        for r in self._records.values():
            if scope_prefix and not r.scope.startswith(scope_prefix):
                continue
            for cat in r.categories:
                counts[cat] = counts.get(cat, 0) + 1
        return counts

    def count(self, scope_prefix: Optional[str] = None) -> int:
        self._warn_session_only()
        if not scope_prefix:
            return len(self._records)
        return sum(1 for r in self._records.values()
                   if r.scope.startswith(scope_prefix))
