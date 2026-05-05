"""SynapDb — Agno BaseDb extension that routes user memories through Synap.

Why extend :class:`InMemoryDb` rather than :class:`BaseDb` directly? BaseDb
is a unified interface covering sessions, traces, evals, metrics, knowledge,
culture, learnings, etc. — 46+ abstract methods. Synap natively backs only
the user-memory subset. Extending InMemoryDb keeps every other concern
working (same as Agno's default DX) while we override just the memory
methods. Users who need durable sessions/traces layer a real DB (Postgres,
Sqlite) underneath; this adapter's scope is memory.

Error policy:
- reads (get_user_memory, get_user_memories) degrade gracefully: log at
  ERROR, return empty result — a Synap blip shouldn't crash an agent turn.
- writes (upsert_user_memory, upsert_memories) surface SynapIntegrationError
  so ingestion outages are observable.
- deletes (delete_user_memory, delete_user_memories, clear_memories) warn
  once and no-op — Synap has no public delete API. Same contract used by
  synap-crewai and SynapStore.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from agno.db.in_memory.in_memory_db import InMemoryDb
from agno.db.schemas.memory import UserMemory
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import wrap_sdk_errors_async, run_async

logger = logging.getLogger(__name__)

# Metadata marker — we tag every Synap memory we create so we can tell
# Synap-originated records apart from non-agno memories in the user's scope.
_MARKER = "agno_user_memory"


class SynapDb(InMemoryDb):
    """Agno BaseDb backed by Synap for user-memory ops.

    Args:
        sdk: Configured :class:`MaximemSynapSDK`.
        customer_id: Optional customer/org scope. When empty, Synap treats
            requests as customer-less.
        mode: Synap fetch mode; ``"accurate"`` (default) or ``"fast"``.
        max_results: Default cap on ``sdk.fetch`` when Agno doesn't specify.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        customer_id: str = "",
        *,
        mode: str = "accurate",
        max_results: int = 50,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapDb requires a non-None sdk")
        super().__init__()
        self.sdk = sdk
        self.customer_id = customer_id
        self.mode = mode
        self.max_results = max_results
        self._delete_warned = False
        self._stats_warned = False

    # ── writes ─────────────────────────────────────────────────────────────

    def upsert_user_memory(
        self,
        memory: UserMemory,
        deserialize: Optional[bool] = True,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        if memory.memory_id is None:
            memory.memory_id = str(uuid4())
        now = int(time.time())
        memory.updated_at = now
        if memory.created_at is None:
            memory.created_at = now

        run_async(self._aupsert(memory))

        # Match InMemoryDb return contract: UserMemory when deserialize=True,
        # dict otherwise.
        if deserialize:
            return memory
        return memory.to_dict() if hasattr(memory, "to_dict") else dict(memory.__dict__)

    def upsert_memories(
        self,
        memories: List[UserMemory],
        deserialize: Optional[bool] = True,
        preserve_updated_at: bool = False,
    ) -> List[Union[UserMemory, Dict[str, Any]]]:
        out: List[Union[UserMemory, Dict[str, Any]]] = []
        now = int(time.time())
        for m in memories:
            if m.memory_id is None:
                m.memory_id = str(uuid4())
            if not preserve_updated_at:
                m.updated_at = now
            if m.created_at is None:
                m.created_at = now
            run_async(self._aupsert(m))
            if deserialize:
                out.append(m)
            else:
                out.append(m.to_dict() if hasattr(m, "to_dict") else dict(m.__dict__))
        return out

    async def _aupsert(self, memory: UserMemory) -> None:
        metadata: Dict[str, Any] = {
            _MARKER: True,
            "memory_id": memory.memory_id,
            "user_id": memory.user_id,
            "agent_id": memory.agent_id,
            "team_id": memory.team_id,
            "input": memory.input,
            "feedback": memory.feedback,
            "topics": list(memory.topics) if memory.topics else [],
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
        }
        async with wrap_sdk_errors_async(
            "agno.upsert_user_memory",
            logger,
            memory_id=memory.memory_id,
            user_id=memory.user_id,
        ):
            await self.sdk.memories.create(
                document=memory.memory,
                user_id=memory.user_id,
                customer_id=self.customer_id or None,
                metadata=metadata,
            )

    # ── reads ──────────────────────────────────────────────────────────────

    def get_user_memory(
        self,
        memory_id: str,
        deserialize: Optional[bool] = True,
        user_id: Optional[str] = None,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        facts = run_async(self._afetch_facts(
            user_id=user_id,
            search_query=[memory_id],
            limit=self.max_results,
        ))
        for fact in facts:
            md = getattr(fact, "metadata", None) or {}
            if md.get(_MARKER) and md.get("memory_id") == memory_id:
                if user_id and md.get("user_id") != user_id:
                    continue
                memory = _fact_to_user_memory(fact)
                return memory if deserialize else memory.to_dict()
        return None

    def get_user_memories(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        topics: Optional[List[str]] = None,
        search_content: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        deserialize: Optional[bool] = True,
    ) -> Union[List[UserMemory], Tuple[List[Dict[str, Any]], int]]:
        fetch_limit = (limit or self.max_results) + ((page or 0) * (limit or 0))
        facts = run_async(self._afetch_facts(
            user_id=user_id,
            search_query=[search_content] if search_content else None,
            limit=max(fetch_limit, self.max_results),
        ))

        matched: List[UserMemory] = []
        for fact in facts:
            md = getattr(fact, "metadata", None) or {}
            if not md.get(_MARKER):
                continue
            if user_id and md.get("user_id") != user_id:
                continue
            if agent_id and md.get("agent_id") != agent_id:
                continue
            if team_id and md.get("team_id") != team_id:
                continue
            if topics:
                fact_topics = md.get("topics") or []
                if not any(t in fact_topics for t in topics):
                    continue
            matched.append(_fact_to_user_memory(fact))

        # Sort (Synap returns by relevance; Agno exposes sort_by/sort_order
        # semantics against created_at/updated_at — apply them client-side).
        if sort_by in {"created_at", "updated_at"}:
            reverse = (sort_order or "desc") == "desc"
            matched.sort(key=lambda m: getattr(m, sort_by) or 0, reverse=reverse)

        # Pagination
        if limit is not None:
            start = (page or 0) * limit
            matched = matched[start : start + limit]

        if deserialize:
            return matched
        return ([m.to_dict() for m in matched], len(matched))

    def get_user_memory_stats(
        self,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        # Synap doesn't expose aggregate stats; warn once then return empty.
        if not self._stats_warned:
            logger.warning(
                "SynapDb.get_user_memory_stats: Synap does not expose "
                "aggregate memory stats; returning empty. This warning "
                "fires once."
            )
            self._stats_warned = True
        return ([], 0)

    def get_all_memory_topics(self, user_id: Optional[str] = None) -> List[str]:
        facts = run_async(self._afetch_facts(
            user_id=user_id,
            search_query=None,
            limit=self.max_results,
        ))
        seen: set[str] = set()
        for fact in facts:
            md = getattr(fact, "metadata", None) or {}
            if not md.get(_MARKER):
                continue
            if user_id and md.get("user_id") != user_id:
                continue
            for t in md.get("topics") or []:
                if isinstance(t, str):
                    seen.add(t)
        return sorted(seen)

    async def _afetch_facts(
        self,
        *,
        user_id: Optional[str],
        search_query: Optional[List[str]],
        limit: int,
    ) -> List[Any]:
        try:
            response = await self.sdk.fetch(
                user_id=user_id,
                customer_id=self.customer_id or None,
                search_query=search_query,
                max_results=limit,
                mode=self.mode,
                include_conversation_context=False,
            )
        except Exception as exc:  # noqa: BLE001 — read-side graceful degrade
            logger.error(
                "SynapDb: sdk.fetch failed user_id=%s error=%s",
                user_id, exc, exc_info=True,
            )
            return []
        return list(getattr(response, "facts", None) or [])

    # ── deletes (warn + no-op) ─────────────────────────────────────────────

    def delete_user_memory(self, memory_id: str, user_id: Optional[str] = None) -> None:
        self._warn_delete_once()

    def delete_user_memories(
        self,
        memory_ids: List[str],
        user_id: Optional[str] = None,
    ) -> None:
        self._warn_delete_once()

    def clear_memories(self) -> None:
        self._warn_delete_once()

    def _warn_delete_once(self) -> None:
        if not self._delete_warned:
            logger.warning(
                "SynapDb: Synap has no public delete API. delete_user_memory, "
                "delete_user_memories, and clear_memories are no-ops. This "
                "warning fires once."
            )
            self._delete_warned = True


# ── helpers ────────────────────────────────────────────────────────────────


def _fact_to_user_memory(fact: Any) -> UserMemory:
    md = getattr(fact, "metadata", None) or {}
    return UserMemory(
        memory=getattr(fact, "content", "") or "",
        memory_id=md.get("memory_id") or getattr(fact, "id", None),
        topics=list(md.get("topics")) if isinstance(md.get("topics"), list) else None,
        user_id=md.get("user_id"),
        input=md.get("input"),
        created_at=md.get("created_at"),
        updated_at=md.get("updated_at"),
        feedback=md.get("feedback"),
        agent_id=md.get("agent_id"),
        team_id=md.get("team_id"),
    )
