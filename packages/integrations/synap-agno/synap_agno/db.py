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
- deletes (delete_user_memory, delete_user_memories) call ``memories.delete``
  for the Synap ids ``get_user_memories`` reports; any other id and
  ``clear_memories`` warn once and no-op. A failed delete is logged.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from agno.db.in_memory.in_memory_db import InMemoryDb
from agno.db.schemas.memory import UserMemory
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import wrap_sdk_errors_async, run_blocking

logger = logging.getLogger(__name__)

# Metadata marker — we tag every Synap memory we create so we can tell
# Synap-originated records apart from non-agno memories in the user's scope.
_MARKER = "agno_user_memory"

DEFAULT_QUERY = "What do you know about this user?"


class SynapDb(InMemoryDb):
    """Agno BaseDb backed by Synap for user-memory ops.

    Args:
        sdk: Configured :class:`MaximemSynapSDK`.
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on a
            B2C instance (user_context_isolation=equals_customer): the server
            rejects a call carrying one with HTTP 400. Leave it unset there. When empty, Synap treats
            requests as customer-less.
        mode: Synap fetch mode; ``"accurate"`` (default) or ``"fast"``.
        max_results: Default cap on ``sdk.fetch`` when Agno doesn't specify.
        default_query: The question asked when Agno loads a user's memories
            without a search term. Synap returns nothing for a fetch with no
            query and retrieval is relevance-ranked, so this returns a
            best-effort sample of what Synap knows about the user, not a full
            list. Pass a topical ``search_content`` to ``get_user_memories`` for
            specifics.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        customer_id: str = "",
        *,
        mode: str = "accurate",
        max_results: int = 50,
        default_query: str = DEFAULT_QUERY,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapDb requires a non-None sdk")
        if not default_query or not default_query.strip():
            raise ValueError(
                "SynapDb requires a non-empty default_query: Synap returns "
                "nothing for a fetch with no query"
            )
        super().__init__()
        self.sdk = sdk
        self.customer_id = customer_id
        self.mode = mode
        self.max_results = max_results
        self.default_query = default_query.strip()
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

        run_blocking(self._aupsert(memory))

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
            run_blocking(self._aupsert(m))
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
        memory = run_blocking(self._aget_memory(memory_id, user_id))
        if memory is None:
            return None
        return memory if deserialize else memory.to_dict()

    async def _aget_memory(self, memory_id: str, user_id: Optional[str]) -> Optional[UserMemory]:
        """Look a memory up by the Synap id that ``get_user_memories`` reports."""
        try:
            parsed = UUID(str(memory_id))
        except ValueError:
            return None  # not a Synap id (e.g. an id Agno minted itself): nothing to look up
        try:
            found = await self.sdk.memories.get(parsed)
        except Exception as exc:  # noqa: BLE001 — read-side graceful degrade
            logger.error("SynapDb: sdk.memories.get failed memory_id=%s error=%s", memory_id, exc)
            return None
        content = getattr(found, "content", None)
        if not content:
            return None
        return UserMemory(
            memory=content,
            memory_id=str(parsed),
            user_id=user_id,
            created_at=_epoch(getattr(found, "created_at", None)),
            updated_at=_epoch(getattr(found, "updated_at", None)),
        )

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
        matched = run_blocking(self._aget_memories(
            user_id, agent_id, team_id, topics, search_content, limit, page, sort_by, sort_order,
        ))
        if deserialize:
            return matched
        return ([m.to_dict() for m in matched], len(matched))

    async def _aget_memories(
        self,
        user_id: Optional[str],
        agent_id: Optional[str],
        team_id: Optional[str],
        topics: Optional[List[str]],
        search_content: Optional[str],
        limit: Optional[int],
        page: Optional[int],
        sort_by: Optional[str],
        sort_order: Optional[str],
    ) -> List[UserMemory]:
        fetch_limit = (limit or self.max_results) + ((page or 0) * (limit or 0))
        response = await self._afetch_response(
            user_id=user_id,
            search_query=[search_content] if search_content else [self.default_query],
            limit=max(fetch_limit, self.max_results),
        )

        matched: List[UserMemory] = []
        for kind, hit, content, _score in iter_hits(response):
            md = getattr(hit, "metadata", None) or {}
            # Synap scopes by user_id server-side and does not echo back custom
            # metadata, so these filters only apply when the hit carries the field.
            if agent_id and md.get("agent_id") not in (None, agent_id):
                continue
            if team_id and md.get("team_id") not in (None, team_id):
                continue
            if topics and md.get("topics") is not None and not any(t in md["topics"] for t in topics):
                continue
            matched.append(hit_to_user_memory(hit, content, user_id=user_id))

        # Sort (Synap returns by relevance; Agno exposes sort_by/sort_order
        # semantics against created_at/updated_at — apply them client-side).
        if sort_by in {"created_at", "updated_at"}:
            reverse = (sort_order or "desc") == "desc"
            matched.sort(key=lambda m: getattr(m, sort_by) or 0, reverse=reverse)

        # Pagination
        if limit is not None:
            start = (page or 0) * limit
            matched = matched[start : start + limit]
        return matched

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
        return run_blocking(self._aget_topics(user_id))

    async def _aget_topics(self, user_id: Optional[str]) -> List[str]:
        # Topics are Agno-side metadata that Synap does not echo back, so this is
        # only non-empty when a hit happens to carry them.
        response = await self._afetch_response(
            user_id=user_id, search_query=[self.default_query], limit=self.max_results,
        )
        seen: set[str] = set()
        for _kind, hit, _content, _score in iter_hits(response):
            for t in (getattr(hit, "metadata", None) or {}).get("topics") or []:
                if isinstance(t, str):
                    seen.add(t)
        return sorted(seen)

    async def _afetch_response(
        self,
        *,
        user_id: Optional[str],
        search_query: Optional[List[str]],
        limit: int,
    ) -> Any:
        try:
            return await self.sdk.fetch(
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
            return None

    # ── deletes ────────────────────────────────────────────────────────────

    def delete_user_memory(self, memory_id: str, user_id: Optional[str] = None) -> None:
        run_blocking(self._adelete([memory_id]))

    def delete_user_memories(
        self,
        memory_ids: List[str],
        user_id: Optional[str] = None,
    ) -> None:
        run_blocking(self._adelete(memory_ids))

    def clear_memories(self) -> None:
        self._warn_delete_once()

    async def _adelete(self, memory_ids: List[str]) -> None:
        """Delete the memories ``get_user_memories`` reported, by their Synap id.

        An id Agno minted itself (``upsert_user_memory`` assigns a uuid4 before
        Synap sees the text) names nothing in Synap and is skipped with the
        one-time warning. A failed delete is logged and does not interrupt the
        agent.
        """
        for memory_id in memory_ids:
            try:
                parsed = UUID(str(memory_id))
            except ValueError:
                self._warn_delete_once()
                continue
            try:
                await self.sdk.memories.delete(parsed)
            except Exception as exc:  # noqa: BLE001: a failed delete must not break a run
                logger.error("SynapDb: sdk.memories.delete failed memory_id=%s error=%s", memory_id, exc)

    def _warn_delete_once(self) -> None:
        if not self._delete_warned:
            logger.warning(
                "SynapDb: Synap has no public delete API for clear_memories or "
                "for ids that did not come from get_user_memories; those are "
                "no-ops. This warning fires once."
            )
            self._delete_warned = True


# ── helpers ────────────────────────────────────────────────────────────────


def iter_hits(response: Any):
    """Yield ``(kind, hit, text, score)`` for every memory type in a fetch response."""
    if response is None:
        return
    for fact in getattr(response, "facts", None) or []:
        yield "fact", fact, getattr(fact, "content", None), getattr(fact, "confidence", None)
    for pref in getattr(response, "preferences", None) or []:
        yield "preference", pref, getattr(pref, "content", None), getattr(pref, "strength", None)
    for episode in getattr(response, "episodes", None) or []:
        yield "episode", episode, getattr(episode, "summary", None), getattr(episode, "significance", None)
    for emotion in getattr(response, "emotions", None) or []:
        etype, ctx = getattr(emotion, "emotion_type", None), getattr(emotion, "context", None)
        yield "emotion", emotion, (f"{etype}: {ctx}" if etype and ctx else (ctx or etype)), getattr(emotion, "intensity", None)
    for event in getattr(response, "temporal_events", None) or []:
        score = getattr(event, "temporal_confidence", None)
        yield "temporal_event", event, getattr(event, "content", None), (score if score is not None else getattr(event, "confidence", None))


def _epoch(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    ts = getattr(value, "timestamp", None)
    return int(ts()) if callable(ts) else None


def hit_to_user_memory(hit: Any, content: Optional[str], *, user_id: Optional[str]) -> UserMemory:
    md = getattr(hit, "metadata", None) or {}
    when = (
        getattr(hit, "extracted_at", None)
        or getattr(hit, "occurred_at", None)
        or getattr(hit, "detected_at", None)
        or getattr(hit, "event_date", None)
    )
    return UserMemory(
        memory=content or "",
        memory_id=md.get("memory_id") or getattr(hit, "id", None),
        topics=list(md.get("topics")) if isinstance(md.get("topics"), list) else None,
        user_id=md.get("user_id") or user_id,
        input=md.get("input"),
        created_at=md.get("created_at") if md.get("created_at") is not None else _epoch(when),
        updated_at=md.get("updated_at") if md.get("updated_at") is not None else _epoch(when),
        feedback=md.get("feedback"),
        agent_id=md.get("agent_id"),
        team_id=md.get("team_id"),
    )
