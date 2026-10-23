"""``AsyncSynapDb``: Agno's async database interface backed by Synap.

Agno picks its code path by the database type. A sync ``BaseDb`` used from
``agent.arun`` is called inside the running event loop, so every Synap round trip
blocks that loop. ``AsyncSynapDb`` subclasses :class:`agno.db.base.AsyncBaseDb`, so
Agno awaits it directly; use it for async agents (``arun``, ``aprint_response``) and
:class:`synap_agno.SynapDb` for sync ones.

Like :class:`SynapDb`, only user memories go to Synap. Sessions, runs, traces and the
rest of Agno's persistence stay in-process, exactly as Agno's ``InMemoryDb`` does.
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from agno.db.base import AsyncBaseDb
from agno.db.in_memory.in_memory_db import InMemoryDb
from agno.db.schemas.memory import UserMemory
from maximem_synap import MaximemSynapSDK

from synap_agno.db import DEFAULT_QUERY, SynapDb

logger = logging.getLogger(__name__)


class AsyncSynapDb(AsyncBaseDb):
    """Agno ``AsyncBaseDb`` backed by Synap for user-memory operations.

    Args:
        sdk: Configured, initialised :class:`MaximemSynapSDK`.
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on a
            B2C instance. When empty, Synap treats requests as customer-less.
        mode: Synap fetch mode; ``"accurate"`` (default) or ``"fast"``.
        max_results: Default cap on ``sdk.fetch`` when Agno doesn't specify.
        default_query: The question asked when Agno loads a user's memories
            without a search term (see :class:`SynapDb`).
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
            raise ValueError("AsyncSynapDb requires a non-None sdk")
        super().__init__()
        # The sync class holds the Synap logic and the in-process stores; this class
        # awaits its async internals directly and never goes through run_async.
        self._synap = SynapDb(
            sdk, customer_id, mode=mode, max_results=max_results, default_query=default_query
        )
        self._inner = InMemoryDb()
        self.sdk = sdk

    # ── user memories ──────────────────────────────────────────────────────

    async def upsert_user_memory(
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
        await self._synap._aupsert(memory)
        return memory if deserialize else memory.to_dict()

    async def upsert_memories(
        self,
        memories: List[UserMemory],
        deserialize: Optional[bool] = True,
        preserve_updated_at: bool = False,
    ) -> List[Union[UserMemory, Dict[str, Any]]]:
        out: List[Union[UserMemory, Dict[str, Any]]] = []
        for memory in memories:
            if memory.memory_id is None:
                memory.memory_id = str(uuid4())
            now = int(time.time())
            if not preserve_updated_at:
                memory.updated_at = now
            if memory.created_at is None:
                memory.created_at = now
            await self._synap._aupsert(memory)
            out.append(memory if deserialize else memory.to_dict())
        return out

    async def get_user_memory(
        self,
        memory_id: str,
        deserialize: Optional[bool] = True,
        user_id: Optional[str] = None,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        memory = await self._synap._aget_memory(memory_id, user_id)
        if memory is None:
            return None
        return memory if deserialize else memory.to_dict()

    async def get_user_memories(
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
        matched = await self._synap._aget_memories(
            user_id, agent_id, team_id, topics, search_content, limit, page, sort_by, sort_order,
        )
        if deserialize:
            return matched
        return ([m.to_dict() for m in matched], len(matched))

    async def get_all_memory_topics(self, user_id: Optional[str] = None) -> List[str]:
        return await self._synap._aget_topics(user_id)

    async def get_user_memory_stats(
        self,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        return self._synap.get_user_memory_stats(limit=limit, page=page, user_id=user_id)

    async def delete_user_memory(self, memory_id: str, user_id: Optional[str] = None) -> None:
        await self._synap._adelete([memory_id])

    async def delete_user_memories(self, memory_ids: List[str], user_id: Optional[str] = None) -> None:
        await self._synap._adelete(memory_ids)

    async def clear_memories(self) -> None:
        self._synap._warn_delete_once()


def _delegate(name: str):
    async def method(self: AsyncSynapDb, *args: Any, **kwargs: Any) -> Any:
        return getattr(self._inner, name)(*args, **kwargs)

    method.__name__ = name
    method.__qualname__ = f"AsyncSynapDb.{name}"
    method.__doc__ = f"Kept in-process (Synap only backs user memories); mirrors InMemoryDb.{name}."
    return method


# Everything that is not a user memory (sessions, runs, traces, evals, knowledge...)
# stays in-process, awaited as a trivial wrapper around the sync InMemoryDb method.
for _name in sorted(AsyncBaseDb.__abstractmethods__):
    if _name not in AsyncSynapDb.__dict__:
        setattr(AsyncSynapDb, _name, _delegate(_name))
abc.update_abstractmethods(AsyncSynapDb)

__all__ = ["AsyncSynapDb"]
