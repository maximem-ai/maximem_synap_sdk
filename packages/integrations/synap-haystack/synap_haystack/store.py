"""SynapMemoryStore — a Haystack-native memory store backed by Synap.

Mirrors the store-centric pattern of the official ``mem0-haystack`` integration
(``Mem0MemoryStore`` + ``Mem0MemoryRetriever`` + ``Mem0MemoryWriter``): the store
is a plain object that owns *all* Synap SDK interaction, and the retriever/writer
``@component``s are thin wrappers that hold a reference to it. This is the same
shape ``synap_langgraph.SynapStore`` follows for LangGraph — the store is the
center of gravity; everything else delegates to it.

Why a memory store (and not the generic ``ChatMessageStore`` protocol)? Memory —
unlike a verbatim chat log — is *semantic and cross-conversation*: writes are
extracted/synthesised server-side, reads are query-driven. So this implements
Mem0's ``add_memories`` / ``search_memories`` contract rather than Haystack's
recency-based ``ChatMessageStore`` protocol (``write_messages`` /
``retrieve_messages`` keyed by ``chat_history_id``), which is designed for
verbatim turn storage. This is the same reason Mem0 deliberately did not conform
to ``ChatMessageStore``.

Scope:

- A ``user_id`` pins the store to **user scope**. With only a ``customer_id``
  (``user_id=None``) it operates on the **customer-wide shared pool** visible to
  every user in the deployment. At least one must be provided. (Same rule as
  ``synap_langgraph.SynapStore``.)

Error policy (identical to ``SynapStore``):

- Writes surface :class:`SynapIntegrationError` when *every* attempt fails — a
  100% failure rate is a broken pipeline, not a partial result, and must stop
  loudly. Partial failures are returned per-message so callers can branch.
- Reads degrade gracefully — return ``[]`` with an ERROR log so an SDK blip
  doesn't poison an agent turn.

Delete: Synap has no public delete API, so :meth:`delete_memory` /
:meth:`delete_all_memories` warn once and no-op (same pattern as ``SynapStore``
and synap-crewai).

The public read/write methods are **sync** (matching Mem0's surface) and bridge
to the async SDK via ``run_async``; ``a``-prefixed async variants are provided
for callers already inside an event loop.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from haystack import Document
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import SynapIntegrationError, run_async

logger = logging.getLogger(__name__)

# Roles we ingest as memory. Synap extracts from the conversation channel, so
# only user/assistant turns carry signal; system/tool messages are skipped.
_WRITE_ROLES = frozenset(("user", "assistant"))


class SynapMemoryStore:
    """Haystack-native memory store backed by Synap semantic memory.

    Example::

        from maximem_synap import MaximemSynapSDK
        from synap_haystack import SynapMemoryStore
        from haystack.dataclasses import ChatMessage

        sdk = MaximemSynapSDK(api_key="sk-...")
        store = SynapMemoryStore(sdk, user_id="alice", customer_id="acme")

        # Write — extracted server-side into long-term memory:
        store.add_memories(
            messages=[ChatMessage.from_user("I prefer window seats")],
            conversation_id="c1",
        )

        # Read — semantic, query-driven:
        memories = store.search_memories(query="seat preference")
        single = store.search_memories_as_single_message(query="seat preference")

    The store is a plain object, not a Haystack ``@component``. Use
    :class:`~synap_haystack.SynapMemoryRetriever` and
    :class:`~synap_haystack.SynapMemoryWriter` to drive it inside a pipeline.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: Optional[str] = None,
        customer_id: str = "",
        *,
        conversation_id: Optional[str] = None,
        mode: str = "accurate",
        max_results: int = 20,
        include_conversation_context: bool = False,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapMemoryStore requires a non-None sdk")
        if not user_id and not customer_id:
            raise ValueError(
                "SynapMemoryStore requires at least one of user_id (user scope) "
                "or customer_id (customer-wide shared scope)"
            )

        self.sdk = sdk
        self.user_id = user_id or ""
        self.customer_id = customer_id
        # Default conversation for writes; can be overridden per add_memories call.
        self.conversation_id = conversation_id
        self.mode = mode
        self.max_results = max_results
        self.include_conversation_context = include_conversation_context
        self._delete_warned = False

    # ── write ────────────────────────────────────────────────────────────────

    def add_memories(
        self,
        *,
        messages: List[ChatMessage],
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Record chat messages to Synap for server-side memory extraction.

        Returns one result dict per message: ``{"role", "status", ...}`` where
        ``status`` is ``"written"`` (with ``message_id`` when the SDK returns
        one), ``"failed"`` (with ``error``), or ``"skipped"`` (role not in
        user/assistant). Raises :class:`SynapIntegrationError` if *every*
        recordable message fails.
        """
        return run_async(
            self.aadd_memories(
                messages=messages,
                conversation_id=conversation_id,
                user_id=user_id,
                customer_id=customer_id,
            )
        )

    async def aadd_memories(
        self,
        *,
        messages: List[ChatMessage],
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        conv_id = conversation_id or self.conversation_id
        if not conv_id:
            raise ValueError(
                "SynapMemoryStore.add_memories requires a conversation_id "
                "(pass one here or set it on the store)"
            )
        uid = user_id if user_id is not None else self.user_id
        cid = customer_id if customer_id is not None else self.customer_id

        results: List[Dict[str, Any]] = []
        written = 0
        failed = 0
        first_error: Optional[str] = None

        for msg in messages:
            role = _role_of(msg)
            if role not in _WRITE_ROLES:
                results.append({"role": role, "status": "skipped"})
                logger.info(
                    "SynapMemoryStore.add_memories: skipping message with "
                    "unsupported role=%r (expected one of %s)",
                    role,
                    sorted(_WRITE_ROLES),
                )
                continue

            try:
                resp = await self.sdk.conversation.record_message(
                    conversation_id=conv_id,
                    role=role,
                    content=msg.text or "",
                    user_id=uid or None,
                    customer_id=cid,
                )
                written += 1
                results.append({
                    "role": role,
                    "status": "written",
                    "message_id": _message_id(resp),
                })
            except Exception as exc:  # noqa: BLE001 — boundary
                failed += 1
                logger.error(
                    "SynapMemoryStore.add_memories: record_message failed "
                    "conversation_id=%s role=%s error=%s",
                    conv_id, role, exc, exc_info=True,
                )
                err = f"{type(exc).__name__}: {exc}"
                if first_error is None:
                    first_error = err
                results.append({"role": role, "status": "failed", "error": err})

        processed = written + failed
        if processed > 0 and written == 0:
            # Every recordable message failed — broken pipeline, not partial.
            raise SynapIntegrationError(
                "haystack.SynapMemoryStore.add_memories",
                f"all {failed} record_message attempts failed; "
                f"first error: {first_error}",
                {"conversation_id": conv_id, "failed": failed},
            )

        return results

    # ── read ─────────────────────────────────────────────────────────────────

    def search_memories(
        self,
        *,
        query: str,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        max_results: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> List[ChatMessage]:
        """Semantic, query-driven memory retrieval.

        Returns one assistant :class:`ChatMessage` per memory, with ``meta``
        carrying ``type`` (fact/preference/episode/emotion/temporal_event),
        ``id``, ``scope``, and a per-type score field. Degrades to ``[]`` on
        SDK failure.
        """
        return run_async(self.asearch_memories(
            query=query, user_id=user_id, customer_id=customer_id,
            max_results=max_results, mode=mode,
        ))

    def search_memories_as_single_message(
        self,
        *,
        query: str,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        max_results: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> Optional[ChatMessage]:
        """Collapse retrieved memories into one system :class:`ChatMessage`,
        ready to prepend to a prompt. Returns ``None`` when nothing matched.
        """
        return run_async(self._asearch_single(
            query=query, user_id=user_id, customer_id=customer_id,
            max_results=max_results, mode=mode,
        ))

    async def asearch_memories(
        self,
        *,
        query: str,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        max_results: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> List[ChatMessage]:
        records = await self._asearch_records(
            query=query, user_id=user_id, customer_id=customer_id,
            max_results=max_results, mode=mode,
        )
        return [ChatMessage.from_assistant(r["content"], meta=r["meta"]) for r in records]

    def search_documents(
        self,
        *,
        query: str,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        max_results: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> List[Document]:
        """Same retrieval as :meth:`search_memories` but returns Haystack
        ``Document``s — the RAG-shaped read path used by
        :class:`~synap_haystack.SynapRetriever`.
        """
        return run_async(self.asearch_documents(
            query=query, user_id=user_id, customer_id=customer_id,
            max_results=max_results, mode=mode,
        ))

    async def asearch_documents(
        self,
        *,
        query: str,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        max_results: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> List[Document]:
        records = await self._asearch_records(
            query=query, user_id=user_id, customer_id=customer_id,
            max_results=max_results, mode=mode,
        )
        return [Document(content=r["content"], meta=r["meta"]) for r in records]

    # ── delete (no public Synap delete API) ────────────────────────────────────

    def delete_memory(self, memory_id: str, **kwargs: Any) -> None:
        """No-op: Synap has no public delete API (warns once)."""
        self._warn_delete()

    def delete_all_memories(self, **kwargs: Any) -> None:
        """No-op: Synap has no public delete API (warns once)."""
        self._warn_delete()

    def _warn_delete(self) -> None:
        if not self._delete_warned:
            logger.warning(
                "SynapMemoryStore: Synap has no public delete API. Delete "
                "operations are no-ops; memory is write-only. This warning "
                "fires once."
            )
            self._delete_warned = True

    # ── serialization (Haystack pipeline save/load) ────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config for Haystack pipeline persistence.

        The live SDK is **not** serialized (it holds credentials and is a
        process-local singleton). Only ``instance_id`` is recorded;
        :meth:`from_dict` re-resolves the SDK from the in-process
        ``MaximemSynapSDK`` registry by that id. Persist API keys via env /
        secrets, not the pipeline YAML.
        """
        return default_to_dict(
            self,
            instance_id=getattr(self.sdk, "instance_id", ""),
            user_id=self.user_id or None,
            customer_id=self.customer_id,
            conversation_id=self.conversation_id,
            mode=self.mode,
            max_results=self.max_results,
            include_conversation_context=self.include_conversation_context,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynapMemoryStore":
        init = dict(data.get("init_parameters", {}))
        instance_id = init.pop("instance_id", "") or ""
        sdk = MaximemSynapSDK(instance_id=instance_id)
        data = {**data, "init_parameters": {**init, "sdk": sdk}}
        return default_from_dict(cls, data)

    # ── internals ──────────────────────────────────────────────────────────────

    async def _asearch_records(
        self,
        *,
        query: str,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        max_results: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        uid = user_id if user_id is not None else self.user_id
        cid = customer_id if customer_id is not None else self.customer_id
        try:
            response = await self.sdk.fetch(
                user_id=uid or None,
                customer_id=cid or None,
                search_query=[query] if query else None,
                max_results=max_results or self.max_results,
                mode=mode or self.mode,
                include_conversation_context=self.include_conversation_context,
            )
        except Exception as exc:  # noqa: BLE001 — read-side degrades gracefully
            logger.error(
                "SynapMemoryStore.search: sdk.fetch failed query=%s error=%s",
                query, exc, exc_info=True,
            )
            return []
        return _records_from_response(response)

    async def _asearch_single(self, **kw: Any) -> Optional[ChatMessage]:
        records = await self._asearch_records(**kw)
        if not records:
            return None
        body = "\n".join(f"- {r['content']}" for r in records if r["content"])
        if not body:
            return None
        return ChatMessage.from_system(f"Relevant memory:\n{body}")


# ── helpers ──────────────────────────────────────────────────────────────────


def _role_of(message: ChatMessage) -> str:
    role = getattr(message, "role", "user")
    return getattr(role, "value", role)


def _message_id(resp: Any) -> Optional[str]:
    if isinstance(resp, dict):
        return resp.get("message_id")
    return getattr(resp, "message_id", None)


def _records_from_response(response: Any) -> List[Dict[str, Any]]:
    """Flatten a Synap fetch response into ``{content, meta}`` records across
    *all* memory types — facts, preferences, episodes, emotions,
    temporal_events — not just facts. Reading only ``facts`` silently drops
    stated preferences (and other types), which Synap routes into their own
    lists.
    """
    scope_map = getattr(response, "scope_map", None) or {}
    records: List[Dict[str, Any]] = []

    for fact in getattr(response, "facts", None) or []:
        records.append({
            "content": fact.content,
            "meta": {"type": "fact", "id": fact.id, "confidence": fact.confidence,
                     "scope": scope_map.get(fact.id, "")},
        })
    for pref in getattr(response, "preferences", None) or []:
        records.append({
            "content": pref.content,
            "meta": {"type": "preference", "id": pref.id, "strength": pref.strength,
                     "scope": scope_map.get(pref.id, "")},
        })
    for ep in getattr(response, "episodes", None) or []:
        records.append({
            "content": ep.summary,
            "meta": {"type": "episode", "id": ep.id, "significance": ep.significance,
                     "scope": scope_map.get(ep.id, "")},
        })
    for em in getattr(response, "emotions", None) or []:
        records.append({
            "content": f"{em.emotion_type}: {em.context}",
            "meta": {"type": "emotion", "id": em.id, "intensity": em.intensity,
                     "scope": scope_map.get(em.id, "")},
        })
    for te in getattr(response, "temporal_events", None) or []:
        records.append({
            "content": te.content,
            "meta": {"type": "temporal_event", "id": te.id,
                     "scope": scope_map.get(te.id, "")},
        })

    return records
