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

## The live stream

``aadd_memories`` is the single funnel every write passes through (the
:class:`~synap_haystack.SynapMemoryWriter` component delegates to it), so it is
where a turn is reported to Synap's live gRPC stream. A Haystack
``ChatMessage`` is a list of content parts, and the parts carry everything the
anticipation agent needs beyond the text: ``ToolCall(tool_name, arguments,
id)`` on an assistant message, ``ToolCallResult(result, origin, error)`` on a
tool message, and ``ReasoningContent(reasoning_text)`` on either.
``ToolCall.id`` is Haystack's own id for the invocation and the result repeats
it as ``origin.id``, which is what ties a call to its result.

Only the turns are also memory, and they follow the one rule that matters:
**stream first, REST only as a fallback, never both**. The server persists
``user_message`` and ``assistant_message`` straight off the stream
(``grpc/servicer.py``), so a ``record_message`` on top of a delivered stream
event writes the turn twice and extracts it twice. Tool calls, tool results
and reasoning are stream-only: there is no REST equivalent and none of them is
a memory. Tool and system messages stay ``"skipped"`` for the write contract,
but a tool message's results are still reported.

The content parts are read through ``getattr`` rather than imported, so a
``haystack-ai`` old enough to predate tool calling or ``ReasoningContent``
reports nothing extra instead of failing to import.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from haystack import Document
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import SynapIntegrationError, run_async
from synap_integrations_common.stream_events import (
    report_reasoning,
    report_tool_call,
    report_tool_result,
    report_turn,
)

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
        ``status`` is ``"written"``, ``"failed"`` (with ``error``), or
        ``"skipped"`` (role not in user/assistant). A written message also
        carries ``transport``: ``"stream"`` when the turn went out on the live
        gRPC stream, which persists it server-side and leaves ``message_id``
        ``None``, or ``"rest"`` when it fell back to ``record_message``, which
        returns one. Raises :class:`SynapIntegrationError` if *every*
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

        ids = {"conversation_id": conv_id, "user_id": uid or "", "customer_id": cid or ""}

        for msg in messages:
            role = _role_of(msg)

            # ── the live stream ──────────────────────────────────────────
            # Reported for every role, including the ones that are never
            # written as memory: a tool result is not a turn, but it is
            # exactly what the anticipation agent needs to see.
            await self._report_reasoning(msg, ids)
            await self._report_tool_results(msg, ids)

            if role not in _WRITE_ROLES:
                results.append({"role": role, "status": "skipped"})
                logger.info(
                    "SynapMemoryStore.add_memories: skipping message with "
                    "unsupported role=%r (expected one of %s)",
                    role,
                    sorted(_WRITE_ROLES),
                )
                continue

            # Stream first, REST only as a fallback, never both.
            #
            # ⚠ Only when there is a user_id. The server refuses to persist a
            # conversation event without one and says so in its own log, and
            # `report_turn` answers "it went out", not "it was stored". A
            # customer-scoped store (user_id=None, customer-wide shared pool)
            # would therefore report a turn, believe it landed, skip the
            # fallback and lose it silently. REST fails loudly instead, which
            # is what this store did before there was a stream at all.
            if uid and await report_turn(
                self.sdk, role=role, content=msg.text or "", **ids
            ):
                written += 1
                results.append({
                    "role": role,
                    "status": "written",
                    "message_id": None,
                    "transport": "stream",
                })
            else:
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
                        "transport": "rest",
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

            # After the turn: the model says its piece, then asks for tools.
            await self._report_tool_calls(msg, ids)

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

    # ── the three events this store never reported ──────────────────────────
    #
    # None of these can raise into a Haystack pipeline: `stream_events`
    # swallows everything by design, and all three are silent when no stream
    # is open.

    async def _report_reasoning(self, msg: Any, ids: Dict[str, str]) -> None:
        """Report each ``ReasoningContent`` part of a message."""
        parts = getattr(msg, "reasonings", None)
        if not isinstance(parts, (list, tuple)):
            return
        step = 0
        for part in parts:
            content = str(getattr(part, "reasoning_text", "") or "").strip()
            if not content:
                continue
            await report_reasoning(
                self.sdk, content=content,
                step_index=step, thought_type="reasoning", **ids,
            )
            step += 1

    async def _report_tool_calls(self, msg: Any, ids: Dict[str, str]) -> None:
        """Report each ``ToolCall`` part of an assistant message.

        ``ToolCall.id`` is Haystack's own id for the invocation, and the
        ``ToolCallResult`` that follows repeats it as ``origin.id``. Without
        it the anticipation agent cannot tell a call and a result belong
        together.
        """
        calls = getattr(msg, "tool_calls", None)
        if not isinstance(calls, (list, tuple)):
            return
        for call in calls:
            args = getattr(call, "arguments", None)
            await report_tool_call(
                self.sdk,
                tool_name=str(getattr(call, "tool_name", "") or "tool"),
                tool_args=args if isinstance(args, dict) else None,
                tool_call_id=str(getattr(call, "id", "") or ""),
                **ids,
            )

    async def _report_tool_results(self, msg: Any, ids: Dict[str, str]) -> None:
        """Report each ``ToolCallResult`` part of a tool message."""
        outcomes = getattr(msg, "tool_call_results", None)
        if not isinstance(outcomes, (list, tuple)):
            return
        for outcome in outcomes:
            origin = getattr(outcome, "origin", None)
            await report_tool_result(
                self.sdk,
                result=getattr(outcome, "result", None),
                tool_name=str(getattr(origin, "tool_name", "") or ""),
                tool_call_id=str(getattr(origin, "id", "") or ""),
                **ids,
            )

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
