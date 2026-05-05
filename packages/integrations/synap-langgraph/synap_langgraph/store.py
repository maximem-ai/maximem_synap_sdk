"""SynapStore — LangGraph ``BaseStore`` backed by Synap semantic memory.

Implements ``batch`` / ``abatch`` (the only abstract methods) and dispatches
individual ops (:class:`PutOp`, :class:`GetOp`, :class:`SearchOp`,
:class:`ListNamespacesOp`) to async helpers. The convenience API
(``put``/``get``/``search``/``delete``/``list_namespaces`` + async variants)
is inherited unchanged from ``BaseStore`` and rides through ``(a)batch``.

Storage strategy:

- Each ``put`` is ingested as a Synap memory via ``sdk.memories.create`` with
  the value JSON-encoded in ``document`` and ``metadata`` tagging the
  namespace tuple (joined by ``/``) plus the key.
- ``get`` retrieves via ``sdk.fetch`` with ``search_query=[namespace+key]``
  and filters the returned facts by the namespace/key metadata markers.
- ``search`` uses ``sdk.fetch(search_query=[query])`` and filters by
  namespace-prefix in metadata. Scores flow through from ``confidence``.
- ``delete`` and ``list_namespaces`` warn + no-op (Synap has no public
  delete API, same pattern as synap-crewai).

Error policy:

- Writes surface ``SynapIntegrationError`` on SDK failure — silent drops
  would hide ingestion outages.
- Reads degrade gracefully — return ``None``/``[]`` with an ERROR log so an
  SDK blip doesn't poison an agent turn.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    SynapIntegrationError,
    run_async,
    wrap_sdk_errors_async,
)

logger = logging.getLogger(__name__)

# Metadata keys we stamp on every Synap memory we create through SynapStore
# so we can identify / filter them on retrieval.
_MARKER = "lg_store"
_NS = "lg_store_ns"
_KEY = "lg_store_key"


def _ns_str(namespace: tuple[str, ...]) -> str:
    """Stringify a namespace tuple for metadata / search tokens."""
    return "/".join(namespace)


def _matches_namespace_prefix(
    item_ns: str,
    prefix: tuple[str, ...],
) -> bool:
    """Return True iff ``item_ns`` sits at or below ``prefix``."""
    if not prefix:
        return True
    pref_str = _ns_str(prefix)
    return item_ns == pref_str or item_ns.startswith(pref_str + "/")


class SynapStore(BaseStore):
    """LangGraph cross-thread long-term memory store backed by Synap.

    Example::

        from langgraph.graph import StateGraph, START, END
        from maximem_synap import MaximemSynapSDK
        from synap_langgraph import SynapStore

        sdk = MaximemSynapSDK(api_key="sk-...")
        store = SynapStore(sdk, user_id="alice", customer_id="acme")

        graph = StateGraph(MyState)
        # ... add nodes / edges ...
        app = graph.compile(store=store)
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: str,
        customer_id: str = "",
        *,
        mode: str = "accurate",
    ) -> None:
        if sdk is None:
            raise ValueError("SynapStore requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapStore requires a non-empty user_id")

        super().__init__()
        self.sdk = sdk
        self.user_id = user_id
        self.customer_id = customer_id
        self.mode = mode
        self._delete_warned = False
        self._listns_warned = False

    # ── abstract methods ────────────────────────────────────────────────────

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return [await self._dispatch(op) for op in ops]

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return run_async(self.abatch(list(ops)))

    # ── dispatcher ──────────────────────────────────────────────────────────

    async def _dispatch(self, op: Op) -> Result:
        if isinstance(op, PutOp):
            # value=None on PutOp signals delete (matches the base-class pattern)
            if op.value is None:
                return await self._adelete(op.namespace, op.key)
            return await self._aput(op.namespace, op.key, op.value, op.index)
        if isinstance(op, GetOp):
            return await self._aget(op.namespace, op.key)
        if isinstance(op, SearchOp):
            return await self._asearch(
                op.namespace_prefix, op.query, op.filter, op.limit, op.offset,
            )
        if isinstance(op, ListNamespacesOp):
            return await self._alist_namespaces(op)
        raise ValueError(f"SynapStore: unsupported op type {type(op).__name__}")

    # ── op implementations ─────────────────────────────────────────────────

    async def _aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Any = None,
    ) -> None:
        ns = _ns_str(namespace)
        document = json.dumps(value, default=str)
        metadata = {
            _MARKER: True,
            _NS: ns,
            _KEY: key,
        }
        async with wrap_sdk_errors_async(
            "langgraph.store.put",
            logger,
            namespace=ns,
            key=key,
        ):
            await self.sdk.memories.create(
                document=document,
                user_id=self.user_id,
                customer_id=self.customer_id or None,
                metadata=metadata,
            )

    async def _aget(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> Optional[Item]:
        ns = _ns_str(namespace)
        try:
            response = await self.sdk.fetch(
                user_id=self.user_id,
                customer_id=self.customer_id or None,
                search_query=[f"{ns} {key}"],
                max_results=50,
                mode=self.mode,
                include_conversation_context=False,
            )
        except Exception as exc:  # noqa: BLE001 — read-side degrades gracefully
            logger.error(
                "SynapStore.get: sdk.fetch failed namespace=%s key=%s error=%s",
                ns, key, exc, exc_info=True,
            )
            return None

        for fact in getattr(response, "facts", None) or []:
            md = getattr(fact, "metadata", None) or {}
            if md.get(_MARKER) and md.get(_NS) == ns and md.get(_KEY) == key:
                return _fact_to_item(fact, namespace, key)
        return None

    async def _asearch(
        self,
        namespace_prefix: tuple[str, ...],
        query: Optional[str],
        filter_: Optional[dict[str, Any]],
        limit: int,
        offset: int,
    ) -> list[SearchItem]:
        q = query or _ns_str(namespace_prefix) or ""
        try:
            response = await self.sdk.fetch(
                user_id=self.user_id,
                customer_id=self.customer_id or None,
                search_query=[q] if q else None,
                max_results=max(limit + offset, 10),
                mode=self.mode,
                include_conversation_context=False,
            )
        except Exception as exc:  # noqa: BLE001 — read-side degrades gracefully
            logger.error(
                "SynapStore.search: sdk.fetch failed namespace_prefix=%s query=%s error=%s",
                _ns_str(namespace_prefix), query, exc, exc_info=True,
            )
            return []

        matches: list[SearchItem] = []
        for fact in getattr(response, "facts", None) or []:
            md = getattr(fact, "metadata", None) or {}
            if not md.get(_MARKER):
                continue
            item_ns_str = md.get(_NS) or ""
            if not _matches_namespace_prefix(item_ns_str, namespace_prefix):
                continue
            if filter_ and not _filter_matches(md, filter_):
                continue
            item_ns = tuple(item_ns_str.split("/")) if item_ns_str else ()
            key = md.get(_KEY) or str(fact.id)
            matches.append(_fact_to_search_item(fact, item_ns, key))

        return matches[offset : offset + limit]

    async def _adelete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        if not self._delete_warned:
            logger.warning(
                "SynapStore.delete: Synap has no public delete API. Delete/update "
                "operations are no-ops; data is write-only. This warning fires once."
            )
            self._delete_warned = True

    async def _alist_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        if not self._listns_warned:
            logger.warning(
                "SynapStore.list_namespaces: Synap does not expose a namespace "
                "enumeration API; returning []. This warning fires once."
            )
            self._listns_warned = True
        return []


# ── helpers ────────────────────────────────────────────────────────────────


def _fact_to_item(fact: Any, namespace: tuple[str, ...], key: str) -> Item:
    value = _parse_value(getattr(fact, "content", "") or "")
    now = _as_datetime(getattr(fact, "extracted_at", None))
    return Item(
        value=value,
        key=key,
        namespace=namespace,
        created_at=now,
        updated_at=now,
    )


def _fact_to_search_item(
    fact: Any,
    namespace: tuple[str, ...],
    key: str,
) -> SearchItem:
    value = _parse_value(getattr(fact, "content", "") or "")
    now = _as_datetime(getattr(fact, "extracted_at", None))
    score = getattr(fact, "confidence", None)
    return SearchItem(
        namespace=namespace,
        key=key,
        value=value,
        created_at=now,
        updated_at=now,
        score=float(score) if isinstance(score, (int, float)) else None,
    )


def _parse_value(document: str) -> dict[str, Any]:
    if not document:
        return {}
    try:
        parsed = json.loads(document)
    except (ValueError, TypeError):
        return {"_raw": document}
    if isinstance(parsed, dict):
        return parsed
    return {"_value": parsed}


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.now(timezone.utc)


# Silence the unused-import complaint — SynapIntegrationError is re-exported
# via wrap_sdk_errors_async raising it. Keep the symbol visible for typing.
_ = SynapIntegrationError
