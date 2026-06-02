"""SynapStore — LangGraph ``BaseStore`` backed by Synap semantic memory.

Implements ``batch`` / ``abatch`` (the only abstract methods) and dispatches
individual ops (:class:`PutOp`, :class:`GetOp`, :class:`SearchOp`,
:class:`ListNamespacesOp`) to async helpers. The convenience API
(``put``/``get``/``search``/``delete``/``list_namespaces`` + async variants)
is inherited unchanged from ``BaseStore`` and rides through ``(a)batch``.

Scope:

- A ``user_id`` pins the store to **user scope**. With only a ``customer_id``
  (``user_id=None``) it operates on the **customer-wide shared pool** visible
  to every user in the deployment. At least one must be provided.

Storage strategy:

- Each ``put`` is ingested as a Synap memory via ``sdk.memories.create`` with
  the value JSON-encoded in ``document`` and ``metadata`` tagging the
  namespace tuple (joined by ``/``) plus the key.
- ``get`` retrieves via ``sdk.fetch`` with ``search_query=[namespace+key]``
  and filters by the namespace/key metadata markers.
- ``search`` uses ``sdk.fetch(search_query=[query])`` and filters by
  namespace-prefix in metadata. Scores flow through from ``confidence``.
- Reads scan **all** memory types Synap returns (facts, preferences,
  episodes, emotions, temporal_events) — not just facts — so stated
  preferences are no longer silently dropped.
- ``delete`` and ``list_namespaces`` warn + no-op (Synap has no public
  delete API, same pattern as synap-crewai).

Anticipation (optional, outside the BaseStore contract): construct with
``include_conversation_context=True`` and drive the conversation channel via
:meth:`SynapStore.record_message` so just-stated context is in play on reads.

Caveat — metadata-stripping backends: ``get``/``search`` match on custom
metadata markers, which instances that atomize content during extraction
(e.g. MACA) strip. When that happens (detected + warned once):
- ``search`` falls back to returning the scope-filtered results Synap ranked
  (semantic retrieval still works; sub-namespace isolation within the
  user/customer scope is not enforced). Disable with ``semantic_fallback=False``.
- ``get`` (exact key) returns ``None`` — there's no reliable way to resolve an
  exact key without the markers, so ``search`` is the supported read path.

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

        # User-scoped private memory:
        store = SynapStore(sdk, user_id="alice", customer_id="acme")
        # Customer-wide shared pool (no user_id):
        shared = SynapStore(sdk, customer_id="acme")

        graph = StateGraph(MyState)
        # ... add nodes / edges ...
        app = graph.compile(store=store)
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: Optional[str] = None,
        customer_id: str = "",
        *,
        mode: str = "accurate",
        include_conversation_context: bool = False,
        semantic_fallback: bool = True,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapStore requires a non-None sdk")
        if not user_id and not customer_id:
            raise ValueError(
                "SynapStore requires at least one of user_id (user scope) or "
                "customer_id (customer-wide shared scope)"
            )

        super().__init__()
        self.sdk = sdk
        self.user_id = user_id or ""
        self.customer_id = customer_id
        self.mode = mode
        self.include_conversation_context = include_conversation_context
        # When the backend strips our namespace/key markers during extraction
        # (e.g. MACA), exact-namespace matching is impossible. With this on
        # (default), ``search`` falls back to returning the scope-filtered
        # results Synap ranked — semantic retrieval still works, but sub-
        # namespace isolation within a scope is NOT enforced (only user/
        # customer scope, applied at the fetch layer). Set False for strict
        # namespace semantics (search returns [] when markers are absent).
        self.semantic_fallback = semantic_fallback
        # A user_id pins the store to user scope; with only a customer_id it
        # operates on the customer-wide shared pool (visible to every user in
        # the deployment). The scope drives both the write owner and which
        # scope `fetch` queries.
        self._scopes = ["user"] if user_id else ["customer"]
        self._delete_warned = False
        self._listns_warned = False
        self._marker_warned = False

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
                user_id=self.user_id or None,
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
                user_id=self.user_id or None,
                customer_id=self.customer_id or None,
                search_query=[f"{ns} {key}"],
                max_results=50,
                mode=self.mode,
                scopes=self._scopes,
                include_conversation_context=self.include_conversation_context,
            )
        except Exception as exc:  # noqa: BLE001 — read-side degrades gracefully
            logger.error(
                "SynapStore.get: sdk.fetch failed namespace=%s key=%s error=%s",
                ns, key, exc, exc_info=True,
            )
            return None

        items = _iter_items(response)
        for item in items:
            md = getattr(item, "metadata", None) or {}
            if md.get(_MARKER) and md.get(_NS) == ns and md.get(_KEY) == key:
                return _item_to_item(item, namespace, key)
        if items and not any((getattr(i, "metadata", None) or {}).get(_MARKER) for i in items):
            self._warn_markers_stripped()
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
                user_id=self.user_id or None,
                customer_id=self.customer_id or None,
                search_query=[q] if q else None,
                max_results=max(limit + offset, 10),
                mode=self.mode,
                scopes=self._scopes,
                include_conversation_context=self.include_conversation_context,
            )
        except Exception as exc:  # noqa: BLE001 — read-side degrades gracefully
            logger.error(
                "SynapStore.search: sdk.fetch failed namespace_prefix=%s query=%s error=%s",
                _ns_str(namespace_prefix), query, exc, exc_info=True,
            )
            return []

        items = _iter_items(response)
        matches: list[SearchItem] = []
        saw_marker = False
        for item in items:
            md = getattr(item, "metadata", None) or {}
            if not md.get(_MARKER):
                continue
            saw_marker = True
            item_ns_str = md.get(_NS) or ""
            if not _matches_namespace_prefix(item_ns_str, namespace_prefix):
                continue
            if filter_ and not _filter_matches(md, filter_):
                continue
            item_ns = tuple(item_ns_str.split("/")) if item_ns_str else ()
            key = md.get(_KEY) or str(getattr(item, "id", ""))
            matches.append(_item_to_search_item(item, item_ns, key))

        if items and not saw_marker:
            self._warn_markers_stripped()
            if self.semantic_fallback:
                # Markers were stripped during extraction, so we can't match by
                # namespace/key. Scope (user/customer) is already enforced by
                # fetch, so return what Synap ranked rather than nothing. Keyed
                # by Synap memory id; namespace stamped as the search prefix.
                # NOTE: sub-namespace isolation within the scope is not enforced
                # in this path (disable via semantic_fallback=False).
                fallback = [
                    _item_to_search_item(it, namespace_prefix, str(getattr(it, "id", "")))
                    for it in items
                ]
                return fallback[offset : offset + limit]
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

    def _warn_markers_stripped(self) -> None:
        if not self._marker_warned:
            logger.warning(
                "SynapStore: fetch returned memories but none carried the "
                "SynapStore markers. This instance is stripping custom metadata "
                "during extraction (e.g. MACA atomization), so exact get/search "
                "by key is unreliable here. This warning fires once."
            )
            self._marker_warned = True

    # ── anticipation (not part of BaseStore) ────────────────────────────────

    async def arecord_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Feed a turn to Synap's conversation channel so anticipation can
        surface just-stated context on subsequent reads (construct the store
        with ``include_conversation_context=True``).

        Anticipation has no key/value analogue, so this lives alongside — not
        inside — the BaseStore contract; it lets a LangGraph node drive the
        conversation channel without reaching for the raw SDK. Best-effort: a
        failure here is logged and swallowed so it never breaks an agent turn.
        """
        if not self.user_id or not self.customer_id:
            logger.warning(
                "SynapStore.record_message needs both user_id and customer_id "
                "(anticipation is user+customer scoped); skipping."
            )
            return
        try:
            await self.sdk.conversation.record_message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                user_id=self.user_id,
                customer_id=self.customer_id,
                session_id=session_id,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 — anticipation is best-effort
            logger.warning("SynapStore.record_message failed (non-fatal): %s", exc)

    def record_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Sync wrapper around :meth:`arecord_message`."""
        run_async(
            self.arecord_message(
                conversation_id, role, content,
                session_id=session_id, metadata=metadata,
            )
        )


# ── helpers ────────────────────────────────────────────────────────────────


def _iter_items(response: Any) -> list[Any]:
    """Every memory item Synap returned, across all types — facts,
    preferences, episodes, emotions, temporal_events — not just ``facts``.

    Reading only ``facts`` silently drops stated *preferences* (and other
    types), which Synap routes into their own lists.
    """
    all_items = getattr(response, "all_items", None)
    if callable(all_items):
        try:
            return list(all_items() or [])
        except Exception:  # noqa: BLE001 — fall back to manual bucket union
            pass
    items: list[Any] = []
    for bucket in ("facts", "preferences", "episodes", "emotions", "temporal_events"):
        items.extend(getattr(response, bucket, None) or [])
    return items


def _filter_matches(metadata: dict[str, Any], filter_: dict[str, Any]) -> bool:
    """LangGraph ``SearchOp.filter`` semantics: every key/value pair in
    ``filter_`` must equal the item's metadata. A list filter value matches
    if the metadata value is one of its members.
    """
    for k, v in filter_.items():
        actual = metadata.get(k)
        if isinstance(v, (list, tuple, set)):
            if actual not in v:
                return False
        elif actual != v:
            return False
    return True


def _item_content(item: Any) -> str:
    """Text of a memory item, regardless of type (episodes use ``summary``)."""
    return getattr(item, "content", None) or getattr(item, "summary", None) or ""


def _item_to_item(item: Any, namespace: tuple[str, ...], key: str) -> Item:
    value = _parse_value(_item_content(item))
    now = _as_datetime(getattr(item, "extracted_at", None))
    return Item(
        value=value,
        key=key,
        namespace=namespace,
        created_at=now,
        updated_at=now,
    )


def _item_to_search_item(
    item: Any,
    namespace: tuple[str, ...],
    key: str,
) -> SearchItem:
    value = _parse_value(_item_content(item))
    now = _as_datetime(getattr(item, "extracted_at", None))
    score = getattr(item, "confidence", None)
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
