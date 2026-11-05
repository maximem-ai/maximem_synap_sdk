"""SynapStore: LangGraph ``BaseStore`` backed by Synap semantic memory.

Implements ``batch`` / ``abatch`` (the only abstract methods) and dispatches
:class:`PutOp` and :class:`SearchOp` to async helpers. ``put`` / ``search``
(+ async variants) ride through ``(a)batch`` as in ``BaseStore``, with optional
keyword-only SDK options on top. ``get`` / ``delete`` / ``list_namespaces``
raise :class:`SynapStoreUnsupportedError` (Synap is search-based; see below).

Who the memory belongs to
-------------------------

* **Shared store (the default)**: ``SynapStore(sdk)``. One store, compiled once
  into a graph that serves many users. The identity is read from the namespace
  on every call: the first ``N`` entries are the identity, mapped to Synap ids
  (and a scope path on a deeper ladder) through the account's scope ladder,
  and everything after them is the app's own sub-namespace::

      store = SynapStore(sdk)                       # B2B: N = 2
      await store.aput(("acme", "alice", "prefs"), "language", {"v": "en"})
      await store.asearch(("acme", "alice"), query="what language?")

  ``N`` is set by ``level``. The default is the innermost level (the user):
  ``N`` is the number of levels on the ladder, so 2 on a default B2B account
  (customer, user), 1 on B2C, 3 on a customer/team/user ladder. A namespace
  shorter than ``N``, an empty one, or an unknown ``level`` raises: the store
  never falls back to another user. A pool at a higher level is a second
  shared store built once at startup: ``SynapStore(sdk, level="customer")``
  (``N`` = 1).

  The ladder comes from ``await sdk.get_scope_ladder()`` (cached by the SDK),
  so this mode needs a ``maximem-synap`` release that has it.

* **Single-user store (backward compatible)**: ``SynapStore(sdk,
  user_id="alice", customer_id="acme")``. Pins one user at construction and
  uses the whole namespace as the app's sub-namespace. ⚠ Compiled into a graph
  that serves several users it reads and writes the construction user's memory
  for everyone: never share such a store between users. Use the shared store.
  With only a ``customer_id`` it operates on the customer-wide pool.

Storage
-------

* ``put`` ingests the value (JSON in ``document``) with ``sdk.memories.create``
  and tags it with its sub-namespace and key in ``metadata``. Optional
  keyword arguments pass the SDK's own create options (``document_type``,
  ``document_id``, ``document_created_at``, ``mode``, ``objective``,
  ``metadata``) and ``wait=True`` waits for the ingestion and returns its
  ``MemoryStatusResponse``. Standard ``put`` calls are unchanged.
* ``search`` calls ``sdk.fetch(response_shape="ranked")`` for exactly
  ``limit + offset`` places (the server fills a top-N differently from a top-M,
  so the store does not ask for more than the caller wants). It asks for at
  least 10 only when it drops items itself afterwards: a sub-namespace in the
  prefix, a ``filter``, or ``semantic_fallback=False``. It returns the SDK's
  ranked list best first, ``rank_score`` as ``score``. Items that carry the
  store's tag (written by ``put``) are filtered by the requested sub-namespace
  prefix; items without it (memories Synap extracted from conversations) belong
  to the identity scope and are always returned. Nothing is dropped because
  another item carries the tag. Extra keyword arguments are ``sdk.fetch``
  options for this one search (``mode``, ``precision_level``, ``types``,
  ``query_date``, ``max_results_per_query``, ``conversation_id`` ...).
* Every item's ``value`` holds the text (the dict for text that is a JSON object,
  which is what ``put`` writes, with the exact text in ``value["_synap"]["text"]``;
  anything else exactly as written, ``{"_raw": text}``, never re-typed: ``1.50``
  stays ``"1.50"``, ``true`` stays ``"true"``) and ``value["_synap"]``: Synap's own data for it (see
  :func:`synap_meta`).
* ``created_at`` / ``updated_at`` are the day the user said it (``said_at``),
  else the event date, never the fetch time. With neither, they are the epoch
  (1970-01-01 UTC) and ``value["_synap"]["dates_known"]`` is ``False``.

Synap is search-based: no lookup by key
---------------------------------------

Agents ask Synap by search. There is no exact ``(namespace, key)`` lookup and
no namespace enumeration, so ``get`` / ``aget``, ``delete`` / ``adelete`` and
``list_namespaces`` / ``alist_namespaces`` (and a ``batch`` holding a
``GetOp``, a delete ``PutOp`` or a ``ListNamespacesOp``) raise
:class:`SynapStoreUnsupportedError`, a ``NotImplementedError``, before
anything is read or written. Use instead:

* to read: ``search`` / ``asearch`` (``query=`` says what to look for);
* to delete one memory: ``adelete_memory(namespace, memory_id)`` with the id
  of a search hit (``item.value["_synap"]["id"]``);
* to delete by description: ``aforget(namespace, "the user's old address")``.

(Before 0.6.0 ``get`` returned ``None`` for everything, ``delete`` raised "no
memory found" and ``list_namespaces`` returned ``[]``: today's Synap server
does not return the metadata a caller attaches to a memory, so the key tag
``put`` writes never came back.) ``put`` still writes the tag because
``search`` reads it to filter by sub-namespace on a server that returns it.

The rest of the SDK, on the store
---------------------------------

The Python SDK's capabilities are methods of the store, identity from the
namespace exactly as for ``search`` (except ``memories.batch_create`` and
``conversation.record_messages_batch``, left out because each item carries
its own user ids and would bypass the namespace identity; excluded by owner
decision, still reachable on ``store.sdk``): ``acreate_memory`` (any
document, file parts included), ``acreate_memory_from_file``,
``aget_ingestion_status``, ``await_ingestion``, ``aget_memory``,
``aupdate_memory``, ``adelete_memory``, ``aforget``, ``adelete_data``,
``arestore``, ``adeletion_status``, ``afetch_raw`` (the SDK's
own ``UnifiedContextResponse``, with the SDK's own defaults),
``aget_profile``, ``aingest_transcript``, ``arecord_message``,
``aget_context_for_prompt``, ``acompact``, ``aget_compacted``,
``aget_compaction_status``, ``aget_scope_ladder``; each has a sync twin
without the ``a``. Identity is never an argument of these methods.

⚠ Calls by id (``aget_memory``, ``aupdate_memory``, ``adelete_memory``,
``aget_ingestion_status``, ``await_ingestion``, ``arestore``,
``adeletion_status``, and the conversation_id methods) are NOT checked against the namespace's identity: the server does
not verify ownership by id, so any caller holding an id can act on it.

Error policy
------------

* Writes and every SDK method of the store (``acreate_memory``,
  ``adelete_memory``, ``afetch_raw`` ...) raise :class:`SynapIntegrationError`
  on SDK failure (the SDK's exception is its ``__cause__``).
* Identity errors (short namespace, unknown level, no ladder) always raise.
* Reads (``search``) follow ``on_error``: ``"fallback"`` (default) logs at
  ERROR and returns ``[]``; ``"raise"`` raises :class:`SynapIntegrationError`, so a
  failed read is distinguishable from an empty memory. This includes a read
  where ``sdk.fetch`` swallowed a failed scope ("non-fatal") and returned only
  the scopes that answered: the store compares the scopes it asked for with the
  answer's ``scopes_queried`` and, with ``"raise"``, raises (naming the scopes);
  with ``"fallback"`` it logs a WARNING and uses what answered.

Anticipation (optional, outside the BaseStore contract): construct with
``include_conversation_context=True`` and drive the conversation channel via
:meth:`SynapStore.record_message` so just-stated context is in play on reads.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Union
from uuid import UUID

from langgraph.store.base import (
    NOT_PROVIDED,
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    NamespacePath,
    NotProvided,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    SynapIntegrationError,
    coerce_conversation_id,
    ensure_ids_match_path,
    record_turn,
    run_async,
    wrap_sdk_errors_async,
)

logger = logging.getLogger(__name__)

# Metadata keys we stamp on every Synap memory we create through SynapStore
# so we can identify / filter them on retrieval.
_MARKER = "lg_store"
_NS = "lg_store_ns"
_KEY = "lg_store_key"
# The identity entries (JSON list) of the store that wrote the memory. Written
# by a shared store only. It is what tells a team-level memory from a
# customer-level one with the same sub-namespace and key: the server labels
# items user / customer / client and nothing finer.
_ID = "lg_store_id"
_TAG_KEYS = (_MARKER, _NS, _KEY, _ID)

#: ``created_at`` / ``updated_at`` of an item Synap gave no date for.
UNKNOWN_DATE = datetime(1970, 1, 1, tzinfo=timezone.utc)

_ROLE_TENANT = "tenant_of_record"  # same string as maximem_synap.scope_ladder.ROLE_TENANT
_CLIENT = "client"

# fetch arguments the store sets itself; fetch_options may not override them.
_RESERVED_FETCH_OPTIONS = frozenset({
    "user_id", "customer_id", "scope_path", "search_query", "response_shape", "max_results",
})

# `search` asks the server for exactly limit + offset places. When the store
# filters afterwards it asks for at least this many.
_FILTER_OVERFETCH_FLOOR = 10

_SINGULAR = {
    "facts": "fact", "preferences": "preference", "episodes": "episode",
    "emotions": "emotion", "temporal_events": "temporal_event", "excerpts": "excerpt",
}


# The SDK this store call resolved from ``sdk_provider`` (store, sdk), and the
# per-call fetch options in force. Context variables, so concurrent calls on one
# store (different users, different turns) never see each other's.
_ACTIVE_SDK: ContextVar[Optional[tuple]] = ContextVar("synap_langgraph_store_sdk", default=None)
_CALL_OPTIONS: ContextVar[Optional[Mapping[str, Any]]] = ContextVar(
    "synap_langgraph_store_fetch_options", default=None)

#: ``config["configurable"]`` key holding per-run fetch options (see ``SynapStore``).
FETCH_OPTIONS_CONFIG_KEY = "synap_fetch_options"

# memories.create options `put` / `acreate_memory` pass through. The identity
# (user_id, customer_id, scope) is the store's own and never an option.
_CREATE_OPTIONS = ("document_type", "document_id", "document_created_at", "mode",
                   "objective", "metadata")
# fetch arguments `afetch_raw` refuses: the identity is the namespace's, the
# query is its own argument.
_RAW_RESERVED = frozenset({"user_id", "customer_id", "scope_path", "search_query"})


class SynapStoreUnsupportedError(NotImplementedError):
    """Raised by the parts of LangGraph's ``BaseStore`` that Synap does not have.

    Synap is a search-based memory: agents ask it by search. It has no exact
    ``(namespace, key)`` lookup and no namespace listing, so ``get`` /
    ``aget``, ``delete`` / ``adelete`` and ``list_namespaces`` /
    ``alist_namespaces`` raise this (a ``NotImplementedError``) instead of
    answering ``None`` / ``[]`` for memory that exists. The message names the
    method to use instead: ``search`` / ``asearch`` to read,
    ``adelete_memory(namespace, memory_id)`` (the id from a search hit's
    ``value["_synap"]["id"]``) or ``aforget(namespace, description)`` to delete.
    Nothing is read or written when it is raised.
    """

    def __init__(self, operation: str, alternative: str) -> None:
        self.operation = operation
        self.alternative = alternative
        super().__init__(
            f"SynapStore.{operation} is not supported: Synap is a search-based memory "
            f"and has no exact key lookup or namespace listing. {alternative}")


_GET_ALTERNATIVE = (
    "Use search()/asearch(namespace, query=...) to find the memory; each hit carries "
    "its id in value['_synap']['id'].")
_DELETE_ALTERNATIVE = (
    "Use adelete_memory(namespace, memory_id) / delete_memory(...) with the id from a "
    "search hit (value['_synap']['id']), or aforget(namespace, description) / "
    "forget(...) to delete by describing the memory.")
_LIST_ALTERNATIVE = (
    "Keep the namespaces you use in your own application; search() accepts any "
    "namespace that starts with the identity.")


def _unsupported_op(op: Op) -> Optional[SynapStoreUnsupportedError]:
    if isinstance(op, GetOp):
        return SynapStoreUnsupportedError("get", _GET_ALTERNATIVE)
    if isinstance(op, PutOp) and op.value is None:
        return SynapStoreUnsupportedError("delete", _DELETE_ALTERNATIVE)
    if isinstance(op, ListNamespacesOp):
        return SynapStoreUnsupportedError("list_namespaces", _LIST_ALTERNATIVE)
    return None


def _refuse_unsupported(ops: list) -> None:
    """Raise before running anything when a batch holds an op Synap lacks."""
    for op in ops:
        error = _unsupported_op(op)
        if error is not None:
            raise error


def _check_options(options: Mapping[str, Any], where: str) -> Dict[str, Any]:
    clash = sorted(_RESERVED_FETCH_OPTIONS & set(options))
    if clash:
        raise ValueError(
            f"{where}: fetch options may not set {clash}: the store sets them itself "
            "(identity from the namespace or constructor, the query and limit from "
            "each call, response_shape='ranked').")
    return dict(options)


def _h(value: Any) -> str:
    """A short stable fingerprint for logs (no user ids or query text)."""
    return hashlib.sha256(str(value).encode("utf-8", "replace")).hexdigest()[:8]


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


@dataclass(frozen=True)
class _Target:
    """Where one operation reads and writes, resolved for that call."""

    read_kwargs: Dict[str, Any]
    write_kwargs: Dict[str, Any]
    #: namespace entries that are identity (empty on a single-user store)
    identity: tuple = ()
    #: the app's own namespace (what is left after the identity)
    sub: tuple = ()
    #: the Synap scope label this level's own memories carry
    own_scope: Optional[str] = None
    user_id: Optional[str] = None
    customer_id: Optional[str] = None
    scope_path: Optional[Dict[str, str]] = field(default=None)
    #: shared store: the JSON identity tag a memory of THIS store carries
    id_tag: Optional[str] = None


class SynapStore(BaseStore):
    """LangGraph cross-thread long-term memory store backed by Synap.

    Example (shared store, one per process, identity in the namespace)::

        from langgraph.graph import StateGraph
        from maximem_synap import MaximemSynapSDK
        from synap_langgraph import SynapStore

        sdk = MaximemSynapSDK(api_key="sk-...")
        store = SynapStore(sdk)                  # was: InMemoryStore()
        app = graph.compile(store=store)

        async def recall(state, config, *, store):
            who = config["configurable"]
            ns = (who["customer_id"], who["user_id"])          # B2B
            hits = await store.asearch(ns, query=state["question"], limit=5)

        # A customer-wide pool, as a second shared store:
        company = SynapStore(sdk, level="customer")
        await company.asearch(("acme",), query="refund policy")

    Args:
        sdk: An initialised :class:`~maximem_synap.MaximemSynapSDK`.
        user_id: Pins the store to ONE user (single-user store). ⚠ Must not be
            shared between users; omit it (and ``customer_id``) for a shared
            store. Everything it does, ``delete`` included, acts on this
            construction identity whatever namespace is passed: ``delete`` from
            a graph that serves another user deletes THIS user's memory. The
            first ``delete`` logs a WARNING naming the identity.
        customer_id: With ``user_id``, the user's customer. Alone, the
            customer-wide pool (single-user store).
        level: Shared store only. The ladder level key whose values the
            namespace starts with. Default: the innermost level. ``"customer"``
            gives the customer-wide pool. ``"client"`` is the whole account's
            shared level and needs ``allow_client_level=True``.
        allow_client_level: Explicit opt-in for ``level="client"``, a store
            with no identity at all. Refused by default.
        mode: Retrieval mode passed to ``sdk.fetch`` (default ``"accurate"``).
        include_conversation_context: Passed to ``sdk.fetch``.
        semantic_fallback: ``True`` (default): items without the store's tag
            are returned by ``search``. ``False``: strict, only items written
            through ``put`` are returned.
        on_error: ``"fallback"`` (default) or ``"raise"``; what a failed READ
            does. See the module docstring.
        fetch_options: Extra keyword arguments for every ``sdk.fetch`` the
            store makes, e.g. ``{"precision_level": "high", "types": [...],
            "max_results_per_query": 5, "scopes": ["user", "client"]}``. Do not
            put per-user values (a ``conversation_id``) here on a shared store:
            the options apply to every user. They override ``mode`` and
            ``include_conversation_context``; the identity, ``search_query``,
            ``response_shape`` and ``max_results`` are the store's own and
            may not be set here.
        scope_path: Single-user stores only: the level to read and write on a
            deeper ladder (``ScopeAddress.scope_path``).
        sdk_provider: A callable (sync or async) returning the SDK, called on
            EVERY store call, instead of ``sdk``. For one store compiled once
            into a graph that serves several SDKs (one per agent), e.g.
            ``SynapStore(sdk_provider=lambda: current_sdk.get())``. Read in the
            caller's thread/context. Returning ``None`` raises. ``SynapStore(sdk)``
            is unchanged; give ``sdk`` or ``sdk_provider``, not both.

    Several queries in one search: ``query`` may be a list of strings
    (``await store.asearch(ns, query=["tea", "cab"], limit=8)``). They go in ONE
    ``sdk.fetch(search_query=[...])``, one server call; ``max_results_per_query``
    (a fetch option) caps what each may contribute. BaseStore's signature is
    unchanged (the type hint says ``str``; a list works at run time).

    Per-call fetch options (``mode``, ``precision_level``, ``types``,
    ``max_results_per_query``, ``query_date``, ``conversation_id`` ...) on a
    shared store, three ways, in increasing priority: the constructor's
    ``fetch_options``; ``config["configurable"]["synap_fetch_options"]`` of the
    graph run (read when the store is called from inside a node); and
    ``with store.with_fetch_options(...):`` or ``store.asearch_with(ns,
    fetch_options={...}, query=...)`` / ``search_with``. The store's own
    arguments (identity, ``search_query``, ``response_shape``, ``max_results``)
    raise ``ValueError``. A ``conversation_id`` may be any string.
    """

    def __init__(
        self,
        sdk: Optional[MaximemSynapSDK] = None,
        user_id: Optional[str] = None,
        customer_id: str = "",
        *,
        level: Optional[str] = None,
        allow_client_level: bool = False,
        mode: str = "accurate",
        include_conversation_context: bool = False,
        semantic_fallback: bool = True,
        on_error: str = "fallback",
        fetch_options: Optional[Mapping[str, Any]] = None,
        scope_path: Optional[Mapping[str, str]] = None,
        sdk_provider: Optional[Callable[[], Union[MaximemSynapSDK, Awaitable[MaximemSynapSDK]]]] = None,
    ) -> None:
        if sdk is not None and sdk_provider is not None:
            raise ValueError("SynapStore takes sdk OR sdk_provider, not both")
        if sdk is None and sdk_provider is None:
            raise ValueError("SynapStore requires a non-None sdk (or an sdk_provider)")
        if sdk_provider is not None and not callable(sdk_provider):
            raise ValueError("SynapStore: sdk_provider must be a callable returning the SDK")
        if on_error not in ("fallback", "raise"):
            raise ValueError(
                f"SynapStore: on_error must be 'fallback' or 'raise', got {on_error!r}")
        shared = not user_id and not customer_id
        if level is not None and not shared:
            raise ValueError(
                "SynapStore: `level` applies to a shared store. A store built "
                "with user_id/customer_id is pinned to that identity; drop the "
                "ids to use `level`.")
        if scope_path and shared:
            raise ValueError(
                "SynapStore: `scope_path` applies to a single-user store "
                "(user_id/customer_id). A shared store reads the path from the "
                "namespace through the scope ladder.")
        # The server lets the path's user/customer win over these ids, so a
        # store pinned to one user next to another user's path would read and
        # file as the path's user. (A shared store takes both from one ladder
        # address, so they agree by construction.)
        ensure_ids_match_path(
            scope_path, user_id=user_id or None, customer_id=customer_id or None,
            where="SynapStore")
        if level == _CLIENT and not allow_client_level:
            raise ValueError(
                "SynapStore(level='client') has no identity: it reads and writes "
                "the whole account's shared memory. Pass allow_client_level=True "
                "if that is what you want.")
        if level is not None and (not isinstance(level, str) or not level.strip()):
            raise ValueError(f"SynapStore: level must be a ladder level key, got {level!r}")
        options = dict(fetch_options or {})
        _check_options(options, "SynapStore: fetch_options")

        super().__init__()
        self._sdk = sdk
        self._sdk_provider = sdk_provider
        self.user_id = user_id or ""
        self.customer_id = customer_id
        self.level = level
        self.allow_client_level = allow_client_level
        self.mode = mode
        self.include_conversation_context = include_conversation_context
        # False: strict namespace semantics, only items written through put.
        self.semantic_fallback = semantic_fallback
        self.on_error = on_error
        self.fetch_options = options
        self.scope_path = dict(scope_path) if scope_path else None
        self._shared = shared
        self._fetch_checked = False

    # ── the SDK of this call ────────────────────────────────────────────────

    @property
    def sdk(self) -> MaximemSynapSDK:
        """The SDK this call uses: the one given to the constructor, or, for a
        store built with ``sdk_provider``, the one resolved for the call in
        progress (there is none outside a store call)."""
        if self._sdk_provider is None:
            return self._sdk
        active = _ACTIVE_SDK.get()
        if active is not None and active[0] is self:
            return active[1]
        raise SynapIntegrationError(
            "langgraph.store.sdk",
            "this store resolves its SDK on every call (sdk_provider=...); there is "
            "no SDK outside a store call.")

    @sdk.setter
    def sdk(self, value: MaximemSynapSDK) -> None:
        self._sdk = value

    async def _resolve_sdk(self, pending: Any = None) -> Any:
        """The SDK for one call: ``sdk_provider()`` (sync or async), or ``pending``
        (what ``sdk_provider()`` returned in the calling thread)."""
        found = pending if pending is not None else self._sdk_provider()
        if inspect.isawaitable(found):
            found = await found
        if found is None:
            raise SynapIntegrationError(
                "langgraph.store.sdk",
                "sdk_provider() returned no SDK for this call. Nothing was read or "
                "written; the store never falls back to another SDK.")
        return found

    # ── per-call fetch options ──────────────────────────────────────────────

    @contextmanager
    def with_fetch_options(self, **options: Any) -> Iterator[None]:
        """Fetch options for every store call made inside the ``with`` block (in
        this task and the tasks it starts), on top of the constructor's::

            with store.with_fetch_options(mode="fast", precision_level="medium",
                                          conversation_id=thread_id):
                hits = await store.asearch(ns, query=["a", "b"])

        Any ``sdk.fetch`` argument except the ones the store sets itself
        (``user_id``, ``customer_id``, ``scope_path``, ``search_query``,
        ``response_shape``, ``max_results``) raises ``ValueError``. Calls
        elsewhere, including concurrent ones for other users, are unaffected.
        A ``conversation_id`` may be any string (it is turned into the UUID the
        SDK needs). Blocks nest; an inner one wins.
        """
        merged = {**(_CALL_OPTIONS.get() or {}), **_check_options(options, "with_fetch_options")}
        token = _CALL_OPTIONS.set(merged)
        try:
            yield
        finally:
            _CALL_OPTIONS.reset(token)

    def _call_options(self) -> Dict[str, Any]:
        """Per-call options in force right now: the run config's
        ``synap_fetch_options`` (when called inside a graph run), then
        :meth:`with_fetch_options`. Read in the caller's context."""
        merged: Dict[str, Any] = {}
        try:
            from langgraph.config import get_config

            conf = (get_config() or {}).get("configurable") or {}
            from_config = conf.get(FETCH_OPTIONS_CONFIG_KEY)
        except Exception:  # noqa: BLE001 — not inside a graph run
            from_config = None
        if from_config:
            if not isinstance(from_config, Mapping):
                raise ValueError(
                    f"config['configurable']['{FETCH_OPTIONS_CONFIG_KEY}'] must be a "
                    f"mapping of sdk.fetch options, got {type(from_config).__name__}")
            merged.update(_check_options(from_config, FETCH_OPTIONS_CONFIG_KEY))
        merged.update(_CALL_OPTIONS.get() or {})
        return merged

    async def asearch_with(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        fetch_options: Mapping[str, Any],
        query: Union[str, Sequence[str], None] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SearchItem]:
        """``asearch`` with ``sdk.fetch`` options for this one search, e.g.
        ``fetch_options={"mode": "accurate", "precision_level": "medium",
        "types": ["facts"], "max_results_per_query": 3,
        "conversation_id": thread_id, "query_date": "2023-03-11"}``. They sit on
        top of the constructor's ``fetch_options`` for this call only; the
        store's own arguments cannot be set (``ValueError``). ``query`` may be a
        list: see :meth:`asearch`."""
        with self.with_fetch_options(**dict(fetch_options)):
            return await self.asearch(
                namespace_prefix, query=query, filter=filter, limit=limit, offset=offset)

    def search_with(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        fetch_options: Mapping[str, Any],
        query: Union[str, Sequence[str], None] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SearchItem]:
        """Sync variant of :meth:`asearch_with`."""
        with self.with_fetch_options(**dict(fetch_options)):
            return self.search(
                namespace_prefix, query=query, filter=filter, limit=limit, offset=offset)

    # ── abstract methods ────────────────────────────────────────────────────

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        ops = list(ops)
        _refuse_unsupported(ops)
        return await self._run_ops(ops, None, self._call_options())

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        ops = list(ops)
        _refuse_unsupported(ops)
        # Provider and options are read here, in the caller's thread and
        # context; the coroutine runs on the bridge's loop, which has neither.
        pending = self._sdk_provider() if self._sdk_provider is not None else None
        return run_async(self._run_ops(ops, pending, self._call_options()))

    async def _run_ops(self, ops: list, pending: Any, options: Mapping[str, Any]) -> list[Result]:
        opt_token = _CALL_OPTIONS.set(dict(options))
        sdk_token = None
        try:
            if self._sdk_provider is not None:
                sdk_token = _ACTIVE_SDK.set((self, await self._resolve_sdk(pending)))
            return [await self._dispatch(op) for op in ops]
        finally:
            if sdk_token is not None:
                _ACTIVE_SDK.reset(sdk_token)
            _CALL_OPTIONS.reset(opt_token)

    # ── dispatcher ──────────────────────────────────────────────────────────

    async def _dispatch(self, op: Op) -> Result:
        error = _unsupported_op(op)
        if error is not None:
            raise error
        if isinstance(op, PutOp):
            await self._aput(op.namespace, op.key, op.value, op.index)
            return None
        if isinstance(op, SearchOp):
            return await self._asearch(
                op.namespace_prefix, op.query, op.filter, op.limit, op.offset,
            )
        raise ValueError(f"SynapStore: unsupported op type {type(op).__name__}")

    # ── who is this call about ──────────────────────────────────────────────

    async def _identity_size(self) -> int:
        """``N``: how many leading namespace entries are identity (shared store)."""
        op = "langgraph.store.identity"
        get_ladder = getattr(self.sdk, "get_scope_ladder", None)
        if not callable(get_ladder):
            raise SynapIntegrationError(
                op, "a shared SynapStore takes the identity from the namespace "
                "through the account's scope ladder, which needs a maximem-synap "
                "release that has `get_scope_ladder`. Upgrade `maximem-synap`, or "
                "build the store with user_id/customer_id (one user per store).")
        async with wrap_sdk_errors_async(f"{op}.get_scope_ladder", logger):
            ladder = await get_ladder()
        if ladder is None:
            raise SynapIntegrationError(
                op, "this account's scope ladder could not be determined, so the "
                "namespace cannot be mapped to a user. Build the store with "
                "user_id/customer_id instead.")
        levels = tuple(ladder.levels)
        # Same rule as ScopeLadder.address: on B2C the tenant level takes the
        # identity's value, so the namespace holds one entry per other level.
        addressable = (
            tuple(lv for lv in levels if lv.role != _ROLE_TENANT) if ladder.b2c else levels
        )
        keys = [lv.key for lv in addressable]
        if self.level is None:
            return len(keys)
        if self.level not in keys:
            hint = (" On a B2C instance the tenant level takes the user's id, so "
                    "it has no pool of its own." if ladder.b2c and
                    any(lv.key == self.level for lv in levels) else "")
            raise SynapIntegrationError(
                op, f"unknown level {self.level!r} for this account's ladder; "
                f"levels, outermost first: {keys}.{hint}")
        return keys.index(self.level) + 1

    async def _target(self, namespace: tuple[str, ...], *, what: str) -> _Target:
        """Resolve the identity of one call. Raises rather than guess."""
        if not isinstance(namespace, tuple) or not all(isinstance(p, str) for p in namespace):
            raise SynapIntegrationError(
                f"langgraph.store.{what}",
                f"namespace must be a tuple of strings, got {type(namespace).__name__}.")
        if not self._shared:
            ids = {"user_id": self.user_id or None, "customer_id": self.customer_id or None}
            read = dict(ids)
            write = dict(ids)
            if self.scope_path:
                read["scope_path"] = dict(self.scope_path)
                write["scope"] = dict(self.scope_path)
            return _Target(
                read_kwargs=read, write_kwargs=write, identity=(), sub=tuple(namespace),
                # a store pinned to a level by a scope_path carries the same
                # identity tag a shared store at that level would: the server's
                # user/customer label cannot tell a team from the customer
                id_tag=(json.dumps(list(self.scope_path.values()))
                        if self.scope_path else None),
                own_scope="user" if self.user_id else "customer",
                user_id=self.user_id or None, customer_id=self.customer_id or None,
                scope_path=dict(self.scope_path) if self.scope_path else None,
            )

        op = f"langgraph.store.{what}"
        if self.level == _CLIENT:  # no identity entries: the whole namespace is the app's
            return _Target(read_kwargs={}, write_kwargs={}, identity=(),
                           sub=tuple(namespace), own_scope=_CLIENT,
                           id_tag=json.dumps([]))
        n = await self._identity_size()
        if len(namespace) < n:
            raise SynapIntegrationError(
                op, f"namespace {namespace!r} is too short: the first {n} "
                f"entr{'y is' if n == 1 else 'ies are'} the identity (level "
                f"{self.level or 'innermost'}), so the namespace must start with "
                f"{n} value{'' if n == 1 else 's'}. The store does not guess a user.")
        identity = tuple(namespace[:n])
        # Imported here so that an older integrations-common (no
        # resolve_address) still imports this package for single-user use.
        try:
            from synap_integrations_common import resolve_address
        except ImportError as exc:
            raise SynapIntegrationError(
                op, "a shared SynapStore needs a maximem-synap-integrations-common "
                "release that has `resolve_address` (the scope ladder). Upgrade "
                "`maximem-synap-integrations-common`, or build the store with "
                "user_id/customer_id (one user per store).") from exc

        address = await resolve_address(self.sdk, values=list(identity))
        return _Target(
            read_kwargs=address.read_kwargs(), write_kwargs=address.write_kwargs(),
            identity=identity, sub=tuple(namespace[n:]),
            own_scope="user" if address.user_id else "customer",
            user_id=address.user_id, customer_id=address.customer_id,
            scope_path=dict(address.scope_path) if address.scope_path else None,
            id_tag=json.dumps(list(identity)),
        )

    # ── reads ───────────────────────────────────────────────────────────────

    async def _fetch(
        self,
        target: _Target,
        queries: Optional[Sequence[str]],
        max_results: int,
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "mode": self.mode,
            "include_conversation_context": self.include_conversation_context,
        }
        kwargs.update(self.fetch_options)
        kwargs.update(_CALL_OPTIONS.get() or {})
        kwargs.update(target.read_kwargs)
        if kwargs.get("conversation_id"):
            # any string is a conversation id; the SDK's read wants the UUID
            kwargs["conversation_id"] = coerce_conversation_id(kwargs["conversation_id"])
        kwargs["search_query"] = list(queries) if queries else None
        kwargs["max_results"] = max_results
        kwargs["response_shape"] = "ranked"
        if target.own_scope == _CLIENT:
            kwargs["scopes"] = [_CLIENT]
        response = await self.sdk.fetch(**kwargs)
        return response, _unanswered_scopes(kwargs, response)

    def _incomplete(self, op: str, missing: list[str], **context: Any) -> None:
        """A scope the fetch asked for did not answer.

        ``sdk.fetch`` catches a scope that raised ("non-fatal") and returns the
        scopes that did answer; the only trace is that the scope is missing from
        ``scopes_queried``. That is an outage, not empty memory. Always logged;
        raised when ``on_error="raise"``. With ``on_error="fallback"`` the
        partial answer is used, as before.
        """
        message = (f"the memory read is incomplete: scope(s) {missing} failed and were "
                   "left out of the answer by sdk.fetch (see the 'non-fatal' warning "
                   "above). An outage looks like empty memory without this check.")
        if self.on_error == "raise":
            logger.error("%s: %s", op, message)
            raise SynapIntegrationError(op, message, {"scopes": missing, **context})
        logger.warning("%s: %s Using the scopes that answered.", op, message)

    def _check_fetch_support(self, target: _Target) -> None:
        """Refuse, whatever ``on_error`` says, an SDK whose ``fetch`` cannot take
        what the store sends: a misconfigured install is not an outage, and
        ``on_error="fallback"`` would show it as empty memory."""
        needed = ["response_shape"]
        if target.read_kwargs.get("scope_path"):
            needed.append("scope_path")
        if self._fetch_checked and needed == ["response_shape"]:
            return
        try:
            params = inspect.signature(self.sdk.fetch).parameters
        except (TypeError, ValueError):
            self._fetch_checked = True
            return  # cannot tell: let the call decide
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            self._fetch_checked = True
            return
        missing = [n for n in needed if n not in params]
        if missing:
            raise SynapIntegrationError(
                "langgraph.store.sdk",
                f"the installed maximem-synap's fetch() does not accept {missing}. "
                "SynapStore needs a newer maximem-synap; upgrade it. This is a "
                "setup error, so it is raised whatever on_error is.")
        if needed == ["response_shape"]:
            self._fetch_checked = True

    def _read_failed(self, op: str, exc: Exception, log_msg: str, *log_args: Any,
                     **context: Any) -> None:
        """Log a failed read; raise it too when ``on_error="raise"``."""
        logger.error(log_msg, *log_args, exc_info=True)
        if self.on_error == "raise":
            raise SynapIntegrationError(op, str(exc), context) from exc

    def _places_to_ask(self, wanted: int, sub_prefix: tuple[str, ...],
                       filter_: Optional[dict[str, Any]]) -> int:
        """How many places to ask the server for: what the caller asked for
        (``limit + offset``), because the server's assembly of a top-N depends on
        N. Only when this store drops items after the fetch (a sub-namespace, a
        metadata ``filter``, or ``semantic_fallback=False``) does it ask for
        more, at least ``_FILTER_OVERFETCH_FLOOR``, so that dropping some still
        leaves ``limit``."""
        wanted = max(int(wanted), 1)
        if sub_prefix or filter_ or not self.semantic_fallback:
            return max(wanted, _FILTER_OVERFETCH_FLOOR)
        return wanted

    async def _asearch(
        self,
        namespace_prefix: tuple[str, ...],
        query: Union[str, Sequence[str], None],
        filter_: Optional[dict[str, Any]],
        limit: int,
        offset: int,
    ) -> list[SearchItem]:
        target = await self._target(namespace_prefix, what="search")
        sub_prefix = target.sub
        queries = _queries_of(query) or ([_ns_str(sub_prefix)] if sub_prefix else [])
        self._check_fetch_support(target)
        try:
            response, missing = await self._fetch(
                target, queries, self._places_to_ask(limit + offset, sub_prefix, filter_))
        except Exception as exc:  # noqa: BLE001 — policy decided by on_error
            self._read_failed(
                "langgraph.store.search", exc,
                "SynapStore.search: sdk.fetch failed namespace_prefix=%s queries=%s query_len=%s error=%s",
                _h(_ns_str(namespace_prefix)), len(queries), sum(map(len, queries)), exc,
                namespace_prefix=_h(_ns_str(namespace_prefix)), queries=len(queries),
                query_len=sum(map(len, queries)))
            return []
        if missing:
            self._incomplete("langgraph.store.search", missing,
                             namespace_prefix=_h(_ns_str(namespace_prefix)))

        matches: list[SearchItem] = []
        for item, category in _iter_items(response):
            md = getattr(item, "metadata", None) or {}
            if md.get(_MARKER):
                item_sub = md.get(_NS) or ""
                if not _matches_namespace_prefix(item_sub, sub_prefix):
                    continue
                if filter_ and not _filter_matches(md, filter_):
                    continue
                key = md.get(_KEY) or str(getattr(item, "id", ""))
                sub_ns = tuple(item_sub.split("/")) if item_sub else ()
                if not self._shared:
                    item_ns = sub_ns
                elif _ID in md:
                    # the memory's TRUE namespace: an ancestor's item keeps its
                    # own (shorter) identity, never the caller's
                    try:
                        ident = tuple(json.loads(md[_ID]))
                    except (ValueError, TypeError):
                        ident = None
                    item_ns = (ident + sub_ns) if ident is not None else namespace_prefix
                else:
                    item_ns = namespace_prefix  # no identity tag: level unknown
                matches.append(_item_to_search_item(item, category, response, item_ns, key))
            else:
                # Not written through put: Synap extracted it from a
                # conversation. It belongs to the identity scope, so it is
                # returned whatever sub-namespace was asked for.
                if not self.semantic_fallback:
                    continue
                if filter_ and not _filter_matches(md, filter_):
                    continue
                matches.append(_item_to_search_item(
                    item, category, response, namespace_prefix, str(getattr(item, "id", ""))))
        return matches[offset : offset + limit]

    # ── writes ──────────────────────────────────────────────────────────────

    async def _aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: Mapping[str, Any],
        index: Any = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Ingest one ``put``; returns the SDK's ``CreateMemoryResponse``.

        ``options``: ``memories.create`` options (:data:`_CREATE_OPTIONS`);
        a caller's ``metadata`` is merged under the store's own tag keys, which
        it may not set.
        """
        target = await self._target(namespace, what="put")
        sub = _ns_str(target.sub)
        # `search` puts a `_synap` block in the value; writing a value back
        # must not store that block in the memory text.
        value = {k: v for k, v in value.items() if k != "_synap"}
        document = json.dumps(value, default=str)
        options = dict(options or {})
        metadata = _caller_metadata(options.pop("metadata", None), "put")
        metadata.update({
            _MARKER: True,
            _NS: sub,
            _KEY: key,
        })
        if target.id_tag is not None:
            metadata[_ID] = target.id_tag
        h_sub, h_key = _h(sub), _h(key)
        async with wrap_sdk_errors_async(
            "langgraph.store.put",
            logger,
            namespace=h_sub,
            key=h_key,
        ):
            return await self.sdk.memories.create(
                document=document,
                metadata=metadata,
                **options,
                **target.write_kwargs,
            )

    # ── BaseStore methods, with the SDK's options on top ────────────────────

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: Mapping[str, Any],
        index: Any = None,
        *,
        ttl: Union[float, None, NotProvided] = NOT_PROVIDED,
        document_type: Optional[str] = None,
        document_id: Optional[str] = None,
        document_created_at: Optional[datetime] = None,
        mode: Optional[str] = None,
        objective: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        wait: bool = False,
        timeout_seconds: int = 300,
        poll_interval_seconds: int = 2,
    ) -> Any:
        """``BaseStore.aput``, plus ``sdk.memories.create``'s options.

        Without the extra arguments this is LangGraph's ``aput``, unchanged
        (through ``abatch``) and returns ``None``. The extras are passed to
        ``sdk.memories.create`` as given (``mode`` left out means the SDK's
        default, ``"long-range"``); ``metadata`` is merged with the store's
        tag. ``wait=True`` waits for the ingestion
        (``sdk.memories.wait_for_completion``) and returns the SDK's
        ``MemoryStatusResponse``; otherwise ``None``. ``ttl`` is not supported
        (Synap keeps memory until it is deleted).
        """
        options = _given_create_options(
            document_type=document_type, document_id=document_id,
            document_created_at=document_created_at, mode=mode, objective=objective,
            metadata=metadata)
        if not options and not wait:
            return await super().aput(namespace, key, value, index, ttl=ttl)
        _check_put(namespace, value, ttl)

        async def go() -> Any:
            created = await self._aput(namespace, str(key), value, index, options)
            if not wait:
                return None
            return await self._await_created(
                created, timeout_seconds, poll_interval_seconds, "put")

        return await self._with_sdk(go)

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: Mapping[str, Any],
        index: Any = None,
        *,
        ttl: Union[float, None, NotProvided] = NOT_PROVIDED,
        document_type: Optional[str] = None,
        document_id: Optional[str] = None,
        document_created_at: Optional[datetime] = None,
        mode: Optional[str] = None,
        objective: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        wait: bool = False,
        timeout_seconds: int = 300,
        poll_interval_seconds: int = 2,
    ) -> Any:
        """Sync variant of :meth:`aput`."""
        options = _given_create_options(
            document_type=document_type, document_id=document_id,
            document_created_at=document_created_at, mode=mode, objective=objective,
            metadata=metadata)
        if not options and not wait:
            return super().put(namespace, key, value, index, ttl=ttl)
        _check_put(namespace, value, ttl)

        async def go() -> Any:
            created = await self._aput(namespace, str(key), value, index, options)
            if not wait:
                return None
            return await self._await_created(
                created, timeout_seconds, poll_interval_seconds, "put")

        return self._sync(go)

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: Union[str, Sequence[str], None] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: Optional[bool] = None,
        **fetch_options: Any,
    ) -> list[SearchItem]:
        """``BaseStore.asearch``, plus ``sdk.fetch`` options for this one search.

        Any extra keyword is an ``sdk.fetch`` argument: ``mode``,
        ``precision_level``, ``types``, ``query_date``,
        ``max_results_per_query``, ``conversation_id``,
        ``include_conversation_context``, ``scopes``, ``include_scope_labels``,
        ``context_mode``, ``include_profile``, ``last_n_conversations``. They
        sit on top of the constructor's ``fetch_options`` and the run config's
        (see the class docstring). The store's own arguments (identity,
        ``search_query``, ``response_shape``, ``max_results``) raise
        ``ValueError``: use :meth:`afetch_raw` for a by-type answer or another
        ``max_results``. ``query`` may be a list of strings (one fetch).
        """
        if not fetch_options:
            return await super().asearch(
                namespace_prefix, query=query, filter=filter, limit=limit,
                offset=offset, refresh_ttl=refresh_ttl)
        with self.with_fetch_options(**fetch_options):
            return await super().asearch(
                namespace_prefix, query=query, filter=filter, limit=limit,
                offset=offset, refresh_ttl=refresh_ttl)

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: Union[str, Sequence[str], None] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: Optional[bool] = None,
        **fetch_options: Any,
    ) -> list[SearchItem]:
        """Sync variant of :meth:`asearch`."""
        if not fetch_options:
            return super().search(
                namespace_prefix, query=query, filter=filter, limit=limit,
                offset=offset, refresh_ttl=refresh_ttl)
        with self.with_fetch_options(**fetch_options):
            return super().search(
                namespace_prefix, query=query, filter=filter, limit=limit,
                offset=offset, refresh_ttl=refresh_ttl)

    # ── BaseStore methods Synap does not have ───────────────────────────────

    async def aget(self, namespace: tuple[str, ...], key: str, *,
                   refresh_ttl: Optional[bool] = None) -> Optional[Item]:
        """Not supported: raises :class:`SynapStoreUnsupportedError`. Synap is
        search-based; use :meth:`asearch`."""
        raise SynapStoreUnsupportedError("aget", _GET_ALTERNATIVE)

    def get(self, namespace: tuple[str, ...], key: str, *,
            refresh_ttl: Optional[bool] = None) -> Optional[Item]:
        """Not supported: raises :class:`SynapStoreUnsupportedError`. Synap is
        search-based; use :meth:`search`."""
        raise SynapStoreUnsupportedError("get", _GET_ALTERNATIVE)

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        """Not supported: raises :class:`SynapStoreUnsupportedError`. Use
        :meth:`adelete_memory` with a search hit's id, or :meth:`aforget`."""
        raise SynapStoreUnsupportedError("adelete", _DELETE_ALTERNATIVE)

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Not supported: raises :class:`SynapStoreUnsupportedError`. Use
        :meth:`delete_memory` with a search hit's id, or :meth:`forget`."""
        raise SynapStoreUnsupportedError("delete", _DELETE_ALTERNATIVE)

    async def alist_namespaces(self, *, prefix: Optional[NamespacePath] = None,
                               suffix: Optional[NamespacePath] = None,
                               max_depth: Optional[int] = None, limit: int = 100,
                               offset: int = 0) -> list[tuple[str, ...]]:
        """Not supported: raises :class:`SynapStoreUnsupportedError`. Synap has
        no namespace listing."""
        raise SynapStoreUnsupportedError("alist_namespaces", _LIST_ALTERNATIVE)

    def list_namespaces(self, *, prefix: Optional[NamespacePath] = None,
                        suffix: Optional[NamespacePath] = None,
                        max_depth: Optional[int] = None, limit: int = 100,
                        offset: int = 0) -> list[tuple[str, ...]]:
        """Not supported: raises :class:`SynapStoreUnsupportedError`. Synap has
        no namespace listing."""
        raise SynapStoreUnsupportedError("list_namespaces", _LIST_ALTERNATIVE)

    # ── running an SDK call with this store's SDK ───────────────────────────

    async def _with_sdk(self, fn: Callable[[], Awaitable[Any]], pending: Any = None) -> Any:
        """Run ``fn`` with the SDK of this call resolved (``sdk_provider``)."""
        if self._sdk_provider is not None and (_ACTIVE_SDK.get() or (None,))[0] is not self:
            token = _ACTIVE_SDK.set((self, await self._resolve_sdk(pending)))
            try:
                return await fn()
            finally:
                _ACTIVE_SDK.reset(token)
        return await fn()

    def _sync(self, fn: Callable[[], Awaitable[Any]]) -> Any:
        """Sync twin of an async store method: the provider is called here, in
        the caller's thread and context, as ``batch`` does."""
        pending = self._sdk_provider() if self._sdk_provider is not None else None
        return run_async(self._with_sdk(fn, pending))

    async def _sdk_call(self, op: str, call: Callable[[], Awaitable[Any]], **context: Any) -> Any:
        async with wrap_sdk_errors_async(f"langgraph.store.{op}", logger, **context):
            return await call()

    async def _await_created(self, created: Any, timeout_seconds: int,
                             poll_interval_seconds: int, op: str) -> Any:
        ingestion_id = getattr(created, "ingestion_id", None)
        if not ingestion_id:
            raise SynapIntegrationError(
                f"langgraph.store.{op}",
                "wait=True: memories.create returned no ingestion_id to wait for.")
        return await self._sdk_call(
            f"{op}.wait", lambda: self.sdk.memories.wait_for_completion(
                ingestion_id, timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds))

    async def _user_target(self, namespace: tuple[str, ...], what: str) -> _Target:
        target = await self._target(tuple(namespace), what=what)
        if not target.user_id:
            raise SynapIntegrationError(
                f"langgraph.store.{what}",
                f"{what} needs a user: the namespace (or the store's construction "
                "identity) must name one. This store's level has no user.")
        return target

    # ── writing memory ──────────────────────────────────────────────────────

    async def acreate_memory(
        self,
        namespace: tuple[str, ...],
        document: Any,
        *,
        document_type: Optional[str] = None,
        document_id: Optional[str] = None,
        document_created_at: Optional[datetime] = None,
        mode: Optional[str] = None,
        objective: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        wait: bool = False,
        timeout_seconds: int = 300,
        poll_interval_seconds: int = 2,
    ) -> Any:
        """``sdk.memories.create`` for the namespace's identity, document as given.

        ``document`` is anything the SDK takes: a string, a dict, a
        ``TextPart``, a ``FilePart`` (an image), or a list of them. Unlike
        :meth:`aput` nothing is JSON-encoded and no store tag is added. Returns
        the SDK's ``CreateMemoryResponse`` (``ingestion_id``, ``status``), or
        with ``wait=True`` the ``MemoryStatusResponse`` once the ingestion
        finished. ``mode`` left out is the SDK's default (``"long-range"``).
        The sub-namespace after the identity is not stored (Synap keeps no
        namespaces).
        """
        options = _given_create_options(
            document_type=document_type, document_id=document_id,
            document_created_at=document_created_at, mode=mode, objective=objective,
            metadata=metadata)
        if "metadata" in options:
            options["metadata"] = _caller_metadata(options["metadata"], "create_memory")

        async def go() -> Any:
            target = await self._target(tuple(namespace), what="create_memory")
            created = await self._sdk_call("create_memory", lambda: self.sdk.memories.create(
                document=document, **options, **target.write_kwargs))
            if not wait:
                return created
            return await self._await_created(
                created, timeout_seconds, poll_interval_seconds, "create_memory")

        return await self._with_sdk(go)

    def create_memory(self, namespace: tuple[str, ...], document: Any, **kwargs: Any) -> Any:
        """Sync variant of :meth:`acreate_memory` (same keyword arguments)."""
        return self._sync(lambda: self.acreate_memory(namespace, document, **kwargs))

    async def acreate_memory_from_file(
        self,
        namespace: tuple[str, ...],
        *,
        file_path: Optional[str] = None,
        file: Any = None,
        filename: Optional[str] = None,
        text: Optional[str] = None,
        document_type: Optional[str] = None,
        mode: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """``sdk.memories.create_from_file`` for the namespace's user: exactly one
        of ``file_path``, ``file`` (a binary file object, with ``filename``) or
        ``text``. ``relationship_type`` follows the identity: ``"b2b"`` when it
        has a customer, else ``"b2c"``. Returns the SDK's
        ``CreateMemoryResponse``. Needs a user in the namespace."""
        options: Dict[str, Any] = {
            k: v for k, v in (("file_path", file_path), ("file", file), ("filename", filename),
                              ("text", text), ("document_type", document_type), ("mode", mode))
            if v is not None}
        if metadata is not None:
            options["metadata"] = _caller_metadata(metadata, "create_memory_from_file")

        async def go() -> Any:
            target = await self._user_target(namespace, "create_memory_from_file")
            extra: Dict[str, Any] = {"scope": dict(target.scope_path)} if target.scope_path else {}
            return await self._sdk_call(
                "create_memory_from_file", lambda: self.sdk.memories.create_from_file(
                    user_id=target.user_id, customer_id=target.customer_id,
                    relationship_type="b2b" if target.customer_id else "b2c",
                    **options, **extra))

        return await self._with_sdk(go)

    def create_memory_from_file(self, namespace: tuple[str, ...], **kwargs: Any) -> Any:
        """Sync variant of :meth:`acreate_memory_from_file`."""
        return self._sync(lambda: self.acreate_memory_from_file(namespace, **kwargs))

    async def aget_ingestion_status(self, ingestion_id: Any) -> Any:
        """``sdk.memories.status``: the SDK's ``MemoryStatusResponse`` for an
        ingestion (the ``ingestion_id`` of :meth:`acreate_memory`).

        ⚠ The ingestion id is not checked against any identity: the server does
        not verify who owns it, so any caller holding one can read its status."""
        return await self._with_sdk(lambda: self._sdk_call(
            "get_ingestion_status", lambda: self.sdk.memories.status(ingestion_id)))

    def get_ingestion_status(self, ingestion_id: Any) -> Any:
        """Sync variant of :meth:`aget_ingestion_status`."""
        return self._sync(lambda: self.aget_ingestion_status(ingestion_id))

    async def await_ingestion(self, ingestion_id: Any, *, timeout_seconds: int = 300,
                              poll_interval_seconds: int = 2) -> Any:
        """``sdk.memories.wait_for_completion``: waits until the ingestion
        finished and returns the SDK's ``MemoryStatusResponse``.

        ⚠ The ingestion id is not checked against any identity: the server does
        not verify who owns it, so any caller holding one can read its status."""
        return await self._with_sdk(lambda: self._sdk_call(
            "await_ingestion", lambda: self.sdk.memories.wait_for_completion(
                ingestion_id, timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds)))

    def wait_for_ingestion(self, ingestion_id: Any, *, timeout_seconds: int = 300,
                           poll_interval_seconds: int = 2) -> Any:
        """Sync variant of :meth:`await_ingestion`."""
        return self._sync(lambda: self.await_ingestion(
            ingestion_id, timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds))

    # ── one memory, by id ───────────────────────────────────────────────────
    # ⚠ The memory id is NOT checked against the namespace's identity. The
    # namespace is only checked for shape; the server keys these calls by id
    # alone and does not verify ownership, so any caller holding an id can
    # read, rewrite or delete that memory, whoever it belongs to.

    async def aget_memory(self, namespace: tuple[str, ...], memory_id: Any) -> Any:
        """``sdk.memories.get``: the SDK's ``Memory`` for a memory id (from a
        search hit's ``value["_synap"]["id"]``).

        ⚠ The memory id is NOT checked against the namespace's identity: the
        server does not verify ownership by id, so any caller holding an id
        (another user's included) can act on it. The namespace is only checked
        for shape. Do not pass ids from untrusted input (e.g. a model's tool
        call) without your own check."""
        async def go() -> Any:
            await self._target(tuple(namespace), what="get_memory")
            return await self._sdk_call(
                "get_memory", lambda: self.sdk.memories.get(memory_id),
                memory_id=str(memory_id))
        return await self._with_sdk(go)

    def get_memory(self, namespace: tuple[str, ...], memory_id: Any) -> Any:
        """Sync variant of :meth:`aget_memory`."""
        return self._sync(lambda: self.aget_memory(namespace, memory_id))

    async def aupdate_memory(
        self,
        namespace: tuple[str, ...],
        memory_id: Any,
        document: str,
        *,
        merge_strategy: str = "smart-merge",
        document_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """``sdk.memories.update``: returns the SDK's updated ``Memory``.

        ⚠ The memory id is NOT checked against the namespace's identity: the
        server does not verify ownership by id, so any caller holding an id
        (another user's included) can act on it. The namespace is only checked
        for shape. Do not pass ids from untrusted input (e.g. a model's tool
        call) without your own check."""
        async def go() -> Any:
            await self._target(tuple(namespace), what="update_memory")
            return await self._sdk_call(
                "update_memory", lambda: self.sdk.memories.update(
                    memory_id, document, merge_strategy=merge_strategy,
                    document_type=document_type,
                    metadata=dict(metadata) if metadata is not None else None),
                memory_id=str(memory_id))
        return await self._with_sdk(go)

    def update_memory(self, namespace: tuple[str, ...], memory_id: Any, document: str,
                      **kwargs: Any) -> Any:
        """Sync variant of :meth:`aupdate_memory`."""
        return self._sync(lambda: self.aupdate_memory(namespace, memory_id, document, **kwargs))

    async def adelete_memory(self, namespace: tuple[str, ...], memory_id: Any, *,
                             permanent: bool = False) -> Any:
        """``sdk.memories.delete``: deletes one memory by id (from a search hit's
        ``value["_synap"]["id"]``) and returns the SDK's answer. This replaces
        ``adelete(namespace, key)``, which Synap cannot do.

        ``permanent`` is the SDK's: ``False`` (default) soft-deletes (hidden at
        once, restorable with :meth:`arestore` until ``purge_after``);
        ``True`` removes it now and needs a management-level key.

        ⚠ The memory id is NOT checked against the namespace's identity: the
        server does not verify ownership by id, so any caller holding an id
        (another user's included) can act on it. The namespace is only checked
        for shape. Do not pass ids from untrusted input (e.g. a model's tool
        call) without your own check."""
        # sent only when set, so a default call works with an SDK that
        # predates the argument
        extra: Dict[str, Any] = {"permanent": True} if permanent else {}

        async def go() -> Any:
            await self._target(tuple(namespace), what="delete_memory")
            return await self._sdk_call(
                "delete_memory", lambda: self.sdk.memories.delete(memory_id, **extra),
                memory_id=str(memory_id))
        return await self._with_sdk(go)

    def delete_memory(self, namespace: tuple[str, ...], memory_id: Any, *,
                      permanent: bool = False) -> Any:
        """Sync variant of :meth:`adelete_memory`."""
        return self._sync(lambda: self.adelete_memory(namespace, memory_id, permanent=permanent))

    # ── deleting data by identity (sdk.data) ────────────────────────────────

    async def adelete_data(
        self,
        namespace: tuple[str, ...],
        scope: str,
        *,
        conversation_id: Optional[str] = None,
        dry_run: bool = False,
        confirm: Optional[str] = None,
        wait: bool = True,
        instances: Optional[Union[str, list[str]]] = None,
        permanent: bool = False,
    ) -> Any:
        """``sdk.data.delete`` for the namespace's identity: delete one user's
        data, one conversation's, or one customer's. Returns the SDK's
        ``DeleteDataResult``.

        The user and customer come ONLY from the namespace (or the
        single-user store's construction identity), never from an argument.
        ``scope`` must match the namespace's own level: ``"user"`` and
        ``"conversation"`` need a user in it (``conversation_id`` then names
        the conversation), ``"customer"`` needs a customer-level namespace
        (no user), so a user's namespace can never delete the whole customer.

        The rest is the SDK's, same names and defaults: ``dry_run`` counts and
        changes nothing; ``confirm`` (required unless ``dry_run``) must equal
        the id of the scope being deleted (user, customer or conversation id);
        ``wait`` repeats while the server says to continue; ``instances``
        (``"all"`` or ids) and ``permanent`` need a management key. Memories
        are soft-deleted by default and restorable with :meth:`arestore`
        (``deletion_id=result.deletion_id``) until ``purge_after``.

        ⚠ ``conversation_id`` is not checked against the user by this store;
        the SDK sends it with the namespace's user and the server scopes the
        delete to that user's instance and client.
        """
        op = "delete_data"
        if scope not in ("user", "conversation", "customer"):
            raise ValueError(
                f"adelete_data: scope must be 'user', 'conversation' or 'customer', got {scope!r}")

        async def go() -> Any:
            target = await self._target(tuple(namespace), what=op)
            if scope in ("user", "conversation") and not target.user_id:
                raise SynapIntegrationError(
                    f"langgraph.store.{op}",
                    f"scope={scope!r} needs a user in the namespace; this one names none.")
            if scope == "customer" and (target.user_id or not target.customer_id):
                raise SynapIntegrationError(
                    f"langgraph.store.{op}",
                    "scope='customer' needs a customer-level namespace (a store with "
                    "level='customer', or one built with customer_id only). A user's "
                    "namespace never deletes the whole customer.")
            if scope == "conversation" and not conversation_id:
                raise SynapIntegrationError(
                    f"langgraph.store.{op}", "scope='conversation' needs conversation_id.")
            kwargs: Dict[str, Any] = {
                "scope": scope, "dry_run": dry_run, "confirm": confirm, "wait": wait,
                "customer_id": target.customer_id,
            }
            if scope != "customer":
                kwargs["user_id"] = target.user_id
            if conversation_id is not None:
                kwargs["conversation_id"] = conversation_id
            if instances is not None:
                kwargs["instances"] = instances
            if permanent:
                kwargs["permanent"] = True
            return await self._sdk_call(op, lambda: self.sdk.data.delete(**kwargs))
        return await self._with_sdk(go)

    def delete_data(self, namespace: tuple[str, ...], scope: str, **kwargs: Any) -> Any:
        """Sync variant of :meth:`adelete_data` (same keyword arguments)."""
        return self._sync(lambda: self.adelete_data(namespace, scope, **kwargs))

    async def arestore(
        self,
        namespace: tuple[str, ...],
        memory_ids: Optional[list[str]] = None,
        deletion_id: Optional[str] = None,
    ) -> Any:
        """``sdk.data.restore``: bring soft-deleted memories back until their
        purge date. Exactly one of ``memory_ids`` (up to 200) or
        ``deletion_id`` (from :meth:`adelete_data`). Returns the SDK's
        ``RestoreResult`` (ids that could not be restored are in
        ``not_restorable_ids``).

        ⚠ The ids are NOT checked against the namespace's identity: the server
        does not verify ownership by id, so any caller holding an id can
        restore it. The namespace is only checked for shape."""
        async def go() -> Any:
            await self._target(tuple(namespace), what="restore")
            return await self._sdk_call("restore", lambda: self.sdk.data.restore(
                memory_ids=memory_ids, deletion_id=deletion_id))
        return await self._with_sdk(go)

    def restore(self, namespace: tuple[str, ...], memory_ids: Optional[list[str]] = None,
                deletion_id: Optional[str] = None) -> Any:
        """Sync variant of :meth:`arestore`."""
        return self._sync(lambda: self.arestore(namespace, memory_ids, deletion_id))

    async def adeletion_status(self, namespace: tuple[str, ...], deletion_id: str) -> Any:
        """``sdk.data.status``: the audit record of a delete (status, counts,
        ``still_present_in``), as the SDK's ``DeleteDataResult``.

        ⚠ The deletion id is NOT checked against the namespace's identity: the
        server does not verify ownership by id. The namespace is only checked
        for shape."""
        async def go() -> Any:
            await self._target(tuple(namespace), what="deletion_status")
            return await self._sdk_call(
                "deletion_status", lambda: self.sdk.data.status(deletion_id))
        return await self._with_sdk(go)

    def deletion_status(self, namespace: tuple[str, ...], deletion_id: str) -> Any:
        """Sync variant of :meth:`adeletion_status`."""
        return self._sync(lambda: self.adeletion_status(namespace, deletion_id))

    async def aforget(
        self,
        namespace: tuple[str, ...],
        description: str,
        *,
        mode: str = "preview",
        conversation_id: Optional[str] = None,
    ) -> Any:
        """``sdk.memories.forget`` for the namespace's user: forget memories by
        describing them. ``mode="preview"`` (the SDK's default) files a request
        a person approves in the dashboard; ``"direct"`` deletes at once.
        Returns the SDK's ``ForgetResponse``. Needs a user in the namespace."""
        async def go() -> Any:
            target = await self._user_target(namespace, "forget")
            extra: Dict[str, Any] = {}
            if conversation_id is not None:
                extra["conversation_id"] = conversation_id
            return await self._sdk_call("forget", lambda: self.sdk.memories.forget(
                description, mode=mode, **extra, **target.write_kwargs))
        return await self._with_sdk(go)

    def forget(self, namespace: tuple[str, ...], description: str, **kwargs: Any) -> Any:
        """Sync variant of :meth:`aforget`."""
        return self._sync(lambda: self.aforget(namespace, description, **kwargs))

    # ── reading, raw ────────────────────────────────────────────────────────

    async def afetch_raw(
        self,
        namespace: tuple[str, ...],
        query: Union[str, Sequence[str], None] = None,
        **fetch_options: Any,
    ) -> Any:
        """``sdk.fetch`` for the namespace's identity; returns the SDK's own
        ``UnifiedContextResponse`` (typed lists, ``items`` when ranked,
        conversation context, profile, ``scope_map``, ``scopes_queried``...).

        Only what is passed is sent, so every default is the SDK's own (``mode
        ="fast"``, ``max_results=20``, ``response_shape="by_type"``,
        ``include_conversation_context=True`` ...), not the store's. ``query``
        is a string or a list of strings (``search_query``). Any ``sdk.fetch``
        argument may be passed except the identity (``user_id``,
        ``customer_id``, ``scope_path``) and ``search_query``, which raise
        ``ValueError``. A ``conversation_id`` may be any string. Errors raise
        :class:`SynapIntegrationError` whatever ``on_error`` says.
        """
        clash = sorted(_RAW_RESERVED & set(fetch_options))
        if clash:
            raise ValueError(
                f"afetch_raw: may not set {clash}: the identity comes from the "
                "namespace and the query is the `query` argument.")
        queries = _queries_of(query)

        async def go() -> Any:
            target = await self._target(tuple(namespace), what="fetch_raw")
            kwargs: Dict[str, Any] = dict(fetch_options)
            if kwargs.get("conversation_id"):
                kwargs["conversation_id"] = coerce_conversation_id(kwargs["conversation_id"])
            if queries:
                kwargs["search_query"] = queries
            if target.own_scope == _CLIENT and "scopes" not in kwargs:
                kwargs["scopes"] = [_CLIENT]
            kwargs.update(target.read_kwargs)
            return await self._sdk_call("fetch_raw", lambda: self.sdk.fetch(**kwargs))

        return await self._with_sdk(go)

    def fetch_raw(self, namespace: tuple[str, ...], query: Union[str, Sequence[str], None] = None,
                  **fetch_options: Any) -> Any:
        """Sync variant of :meth:`afetch_raw`."""
        return self._sync(lambda: self.afetch_raw(namespace, query, **fetch_options))

    async def aget_profile(self, namespace: tuple[str, ...]) -> Any:
        """``sdk.user.get_profile`` for the namespace's user: the SDK's
        ``UserProfileModel``. Needs a user in the namespace."""
        async def go() -> Any:
            target = await self._user_target(namespace, "get_profile")
            return await self._sdk_call("get_profile", lambda: self.sdk.user.get_profile(
                user_id=target.user_id, customer_id=target.customer_id))
        return await self._with_sdk(go)

    def get_profile(self, namespace: tuple[str, ...]) -> Any:
        """Sync variant of :meth:`aget_profile`."""
        return self._sync(lambda: self.aget_profile(namespace))

    async def aget_scope_ladder(self, *, refresh: bool = False) -> Any:
        """``sdk.get_scope_ladder``: the account's ``ScopeLadder`` (``None`` when
        it cannot be determined)."""
        return await self._with_sdk(lambda: self._sdk_call(
            "get_scope_ladder", lambda: self.sdk.get_scope_ladder(refresh=refresh)))

    def get_scope_ladder(self, *, refresh: bool = False) -> Any:
        """Sync variant of :meth:`aget_scope_ladder`."""
        return self._sync(lambda: self.aget_scope_ladder(refresh=refresh))

    # ── conversations ───────────────────────────────────────────────────────

    async def aingest_transcript(
        self,
        namespace: tuple[str, ...],
        conversation_id: str,
        transcript: Any,
        *,
        conversation_type: Optional[str] = None,
        analysis: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
    ) -> Any:
        """``sdk.conversation.ingest_transcript`` for the namespace's user: push a
        whole conversation (a string or a list of turns) for ingestion. Returns
        the SDK's ``TranscriptIngestResponse``; wait on its ingestion with
        :meth:`await_ingestion`. Needs a user in the namespace. The
        transcript is filed under the namespace's user; ``conversation_id``
        itself is not checked against that identity by the server."""
        async def go() -> Any:
            target = await self._user_target(namespace, "ingest_transcript")
            extra: Dict[str, Any] = {"scope_path": dict(target.scope_path)} if target.scope_path else {}
            return await self._sdk_call(
                "ingest_transcript", lambda: self.sdk.conversation.ingest_transcript(
                    conversation_id=conversation_id, user_id=target.user_id,
                    transcript=transcript, customer_id=target.customer_id,
                    conversation_type=conversation_type, analysis=analysis,
                    metadata=metadata, started_at=started_at, ended_at=ended_at, **extra))
        return await self._with_sdk(go)

    def ingest_transcript(self, namespace: tuple[str, ...], conversation_id: str,
                          transcript: Any, **kwargs: Any) -> Any:
        """Sync variant of :meth:`aingest_transcript`."""
        return self._sync(lambda: self.aingest_transcript(
            namespace, conversation_id, transcript, **kwargs))

    async def aget_context_for_prompt(
        self,
        conversation_id: str,
        *,
        style: str = "structured",
        search_history: bool = True,
        query: Optional[str] = None,
    ) -> Any:
        """``sdk.conversation.context.get_context_for_prompt``: the compacted
        conversation plus recent turns, ready for a prompt. Returns the SDK's
        ``ContextForPromptResponse``. (``synap_st_prompt`` /
        ``create_synap_st_node`` wire the same call into a graph.)

        ⚠ ``conversation_id`` is not checked against any identity: the server
        does not verify who owns a conversation id, so any caller holding one
        can act on it."""
        return await self._with_sdk(lambda: self._sdk_call(
            "get_context_for_prompt",
            lambda: self.sdk.conversation.context.get_context_for_prompt(
                conversation_id=conversation_id, style=style,
                search_history=search_history, query=query)))

    def get_context_for_prompt(self, conversation_id: str, **kwargs: Any) -> Any:
        """Sync variant of :meth:`aget_context_for_prompt`."""
        return self._sync(lambda: self.aget_context_for_prompt(conversation_id, **kwargs))

    async def acompact(
        self,
        conversation_id: str,
        *,
        strategy: Optional[str] = None,
        compaction_level: Optional[str] = None,
        target_tokens: Optional[int] = None,
        force: bool = False,
    ) -> Any:
        """``sdk.conversation.context.compact``: start compacting a conversation
        (asynchronous on the server). ``compaction_level`` is the SDK's
        backward-compatible alias for ``strategy``. Returns the SDK's
        ``CompactionTriggerResponse``.

        ⚠ ``conversation_id`` is not checked against any identity: the server
        does not verify who owns a conversation id, so any caller holding one
        can act on it."""
        return await self._with_sdk(lambda: self._sdk_call(
            "compact", lambda: self.sdk.conversation.context.compact(
                conversation_id=conversation_id, strategy=strategy,
                compaction_level=compaction_level, target_tokens=target_tokens,
                force=force)))

    def compact(self, conversation_id: str, **kwargs: Any) -> Any:
        """Sync variant of :meth:`acompact`."""
        return self._sync(lambda: self.acompact(conversation_id, **kwargs))

    async def aget_compacted(self, conversation_id: str, *, version: Optional[int] = None,
                             format: str = "structured") -> Any:
        """``sdk.conversation.context.get_compacted``: the SDK's
        ``CompactionResponse``, or ``None`` when there is none yet.

        ⚠ ``conversation_id`` is not checked against any identity: the server
        does not verify who owns a conversation id, so any caller holding one
        can act on it."""
        return await self._with_sdk(lambda: self._sdk_call(
            "get_compacted", lambda: self.sdk.conversation.context.get_compacted(
                conversation_id=conversation_id, version=version, format=format)))

    def get_compacted(self, conversation_id: str, **kwargs: Any) -> Any:
        """Sync variant of :meth:`aget_compacted`."""
        return self._sync(lambda: self.aget_compacted(conversation_id, **kwargs))

    async def aget_compaction_status(self, conversation_id: str) -> Any:
        """``sdk.conversation.context.get_compaction_status``: the SDK's
        ``CompactionStatusResponse``.

        ⚠ ``conversation_id`` is not checked against any identity: the server
        does not verify who owns a conversation id, so any caller holding one
        can act on it."""
        return await self._with_sdk(lambda: self._sdk_call(
            "get_compaction_status", lambda: self.sdk.conversation.context.get_compaction_status(
                conversation_id=conversation_id)))

    def get_compaction_status(self, conversation_id: str) -> Any:
        """Sync variant of :meth:`aget_compaction_status`."""
        return self._sync(lambda: self.aget_compaction_status(conversation_id))

    # ── anticipation (not part of BaseStore) ────────────────────────────────

    async def arecord_message(
        self,
        conversation_id: str,
        role: str,
        content: Union[str, Sequence[Any]],
        *,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        namespace: Optional[tuple[str, ...]] = None,
        model: Optional[str] = None,
        context_window: Optional[int] = None,
    ) -> None:
        """Feed a turn to Synap's conversation channel so anticipation can
        surface just-stated context on subsequent reads (construct the store
        with ``include_conversation_context=True``).

        Anticipation has no key/value analogue, so this lives alongside, not
        inside, the BaseStore contract; it lets a LangGraph node drive the
        conversation channel without reaching for the raw SDK. Best-effort: a
        failure here is logged and swallowed so it never breaks an agent turn.

        On the gRPC stream, which the first call opens, and over REST when the
        stream cannot carry it; never both. The server persists
        ``user_message`` and ``assistant_message`` from the stream itself, so a
        REST write on top of a delivered stream event would write the turn
        twice. ``customer_id`` is optional: a B2C instance refuses one, so the
        store is built with ``user_id`` alone there.

        On a shared store pass ``namespace`` (it starts with the identity, as
        for every other call); without it the call raises, because a turn must
        not be filed under a guessed user.

        ⚠ ``session_id`` is not sent on that path: the stream owns the
        session, and the shared turn writer has no field for one.

        ``model``, ``context_window`` and list-of-parts ``content`` (strings,
        ``TextPart``, ``FilePart``) are ``sdk.conversation.record_message``'s
        own arguments, with its names and defaults (``None``). A call that uses
        any of them is sent with ``sdk.conversation.record_message`` (REST),
        exactly as the SDK sends it, ``session_id`` included, and not on the
        stream (the shared stream writer carries text only). Still
        best-effort: a failure is logged and swallowed. A call without them
        is unchanged.

        ⚠ ``conversation_id`` is not checked against the namespace's identity:
        the server does not verify who owns a conversation id.
        """
        if self._sdk_provider is not None and (_ACTIVE_SDK.get() or (None,))[0] is not self:
            token = _ACTIVE_SDK.set((self, await self._resolve_sdk()))
            try:
                return await self.arecord_message(
                    conversation_id, role, content, session_id=session_id,
                    metadata=metadata, namespace=namespace, model=model,
                    context_window=context_window)
            finally:
                _ACTIVE_SDK.reset(token)
        if self._shared:
            if namespace is None:
                raise SynapIntegrationError(
                    "langgraph.store.record_message",
                    "a shared SynapStore needs `namespace=` (starting with the "
                    "identity) to know whose turn this is.")
            target = await self._target(tuple(namespace), what="record_message")
            user_id, customer_id, path = target.user_id, target.customer_id, target.scope_path
        else:
            user_id, customer_id, path = self.user_id, self.customer_id, self.scope_path
        if not user_id:
            logger.warning(
                "SynapStore.record_message needs a user_id (a conversation turn "
                "belongs to a user); skipping."
            )
            return
        if model is not None or context_window is not None or not isinstance(content, str):
            # The SDK's own record_message carries these; the shared stream
            # writer does not. One transport only, so never written twice.
            rest: Dict[str, Any] = {
                k: v for k, v in (("customer_id", customer_id or None),
                                  ("session_id", session_id), ("metadata", metadata),
                                  ("scope_path", dict(path) if path else None),
                                  ("model", model), ("context_window", context_window))
                if v is not None}
            try:
                await self.sdk.conversation.record_message(
                    conversation_id=coerce_conversation_id(conversation_id), role=role,
                    content=list(content) if not isinstance(content, str) else content,
                    user_id=user_id, **rest)
            except Exception as exc:  # noqa: BLE001 — best-effort, like the stream path
                logger.error("SynapStore.record_message failed (REST): %s", exc, exc_info=True)
            return
        # Stream metadata is string key/value on the wire; coerce rather than
        # let a non-string value fail the send and silently demote the turn to
        # the REST path.
        stream_metadata = {str(k): str(v) for k, v in (metadata or {}).items()}
        extra: Dict[str, Any] = {"scope_path": path} if path else {}
        await record_turn(
            self.sdk,
            role=role,
            content=content,
            conversation_id=conversation_id,
            user_id=user_id,
            customer_id=customer_id or "",
            metadata=stream_metadata,
            **extra,
        )

    def record_message(
        self,
        conversation_id: str,
        role: str,
        content: Union[str, Sequence[Any]],
        *,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        namespace: Optional[tuple[str, ...]] = None,
        model: Optional[str] = None,
        context_window: Optional[int] = None,
    ) -> None:
        """Sync wrapper around :meth:`arecord_message`."""
        pending = self._sdk_provider() if self._sdk_provider is not None else None

        async def _go() -> None:
            token = None
            if pending is not None:
                token = _ACTIVE_SDK.set((self, await self._resolve_sdk(pending)))
            try:
                await self.arecord_message(
                    conversation_id, role, content,
                    session_id=session_id, metadata=metadata, namespace=namespace,
                    model=model, context_window=context_window)
            finally:
                if token is not None:
                    _ACTIVE_SDK.reset(token)

        run_async(_go())


# ── helpers ────────────────────────────────────────────────────────────────

_BUCKETS = ("facts", "preferences", "episodes", "emotions", "temporal_events")


def _given_create_options(**options: Any) -> Dict[str, Any]:
    """The ``memories.create`` options a caller actually gave (``None`` = not given,
    so the SDK's own default applies)."""
    return {k: v for k, v in options.items() if v is not None}


def _caller_metadata(metadata: Optional[Mapping[str, Any]], where: str) -> Dict[str, Any]:
    """A caller's ``metadata`` for ``memories.create``; it may not set the
    store's own tag keys."""
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{where}: metadata must be a mapping, got {type(metadata).__name__}")
    clash = sorted(set(metadata) & set(_TAG_KEYS))
    if clash:
        raise ValueError(f"{where}: metadata may not set the store's own keys {clash}")
    return dict(metadata)


def _check_put(namespace: tuple[str, ...], value: Any, ttl: Any) -> None:
    """``BaseStore.put``'s own checks, for a put that does not go through it."""
    from langgraph.store.base import _validate_namespace

    _validate_namespace(namespace)
    if value is None:
        raise SynapStoreUnsupportedError("delete", _DELETE_ALTERNATIVE)
    if ttl not in (NOT_PROVIDED, None):
        raise NotImplementedError(
            "TTL is not supported by SynapStore. Use a store implementation that "
            "supports TTL or set ttl=None.")


def _iter_items(response: Any) -> list[tuple[Any, str]]:
    """``(item, category)`` for every memory in a fetch response, best first.

    Reads ``response.items`` (the ranked list, ``response_shape="ranked"``) in
    the order the server ranked it. A response without one (a server that
    predates ranking, or a by-type answer) falls back to the typed lists in
    bucket order, which is NOT a ranking; those items carry no rank.
    """
    ranked = getattr(response, "items", None)
    if ranked:
        return [(it, str(getattr(it, "category", "") or "")) for it in ranked]
    out: list[tuple[Any, str]] = []
    for bucket in _BUCKETS:
        out.extend((it, bucket) for it in getattr(response, bucket, None) or [])
    return out


def _queries_of(query: Union[str, Sequence[str], None]) -> list[str]:
    """The search queries of one ``search`` call: a string is one query, a list or
    tuple of strings is several (sent in ONE fetch). Blank and repeated entries
    are dropped, order kept. Anything else raises."""
    if query is None:
        return []
    if isinstance(query, str):
        parts: Sequence[Any] = [query]
    elif isinstance(query, (list, tuple)):
        parts = query
    else:
        raise SynapIntegrationError(
            "langgraph.store.search",
            f"query must be a string or a list of strings, got {type(query).__name__}.")
    out: list[str] = []
    for part in parts:
        if not isinstance(part, str):
            raise SynapIntegrationError(
                "langgraph.store.search",
                f"every query must be a string, got {type(part).__name__} in the list.")
        if part.strip() and part not in out:
            out.append(part)
    return out


def _unanswered_scopes(kwargs: Mapping[str, Any], response: Any) -> list[str]:
    """Scopes ``sdk.fetch`` was asked to read that are not in the answer's
    ``scopes_queried``.

    ``sdk.fetch`` reads each scope separately and drops one that raised from the
    merged answer without raising; ``scopes_queried`` lists the scopes that
    answered. Which scopes it reads follows from its arguments: ``user`` when a
    ``user_id`` is given, ``customer`` for a ``customer_id``, ``client`` when
    asked for by ``scopes``, and ``conversation`` only when a ``conversation_id``
    AND a query are given. With no query the conversation scope is not a
    sub-fetch at all: its state comes from ``get_context_for_prompt``, whose
    failure is only debug-logged by the SDK and is not an outage of a memory
    scope, so it must not be expected here.

    An EMPTY ``scopes_queried`` from the real ``sdk.fetch`` means every scope
    it ran failed: the SDK builds the list from the scopes that answered
    (``UnifiedContextResponse.merge``), a healthy empty memory still lists them,
    and a call that runs no scope at all is the case where nothing was asked
    (the list of asked scopes above is then empty too). So empty means "all
    failed" and is reported. The one exception is an answer that carries items
    while listing no scopes: an answer cannot exist if no scope answered, so
    that is a stand-in SDK that does not fill the list, and it is taken as
    complete. A response without a ``scopes_queried`` list at all cannot be
    checked and counts as complete too.
    """
    answered = getattr(response, "scopes_queried", None)
    if not isinstance(answered, (list, tuple)):
        return []
    if not answered and (getattr(response, "items", None) or any(
            getattr(response, bucket, None) for bucket in _BUCKETS)):
        return []  # items without any answering scope: a stand-in that leaves the list empty
    limit = kwargs.get("scopes")
    asked: list[str] = []
    has_query = any(isinstance(q, str) and q.strip() for q in (kwargs.get("search_query") or []))
    for scope, wanted in (
        ("conversation", bool(kwargs.get("conversation_id")) and has_query),
        ("user", bool(kwargs.get("user_id"))),
        ("customer", bool(kwargs.get("customer_id"))),
        ("client", _CLIENT in (limit or ())),
    ):
        if wanted and (not limit or scope in limit):
            asked.append(scope)
    return [s for s in asked if s not in answered]


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


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if isinstance(value, datetime) else None


def _item_date(item: Any) -> tuple[datetime, bool]:
    """``(date, known)``: the day the user said it, else the event date.

    Never the fetch time (the SDK fills ``extracted_at`` with it when the
    server sends none). With neither date: :data:`UNKNOWN_DATE`, ``False``.
    """
    for attr in ("said_at", "event_date"):
        value = getattr(item, attr, None)
        if isinstance(value, datetime):
            return value, True
    return UNKNOWN_DATE, False


def synap_meta(item: Any, category: str = "", response: Any = None) -> dict[str, Any]:
    """Synap's own data for a memory item, as stored in ``value["_synap"]``.

    Keys: ``id``; ``category`` (the server's, e.g. ``"facts"``) and ``type``
    (singular, e.g. ``"fact"``); ``scope`` (``metadata.scope``, else the
    response's ``scope_map``: ``"user"`` / ``"customer"`` / ``"client"`` ...);
    ``assertion_source`` (who said it); ``said_at`` / ``event_date`` /
    ``valid_until`` (ISO strings or ``None``); ``dates_known``; ``source``;
    ``rank`` / ``rank_score`` (``None`` when the response was not ranked);
    ``confidence``; and ``metadata``: any other metadata keys the item carries
    (the store's own tag keys are left out).
    """
    md = dict(getattr(item, "metadata", None) or {})
    item_id = str(getattr(item, "id", "") or "")
    scope_map = getattr(response, "scope_map", None) or {}
    _, known = _item_date(item)
    confidence = getattr(item, "confidence", None)
    if confidence is None:
        confidence = getattr(item, "strength", None)
    rank_score = getattr(item, "rank_score", None)
    rank = getattr(item, "rank", None)
    out: dict[str, Any] = {
        "id": item_id,
        "category": category,
        "type": _SINGULAR.get(category, category),
        "scope": md.get("scope") or scope_map.get(item_id),
        "assertion_source": md.get("assertion_source") or getattr(item, "assertion_source", None),
        "said_at": _iso(getattr(item, "said_at", None)),
        "event_date": _iso(getattr(item, "event_date", None)),
        "valid_until": _iso(getattr(item, "valid_until", None)),
        "dates_known": known,
        "source": getattr(item, "source", None) or None,
        "rank": rank if isinstance(rank, int) and rank > 0 else None,
        "rank_score": float(rank_score) if isinstance(rank_score, (int, float)) else None,
        "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
    }
    rest = {k: v for k, v in md.items()
            if k not in ("scope", "assertion_source") + _TAG_KEYS}
    if rest:
        out["metadata"] = rest
    return out


def _value_of(item: Any, category: str, response: Any) -> dict[str, Any]:
    text = _item_content(item)
    value = _parse_value(text)
    value["_synap"] = synap_meta(item, category, response)
    if "_raw" not in value and text:
        # an object was parsed: keep the exact text it came from
        value["_synap"]["text"] = text
    return value


def _item_to_item(
    item: Any, category: str, response: Any, namespace: tuple[str, ...], key: str,
) -> Item:
    when, _ = _item_date(item)
    return Item(
        value=_value_of(item, category, response),
        key=key,
        namespace=namespace,
        created_at=when,
        updated_at=when,
    )


def _item_to_search_item(
    item: Any, category: str, response: Any, namespace: tuple[str, ...], key: str,
) -> SearchItem:
    when, _ = _item_date(item)
    score = getattr(item, "rank_score", None)
    return SearchItem(
        namespace=namespace,
        key=key,
        value=_value_of(item, category, response),
        created_at=when,
        updated_at=when,
        score=float(score) if isinstance(score, (int, float)) else None,
    )


def _parse_value(document: str) -> dict[str, Any]:
    """The ``value`` dict for a memory's text.

    ``put`` stores ``json.dumps(value)``, always an object, so text that parses
    as a JSON **object** is read back as the dict that was put (the exact text
    stays in ``value["_synap"]["text"]``). Any other text is returned exactly as
    written, as ``{"_raw": text}``: a bare ``1.50`` is not ``1.5``, ``true`` is
    not ``True``, ``"quoted"`` keeps its quotes, ``1e3`` and ``007`` stay as
    written. (Before 0.5.1 a scalar or list came back parsed, as ``_value``.)
    """
    if not document:
        return {}
    try:
        parsed = json.loads(document)
    except (ValueError, TypeError):
        return {"_raw": document}
    return parsed if isinstance(parsed, dict) else {"_raw": document}


__all__ = ["SynapStore", "SynapStoreUnsupportedError", "synap_meta", "UNKNOWN_DATE"]
