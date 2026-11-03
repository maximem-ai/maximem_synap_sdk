"""SynapStore: LangGraph ``BaseStore`` backed by Synap semantic memory.

Implements ``batch`` / ``abatch`` (the only abstract methods) and dispatches
individual ops (:class:`PutOp`, :class:`GetOp`, :class:`SearchOp`,
:class:`ListNamespacesOp`) to async helpers. The convenience API
(``put``/``get``/``search``/``delete``/``list_namespaces`` + async variants)
is inherited unchanged from ``BaseStore`` and rides through ``(a)batch``.

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
  and tags it with its sub-namespace and key in ``metadata``.
* ``get`` retrieves with ``sdk.fetch`` and returns the item carrying the exact
  sub-namespace and key tag.
* ``search`` calls ``sdk.fetch(response_shape="ranked")`` for exactly
  ``limit + offset`` places (the server fills a top-N differently from a top-M,
  so the store does not ask for more than the caller wants). It asks for at
  least 10 only when it drops items itself afterwards: a sub-namespace in the
  prefix, a ``filter``, or ``semantic_fallback=False``. It returns the SDK's
  ranked list best first, ``rank_score`` as ``score``. Items that carry the
  store's tag (written by ``put``) are filtered by the requested sub-namespace
  prefix; items without it (memories Synap extracted from conversations) belong
  to the identity scope and are always returned. Nothing is dropped because
  another item carries the tag.
* Every item's ``value`` holds the text (the dict for text that is a JSON object,
  which is what ``put`` writes, with the exact text in ``value["_synap"]["text"]``;
  anything else exactly as written, ``{"_raw": text}``, never re-typed: ``1.50``
  stays ``"1.50"``, ``true`` stays ``"true"``) and ``value["_synap"]``: Synap's own data for it (see
  :func:`synap_meta`).
* ``created_at`` / ``updated_at`` are the day the user said it (``said_at``),
  else the event date, never the fetch time. With neither, they are the epoch
  (1970-01-01 UTC) and ``value["_synap"]["dates_known"]`` is ``False``.
* ``delete`` finds the memory by its tag under the identity and calls
  ``sdk.memories.delete``. A memory that cannot be found raises, and no
  ``delete`` call is ever made on a guess.
* ``get`` and ``delete`` act on the store's OWN level only. A shared store (and
  a single-user store pinned by a ``scope_path``) tags each ``put`` with its
  identity entries (``lg_store_id``); a memory without that tag or with another
  level's is never returned by ``get`` and never deleted. ``search`` returns
  ancestors' memories too, each with its true namespace and scope.

⚠ **Today's Synap server does not return caller-supplied memory metadata.**
The tag ``put`` writes (``lg_store*``) is accepted but never comes back from
``fetch``. So against today's server ``get(namespace, key)`` returns ``None``
for everything, ``delete(namespace, key)`` raises (it cannot find the memory)
and ``search`` is the read path: it returns every memory as an ordinary
memory, without sub-namespace filtering. ``get`` and ``delete`` work only
against a server that returns the tags. To delete today, take the memory's id
from the ``search`` results (``value["_synap"]["id"]``) and call
``sdk.memories.delete(memory_id)``, or ``sdk.memories.forget(...)``.

* ``list_namespaces`` is a documented no-op that returns ``[]``: Synap has no
  namespace enumeration.

Error policy
------------

* Writes and ``delete`` raise :class:`SynapIntegrationError` on SDK failure.
* Identity errors (short namespace, unknown level, no ladder) always raise.
* Reads follow ``on_error``: ``"fallback"`` (default) logs at ERROR and returns
  ``None`` / ``[]``; ``"raise"`` raises :class:`SynapIntegrationError`, so a
  failed read is distinguishable from an empty memory. This includes a read
  where ``sdk.fetch`` swallowed a failed scope ("non-fatal") and returned only
  the scopes that answered: the store compares the scopes it asked for with the
  answer's ``scopes_queried`` and, with ``"raise"``, raises (naming the scopes);
  with ``"fallback"`` it logs a WARNING and uses what answered. A ``delete``
  that meets such a gap always raises (it would otherwise say "no memory found").

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
        self._listns_warned = False
        self._delete_warned = False
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
        return await self._run_ops(list(ops), None, self._call_options())

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        # Provider and options are read here, in the caller's thread and
        # context; the coroutine runs on the bridge's loop, which has neither.
        pending = self._sdk_provider() if self._sdk_provider is not None else None
        return run_async(self._run_ops(list(ops), pending, self._call_options()))

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
        *,
        own_scope_only: bool = False,
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
        elif own_scope_only:
            kwargs["scopes"] = [target.own_scope]
        response = await self.sdk.fetch(**kwargs)
        return response, _unanswered_scopes(kwargs, response)

    def _incomplete(self, op: str, missing: list[str], *, strict: bool, **context: Any) -> None:
        """A scope the fetch asked for did not answer.

        ``sdk.fetch`` catches a scope that raised ("non-fatal") and returns the
        scopes that did answer; the only trace is that the scope is missing from
        ``scopes_queried``. That is an outage, not empty memory. Always logged;
        raised when ``on_error="raise"`` (or ``strict``: a delete, which would
        otherwise report a missing memory). With ``on_error="fallback"`` the
        partial answer is used, as before.
        """
        message = (f"the memory read is incomplete: scope(s) {missing} failed and were "
                   "left out of the answer by sdk.fetch (see the 'non-fatal' warning "
                   "above). An outage looks like empty memory without this check.")
        if strict or self.on_error == "raise":
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

    async def _aget(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> Optional[Item]:
        target = await self._target(namespace, what="get")
        sub = _ns_str(target.sub)
        self._check_fetch_support(target)
        try:
            response, missing = await self._fetch(target, [f"{sub} {key}"], 50)
        except Exception as exc:  # noqa: BLE001 — policy decided by on_error
            self._read_failed(
                "langgraph.store.get", exc,
                "SynapStore.get: sdk.fetch failed namespace=%s key=%s error=%s",
                _h(sub), _h(key), exc, namespace=_h(sub), key=_h(key))
            return None
        if missing:
            self._incomplete("langgraph.store.get", missing, strict=False,
                             namespace=_h(sub), key=_h(key))

        # Only an item at this store's OWN level: an ancestor's memory (a
        # customer pool item with the same sub-namespace and key) is never
        # presented as the caller's own. `search` still returns ancestors, with
        # their true namespace and scope.
        for item, category in _iter_items(response):
            md = getattr(item, "metadata", None) or {}
            if (md.get(_MARKER) and md.get(_NS, "") == sub and md.get(_KEY) == key
                    and self._is_own(target, md)):
                return _item_to_item(item, category, response, target.identity + target.sub, key)
        return None

    def _is_own(self, target: _Target, md: Dict[str, Any]) -> bool:
        """Whether a tagged memory is provably at THIS store's own level.

        Shared store: its identity tag equals this call's identity entries
        (a team memory and a customer memory differ there; the server's scope
        label cannot tell them apart). A memory with no identity tag (written
        by a single-user store) is not provably ours. Single-user store: the
        server's scope label, when there is one, must be the store's own.
        """
        if self._shared:
            if md.get(_ID) != target.id_tag:
                return False
            if target.user_id and md.get("scope") and md.get("scope") != target.own_scope:
                return False
            return True
        if target.id_tag is not None:  # single-user store pinned by a scope_path
            return md.get(_ID) == target.id_tag and (
                not md.get("scope") or md.get("scope") == target.own_scope)
        label = md.get("scope")
        return not label or label == target.own_scope

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
            self._incomplete("langgraph.store.search", missing, strict=False,
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
        value: dict[str, Any],
        index: Any = None,
    ) -> None:
        target = await self._target(namespace, what="put")
        sub = _ns_str(target.sub)
        # `get` / `search` put a `_synap` block in the value; writing a value
        # back must not store that block in the memory text.
        value = {k: v for k, v in value.items() if k != "_synap"}
        document = json.dumps(value, default=str)
        metadata = {
            _MARKER: True,
            _NS: sub,
            _KEY: key,
        }
        if target.id_tag is not None:
            metadata[_ID] = target.id_tag
        h_sub, h_key = _h(sub), _h(key)
        async with wrap_sdk_errors_async(
            "langgraph.store.put",
            logger,
            namespace=h_sub,
            key=h_key,
        ):
            await self.sdk.memories.create(
                document=document,
                metadata=metadata,
                **target.write_kwargs,
            )

    async def _adelete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        """Delete what ``put`` stored under ``(namespace, key)``.

        Finds the memories by the store's tag among the identity's OWN level
        (never a level above it: a user store cannot delete the customer's
        pool), then ``sdk.memories.delete`` each. Raises when none is found:
        ``put`` is ingested asynchronously and extraction may rewrite the
        text, so a very recent ``put`` may not be findable yet.
        """
        op = "langgraph.store.delete"
        target = await self._target(namespace, what="delete")
        sub = _ns_str(target.sub)
        if not self._shared and not self._delete_warned:
            self._delete_warned = True
            logger.warning(
                "SynapStore.delete on a store built with user_id=%s customer_id=%s: "
                "deletes act on that construction identity whatever namespace is "
                "passed. Use a shared store (no ids) if this graph serves several "
                "users. This warning fires once.",
                _h(self.user_id) if self.user_id else None,
                _h(self.customer_id) if self.customer_id else None)
        self._check_fetch_support(target)
        h_sub, h_key = _h(sub), _h(key)
        async with wrap_sdk_errors_async(op, logger, namespace=h_sub, key=h_key):
            response, missing = await self._fetch(
                target, [f"{sub} {key}"], 50, own_scope_only=True)
        if missing:
            self._incomplete(op, missing, strict=True, namespace=h_sub, key=h_key)

        ids: list[UUID] = []
        for item, _category in _iter_items(response):
            md = getattr(item, "metadata", None) or {}
            if not (md.get(_MARKER) and md.get(_NS, "") == sub and md.get(_KEY) == key):
                continue
            # Provably this store's own level, and labelled by the server. The
            # response's scope_map is NOT used: the SDK fills it with the name
            # of the fetch that returned the item, which says nothing about
            # the item's own level.
            if not md.get("scope") or not self._is_own(target, md):
                continue
            try:
                ids.append(UUID(str(getattr(item, "id", ""))))
            except ValueError:
                continue  # e.g. a chunk id: not a memory
        if not ids:
            raise SynapIntegrationError(
                op,
                f"no memory found for namespace={tuple(namespace)!r} key={key!r}. "
                "Nothing was deleted. Synap's server does not return the metadata "
                "a caller attaches to a memory, so the tag this store writes on "
                "`put` normally cannot be read back and `delete(namespace, key)` "
                "cannot find anything. To delete, take the memory's id from the "
                "`fetch`/`search` results (`value[\"_synap\"][\"id\"]`) and call "
                "`sdk.memories.delete(memory_id)`, or describe it to "
                "`sdk.memories.forget(description, user_id=..., ...)`. (Also: a "
                "put is ingested asynchronously; a key never put is not found; a "
                "memory with no identity tag is not provably this level's.)",
                {"namespace": h_sub, "key": h_key},
            )
        for memory_id in dict.fromkeys(ids):
            async with wrap_sdk_errors_async(
                op, logger, namespace=h_sub, key=h_key, memory_id=str(memory_id),
            ):
                await self.sdk.memories.delete(memory_id)

    async def _alist_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        if not self._listns_warned:
            logger.warning(
                "SynapStore.list_namespaces: Synap does not expose a namespace "
                "enumeration API; returning []. This warning fires once."
            )
            self._listns_warned = True
        return []

    # ── anticipation (not part of BaseStore) ────────────────────────────────

    async def arecord_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        namespace: Optional[tuple[str, ...]] = None,
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

        ⚠ ``session_id`` is not sent: the stream owns the session, and the
        shared turn writer has no field for one.
        """
        if self._sdk_provider is not None and (_ACTIVE_SDK.get() or (None,))[0] is not self:
            token = _ACTIVE_SDK.set((self, await self._resolve_sdk()))
            try:
                return await self.arecord_message(
                    conversation_id, role, content, session_id=session_id,
                    metadata=metadata, namespace=namespace)
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
        content: str,
        *,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        namespace: Optional[tuple[str, ...]] = None,
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
                    session_id=session_id, metadata=metadata, namespace=namespace)
            finally:
                if token is not None:
                    _ACTIVE_SDK.reset(token)

        run_async(_go())


# ── helpers ────────────────────────────────────────────────────────────────

_BUCKETS = ("facts", "preferences", "episodes", "emotions", "temporal_events")


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


__all__ = ["SynapStore", "synap_meta", "UNKNOWN_DATE"]
