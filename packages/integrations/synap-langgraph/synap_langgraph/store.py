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
* ``search`` calls ``sdk.fetch(response_shape="ranked")`` and returns the SDK's
  ranked list best first, ``rank_score`` as ``score``. Items that carry the
  store's tag (written by ``put``) are filtered by the requested sub-namespace
  prefix; items without it (memories Synap extracted from conversations) belong
  to the identity scope and are always returned. Nothing is dropped because
  another item carries the tag.
* Every item's ``value`` holds the text (the parsed JSON for a ``put`` item, or
  ``{"_raw": text}``) and ``value["_synap"]``: Synap's own data for it (see
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
  failed read is distinguishable from an empty memory.

Anticipation (optional, outside the BaseStore contract): construct with
``include_conversation_context=True`` and drive the conversation channel via
:meth:`SynapStore.record_message` so just-stated context is in play on reads.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional
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

_SINGULAR = {
    "facts": "fact", "preferences": "preference", "episodes": "episode",
    "emotions": "emotion", "temporal_events": "temporal_event", "excerpts": "excerpt",
}


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
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
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
    ) -> None:
        if sdk is None:
            raise ValueError("SynapStore requires a non-None sdk")
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
        if level == _CLIENT and not allow_client_level:
            raise ValueError(
                "SynapStore(level='client') has no identity: it reads and writes "
                "the whole account's shared memory. Pass allow_client_level=True "
                "if that is what you want.")
        if level is not None and (not isinstance(level, str) or not level.strip()):
            raise ValueError(f"SynapStore: level must be a ladder level key, got {level!r}")
        options = dict(fetch_options or {})
        clash = sorted(_RESERVED_FETCH_OPTIONS & set(options))
        if clash:
            raise ValueError(
                f"SynapStore: fetch_options may not set {clash}: the store sets "
                "them itself (identity from the namespace or constructor, the "
                "query and limit from each call, response_shape='ranked').")

        super().__init__()
        self.sdk = sdk
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
        query: Optional[str],
        max_results: int,
        *,
        own_scope_only: bool = False,
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "mode": self.mode,
            "include_conversation_context": self.include_conversation_context,
        }
        kwargs.update(self.fetch_options)
        kwargs.update(target.read_kwargs)
        kwargs["search_query"] = [query] if query else None
        kwargs["max_results"] = max_results
        kwargs["response_shape"] = "ranked"
        if target.own_scope == _CLIENT:
            kwargs["scopes"] = [_CLIENT]
        elif own_scope_only:
            kwargs["scopes"] = [target.own_scope]
        return await self.sdk.fetch(**kwargs)

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
            response = await self._fetch(target, f"{sub} {key}", 50)
        except Exception as exc:  # noqa: BLE001 — policy decided by on_error
            self._read_failed(
                "langgraph.store.get", exc,
                "SynapStore.get: sdk.fetch failed namespace=%s key=%s error=%s",
                _h(sub), _h(key), exc, namespace=_h(sub), key=_h(key))
            return None

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

    async def _asearch(
        self,
        namespace_prefix: tuple[str, ...],
        query: Optional[str],
        filter_: Optional[dict[str, Any]],
        limit: int,
        offset: int,
    ) -> list[SearchItem]:
        target = await self._target(namespace_prefix, what="search")
        sub_prefix = target.sub
        q = query or _ns_str(sub_prefix) or ""
        self._check_fetch_support(target)
        try:
            response = await self._fetch(target, q, max(limit + offset, 10))
        except Exception as exc:  # noqa: BLE001 — policy decided by on_error
            self._read_failed(
                "langgraph.store.search", exc,
                "SynapStore.search: sdk.fetch failed namespace_prefix=%s query_len=%s error=%s",
                _h(_ns_str(namespace_prefix)), len(query or ""), exc,
                namespace_prefix=_h(_ns_str(namespace_prefix)), query_len=len(query or ""))
            return []

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
            response = await self._fetch(target, f"{sub} {key}", 50, own_scope_only=True)

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
        run_async(
            self.arecord_message(
                conversation_id, role, content,
                session_id=session_id, metadata=metadata, namespace=namespace,
            )
        )


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
    value = _parse_value(_item_content(item))
    value["_synap"] = synap_meta(item, category, response)
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
    if not document:
        return {}
    try:
        parsed = json.loads(document)
    except (ValueError, TypeError):
        return {"_raw": document}
    if isinstance(parsed, dict):
        return parsed
    return {"_value": parsed}


__all__ = ["SynapStore", "synap_meta", "UNKNOWN_DATE"]
