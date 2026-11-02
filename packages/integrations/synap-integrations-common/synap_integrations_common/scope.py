"""Scope path construction for Synap integrations.

Synap's scope chain narrows from WORLD -> CLIENT -> CUSTOMER -> USER. When
an integration caches a record locally (CrewAI) or annotates a framework
record with its origin scope (LlamaIndex, LangChain), it needs a consistent
path string.

Historically the integrations did this ad-hoc: llamaindex retriever built
one path, llamaindex memory built another; CrewAI defaulted to ``/user``
while LangChain left it blank. This module is the single source of truth.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from synap_integrations_common.errors import SynapIntegrationError, wrap_sdk_errors_async

logger = logging.getLogger(__name__)

try:  # SDK >= the release that added the scope ladder
    from maximem_synap import ScopeAddress

    _SDK_HAS_LADDER = True
except ImportError:
    _SDK_HAS_LADDER = False  # older SDK: same shape, so ids and an explicit path still work

    @dataclass(frozen=True)  # type: ignore[no-redef]
    class ScopeAddress:  # noqa: D101 - mirrors maximem_synap.ScopeAddress
        user_id: Optional[str] = None
        customer_id: Optional[str] = None
        scope_path: Optional[Dict[str, str]] = None

        def read_kwargs(self) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            if self.user_id:
                out["user_id"] = self.user_id
            if self.customer_id:
                out["customer_id"] = self.customer_id
            if self.scope_path:
                out["scope_path"] = dict(self.scope_path)
            return out

        def write_kwargs(self) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            if self.user_id:
                out["user_id"] = self.user_id
            if self.customer_id:
                out["customer_id"] = self.customer_id
            if self.scope_path:
                out["scope"] = dict(self.scope_path)
            return out


def default_scope(user_id: str, customer_id: Optional[str] = None) -> str:
    """Return the canonical default scope path for a record.

    - ``default_scope("alice")`` -> ``"/alice"``
    - ``default_scope("alice", "acme")`` -> ``"/acme/alice"``
    - ``default_scope("alice", "")`` -> ``"/alice"`` (empty string treated as absent)

    ``user_id`` is required; an empty ``user_id`` raises ``ValueError``
    because a record with no user scope is ambiguous in Synap's model.
    """
    if not user_id:
        raise ValueError("default_scope requires a non-empty user_id")
    if customer_id:
        return f"/{customer_id}/{user_id}"
    return f"/{user_id}"


def scope_string(values: Sequence[str]) -> str:
    """The ``/a/b/c`` annotation string for ladder values, outermost first.

    Works at any depth: ``scope_string(["acme"])`` -> ``"/acme"``,
    ``scope_string(["acme", "payments", "dana"])`` -> ``"/acme/payments/dana"``.
    It is a label for records an integration caches or annotates, not an
    address: nothing is sent to Synap from it. Use ``resolve_address`` for that.

    Raises ``ValueError`` for no values or an empty / non-string value, because
    a label that silently drops a level names the wrong node. Values are not
    escaped, so a value containing ``/`` makes the label ambiguous.
    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("scope_string requires a list of values, outermost level first")
    vals = list(values)
    if not vals:
        raise ValueError("scope_string requires at least one value")
    for v in vals:
        if not isinstance(v, str) or not v:
            raise ValueError(f"scope_string values must be non-empty strings, got {v!r}")
    return "/" + "/".join(vals)


async def resolve_address(
    sdk: Any,
    *,
    user_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    scope_path: Optional[Mapping[str, str]] = None,
    values: Optional[Sequence[str]] = None,
) -> "ScopeAddress":
    """Turn whatever identity a caller gave into a ``ScopeAddress``.

    Priority, first match wins:

    1. ``scope_path`` set: used as given, together with the ids as given. No
       network call. An empty dict counts as not set, like the SDK's
       ``if scope_path:``: it silently falls back to the ids / ``values``
       (a WARNING is logged when it is passed together with either).
    2. ``values`` set (outermost level first): ``await sdk.get_scope_ladder()``
       then ``ladder.address(values)``. Needs a ladder; with none this raises
       rather than guess.
    3. Otherwise the plain ids, no network call, no path.

    Then use ``address.read_kwargs()`` for ``fetch`` / ``send_message`` /
    ``record_message`` and ``address.write_kwargs()`` for ``memories.create`` /
    ``forget`` (the path is ``scope_path`` for reads, ``scope`` for memories).

    Raises ``SynapIntegrationError`` for: ``values`` together with
    ``scope_path`` or with ids; nothing given at all; a malformed
    ``scope_path``; an SDK without ``get_scope_ladder`` (the message says to
    upgrade ``maximem-synap``); no ladder available; a failed ladder read.
    ``maximem_synap.ScopeLadderError`` (an ``InvalidInputError``) from
    ``ladder.address`` propagates unchanged: it names the levels.
    """
    op = "resolve_address"
    user_id = user_id or None
    customer_id = customer_id or None
    has_path = bool(scope_path)
    has_values = values is not None
    if scope_path is not None and not scope_path and (user_id or customer_id or has_values):
        logger.warning(
            "resolve_address: an empty scope_path was passed together with %s; it "
            "is treated as not set, so the address falls back to those, with no path.",
            "values" if has_values else "ids")

    if has_values and has_path:
        raise SynapIntegrationError(
            op, "pass either `values` or `scope_path`, not both: `scope_path` "
            "is already an address and `values` would be resolved to another one.")
    if has_values and (user_id or customer_id):
        raise SynapIntegrationError(
            op, "pass either `values` or user_id/customer_id, not both: "
            "`values` already name the user (and customer) by level.")

    if has_path:
        if not isinstance(scope_path, Mapping) or not all(
                isinstance(k, str) and k and isinstance(v, str) and v
                for k, v in scope_path.items()):
            raise SynapIntegrationError(
                op, "`scope_path` must map level keys to non-empty string values, "
                f"got {scope_path!r}.")
        return ScopeAddress(user_id=user_id, customer_id=customer_id,
                            scope_path=dict(scope_path))

    if has_values:
        get_ladder = getattr(sdk, "get_scope_ladder", None)
        if not _SDK_HAS_LADDER or not callable(get_ladder):
            raise SynapIntegrationError(
                op, "addressing by level values needs a maximem-synap release "
                "that has `get_scope_ladder` (the scope ladder); the installed one "
                "does not. Upgrade `maximem-synap`, or pass user_id/customer_id "
                "or an explicit `scope_path`.")
        async with wrap_sdk_errors_async(f"{op}.get_scope_ladder", logger):
            ladder = await get_ladder()
        if ladder is None:
            raise SynapIntegrationError(
                op, "this account's scope ladder could not be determined "
                "(the server and the instance's isolation mode gave no ladder), so "
                "`values` cannot be placed on a level. Pass user_id/customer_id or "
                "an explicit `scope_path` instead.")
        return ladder.address(values)

    if not (user_id or customer_id):
        raise SynapIntegrationError(
            op, "no identity given: pass user_id/customer_id, `scope_path` or `values`.")
    return ScopeAddress(user_id=user_id, customer_id=customer_id)


def _fingerprint(value: Any) -> str:
    """A short hash of an id, safe to put in an error message or a log."""
    return hashlib.sha256(str(value).encode("utf-8", "replace")).hexdigest()[:8]


def ensure_ids_match_path(
    scope_path: Optional[Mapping[str, str]],
    *,
    user_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    where: str = "",
) -> None:
    """Refuse per-call ids that disagree with a scope path fixed elsewhere.

    The server lets the path win: ``resolve_identity`` takes ``user_id`` and
    ``customer_id`` from the path keys ``user`` and ``customer`` over the flat
    ids. A package that fixes a path at construction but takes ids per call
    would therefore let one caller read and write another's memory while
    believing it addressed its own. Call this before every SDK call that sends
    both.

    * ``scope_path`` falsy: no-op.
    * For each reserved key (``user``, ``customer``) present in the path: an id
      argument that is set and differs raises ``SynapIntegrationError``; an
      empty / ``None`` id is fine (the path supplies it).
    * B2C: a path whose ``customer`` equals its ``user`` with ``customer_id``
      left out passes, as does ``customer_id`` equal to that value.
    * The path itself must map non-empty string keys to non-empty string values
      without ``/``; otherwise ``SynapIntegrationError``.

    Messages carry short fingerprints of the ids, never the ids themselves.
    """
    op = "ensure_ids_match_path"
    where_txt = f" ({where})" if where else ""
    if not scope_path:
        return
    if not isinstance(scope_path, Mapping):
        raise SynapIntegrationError(op, f"scope_path must be a mapping{where_txt}.")
    for key, value in scope_path.items():
        if not isinstance(key, str) or not key:
            raise SynapIntegrationError(
                op, f"scope_path keys must be non-empty strings{where_txt}.")
        if not isinstance(value, str) or not value or "/" in value:
            raise SynapIntegrationError(
                op, f"scope_path value for level {key!r} must be a non-empty string "
                f"without '/'{where_txt}.")
    for key, given in (("user", user_id), ("customer", customer_id)):
        if key in scope_path and given not in (None, "") and given != scope_path[key]:
            raise SynapIntegrationError(
                op,
                f"{key}_id does not match the scope_path's {key!r} level{where_txt}: "
                f"id fingerprint {_fingerprint(given)}, path fingerprint "
                f"{_fingerprint(scope_path[key])}. The server would use the path, "
                "so this call would read or write another identity's memory.",
                {"key": key, "id_fp": _fingerprint(given),
                 "path_fp": _fingerprint(scope_path[key]), "where": where})


__all__ = ["default_scope", "scope_string", "resolve_address", "ensure_ids_match_path",
           "ScopeAddress"]
