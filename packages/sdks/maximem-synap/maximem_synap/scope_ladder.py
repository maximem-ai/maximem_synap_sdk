"""The account's scope ladder, as the SDK sees it.

A ladder is the ordered list of levels an account files memory under. The
default has two non-root levels, ``customer`` then ``user`` (B2C: ``user``
alone). An account may add levels in between, for example ``team``.

What this module does: turn "values, outermost first" into the arguments the
SDK calls take (``user_id``, ``customer_id`` and, on a deeper ladder,
``scope_path`` for reads or ``scope`` for memories), and refuse before any
network call when the values cannot be addressed.

Rules, each read from the server code (``synap/cloud/shared/scope_ladder.py``
``_normalise`` and ``synap/cloud/shared/scope.py`` ``resolve_identity``):

* A path never names the root level.
* A path must be contiguous from the first level; a shorter path (a prefix) is
  valid and means that level.
* The server reads the path keys ``customer`` and ``user`` over the flat ids,
  so ids are always sent as well.
* A path is honoured only when nested scoping is on for the instance AND the
  account's ladder is ratified; otherwise the server refuses it. So on a
  default ladder this module never produces a path.
* B2C (``user_context_isolation == "equals_customer"``): ``resolve_identity``
  collapses customer and user onto the user's id, and ``customer_id`` is
  refused by the SDK. The tenant level of a path therefore carries the
  identity's value, and only ``user_id`` is sent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .models.errors import InvalidInputError

logger = logging.getLogger(__name__)

ROLE_TENANT = "tenant_of_record"
ROLE_IDENTITY = "identity"

#: How long a ladder is trusted before whoami is read again. Same as the
#: server's own ladder cache.
LADDER_TTL_SECONDS = 300.0
#: After a failed re-read, how long to keep serving the last known ladder
#: before trying again.
LADDER_RETRY_BACKOFF_SECONDS = 60.0

#: Minimum time between re-reads that an invalidation triggers. A caller whose
#: every request is refused would otherwise read whoami once per request.
LADDER_INVALIDATION_MIN_INTERVAL_SECONDS = 5.0


class ScopeLadderError(InvalidInputError):
    """The values cannot be addressed on this account's ladder.

    Raised before any network call. An ``InvalidInputError``, so one
    ``except InvalidInputError`` covers it.
    """


@dataclass(frozen=True)
class ScopeLevel:
    key: str
    label: str
    position: int
    role: Optional[str]
    required: bool


@dataclass(frozen=True)
class ScopeAddress:
    """Where a call reads or writes: ids, plus a path on a deeper ladder.

    ``scope_path`` is ``None`` on a default ladder.
    """

    user_id: Optional[str] = None
    customer_id: Optional[str] = None
    scope_path: Optional[Dict[str, str]] = None

    def read_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for reads: ``sdk.fetch``, ``send_message``,
        ``record_message``, ``user.context.fetch``,
        ``conversation.context.fetch`` and ``client.context.fetch``. The path
        is ``scope_path``.

        Not for ``customer.context.fetch``: that route refuses a path that
        names the identity rung (HTTP 400, it reads shared material only), and
        it takes no ``user_id``, so ``customer.context.fetch(**addr.read_kwargs())``
        raises ``TypeError``. Call ``sdk.fetch(**addr.read_kwargs())``, which
        sends the customer route the path cut above the identity rung."""
        out: Dict[str, Any] = {}
        if self.user_id:
            out["user_id"] = self.user_id
        if self.customer_id:
            out["customer_id"] = self.customer_id
        if self.scope_path:
            out["scope_path"] = dict(self.scope_path)
        return out

    def write_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for ``memories.create`` / ``create_from_file`` /
        ``forget``: the path is ``scope``."""
        out: Dict[str, Any] = {}
        if self.user_id:
            out["user_id"] = self.user_id
        if self.customer_id:
            out["customer_id"] = self.customer_id
        if self.scope_path:
            out["scope"] = dict(self.scope_path)
        return out


#: The reserved key of the identity level, used when the ladder is not known.
RESERVED_IDENTITY_KEY = "user"


def path_above_identity(
    scope_path: Optional[Mapping[str, str]], ladder: Optional["ScopeLadder"] = None,
) -> Optional[Dict[str, str]]:
    """``scope_path`` cut above the identity rung, or ``None`` when nothing is left.

    ``POST /v1/context/customer/fetch`` reads shared material only and refuses
    (HTTP 400) any path that resolves to a rung holding the ``identity`` role
    (``customer_scope_reaches_an_identity``): one person is not the tenant they
    belong to. The customer-scope read of a person's path is therefore the
    same path "stopped above the identity rung", which is this.

    The identity level is found by ROLE on ``ladder`` (the account may have
    renamed it), and every level at or below it is removed. Without a ladder,
    or one with no identity level, the reserved key ``user`` is removed. A path
    that names no identity rung comes back unchanged (as a copy).
    """
    if not scope_path:
        return None
    identity = None
    if ladder is not None:
        identity = next((lv for lv in ladder.levels if lv.role == ROLE_IDENTITY), None)
    if identity is None:
        out = {k: v for k, v in scope_path.items() if k != RESERVED_IDENTITY_KEY}
    else:
        position = {lv.key: lv.position for lv in ladder.levels}  # type: ignore[union-attr]
        out = {
            k: v for k, v in scope_path.items()
            if position.get(k, -1) < identity.position
        }
    return out or None


_REFRESH_HINT = (
    "The account's ladder may not be accepted yet, or nested scoping may be "
    "off for this instance. If that changed recently, "
    "`await sdk.get_scope_ladder(refresh=True)` re-reads it."
)


@dataclass(frozen=True)
class ScopeLadder:
    levels: Tuple[ScopeLevel, ...]
    is_default: bool
    ladder_ratified: bool
    nested_scoping_enabled: bool
    source: str  # "server" | "inferred"
    #: True when the instance is B2C (``user_context_isolation ==
    #: "equals_customer"``): the tenant and identity level are one node.
    b2c: bool = False

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(level.key for level in self.levels)

    def address(self, values: Sequence[str]) -> ScopeAddress:
        """Map ``values`` (outermost level first) to ids and, on a deeper
        ladder, the full prefix path.

        A shorter list addresses a higher level. Raises ``ScopeLadderError``
        before any call when the values cannot be addressed.
        """
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ScopeLadderError(
                "address() takes a list of values, outermost level first, "
                f"got {type(values).__name__}."
            )
        vals = list(values)
        if not vals:
            raise ScopeLadderError(
                "address() needs at least one value. The client level is "
                "addressed explicitly elsewhere (scope='client'), not through "
                f"the ladder. Levels, outermost first: {list(self.keys)}."
            )
        for v in vals:
            if not isinstance(v, str) or not v:
                raise ScopeLadderError(
                    f"address() values must be non-empty strings, got {v!r}."
                )

        levels = self.levels
        # B2C: the tenant level takes the identity's value, so the caller gives
        # one value per level except the tenant level.
        expected = (
            tuple(lv for lv in levels if lv.role != ROLE_TENANT) if self.b2c else levels
        )
        if len(vals) > len(expected):
            raise ScopeLadderError(
                f"{len(vals)} values for {len(expected)} levels "
                f"({[lv.key for lv in expected]}, outermost first)."
            )

        if not self.is_default:
            if not self.nested_scoping_enabled or not self.ladder_ratified:
                raise ScopeLadderError(
                    "This account has a deeper scope ladder "
                    f"({list(self.keys)}), but the server would refuse a scope "
                    f"path (nested_scoping_enabled={self.nested_scoping_enabled}, "
                    f"ladder_ratified={self.ladder_ratified}). " + _REFRESH_HINT
                )

        user_id: Optional[str] = None
        customer_id: Optional[str] = None
        path: Dict[str, str] = {}

        if self.b2c:
            identity = next((lv for lv in levels if lv.role == ROLE_IDENTITY), None)
            if identity is None:
                raise ScopeLadderError(
                    "This ladder has no identity level, so a B2C address cannot "
                    "be built."
                )
            if len(vals) < len(expected):
                # The tenant level carries the identity's value, which a
                # shorter list does not reach. Guessing would file under the
                # wrong node.
                raise ScopeLadderError(
                    "On a B2C instance the tenant level takes the identity's "
                    "value, so give one value for every level after it "
                    f"({[lv.key for lv in expected]}); got {len(vals)}."
                )
            by_key = {lv.key: v for lv, v in zip(expected, vals)}
            identity_value = by_key[identity.key]
            user_id = identity_value
            if not self.is_default:
                for lv in levels:
                    path[lv.key] = (
                        identity_value if lv.role == ROLE_TENANT else by_key[lv.key]
                    )
        else:
            for lv, v in zip(levels, vals):
                if lv.role == ROLE_TENANT:
                    customer_id = v
                elif lv.role == ROLE_IDENTITY:
                    user_id = v
                if not self.is_default:
                    path[lv.key] = v

        return ScopeAddress(
            user_id=user_id,
            customer_id=customer_id,
            scope_path=path or None,
        )


# ---------------------------------------------------------------------------
# Building a ladder from a whoami answer
# ---------------------------------------------------------------------------

_STRICT_LEVELS = (
    ScopeLevel("customer", "Customer", 1, ROLE_TENANT, True),
    ScopeLevel("user", "User", 2, ROLE_IDENTITY, True),
)
_B2C_LEVELS = (ScopeLevel("user", "User", 1, ROLE_IDENTITY, True),)


def _parse_levels(raw: Any) -> Optional[Tuple[ScopeLevel, ...]]:
    if not isinstance(raw, list) or not raw:
        return None
    out = []
    for item in raw:
        if not isinstance(item, Mapping) or not item.get("key"):
            return None
        try:
            position = int(item.get("position"))
        except (TypeError, ValueError):
            return None
        out.append(
            ScopeLevel(
                key=str(item["key"]),
                label=str(item.get("label") or item["key"]),
                position=position,
                role=item.get("role") or None,
                required=bool(item.get("required", True)),
            )
        )
    out.sort(key=lambda lv: lv.position)
    return tuple(out)


def ladder_from_whoami(
    whoami: Mapping[str, Any], fallback_isolation: Optional[str] = None
) -> Optional[ScopeLadder]:
    """The ladder a whoami answer describes, or ``None`` when it does not say.

    Uses the ``scope_ladder`` block when present and well formed. Otherwise
    infers the default ladder from ``user_context_isolation`` (``strict``:
    customer, user; ``equals_customer``: user), marked ``source="inferred"``.
    """
    isolation = whoami.get("user_context_isolation") or fallback_isolation
    b2c = isolation == "equals_customer"

    # A missing key and an explicit null both mean "no block". Anything else
    # that is not a well formed block is treated as absent, with a warning.
    block = whoami.get("scope_ladder")
    if block is not None:
        levels = _parse_levels(block.get("levels")) if isinstance(block, Mapping) else None
        if levels is None:
            logger.warning(
                "whoami returned a malformed scope_ladder block; ignoring it "
                "and inferring the default ladder."
            )
        else:
            return ScopeLadder(
                levels=levels,
                is_default=bool(block.get("is_default")),
                ladder_ratified=bool(block.get("ladder_ratified")),
                nested_scoping_enabled=bool(block.get("nested_scoping_enabled")),
                source="server",
                b2c=b2c,
            )

    if isolation is None:
        return None
    return ScopeLadder(
        levels=_B2C_LEVELS if b2c else _STRICT_LEVELS,
        is_default=True,
        ladder_ratified=False,
        nested_scoping_enabled=False,
        source="inferred",
        b2c=b2c,
    )
