"""Credits sub-interface for the Python SDK.

Usage:
    sdk = MaximemSynapSDK()
    await sdk.initialize()

    balance = await sdk.credits.get_balance()
    print(f"You have {balance.balance_credits} credits")

    if balance.warning_low:
        # Prompt user to redeem a code:
        result = await sdk.credits.redeem("SYN-XXXX-XXXX-XXXX-X")
        print(f"Added {result.credits_granted} credits")

    # Dry-run cost of a planned operation
    quote = await sdk.credits.estimate(
        metric_type="llm_input_tokens", units=1000,
    )
    print(f"Would cost {quote.credits_estimate} credits")

All methods translate HTTP 402 into
:class:`~synap.sdk.python.maximem_synap.models.errors.InsufficientCreditsError`
via the shared transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .sdk import MaximemSynapSDK


# =============================================================================
# Public data classes — mirror the server's Client* Pydantic schemas
# =============================================================================


@dataclass(slots=True)
class CreditBucket:
    """One non-zero (source_type, expires_at) bucket in a wallet."""

    source_type: str
    balance: float
    expires_at: Optional[datetime]


@dataclass(slots=True)
class CreditBalance:
    """Snapshot of a client's credit wallet."""

    client_id: str
    balance_credits: float
    warning_low: bool
    buckets: List[CreditBucket]


@dataclass(slots=True)
class CreditLedgerEntry:
    """One row from the client-facing ledger (no USD / rate detail)."""

    ledger_id: str
    entry_type: str
    delta: float
    metric_type: Optional[str]
    category: Optional[str]
    created_at: datetime


@dataclass(slots=True)
class CreditLedgerPage:
    entries: List[CreditLedgerEntry]
    total: int
    limit: int
    offset: int


@dataclass(slots=True)
class CreditEstimate:
    credits_estimate: float


@dataclass(slots=True)
class RedeemResult:
    """Successful redemption return value."""

    redemption_id: str
    credits_granted: float
    new_balance_credits: float
    expires_at: Optional[datetime]


# =============================================================================
# Helpers
# =============================================================================


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


# =============================================================================
# The sub-interface
# =============================================================================


class CreditsInterface:
    """``client.credits.*`` — balance, ledger, estimate, redeem.

    Attached to :class:`MaximemSynapSDK` during construction. The
    interface holds a back-ref to the SDK so it can reuse its
    authenticated HTTP transport.
    """

    def __init__(self, sdk: "MaximemSynapSDK"):
        self._sdk = sdk

    # --------------------------------------------------------------

    async def get_balance(self) -> CreditBalance:
        """Return the current credit balance + bucket breakdown."""
        result = await self._sdk._http_transport.get(path="/v1/credits/balance")
        return CreditBalance(
            client_id=result["client_id"],
            balance_credits=float(result["balance_credits"]),
            warning_low=bool(result["warning_low"]),
            buckets=[
                CreditBucket(
                    source_type=b["source_type"],
                    balance=float(b["balance"]),
                    expires_at=_parse_iso(b.get("expires_at")),
                )
                for b in result.get("buckets", [])
            ],
        )

    # --------------------------------------------------------------

    async def get_ledger(
        self,
        *,
        entry_type: Optional[str] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> CreditLedgerPage:
        """Paginated ledger for the caller's wallet."""
        params: dict = {"limit": limit, "offset": offset}
        if entry_type is not None:
            params["entry_type"] = entry_type
        if from_time is not None:
            params["from"] = from_time.isoformat()
        if to_time is not None:
            params["to"] = to_time.isoformat()

        result = await self._sdk._http_transport.get(
            path="/v1/credits/ledger", params=params
        )
        entries = [
            CreditLedgerEntry(
                ledger_id=e["ledger_id"],
                entry_type=e["entry_type"],
                delta=float(e["delta"]),
                metric_type=e.get("metric_type"),
                category=e.get("category"),
                created_at=_parse_iso(e["created_at"]) or datetime.now(),
            )
            for e in result.get("entries", [])
        ]
        return CreditLedgerPage(
            entries=entries,
            total=int(result.get("total", 0)),
            limit=int(result.get("limit", limit)),
            offset=int(result.get("offset", offset)),
        )

    # --------------------------------------------------------------

    async def estimate(
        self,
        *,
        metric_type: str,
        units: float,
        item_count: Optional[int] = None,
        endpoint: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> CreditEstimate:
        """Dry-run cost of a proposed operation (credits only)."""
        body = {
            "metric_type": metric_type,
            "units": units,
            "item_count": item_count,
            "endpoint": endpoint,
            "mode": mode,
        }
        result = await self._sdk._http_transport.post(
            path="/v1/credits/estimate", body=body
        )
        return CreditEstimate(
            credits_estimate=float(result["credits_estimate"]),
        )

    # --------------------------------------------------------------

    async def redeem(self, code: str) -> RedeemResult:
        """Apply a redeem code to the current wallet.

        Raises one of the SDK's permanent-error subclasses on rejection
        (invalid format, unknown code, already redeemed, etc.). Callers
        can distinguish by catching :class:`InvalidInputError` (bad
        format / unknown code) vs :class:`InsufficientCreditsError`
        won't fire here — 402 on this endpoint is used for
        "already_redeemed / expired / exhausted" and the transport
        layer converts all of them into ``InsufficientCreditsError``
        with ``balance_credits=None`` so the caller can print the
        error cleanly.
        """
        result = await self._sdk._http_transport.post(
            path="/v1/credits/redeem",
            body={"code": code},
        )
        return RedeemResult(
            redemption_id=result["redemption_id"],
            credits_granted=float(result["credits_granted"]),
            new_balance_credits=float(result["new_balance_credits"]),
            expires_at=_parse_iso(result.get("expires_at")),
        )
