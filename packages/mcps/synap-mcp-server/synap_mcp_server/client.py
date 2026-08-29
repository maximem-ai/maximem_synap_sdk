"""Thin httpx wrapper around the Synap public REST API.

Operations wrapped:
  - create_memory  -> POST /api/v1/memories/create   (long-range, async/queued ingestion)
  - fetch_context  -> POST /v1/context/{scope}/fetch  (fast retrieval)
  - get_isolation  -> GET  /api/v1/auth/whoami        (which id shape this key expects)

The Bearer token is read from the per-request ContextVar and forwarded verbatim; this
server never validates or stores it. The REST API owns auth.

The identifier contract, enforced by the REST API on production since
2026-08-26 07:07 UTC:

  B2C, ``user_context_isolation = "equals_customer"``
      Send ``user_id`` and nothing else. The user id IS the whole identity.
      A ``customer_id`` is refused with HTTP 400, and ``/v1/context/customer/fetch``
      is not available on the instance at all.

  B2B, ``user_context_isolation = "strict"``
      ``customer_id`` is REQUIRED. A ``user_id`` on its own is an error. Unchanged.

``GET /api/v1/auth/whoami`` publishes the mode, so this server refuses a customer_id
locally rather than letting a coding agent discover the rule from a 400 it cannot read.
The mode is only read when a customer_id was actually supplied, so the correct B2C
shape costs no extra round trip.

Error policy: every transport/HTTP failure surfaces as a SynapAPIError carrying the
upstream status (and Retry-After when present), plus a synthetic status for timeouts
(``TIMEOUT_STATUS``) and network errors (``NETWORK_STATUS``). The tools layer turns
those into human-readable messages, see tools._describe_api_error. A contract
violation is raised as CustomerIdNotAcceptedError before any request is sent.
"""

import hashlib
import time
from typing import Optional

import httpx

from .config import settings
from .context import MissingTokenError, get_token

# Synthetic statuses for failures that never reach an HTTP response, so the tools
# layer can branch on exc.status uniformly.
TIMEOUT_STATUS = 408
NETWORK_STATUS = 503

# The MACA value that means "customer and user are one node", i.e. B2C.
B2C_ISOLATION = "equals_customer"

# whoami is a small identity lookup, so it gets a tighter timeout than a fetch.
_WHOAMI_TIMEOUT_S = 5.0
# Mirrors the server's own MACA cache TTL (_MACA_CACHE_TTL_SECONDS = 300), so this
# server never believes a stale mode for longer than the server itself does.
_ISOLATION_TTL_S = 300.0
# A mode we could NOT read is retried sooner: it is the state in which the contract
# check is inactive, so lingering in it is the expensive mistake.
_ISOLATION_MISS_TTL_S = 30.0
_ISOLATION_CACHE_MAX = 512

# token digest -> (isolation or None, expiry monotonic seconds).
# Keyed by digest rather than by the token so a long-lived process never holds a
# table of raw client keys; the token itself lives only in the per-request ContextVar.
_isolation_cache: dict[str, tuple[Optional[str], float]] = {}


class SynapAPIError(Exception):
    def __init__(self, status: int, detail: str, retry_after: str | None = None):
        self.status = status
        self.detail = detail
        # Seconds the caller should back off, parsed from the Retry-After header (429s).
        self.retry_after = retry_after
        super().__init__(f"Synap API {status}: {detail}")


class CustomerIdNotAcceptedError(Exception):
    """A customer_id was supplied for a B2C instance, which refuses it.

    Raised before the request leaves this process. The REST API rejects the same
    shape with HTTP 400, so this changes nothing about what is allowed; it only
    means the model is told what to do instead of reading a status code.

    The message is written for a model to act on: it names the offending value,
    names the mode, and says exactly which id to send instead.
    """

    def __init__(self, where: str, customer_id: str, *, extra: str = ""):
        self.where = where
        self.customer_id = customer_id
        msg = (
            "customer_id is not accepted on this Synap instance, which is B2C "
            f"(user_context_isolation={B2C_ISOLATION!r}). Send user_id on its own: "
            "on a B2C instance the user id is the whole identity, and the API "
            f"rejects a customer_id with HTTP 400. Received customer_id={customer_id!r} "
            f"at {where}."
        )
        if extra:
            msg = f"{msg} {extra}"
        super().__init__(msg)


def scope_for(user_id: str | None, customer_id: str | None) -> str:
    """The scope a call resolves to, mirroring the REST routing:
    no IDs -> client (shared per-key, the no-code default); user_id -> user;
    customer_id only -> customer. Kept here so the tools layer can echo it back.

    The ``customer`` branch is reachable on B2B instances only. On a B2C instance
    ``/v1/context/customer/fetch`` does not exist, and a customer_id never gets
    this far because ``assert_customer_id_allowed`` raises first. Do not "fix"
    this by collapsing customer to user: silently rewriting the scope a caller
    asked for is the failure this contract exists to remove.
    """
    if user_id:
        return "user"
    if customer_id:
        return "customer"
    return "client"


def _auth_headers() -> dict:
    token = get_token()
    if not token:
        raise MissingTokenError(
            "No Synap token provided. Set 'Authorization: Bearer synap_<key>' "
            "on the MCP connection."
        )
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


async def _request(
    method: str, path: str, *, timeout: float, json: dict | None = None
) -> dict:
    """Issue an authenticated request, normalizing every failure to SynapAPIError.

    Timeouts and network errors map to synthetic statuses so a caller never has to
    distinguish ``httpx`` exception types, only ``exc.status``.
    """
    headers = _auth_headers()
    try:
        async with httpx.AsyncClient(
            base_url=settings.synap_api_url, timeout=timeout
        ) as client:
            resp = await client.request(method, path, json=json, headers=headers)
    except httpx.TimeoutException as exc:
        raise SynapAPIError(TIMEOUT_STATUS, f"timeout: {exc}") from exc
    except httpx.HTTPError as exc:
        raise SynapAPIError(NETWORK_STATUS, f"network error: {exc}") from exc

    if resp.status_code >= 400:
        raise SynapAPIError(
            # 2000, not 500. The contract rejection body carries the message AND the
            # fix, and truncating at 500 cut the fix off mid-sentence, which turned
            # the one error that explains itself back into a bare status code.
            resp.status_code,
            resp.text[:2000],
            retry_after=resp.headers.get("retry-after"),
        )
    return resp.json()


def _cache_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def get_isolation() -> Optional[str]:
    """This key's scoping mode from ``GET /api/v1/auth/whoami``, or None.

    **None means the server did not say**, which is the case against any deployment
    older than the field and on any transient failure, and it must mean "do
    nothing". Guessing B2C there would make this server refuse a B2B caller's
    MANDATORY customer_id, which is a worse failure than the one being prevented,
    and guessing B2B would tell a B2C caller to send an id that is refused.

    Cached per token so the check costs one request per key per TTL, not one per
    tool call. Never raises: an unreadable mode is an answer, not an error.
    """
    token = get_token()
    if not token:
        return None
    key = _cache_key(token)
    now = time.monotonic()
    hit = _isolation_cache.get(key)
    if hit is not None and hit[1] > now:
        return hit[0]

    isolation: Optional[str] = None
    try:
        data = await _request(
            "GET", "/api/v1/auth/whoami", timeout=_WHOAMI_TIMEOUT_S
        )
        raw = (data or {}).get("user_context_isolation")
        isolation = str(raw) if raw else None
    except (SynapAPIError, MissingTokenError, ValueError):
        # ValueError covers a non-JSON body (json.JSONDecodeError subclasses it).
        isolation = None

    # Bounded, and cleared wholesale rather than evicted one by one: this table is a
    # latency optimisation, so the cheapest correct eviction is the right one.
    if len(_isolation_cache) >= _ISOLATION_CACHE_MAX:
        _isolation_cache.clear()
    ttl = _ISOLATION_TTL_S if isolation else _ISOLATION_MISS_TTL_S
    _isolation_cache[key] = (isolation, now + ttl)
    return isolation


async def assert_customer_id_allowed(
    customer_id: Optional[str], *, where: str, extra: str = ""
) -> None:
    """Refuse a customer_id on a B2C instance, before the request goes out.

    Returns immediately when no customer_id was supplied, so the correct B2C shape
    never triggers the whoami lookup. That matters: the shape this contract wants
    people to send is the one that must not get slower.
    """
    if not customer_id:
        return
    if await get_isolation() != B2C_ISOLATION:
        return
    raise CustomerIdNotAcceptedError(where, customer_id, extra=extra)


async def create_memory(
    document: str,
    *,
    document_type: str = "ai-chat-conversation",
    user_id: str | None = None,
    customer_id: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Forward a conversation turn into long-range ingestion. Fire-and-forget on the
    REST side: it returns a queued ingestion_id immediately and extraction decides what
    persists.

    ``customer_id`` is B2B only. On a B2B instance it is required; on a B2C instance
    it is refused here and by the API.
    """
    await assert_customer_id_allowed(
        customer_id, where="POST /api/v1/memories/create"
    )

    body: dict = {
        "document": document,
        "document_type": document_type,
        "mode": "long-range",
        "metadata": {"source": "mcp-server", **(metadata or {})},
    }
    if user_id:
        body["user_id"] = user_id
    if customer_id:
        body["customer_id"] = customer_id

    return await _request(
        "POST",
        "/api/v1/memories/create",
        json=body,
        timeout=settings.ingest_timeout_s,
    )


async def fetch_context(
    search_query: list[str] | None,
    *,
    max_results: int,
    user_id: str | None = None,
    customer_id: str | None = None,
) -> dict:
    """Fetch ranked context. Scope is derived from the supplied IDs (see scope_for).

    On a B2C instance a customer_id is refused before the route is chosen. Without
    that guard, a customer-only call routes to ``/v1/context/customer/fetch``, which
    a B2C instance does not serve: it used to answer 200 with an empty body, and now
    answers 400. Neither is something a model can act on, so the refusal happens
    here with a message that names the working route instead.
    """
    scope = scope_for(user_id, customer_id)
    await assert_customer_id_allowed(
        customer_id,
        where=f"POST /v1/context/{scope}/fetch",
        extra=(
            "Customer-scoped retrieval does not exist on a B2C instance at all: "
            "recall at user scope with user_id."
            if scope == "customer"
            else ""
        ),
    )

    body: dict = {
        "search_query": search_query,  # List[str] | None
        "max_results": max_results,
        "types": ["all"],
        "mode": "fast",
    }
    if user_id:
        body["user_id"] = user_id
    if customer_id:
        body["customer_id"] = customer_id

    return await _request(
        "POST",
        f"/v1/context/{scope}/fetch",
        json=body,
        timeout=settings.recall_timeout_s,
    )


async def get_ingestion_status(ingestion_id: str) -> dict:
    """Poll the status of a queued ingestion (the long-range pipeline is async).
    Returns the REST status payload: { status, memories_created, completed_at, ... }.

    Takes no scope ids, so the identifier contract does not apply to it.
    """
    return await _request(
        "GET",
        f"/api/v1/memories/status/{ingestion_id}",
        timeout=settings.recall_timeout_s,
    )
