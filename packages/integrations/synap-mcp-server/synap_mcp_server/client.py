"""Thin httpx wrapper around the Synap public REST API.

Two operations are wrapped:
  - create_memory  -> POST /api/v1/memories/create   (long-range, async/queued ingestion)
  - fetch_context  -> POST /v1/context/{scope}/fetch  (fast retrieval)

The Bearer token is read from the per-request ContextVar and forwarded verbatim; this
server never validates or stores it — the REST API owns auth.

Error policy: every transport/HTTP failure surfaces as a SynapAPIError carrying the
upstream status (and Retry-After when present), plus a synthetic status for timeouts
(``TIMEOUT_STATUS``) and network errors (``NETWORK_STATUS``). The tools layer turns
those into human-readable messages — see tools._describe_api_error.
"""

import httpx

from .config import settings
from .context import MissingTokenError, get_token

# Synthetic statuses for failures that never reach an HTTP response, so the tools
# layer can branch on exc.status uniformly.
TIMEOUT_STATUS = 408
NETWORK_STATUS = 503


class SynapAPIError(Exception):
    def __init__(self, status: int, detail: str, retry_after: str | None = None):
        self.status = status
        self.detail = detail
        # Seconds the caller should back off, parsed from the Retry-After header (429s).
        self.retry_after = retry_after
        super().__init__(f"Synap API {status}: {detail}")


def scope_for(user_id: str | None, customer_id: str | None) -> str:
    """The scope a call resolves to, mirroring the REST routing:
    no IDs -> client (shared per-key, the no-code default); user_id -> user;
    customer_id only -> customer. Kept here so the tools layer can echo it back."""
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
    distinguish ``httpx`` exception types — only ``exc.status``.
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
            resp.status_code,
            resp.text[:500],
            retry_after=resp.headers.get("retry-after"),
        )
    return resp.json()


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
    persists."""
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
    """Fetch ranked context. Scope is derived from the supplied IDs (see scope_for)."""
    scope = scope_for(user_id, customer_id)

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
    Returns the REST status payload: { status, memories_created, completed_at, ... }."""
    return await _request(
        "GET",
        f"/api/v1/memories/status/{ingestion_id}",
        timeout=settings.recall_timeout_s,
    )
