"""Thin httpx wrapper around the Synap public REST API.

Two operations are wrapped:
  - create_memory  -> POST /api/v1/memories/create   (long-range, async/queued ingestion)
  - fetch_context  -> POST /v1/context/{scope}/fetch  (fast retrieval)

The Bearer token is read from the per-request ContextVar and forwarded verbatim; this
server never validates or stores it — the REST API owns auth.
"""

import httpx

from .config import settings
from .context import MissingTokenError, get_token


class SynapAPIError(Exception):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"Synap API {status}: {detail}")


def _auth_headers() -> dict:
    token = get_token()
    if not token:
        raise MissingTokenError(
            "No Synap token provided. Set 'Authorization: Bearer synap_<key>' "
            "on the MCP connection."
        )
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


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

    async with httpx.AsyncClient(
        base_url=settings.synap_api_url, timeout=settings.ingest_timeout_s
    ) as client:
        resp = await client.post(
            "/api/v1/memories/create", json=body, headers=_auth_headers()
        )
    if resp.status_code >= 400:
        raise SynapAPIError(resp.status_code, resp.text[:500])
    return resp.json()


async def fetch_context(
    search_query: list[str] | None,
    *,
    max_results: int,
    user_id: str | None = None,
    customer_id: str | None = None,
) -> dict:
    """Fetch ranked context. Scope is derived from the supplied IDs:
    no IDs -> client (shared per-key, the no-code default); user_id -> user;
    customer_id only -> customer."""
    if user_id:
        scope = "user"
    elif customer_id:
        scope = "customer"
    else:
        scope = "client"

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

    async with httpx.AsyncClient(
        base_url=settings.synap_api_url, timeout=settings.recall_timeout_s
    ) as client:
        resp = await client.post(
            f"/v1/context/{scope}/fetch", json=body, headers=_auth_headers()
        )
    if resp.status_code >= 400:
        raise SynapAPIError(resp.status_code, resp.text[:500])
    return resp.json()


async def get_ingestion_status(ingestion_id: str) -> dict:
    """Poll the status of a queued ingestion (the long-range pipeline is async).
    Returns the REST status payload: { status, memories_created, completed_at, ... }."""
    async with httpx.AsyncClient(
        base_url=settings.synap_api_url, timeout=settings.recall_timeout_s
    ) as client:
        resp = await client.get(
            f"/api/v1/memories/status/{ingestion_id}", headers=_auth_headers()
        )
    if resp.status_code >= 400:
        raise SynapAPIError(resp.status_code, resp.text[:500])
    return resp.json()
