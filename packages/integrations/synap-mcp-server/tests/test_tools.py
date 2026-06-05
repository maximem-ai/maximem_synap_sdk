"""Adapter mapping tests (TC-ADP-*) — assert each tool calls the right REST route with
the right body. The REST API is mocked with respx; no backend required."""

import httpx
import pytest
import respx

from synap_mcp_server import client
from tests.conftest import API_BASE

pytestmark = pytest.mark.asyncio


@respx.mock
async def test_log_exchange_maps_to_create(with_token):
    """TC-ADP-01 / TC-ADP-06: long-range create, ai-chat-conversation, no IDs."""
    route = respx.post(f"{API_BASE}/api/v1/memories/create").mock(
        return_value=httpx.Response(200, json={"ingestion_id": "ing_1", "status": "completed"})
    )
    res = await client.create_memory("User: hi\nAssistant: hello")
    assert route.called
    body = route.calls.last.request.read().decode()
    import json

    sent = json.loads(body)
    assert sent["mode"] == "long-range"
    assert sent["document_type"] == "ai-chat-conversation"
    assert "user_id" not in sent and "customer_id" not in sent  # client scope
    assert sent["metadata"]["source"] == "mcp-server"
    assert res["ingestion_id"] == "ing_1"


@respx.mock
async def test_recall_maps_to_client_fetch_fast(with_token):
    """TC-ADP-02 / TC-ADP-03: client scope, fast mode, honors max_results."""
    route = respx.post(f"{API_BASE}/v1/context/client/fetch").mock(
        return_value=httpx.Response(200, json={"context": {"facts": []}})
    )
    await client.fetch_context(["favorite color"], max_results=3)
    assert route.called
    import json

    sent = json.loads(route.calls.last.request.read().decode())
    assert sent["mode"] == "fast"
    assert sent["max_results"] == 3
    assert sent["search_query"] == ["favorite color"]


@respx.mock
async def test_recall_user_scope_routes_to_user_fetch(with_token):
    """TC-ADP-04: user_id routes to /v1/context/user/fetch."""
    route = respx.post(f"{API_BASE}/v1/context/user/fetch").mock(
        return_value=httpx.Response(200, json={"context": {}})
    )
    await client.fetch_context(["q"], max_results=10, user_id="u1")
    assert route.called
    import json

    sent = json.loads(route.calls.last.request.read().decode())
    assert sent["user_id"] == "u1"


@respx.mock
async def test_list_recent_uses_broad_fetch(with_token):
    """TC-ADP-05: no search_query, client scope."""
    route = respx.post(f"{API_BASE}/v1/context/client/fetch").mock(
        return_value=httpx.Response(200, json={"context": {}})
    )
    await client.fetch_context(None, max_results=5)
    assert route.called
    import json

    sent = json.loads(route.calls.last.request.read().decode())
    assert sent["search_query"] is None
    assert sent["max_results"] == 5
