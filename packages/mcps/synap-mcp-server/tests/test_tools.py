"""Adapter mapping tests (TC-ADP-*): assert each tool calls the right REST route with
the right body. The REST API is mocked with respx; no backend required.

The identifier contract is exercised here rather than described: a customer_id is
required on B2B, refused on B2C, and passed through untouched when the server does
not publish a mode. All three are asserted, because a check that only ever sees one
mode cannot tell "refuses on B2C" from "refuses always".
"""

import httpx
import pytest
import respx

from synap_mcp_server import client
from synap_mcp_server.client import CustomerIdNotAcceptedError
from tests.conftest import API_BASE, B2B, B2C, mock_whoami

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
async def test_create_includes_user_and_customer_on_b2b(with_token):
    """B2B is unchanged: customer_id is required there, so it must reach the body.

    This is the half of the contract that must NOT move. Rejecting customer_id
    everywhere would break every B2B client on the platform, which is the larger
    failure of the two.
    """
    mock_whoami(B2B)
    route = respx.post(f"{API_BASE}/api/v1/memories/create").mock(
        return_value=httpx.Response(200, json={"ingestion_id": "ing_2"})
    )
    await client.create_memory("doc", user_id="u1", customer_id="c1")
    import json

    sent = json.loads(route.calls.last.request.read().decode())
    assert sent["user_id"] == "u1"
    assert sent["customer_id"] == "c1"
    assert sent["mode"] == "long-range"


@respx.mock
async def test_create_refuses_customer_id_on_b2c(with_token):
    """B2C refuses a customer_id, and refuses it BEFORE the request is sent.

    Asserting the create route was never called is the point: a check that raised
    after the write would leave the row behind and only make the error louder.
    """
    mock_whoami(B2C)
    route = respx.post(f"{API_BASE}/api/v1/memories/create").mock(
        return_value=httpx.Response(200, json={"ingestion_id": "ing_never"})
    )
    with pytest.raises(CustomerIdNotAcceptedError) as exc:
        await client.create_memory("doc", user_id="u1", customer_id="c1")
    assert not route.called, "the write must not go out before being refused"
    msg = str(exc.value)
    assert "'c1'" in msg, "the rejected value must be echoed or it cannot be traced"
    assert "user_id" in msg, "the message must say what to send instead"


@respx.mock
async def test_create_forwards_customer_id_when_the_mode_is_unknown(with_token):
    """A server that does not publish user_context_isolation means "do not know".

    Not knowing must mean not acting. Treating an absent field as B2C would make
    this server refuse a B2B client's mandatory id against every deployment that
    has not shipped the field yet.
    """
    mock_whoami(None)
    route = respx.post(f"{API_BASE}/api/v1/memories/create").mock(
        return_value=httpx.Response(200, json={"ingestion_id": "ing_3"})
    )
    await client.create_memory("doc", user_id="u1", customer_id="c1")
    import json

    sent = json.loads(route.calls.last.request.read().decode())
    assert sent["customer_id"] == "c1"


@respx.mock
async def test_create_forwards_customer_id_when_whoami_is_down(with_token):
    """Same rule when whoami itself fails: unreadable is not the same as B2C."""
    respx.get(f"{API_BASE}/api/v1/auth/whoami").mock(
        return_value=httpx.Response(500, text="boom")
    )
    route = respx.post(f"{API_BASE}/api/v1/memories/create").mock(
        return_value=httpx.Response(200, json={"ingestion_id": "ing_4"})
    )
    await client.create_memory("doc", user_id="u1", customer_id="c1")
    import json

    sent = json.loads(route.calls.last.request.read().decode())
    assert sent["customer_id"] == "c1"


@respx.mock
async def test_the_correct_b2c_shape_never_asks_for_the_mode(with_token):
    """user_id alone must not pay for the contract check with a round trip.

    The shape the contract wants everyone to send is the one that must not get
    slower, so the mode is only read when a customer_id was actually supplied.
    """
    whoami = mock_whoami(B2C)
    respx.post(f"{API_BASE}/api/v1/memories/create").mock(
        return_value=httpx.Response(200, json={"ingestion_id": "ing_5"})
    )
    respx.post(f"{API_BASE}/v1/context/user/fetch").mock(
        return_value=httpx.Response(200, json={"context": {}})
    )
    await client.create_memory("doc", user_id="u1")
    await client.fetch_context(["q"], max_results=5, user_id="u1")
    assert not whoami.called


@respx.mock
async def test_the_mode_is_read_once_per_token_not_once_per_call(with_token):
    """Otherwise every B2B tool call pays an extra request forever."""
    whoami = mock_whoami(B2B)
    respx.post(f"{API_BASE}/api/v1/memories/create").mock(
        return_value=httpx.Response(200, json={"ingestion_id": "ing_6"})
    )
    for _ in range(3):
        await client.create_memory("doc", user_id="u1", customer_id="c1")
    assert whoami.call_count == 1


@respx.mock
async def test_b2c_customer_only_recall_never_reaches_the_customer_route(with_token):
    """/v1/context/customer/fetch does not exist on a B2C instance.

    It used to answer 200 with an empty body there and now answers 400; neither is
    something a model can act on, so the call is refused before it is routed.
    """
    mock_whoami(B2C)
    route = respx.post(f"{API_BASE}/v1/context/customer/fetch").mock(
        return_value=httpx.Response(200, json={"context": {}})
    )
    with pytest.raises(CustomerIdNotAcceptedError) as exc:
        await client.fetch_context(["q"], max_results=5, customer_id="c1")
    assert not route.called
    assert "user scope" in str(exc.value), (
        "a customer-only recall must be told which route does work"
    )


@respx.mock
async def test_b2b_customer_only_recall_still_uses_the_customer_route(with_token):
    """The B2B path this whole change must leave alone."""
    mock_whoami(B2B)
    route = respx.post(f"{API_BASE}/v1/context/customer/fetch").mock(
        return_value=httpx.Response(200, json={"context": {}})
    )
    await client.fetch_context(["q"], max_results=5, customer_id="c1")
    assert route.called
    import json

    sent = json.loads(route.calls.last.request.read().decode())
    assert sent["customer_id"] == "c1"


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
