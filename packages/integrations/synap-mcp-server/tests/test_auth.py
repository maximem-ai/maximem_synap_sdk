"""Auth pass-through tests (TC-AUTH-*)."""

import httpx
import pytest
import respx

from synap_mcp_server import client
from synap_mcp_server.context import MissingTokenError
from tests.conftest import API_BASE

pytestmark = pytest.mark.asyncio


@respx.mock
async def test_bearer_forwarded_verbatim(with_token):
    """TC-AUTH-01 / TC-AUTH-05: token forwarded as Bearer, no custom header."""
    route = respx.post(f"{API_BASE}/v1/context/client/fetch").mock(
        return_value=httpx.Response(200, json={"context": {}})
    )
    await client.fetch_context(["q"], max_results=10)
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer synap_testkey"
    assert "x-api-key" not in {k.lower() for k in req.headers}


async def test_missing_token_raises(no_token):
    """TC-AUTH-02: no token -> MissingTokenError before any REST call."""
    with pytest.raises(MissingTokenError):
        await client.fetch_context(["q"], max_results=10)


@respx.mock
async def test_rest_401_becomes_api_error(with_token):
    """TC-AUTH-03 / TC-AUTH-04: REST 401 surfaces as SynapAPIError(401)."""
    respx.post(f"{API_BASE}/v1/context/client/fetch").mock(
        return_value=httpx.Response(401, text="unauthorized")
    )
    with pytest.raises(client.SynapAPIError) as exc:
        await client.fetch_context(["q"], max_results=10)
    assert exc.value.status == 401
