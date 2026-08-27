import os

# Pin the backing API base URL before importing app modules so settings is deterministic.
os.environ.setdefault("SYNAP_API_URL", "http://synap-cloud-test:8000")

import httpx
import pytest
import respx

from synap_mcp_server import client, context

API_BASE = "http://synap-cloud-test:8000"

# The two modes, spelled once. A test that hardcodes "strict" in ten places is a
# test that will keep passing after the string changes.
B2C = "equals_customer"
B2B = "strict"


@pytest.fixture(autouse=True)
def clear_isolation_cache():
    """Drop the cached whoami answer around every test.

    The mode is cached per token for five minutes in a module-global table. Left
    alone, the first test to learn a mode would decide it for every test after it,
    and the suite would pass or fail on ordering rather than on behaviour.
    """
    client._isolation_cache.clear()
    yield
    client._isolation_cache.clear()


def mock_whoami(isolation):
    """Mock ``GET /api/v1/auth/whoami`` as reporting ``isolation``.

    Pass ``None`` for a server that does not publish the field at all, which is
    what any deployment older than the contract looks like. Returns the respx
    route so a test can assert whether it was called.
    """
    body = {
        "client_id": "cli_test",
        "instance_id": "inst_test",
        "credential_id": "cred_test",
    }
    if isolation is not None:
        body["user_context_isolation"] = isolation
    return respx.get(f"{API_BASE}/api/v1/auth/whoami").mock(
        return_value=httpx.Response(200, json=body)
    )


@pytest.fixture
def with_token():
    """Set a valid Bearer token for the duration of a test, then clear it."""
    context.set_token("synap_testkey")
    yield "synap_testkey"
    context.set_token(None)


@pytest.fixture
def no_token():
    context.set_token(None)
    yield
    context.set_token(None)
