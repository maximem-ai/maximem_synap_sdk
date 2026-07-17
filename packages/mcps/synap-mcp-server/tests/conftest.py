import os

# Pin the backing API base URL before importing app modules so settings is deterministic.
os.environ.setdefault("SYNAP_API_URL", "http://synap-cloud-test:8000")

import pytest

from synap_mcp_server import context

API_BASE = "http://synap-cloud-test:8000"


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
