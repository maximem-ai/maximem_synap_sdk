"""Two production-checklist findings.

1. A context fetch is a POST that only reads, and `transport/endpoints.py`
   declares it idempotent, as the JS SDK's table does. The transport never read
   the table, so a fetch that timed out after sending was tried once in Python
   and three times in JS. A write that times out is still not retried.
2. An unwritable storage path (a read-only container) let a bare
   PermissionError out of the first fetch, past `except SynapError`. The cache
   now turns itself off.
"""
from __future__ import annotations

import asyncio
import os

import httpx
import pytest

from maximem_synap.auth.models import AuthContext
from maximem_synap.cache.manager import CacheManager, CacheScope
from maximem_synap.models.config import RetryPolicy
from maximem_synap.models.errors import NetworkTimeoutError
from maximem_synap.transport.http_client import HTTPTransport, _declared_idempotent

AUTH = AuthContext(client_id="c", instance_id="inst_1", api_key="k")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def _sleep(_s):
        return None
    monkeypatch.setattr(asyncio, "sleep", _sleep)


def _hanging():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        raise httpx.ReadTimeout("no answer", request=request)

    t = HTTPTransport(instance_id="inst_1", base_url="http://x", heartbeat_interval=0,
                      retry_policy=RetryPolicy(max_attempts=3, backoff_base=0.0, backoff_jitter=False))
    t._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")
    return t, seen


@pytest.mark.parametrize("path", ["/v1/context/user/fetch", "/v1/context/customer/fetch",
                                  "/v1/context/client/fetch", "/v1/context/conversation/fetch"])
@pytest.mark.asyncio
async def test_a_timed_out_fetch_is_retried(path):
    t, seen = _hanging()
    with pytest.raises(NetworkTimeoutError):
        await t.request("POST", path, AUTH, json={})
    assert len(seen) == 3


@pytest.mark.parametrize("path", ["/api/v1/memories/create", "/v1/conversations/messages"])
@pytest.mark.asyncio
async def test_a_timed_out_write_is_still_not_retried(path):
    t, seen = _hanging()
    with pytest.raises(NetworkTimeoutError):
        await t.request("POST", path, AUTH, json={})
    assert len(seen) == 1


def test_the_table_is_matched_by_method_and_path():
    assert _declared_idempotent("POST", "/v1/context/user/fetch")
    assert _declared_idempotent("GET", "/api/v1/memories/status/abc-123")
    assert not _declared_idempotent("POST", "/api/v1/memories/create")
    assert not _declared_idempotent("GET", "/v1/context/user/fetch")


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_an_unwritable_storage_path_turns_the_cache_off(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        m = CacheManager(client_id="c1", instance_id="inst_1", storage_path=str(ro))
        assert m.get(CacheScope.USER, "u1", "context") is None
        m.set(CacheScope.USER, "u1", "context", b"x")
        assert m.enabled is False
    finally:
        ro.chmod(0o700)
