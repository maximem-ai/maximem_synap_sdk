"""A 429 or 5xx the server answered with is retried, as the JS SDK does.

`_handle_response` raises the mapped error, and the request loop's
`except SynapError: raise` re-raised it before the retry check, so no HTTP
status was ever retried in Python on any method. JS twin:
js/src/__tests__/http-status-retry.test.ts.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from maximem_synap.auth.models import AuthContext
from maximem_synap.models.config import RetryPolicy
from maximem_synap.models.errors import (
    AuthenticationError,
    InvalidInputError,
    RateLimitError,
    ServiceUnavailableError,
)
from maximem_synap.transport.http_client import HTTPTransport

AUTH = AuthContext(client_id="c", instance_id="inst_1", api_key="k")
RETRY = RetryPolicy(max_attempts=3, backoff_base=0.0, backoff_max=10.0, backoff_jitter=False)


class Fake:
    def __init__(self, *script):
        self.script = list(script)
        self.seen = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.seen += 1
        status, body, headers = (self.script.pop(0) + ({},))[:3] if self.script else (200, {"ok": True}, {})
        return httpx.Response(status, json=body, headers=headers)


def make(fake, retry=RETRY):
    t = HTTPTransport(instance_id="inst_1", base_url="http://x", heartbeat_interval=0, retry_policy=retry)
    t._client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler), base_url="http://x")
    return t


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept = []

    async def _sleep(s):
        slept.append(s)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    return slept


@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
@pytest.mark.parametrize("status", [429, 500, 503])
@pytest.mark.asyncio
async def test_a_transient_status_is_retried_then_succeeds(method, status):
    fake = Fake((status, {"detail": "busy"}))
    result = await make(fake).request(method, "/api/v1/memories/forget", AUTH, json={})
    assert result == {"ok": True}
    assert fake.seen == 2


@pytest.mark.asyncio
async def test_it_gives_up_after_max_attempts():
    fake = Fake(*[(503, {"detail": "down"})] * 5)
    with pytest.raises(ServiceUnavailableError):
        await make(fake).request("POST", "/api/v1/memories/forget", AUTH, json={})
    assert fake.seen == 3


@pytest.mark.parametrize("status,error", [(400, InvalidInputError), (401, AuthenticationError)])
@pytest.mark.asyncio
async def test_a_permanent_status_is_not_retried(status, error):
    fake = Fake((status, {"detail": "no"}))
    with pytest.raises(error):
        await make(fake).request("POST", "/api/v1/memories/forget", AUTH, json={})
    assert fake.seen == 1


@pytest.mark.parametrize("reason", ["trial_limit_reached", "subscription_inactive"])
@pytest.mark.asyncio
async def test_a_credit_stop_is_not_retried(reason):
    fake = Fake((429, {"detail": {"error": reason, "reason": reason, "message": "m"}}))
    with pytest.raises(RateLimitError) as exc:
        await make(fake).request("POST", "/api/v1/memories/forget", AUTH, json={})
    assert exc.value.reason == reason
    assert fake.seen == 1


@pytest.mark.asyncio
async def test_retry_after_is_honoured_and_capped(no_sleep):
    fake = Fake((429, {"detail": "slow"}, {"Retry-After": "4"}), (429, {"detail": "slow"}, {"Retry-After": "60"}))
    await make(fake).request("GET", "/x", AUTH)
    assert no_sleep == [4.0, 10.0]


@pytest.mark.asyncio
async def test_multipart_upload_retries_a_503_too():
    fake = Fake((503, {"detail": "busy"}))
    result = await make(fake).post_multipart("/api/v1/memories/upload", AUTH, data={"text": "x"})
    assert result == {"ok": True}
    assert fake.seen == 2
