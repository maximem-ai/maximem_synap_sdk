"""A 504 is the gateway giving up, not the origin refusing: the origin may have finished the work.
Retrying a POST create after it can store and bill the same memory twice, so Python does not,
matching JS (js/src/transport/http.ts `isOutcomeUnknown`). Reads still retry. 500 and 503 are the
origin's own answer, so they keep retrying (test_http_status_retry.py)."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from maximem_synap.auth.models import AuthContext
from maximem_synap.models.config import RetryPolicy
from maximem_synap.models.errors import ServiceUnavailableError
from maximem_synap.transport.http_client import HTTPTransport

AUTH = AuthContext(client_id="c", instance_id="inst_1", api_key="k")
RETRY = RetryPolicy(max_attempts=3, backoff_base=0.0, backoff_max=10.0, backoff_jitter=False)


def make(statuses):
    seen = []

    def handler(request):
        seen.append(request.method)
        return httpx.Response(statuses.pop(0) if statuses else 200, json={"ok": True, "detail": "gw"})

    t = HTTPTransport(instance_id="inst_1", base_url="http://x", heartbeat_interval=0, retry_policy=RETRY)
    t._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")
    return t, seen


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def _sleep(s):
        return None

    monkeypatch.setattr(asyncio, "sleep", _sleep)


@pytest.mark.parametrize("path", ["/api/v1/memories/create", "/api/v1/memories/batch",
                                  "/api/v1/conversations/messages"])
@pytest.mark.asyncio
async def test_a_504_on_a_non_idempotent_post_is_sent_once(path):
    t, seen = make([504, 504, 504])
    with pytest.raises(ServiceUnavailableError):
        await t.request("POST", path, AUTH, json={"document": "x"})
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_a_504_on_a_read_is_still_retried():
    t, seen = make([504])
    assert await t.request("GET", "/api/v1/memories/abc", AUTH) == {"ok": True, "detail": "gw"}
    assert len(seen) == 2


@pytest.mark.parametrize("status", [500, 503])
@pytest.mark.asyncio
async def test_a_503_or_500_on_a_post_still_retries(status):
    t, seen = make([status])
    await t.request("POST", "/api/v1/memories/create", AUTH, json={"document": "x"})
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_a_504_on_a_multipart_upload_is_sent_once():
    t, seen = make([504, 504, 504])
    with pytest.raises(ServiceUnavailableError):
        await t.post_multipart("/api/v1/memories/upload", AUTH, data={"text": "x"})
    assert len(seen) == 1


def test_outcome_unknown_is_a_declared_attribute_defaulting_to_false():
    from maximem_synap.models.errors import (
        NetworkTimeoutError, RateLimitError, SynapTransientError,
    )

    assert SynapTransientError.outcome_unknown is False
    assert SynapTransientError("x").outcome_unknown is False
    assert ServiceUnavailableError("x").outcome_unknown is False
    assert RateLimitError("x").outcome_unknown is False
    assert NetworkTimeoutError("x").outcome_unknown is False


@pytest.mark.parametrize("status,expected", [(504, True), (503, False), (500, False), (502, False)])
@pytest.mark.asyncio
async def test_only_a_504_marks_the_raised_error_outcome_unknown(status, expected):
    t, _ = make([status, status, status])
    t.retry_policy = None  # one attempt, so the error comes straight out
    with pytest.raises(ServiceUnavailableError) as exc:
        await t.request("GET", "/api/v1/memories/abc", AUTH)
    assert exc.value.outcome_unknown is expected
