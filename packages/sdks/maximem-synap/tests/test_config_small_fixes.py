"""SDK configuration findings from the configuration page review.

- RetryPolicy(max_attempts=0) skipped the request loop and raised
  `TypeError: exceptions must derive from BaseException`.
- With jitter on, a wait could reach 125% of backoff_max.
- The constructor wrote env fallbacks (base URL, gRPC host/port/TLS) into the
  SDKConfig it was given, so a config shared between clients changed.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from maximem_synap import MaximemSynapSDK, SDKConfig
from maximem_synap.auth.models import AuthContext
from maximem_synap.models.config import RetryPolicy
from maximem_synap.models.errors import ServiceUnavailableError
from maximem_synap.transport.http_client import HTTPTransport

AUTH = AuthContext(client_id="c", instance_id="inst_1", api_key="k")


def _transport(policy, handler):
    t = HTTPTransport(instance_id="inst_1", base_url="http://x", heartbeat_interval=0, retry_policy=policy)
    t._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")
    return t


@pytest.mark.asyncio
async def test_zero_attempts_means_one_attempt_and_the_real_error():
    seen = []

    def handler(request):
        seen.append(1)
        return httpx.Response(503, json={"detail": "down"})

    with pytest.raises(ServiceUnavailableError):
        await _transport(RetryPolicy(max_attempts=0), handler).request("GET", "/x", AUTH)
    assert len(seen) == 1


def test_jitter_never_goes_past_backoff_max(monkeypatch):
    t = _transport(RetryPolicy(backoff_base=100.0, backoff_max=10.0, backoff_jitter=True),
                   lambda r: httpx.Response(200, json={}))
    monkeypatch.setattr("maximem_synap.transport.http_client.random.random", lambda: 0.999)
    assert t._calculate_backoff(1) == 10.0


def test_the_constructor_does_not_change_the_config_you_pass(monkeypatch):
    monkeypatch.setenv("SYNAP_BASE_URL", "http://from-env")
    monkeypatch.setenv("SYNAP_GRPC_HOST", "grpc.from-env")
    config = SDKConfig()
    MaximemSynapSDK(api_key="synap_k", instance_id="inst_0123456789abcdef", config=config, _force_new=True)
    assert config.api_base_url is None and config.grpc_host is None
