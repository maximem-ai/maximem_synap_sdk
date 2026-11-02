"""Local mode: the SDK against the open-source server, which has no auth.

On when there is no API key (argument or SYNAP_API_KEY) and a base URL is set.
The JS twin is js/src/__tests__/local-mode.test.ts.

The server here is a real HTTP server on 127.0.0.1 in this process, so the
tests see the headers and paths the SDK actually puts on the wire.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from maximem_synap import (
    AuthenticationError,
    ContextNotFoundError,
    MaximemSynapSDK,
    SynapPermanentError,
)
from maximem_synap.models.config import SDKConfig
from maximem_synap.telemetry.collector import TelemetryCollector
from maximem_synap.transport import grpc_client
from maximem_synap.transport.http_client import HTTPTransport

CLOUD_ONLY = "Not available in the open-source server"


class FakeCore:
    """Records every request; answers whoami and the Cloud-only routes."""

    def __init__(self, whoami_status: int = 200):
        self.requests: list[tuple[str, str, dict]] = []
        self.whoami_status = whoami_status
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep the test output quiet
                pass

            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                outer.requests.append(
                    (self.command, self.path, {k.lower(): v for k, v in self.headers.items()})
                )
                if self.path == "/api/v1/auth/whoami":
                    status = outer.whoami_status
                    body = {"client_id": "cli_core", "instance_id": "inst_core",
                            "user_context_isolation": "equals_customer"}
                    if status != 200:
                        body = {"detail": "no"}
                elif self.path == "/v1/data/delete":
                    status, body = 404, {"detail": CLOUD_ONLY}
                else:
                    status, body = 200, {}
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = do_POST = do_PUT = do_DELETE = _answer

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()

    def auth_headers(self):
        return [h.get("authorization") for _, _, h in self.requests]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for k in ("SYNAP_API_KEY", "SYNAP_API_TOKEN", "SYNAP_BASE_URL",
              "SYNAP_INSTANCE_ID", "SYNAP_CLIENT_ID"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SYNAP_SDK_LOCAL_MODEL", "off")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def core():
    server = FakeCore()
    yield server
    server.close()


def _sdk(url=None, api_key=None, retry=False):
    cfg = SDKConfig(api_base_url=url, cache_backend=None,
                    retry_policy=None if not retry else SDKConfig().retry_policy)
    return MaximemSynapSDK(api_key=api_key, config=cfg, _force_new=True)


@pytest.mark.asyncio
async def test_keyless_with_base_url_learns_identity_from_whoami(core):
    sdk = _sdk(core.url)
    await sdk.initialize()
    try:
        assert sdk._client_id == "cli_core"
        assert sdk.instance_id == "inst_core"
        assert sdk._user_context_isolation == "equals_customer"
        assert [p for _, p, _ in core.requests] == ["/api/v1/auth/whoami"]
    finally:
        await sdk.shutdown()


@pytest.mark.asyncio
async def test_base_url_from_env_also_turns_local_mode_on(core, monkeypatch):
    monkeypatch.setenv("SYNAP_BASE_URL", core.url)
    sdk = _sdk()
    await sdk.initialize()
    try:
        assert sdk.instance_id == "inst_core"
    finally:
        await sdk.shutdown()


@pytest.mark.asyncio
async def test_keyless_without_base_url_fails_as_before():
    sdk = _sdk()
    with pytest.raises(AuthenticationError, match="No Synap API key found"):
        await sdk.initialize()


@pytest.mark.asyncio
async def test_empty_key_env_counts_as_no_key(core, monkeypatch):
    monkeypatch.setenv("SYNAP_API_KEY", "")
    sdk = _sdk(core.url)
    await sdk.initialize()
    try:
        assert sdk._local_mode is True
    finally:
        await sdk.shutdown()


@pytest.mark.asyncio
async def test_with_a_key_the_authorization_header_is_the_key(core, monkeypatch):
    # A token in the environment must not displace the key.
    monkeypatch.setenv("SYNAP_API_TOKEN", "tok-should-not-be-used")
    sdk = _sdk(core.url, api_key="sk-cloud")
    await sdk.initialize()
    try:
        assert sdk._local_mode is False
        assert core.auth_headers() == ["Bearer sk-cloud"]
    finally:
        await sdk.shutdown()


@pytest.mark.asyncio
async def test_token_is_sent_as_bearer_on_every_request(core, monkeypatch):
    monkeypatch.setenv("SYNAP_API_TOKEN", "tok-1")
    sdk = _sdk(core.url)
    await sdk.initialize()
    try:
        with pytest.raises(ContextNotFoundError):
            await sdk.data.delete("user", user_id="u1", dry_run=True)
        assert len(core.requests) == 2
        assert core.auth_headers() == ["Bearer tok-1", "Bearer tok-1"]
    finally:
        await sdk.shutdown()


@pytest.mark.asyncio
async def test_no_token_means_no_authorization_header_at_all(core):
    sdk = _sdk(core.url)
    await sdk.initialize()
    try:
        with pytest.raises(ContextNotFoundError):
            await sdk.data.delete("user", user_id="u1", dry_run=True)
        assert len(core.requests) == 2
        assert all("authorization" not in h for _, _, h in core.requests)
    finally:
        await sdk.shutdown()


@pytest.mark.asyncio
async def test_multipart_requests_follow_the_same_header_rule(core):
    sdk = _sdk(core.url)
    await sdk.initialize()
    try:
        ctx = await sdk._get_auth_context()
        await sdk._http_transport.post_multipart("/api/v1/memories/upload", ctx, data={})
        assert "authorization" not in core.requests[-1][2]
    finally:
        await sdk.shutdown()


@pytest.mark.asyncio
async def test_whoami_failure_in_local_mode_raises():
    # Nothing listens on this port: the server cannot be reached.
    sdk = _sdk("http://127.0.0.1:9")
    with pytest.raises(AuthenticationError, match="Could not reach the Synap server"):
        await sdk.initialize()


@pytest.mark.asyncio
async def test_whoami_error_status_in_local_mode_raises():
    server = FakeCore(whoami_status=500)
    try:
        sdk = _sdk(server.url)
        with pytest.raises(AuthenticationError, match="whoami"):
            await sdk.initialize()
    finally:
        server.close()


@pytest.mark.asyncio
async def test_whoami_failure_with_a_key_stays_non_fatal():
    sdk = _sdk("http://127.0.0.1:9", api_key="sk-cloud")
    await sdk.initialize()
    try:
        assert sdk._local_mode is False
    finally:
        await sdk.shutdown()


@pytest.mark.asyncio
async def test_listen_in_local_mode_raises_and_opens_no_channel(core, monkeypatch):
    opened = []
    monkeypatch.setattr(grpc_client.GRPCTransport, "__init__",
                        lambda self, *a, **k: opened.append("transport"))
    monkeypatch.setattr(grpc_client.GRPCTransport, "connect",
                        lambda self, *a, **k: opened.append("connect"), raising=False)
    sdk = _sdk(core.url)
    await sdk.initialize()
    try:
        assert sdk._grpc_transport is None  # initialize opened nothing
        with pytest.raises(SynapPermanentError, match=CLOUD_ONLY):
            await sdk.instance.listen()
        assert opened == []
        assert sdk._grpc_transport is None
    finally:
        await sdk.shutdown()


@pytest.mark.asyncio
async def test_a_cloud_only_error_body_reaches_the_caller_verbatim(core):
    sdk = _sdk(core.url)
    await sdk.initialize()
    try:
        with pytest.raises(ContextNotFoundError) as ei:
            await sdk.data.delete("user", user_id="u1", dry_run=True)
        assert str(ei.value) == CLOUD_ONLY
        assert ei.value.status_code == 404
    finally:
        await sdk.shutdown()


def test_a_501_cloud_only_body_is_not_replaced_by_generic_text():
    import httpx

    resp = httpx.Response(501, json={"detail": CLOUD_ONLY})
    transport = object.__new__(HTTPTransport)
    with pytest.raises(Exception) as ei:
        HTTPTransport._handle_response(transport, resp, "corr-1")
    assert str(ei.value) == CLOUD_ONLY
    assert ei.value.status_code == 501


@pytest.mark.asyncio
async def test_no_telemetry_collector_runs_in_local_mode(core, monkeypatch):
    started = []
    real_start = TelemetryCollector.start

    async def spy(self):
        started.append(self)
        await real_start(self)

    monkeypatch.setattr(TelemetryCollector, "start", spy)
    sdk = _sdk(core.url)
    await sdk.initialize()
    assert started == []
    assert sdk._telemetry_transport is None
    assert sdk._telemetry_collector.enabled is False
    assert sdk._telemetry_collector._flush_task is None
    await sdk.shutdown()
    assert not any(p.startswith("/v1/telemetry") for _, p, _ in core.requests)


@pytest.mark.asyncio
async def test_telemetry_collector_starts_with_a_key(core, monkeypatch):
    started = []
    real_start = TelemetryCollector.start

    async def spy(self):
        started.append(self)
        await real_start(self)

    monkeypatch.setattr(TelemetryCollector, "start", spy)
    sdk = _sdk(core.url, api_key="sk-cloud")
    await sdk.initialize()
    try:
        assert len(started) == 1
        assert sdk._telemetry_collector.enabled is True
        assert sdk._telemetry_collector._flush_task is not None
    finally:
        await sdk.shutdown()
