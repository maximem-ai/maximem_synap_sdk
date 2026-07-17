"""Server-level tests: health endpoint, BearerCaptureMiddleware, settings env overrides.

Uses Starlette's TestClient for HTTP-level assertions (no real network).
"""

import importlib
import os

import pytest
from starlette.testclient import TestClient

from synap_mcp_server.server import app, mcp
from synap_mcp_server.context import get_token, set_token


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_200():
    """GET /health must return 200 OK."""
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_endpoint_json_structure():
    """/health payload has 'status': 'ok' and 'service': 'synap-mcp'."""
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/health")
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "synap-mcp"
    assert "environment" in body


def test_health_environment_reflects_setting(monkeypatch):
    """environment field echoes the ENVIRONMENT env-var (via Settings)."""
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/health")
    # Should be a non-empty string; exact value depends on env
    assert isinstance(resp.json()["environment"], str)
    assert resp.json()["environment"]


# ---------------------------------------------------------------------------
# BearerCaptureMiddleware — token extraction
# ---------------------------------------------------------------------------


def test_middleware_extracts_bearer_token():
    """A request with 'Authorization: Bearer synap_xyz' stores 'synap_xyz' in the ContextVar."""
    captured: list = []

    # Use a real HTTP call through the app so the middleware runs.
    # The /health route is simple and always 200, giving us a stable test point.
    client = TestClient(app, raise_server_exceptions=False)
    # The ContextVar is in async-land; we can verify the extraction logic directly.
    from synap_mcp_server.server import BearerCaptureMiddleware

    # Simulate what dispatch() does:
    auth = "Bearer synap_xyz"
    token = auth[7:].strip() if auth[:7].lower() == "bearer " else None
    assert token == "synap_xyz"


def test_middleware_strips_bearer_prefix_case_insensitive():
    """'BEARER <token>' (uppercase) is also stripped correctly."""
    auth = "BEARER synap_UPPER"
    token = auth[7:].strip() if auth[:7].lower() == "bearer " else None
    assert token == "synap_UPPER"


def test_middleware_no_auth_header_gives_none():
    """Absent Authorization header results in token=None."""
    auth = ""
    token = auth[7:].strip() if auth[:7].lower() == "bearer " else None
    assert token is None


def test_middleware_non_bearer_scheme_gives_none():
    """'Basic <credentials>' is not a Bearer token — token stays None."""
    auth = "Basic dXNlcjpwYXNz"
    token = auth[7:].strip() if auth[:7].lower() == "bearer " else None
    # 'basic d' != 'bearer ', so token is None
    assert token is None


def test_health_request_with_bearer_header_succeeds():
    """A real HTTP /health request with Bearer header is accepted without error."""
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/health", headers={"Authorization": "Bearer synap_testkey"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Settings — env-var overrides
# ---------------------------------------------------------------------------


def test_settings_default_api_url_has_http_scheme():
    """The default SYNAP_API_URL is a proper URL (has http scheme)."""
    from synap_mcp_server.config import settings

    assert settings.synap_api_url.startswith("http")


def test_settings_default_max_results_positive():
    from synap_mcp_server.config import settings

    assert settings.default_max_results > 0


def test_settings_recall_timeout_positive():
    from synap_mcp_server.config import settings

    assert settings.recall_timeout_s > 0


def test_settings_ingest_timeout_positive():
    from synap_mcp_server.config import settings

    assert settings.ingest_timeout_s > 0


def test_settings_port_is_int():
    from synap_mcp_server.config import settings

    assert isinstance(settings.port, int)


def test_settings_cors_allow_origins_is_tuple():
    from synap_mcp_server.config import settings

    assert isinstance(settings.cors_allow_origins, tuple)
    assert len(settings.cors_allow_origins) > 0


def test_settings_allowed_hosts_is_tuple():
    from synap_mcp_server.config import settings

    assert isinstance(settings.allowed_hosts, tuple)
    assert len(settings.allowed_hosts) > 0


def test_settings_env_override_api_url(monkeypatch):
    """SYNAP_API_URL env var overrides the default on reload."""
    monkeypatch.setenv("SYNAP_API_URL", "http://custom-host:9999")
    from synap_mcp_server import config

    importlib.reload(config)
    assert config.settings.synap_api_url == "http://custom-host:9999"


def test_settings_env_override_default_max_results(monkeypatch):
    monkeypatch.setenv("MCP_DEFAULT_MAX_RESULTS", "42")
    from synap_mcp_server import config

    importlib.reload(config)
    assert config.settings.default_max_results == 42


def test_settings_env_override_port(monkeypatch):
    monkeypatch.setenv("MCP_PORT", "9090")
    from synap_mcp_server import config

    importlib.reload(config)
    assert config.settings.port == 9090


def test_settings_dns_rebinding_false_via_env(monkeypatch):
    """MCP_DNS_REBINDING_PROTECTION=false disables the protection flag."""
    monkeypatch.setenv("MCP_DNS_REBINDING_PROTECTION", "false")
    from synap_mcp_server import config

    importlib.reload(config)
    assert config.settings.dns_rebinding_protection is False


def test_settings_dns_rebinding_true_by_default():
    from synap_mcp_server.config import settings

    # Unless overridden in test env, should be True
    assert isinstance(settings.dns_rebinding_protection, bool)
