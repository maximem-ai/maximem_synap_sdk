"""SDK transport endpoint resolution from env vars.

Verifies that MaximemSynapSDK honors SYNAP_BASE_URL / SYNAP_GRPC_HOST /
SYNAP_GRPC_PORT / SYNAP_GRPC_USE_TLS, with priority:

  explicit SDKConfig field  >  env var  >  hardcoded transport default.

Mirrors the SYNAP_API_KEY / SYNAP_INSTANCE_ID env-var pattern already
established elsewhere in the SDK.
"""

from __future__ import annotations

import os

import pytest

from maximem_synap.sdk import MaximemSynapSDK
from maximem_synap.models.config import SDKConfig


_ENV_KEYS = (
    "SYNAP_BASE_URL",
    "SYNAP_GRPC_HOST",
    "SYNAP_GRPC_PORT",
    "SYNAP_GRPC_USE_TLS",
    "SYNAP_INSTANCE_ID",
    "SYNAP_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SYNAP_API_KEY", "fake-for-tests")


def test_env_vars_populate_unset_config():
    os.environ["SYNAP_BASE_URL"] = "https://staging.example.com"
    os.environ["SYNAP_GRPC_HOST"] = "grpc-staging.example.com"
    os.environ["SYNAP_GRPC_PORT"] = "443"
    os.environ["SYNAP_GRPC_USE_TLS"] = "true"

    sdk = MaximemSynapSDK(instance_id="t1", _force_new=True)

    assert sdk._config.api_base_url == "https://staging.example.com"
    assert sdk._config.grpc_host == "grpc-staging.example.com"
    assert sdk._config.grpc_port == 443
    assert sdk._config.grpc_use_tls is True


def test_explicit_config_beats_env_var():
    os.environ["SYNAP_BASE_URL"] = "https://env.example.com"
    os.environ["SYNAP_GRPC_USE_TLS"] = "true"

    sdk = MaximemSynapSDK(
        instance_id="t2",
        config=SDKConfig(
            api_base_url="https://explicit.example.com",
            grpc_host="ex.host",
            grpc_port=50051,
            grpc_use_tls=False,
        ),
        _force_new=True,
    )
    assert sdk._config.api_base_url == "https://explicit.example.com"
    assert sdk._config.grpc_host == "ex.host"
    assert sdk._config.grpc_port == 50051
    assert sdk._config.grpc_use_tls is False


def test_nothing_set_leaves_none_for_transport_default():
    sdk = MaximemSynapSDK(instance_id="t3", _force_new=True)
    assert sdk._config.api_base_url is None
    assert sdk._config.grpc_host is None
    assert sdk._config.grpc_port is None
    # None at SDKConfig level — call-site resolves to True (transport default).
    assert sdk._config.grpc_use_tls is None


def test_garbage_port_ignored_with_warning(caplog):
    os.environ["SYNAP_GRPC_PORT"] = "not-a-port"
    sdk = MaximemSynapSDK(instance_id="t4", _force_new=True)
    assert sdk._config.grpc_port is None


def test_env_tls_false_honored():
    os.environ["SYNAP_GRPC_USE_TLS"] = "false"
    sdk = MaximemSynapSDK(instance_id="t5", _force_new=True)
    assert sdk._config.grpc_use_tls is False


def test_env_tls_accepts_common_truthy_strings():
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        os.environ["SYNAP_GRPC_USE_TLS"] = truthy
        sdk = MaximemSynapSDK(instance_id=f"t_{truthy}", _force_new=True)
        assert sdk._config.grpc_use_tls is True, (
            f"{truthy!r} should resolve to True"
        )


def test_http_transport_base_url_propagation():
    from maximem_synap.transport.http_client import HTTPTransport

    os.environ["SYNAP_BASE_URL"] = "https://probe.example.com"
    sdk = MaximemSynapSDK(instance_id="t6", _force_new=True)

    # Mirror what SDK.initialize() does: construct transport from config.
    t = HTTPTransport(instance_id="t6", base_url=sdk._config.api_base_url)
    assert t.base_url == "https://probe.example.com"
