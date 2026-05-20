"""Regression tests for the SDK's whoami-on-initialize bootstrap.

Before this fix, the SDK only read client_id from the SYNAP_CLIENT_ID env
var. That silently broke the AnticipationCache scope filter for any caller
who didn't set the env (most real users plus the entire multi-tenant
playground), causing client-shared bundles to be excluded from every
lookup even when BM25 scored them well above threshold.

The whoami bootstrap pings GET /api/v1/auth/whoami after the api_key is
loaded; the server resolves the api_key to its canonical (client_id,
instance_id) pair and returns them. The SDK uses that to fill any
identity fields it doesn't already have.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from maximem_synap.sdk import MaximemSynapSDK


@pytest.mark.asyncio
async def test_whoami_populates_empty_client_id():
    """The original bug: api_key set but client_id env unset → cache scope-excluded.
    After fix: SDK calls whoami, fills client_id from the response."""
    os.environ.pop("SYNAP_CLIENT_ID", None)
    os.environ["SYNAP_INSTANCE_ID"] = "inst_test_bootstrap"

    sdk = MaximemSynapSDK(api_key="synap_test_key", _force_new=True)

    whoami_resp = {
        "client_id": "client_uuid_from_server",
        "instance_id": "inst_test_bootstrap",
        "credential_id": "cred_uuid",
    }
    with patch("maximem_synap.transport.http_client.HTTPTransport.get",
               new_callable=AsyncMock, return_value=whoami_resp):
        await sdk.initialize()

    assert sdk._client_id == "client_uuid_from_server", (
        "whoami should have populated client_id — empty client_id is the "
        "smoking gun for cache scope-exclusion"
    )
    assert sdk.instance_id == "inst_test_bootstrap"


@pytest.mark.asyncio
async def test_whoami_does_not_overwrite_explicit_env_client_id():
    """SYNAP_CLIENT_ID env still wins over whoami — preserves the offline
    / explicit-config path for callers that prefer it."""
    os.environ["SYNAP_CLIENT_ID"] = "explicit_client_id"
    os.environ["SYNAP_INSTANCE_ID"] = "inst_test_explicit"

    sdk = MaximemSynapSDK(api_key="synap_test_key", _force_new=True)

    whoami_resp = {"client_id": "different_from_whoami", "instance_id": "x"}
    with patch("maximem_synap.transport.http_client.HTTPTransport.get",
               new_callable=AsyncMock, return_value=whoami_resp):
        await sdk.initialize()

    assert sdk._client_id == "explicit_client_id"

    os.environ.pop("SYNAP_CLIENT_ID", None)


@pytest.mark.asyncio
async def test_whoami_failure_is_non_fatal():
    """A whoami failure must not break SDK init — it leaves whatever the
    env / explicit path produced and falls through."""
    os.environ.pop("SYNAP_CLIENT_ID", None)
    os.environ["SYNAP_INSTANCE_ID"] = "inst_test_failure"

    sdk = MaximemSynapSDK(api_key="synap_test_key", _force_new=True)

    with patch("maximem_synap.transport.http_client.HTTPTransport.get",
               new_callable=AsyncMock, side_effect=RuntimeError("network down")):
        await sdk.initialize()  # should NOT raise

    assert sdk._initialized
    assert sdk._client_id == ""  # nothing to fill from, but didn't crash
