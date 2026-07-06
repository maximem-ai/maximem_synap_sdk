"""Tests for the precision_level fetch parameter (SDK side).

precision_level ("high" default | "medium") is validated client-side like
mode, included in the SDK's local cache params, and emitted in the HTTP body
only when non-default so existing server contracts are untouched.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from maximem_synap.models.errors import InvalidInputError
from maximem_synap.sdk import (
    ClientContextInterface,
    ConversationContextInterface,
    CustomerContextInterface,
    UserContextInterface,
)


def _fake_sdk():
    # MagicMock base: incidental attrs (turn counters, telemetry, summary
    # gates) auto-mock; everything awaited or branch-relevant is explicit.
    sdk = MagicMock()
    sdk._ensure_initialized = lambda: None
    sdk.instance_id = "inst_test"
    sdk._client_id = ""
    sdk._anticipation_cache.lookup.return_value = None
    sdk._cache_manager.get.return_value = None
    sdk._get_auth_context = AsyncMock(return_value=MagicMock())
    sdk._http_transport.post = AsyncMock(
        return_value={"context": {"facts": []}, "ttl_seconds": 300}
    )
    sdk.instance.is_listening = False
    sdk._is_st_authoritative = lambda: False
    sdk._should_inject_user_summary = MagicMock(return_value=False)
    sdk._maybe_trigger_compaction = AsyncMock()
    return sdk


_INTERFACES = [
    (UserContextInterface, {"user_id": "u1", "customer_id": "c1"}),
    (CustomerContextInterface, {"customer_id": "c1"}),
    (ClientContextInterface, {}),
    (ConversationContextInterface, {"conversation_id": "conv1"}),
]


class TestValidation:
    @pytest.mark.parametrize("iface_cls,kwargs", _INTERFACES)
    @pytest.mark.asyncio
    async def test_invalid_precision_level_raises(self, iface_cls, kwargs):
        iface = iface_cls(_fake_sdk())
        with pytest.raises(InvalidInputError):
            await iface.fetch(**kwargs, precision_level="low")

    @pytest.mark.parametrize("iface_cls,kwargs", _INTERFACES)
    @pytest.mark.asyncio
    async def test_invalid_mode_still_raises(self, iface_cls, kwargs):
        iface = iface_cls(_fake_sdk())
        with pytest.raises(InvalidInputError):
            await iface.fetch(**kwargs, mode="turbo")


class TestBodyEmission:
    @pytest.mark.asyncio
    async def test_medium_included_in_body(self):
        sdk = _fake_sdk()
        iface = UserContextInterface(sdk)
        await iface.fetch(user_id="u1", customer_id="c1", precision_level="medium")
        _, call_kwargs = sdk._http_transport.post.call_args
        assert call_kwargs["json"]["precision_level"] == "medium"

    @pytest.mark.asyncio
    async def test_high_default_omitted_from_body(self):
        """Default "high" is NOT sent — the wire contract stays identical to
        pre-feature SDKs, so old servers never see an unknown field."""
        sdk = _fake_sdk()
        iface = UserContextInterface(sdk)
        await iface.fetch(user_id="u1", customer_id="c1")
        _, call_kwargs = sdk._http_transport.post.call_args
        assert "precision_level" not in call_kwargs["json"]

    @pytest.mark.asyncio
    async def test_precision_level_keys_local_cache(self):
        """medium and high fetches must not share a local cache entry."""
        sdk = _fake_sdk()
        iface = UserContextInterface(sdk)
        await iface.fetch(user_id="u1", customer_id="c1", precision_level="medium")
        _, get_kwargs = sdk._cache_manager.get.call_args
        assert get_kwargs["query"]["precision_level"] == "medium"
