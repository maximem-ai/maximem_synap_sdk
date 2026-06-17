"""Tests for SynapPlugin — public surface, arg propagation, failure paths.

Uses the shared harness (``mock_sdk`` / ``failing_sdk`` from
``synap_integrations_common.testing``, re-exported via conftest). Both
kernel functions wrap their SDK call in ``wrap_sdk_errors_async``, so an
SDK failure must surface as ``SynapIntegrationError`` — never an
unhandled crash.
"""

import pytest

from synap_integrations_common import SynapIntegrationError
from synap_integrations_common.testing import make_unified_response
from synap_semantic_kernel.plugin import SynapPlugin


@pytest.fixture
def plugin(mock_sdk):
    return SynapPlugin(sdk=mock_sdk, user_id="u1")


class TestConstruction:
    def test_public_surface_exported(self):
        import synap_semantic_kernel

        assert synap_semantic_kernel.SynapPlugin is SynapPlugin

    def test_requires_non_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapPlugin(sdk=None, user_id="u1")  # type: ignore[arg-type]

    def test_requires_non_empty_user_id(self, mock_sdk):
        with pytest.raises(ValueError, match="non-empty user_id"):
            SynapPlugin(sdk=mock_sdk, user_id="")


class TestSearchMemory:
    @pytest.mark.asyncio
    async def test_returns_formatted_context(self, plugin, mock_sdk):
        result = await plugin.search_memory(query="coffee")

        assert result == mock_sdk.fetch.return_value.formatted_context
        mock_sdk.fetch.assert_awaited_once()
        kwargs = mock_sdk.fetch.call_args.kwargs
        assert kwargs["search_query"] == ["coffee"]
        assert kwargs["user_id"] == "u1"
        assert kwargs["mode"] == "accurate"
        assert kwargs["include_conversation_context"] is False

    @pytest.mark.asyncio
    async def test_empty_context_returns_fallback(self, plugin, mock_sdk):
        mock_sdk.fetch.return_value = make_unified_response(formatted_context="")

        result = await plugin.search_memory(query="nothing")

        assert result == "No relevant memories found."

    @pytest.mark.asyncio
    async def test_customer_id_propagated_when_set(self, mock_sdk):
        plugin = SynapPlugin(sdk=mock_sdk, user_id="u1", customer_id="c1")

        await plugin.search_memory(query="q")

        assert mock_sdk.fetch.call_args.kwargs["customer_id"] == "c1"

    @pytest.mark.asyncio
    async def test_empty_customer_id_passes_none(self, plugin, mock_sdk):
        await plugin.search_memory(query="q")

        # plugin coerces "" -> None so the SDK searches without a customer scope
        assert mock_sdk.fetch.call_args.kwargs["customer_id"] is None

    @pytest.mark.asyncio
    async def test_sdk_failure_surfaces_wrapped(self, failing_sdk):
        plugin = SynapPlugin(sdk=failing_sdk, user_id="u1")

        with pytest.raises(SynapIntegrationError):
            await plugin.search_memory(query="coffee")


class TestStoreMemory:
    @pytest.mark.asyncio
    async def test_stores_and_returns_ingestion_id(self, plugin, mock_sdk):
        result = await plugin.store_memory(content="prefers dark mode")

        assert "ing-001" in result
        mock_sdk.memories.create.assert_awaited_once()
        kwargs = mock_sdk.memories.create.call_args.kwargs
        assert kwargs["document"] == "prefers dark mode"
        assert kwargs["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_sdk_failure_surfaces_wrapped(self, failing_sdk):
        plugin = SynapPlugin(sdk=failing_sdk, user_id="u1")

        with pytest.raises(SynapIntegrationError):
            await plugin.store_memory(content="x")
