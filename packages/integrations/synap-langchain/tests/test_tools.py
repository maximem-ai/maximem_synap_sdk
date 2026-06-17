"""Tests for SynapSearchTool and SynapStoreTool.

Documented error-handling contract (from tools.py / wrap_sdk_errors_async):
- Both tools wrap SDK errors as SynapIntegrationError and re-raise.
- Callers (agent frameworks) decide how to handle the error.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from synap_langchain.tools import SynapSearchTool, SynapStoreTool, _SearchInput, _StoreInput
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sdk():
    from maximem_synap import MaximemSynapSDK
    sdk = MagicMock(spec=MaximemSynapSDK)
    sdk.fetch = AsyncMock()
    sdk.memories = MagicMock()
    sdk.memories.create = AsyncMock()
    return sdk


def _search_tool(sdk, **overrides):
    defaults = dict(
        sdk=sdk, user_id="u1", customer_id=None,
        conversation_id=None, mode="accurate", max_results=10,
        name="search_memory", description="Search memory",
        args_schema=_SearchInput,
    )
    defaults.update(overrides)
    return SynapSearchTool.model_construct(**defaults)


def _store_tool(sdk, **overrides):
    defaults = dict(
        sdk=sdk, user_id="u1", customer_id="c1",
        name="store_memory", description="Store memory",
        args_schema=_StoreInput,
    )
    defaults.update(overrides)
    return SynapStoreTool.model_construct(**defaults)


# ---------------------------------------------------------------------------
# SynapSearchTool — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_tool_returns_formatted_context(mock_sdk):
    """_arun returns the response's formatted_context string."""
    mock_sdk.fetch.return_value = MagicMock(formatted_context="User likes coffee")
    tool = _search_tool(mock_sdk)

    result = await tool._arun("coffee preferences")

    assert result == "User likes coffee"


@pytest.mark.asyncio
async def test_search_tool_forwards_correct_fetch_kwargs(mock_sdk):
    """SDK is called with the correct keyword arguments."""
    mock_sdk.fetch.return_value = MagicMock(formatted_context="ctx")
    tool = _search_tool(mock_sdk)

    await tool._arun("coffee preferences")

    mock_sdk.fetch.assert_awaited_once()
    call_kwargs = mock_sdk.fetch.call_args.kwargs
    assert call_kwargs["search_query"] == ["coffee preferences"]
    assert call_kwargs["user_id"] == "u1"
    assert call_kwargs["include_conversation_context"] is False


@pytest.mark.asyncio
async def test_search_tool_none_formatted_context_returns_fallback(mock_sdk):
    """None formatted_context → 'No relevant memories found.' sentinel string."""
    mock_sdk.fetch.return_value = MagicMock(formatted_context=None)
    tool = _search_tool(mock_sdk)

    result = await tool._arun("unknown topic")
    assert result == "No relevant memories found."


@pytest.mark.asyncio
async def test_search_tool_empty_string_context_returns_fallback(mock_sdk):
    """Empty-string formatted_context → same fallback sentinel."""
    mock_sdk.fetch.return_value = MagicMock(formatted_context="")
    tool = _search_tool(mock_sdk)

    result = await tool._arun("unknown topic")
    assert result == "No relevant memories found."


# ---------------------------------------------------------------------------
# SynapSearchTool — failure path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_tool_raises_synap_integration_error_on_sdk_failure(mock_sdk):
    """SDK failure is wrapped as SynapIntegrationError (wrap_sdk_errors_async contract)."""
    mock_sdk.fetch.side_effect = RuntimeError("sdk boom")
    tool = _search_tool(mock_sdk)

    with pytest.raises(SynapIntegrationError):
        await tool._arun("any query")


# ---------------------------------------------------------------------------
# SynapStoreTool — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_tool_returns_ingestion_id_string(mock_sdk):
    """_arun returns a string containing the ingestion_id."""
    mock_sdk.memories.create.return_value = MagicMock(ingestion_id="ing-123")
    tool = _store_tool(mock_sdk)

    result = await tool._arun("User prefers dark mode")

    assert "ing-123" in result


@pytest.mark.asyncio
async def test_store_tool_forwards_correct_create_kwargs(mock_sdk):
    """memories.create receives the right document, user_id, and customer_id."""
    mock_sdk.memories.create.return_value = MagicMock(ingestion_id="ing-xyz")
    tool = _store_tool(mock_sdk, user_id="u1", customer_id="c1")

    await tool._arun("User prefers dark mode")

    mock_sdk.memories.create.assert_awaited_once_with(
        document="User prefers dark mode",
        user_id="u1",
        customer_id="c1",
    )


@pytest.mark.asyncio
async def test_store_tool_none_customer_id_falls_back_to_empty_string(mock_sdk):
    """None customer_id is coerced to empty string before calling memories.create."""
    mock_sdk.memories.create.return_value = MagicMock(ingestion_id="ing-000")
    tool = _store_tool(mock_sdk, customer_id=None)

    await tool._arun("Some content")

    kw = mock_sdk.memories.create.call_args.kwargs
    assert kw["customer_id"] == ""


# ---------------------------------------------------------------------------
# SynapStoreTool — failure path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_tool_raises_synap_integration_error_on_sdk_failure(mock_sdk):
    """SDK failure is wrapped as SynapIntegrationError (wrap_sdk_errors_async contract)."""
    mock_sdk.memories.create.side_effect = RuntimeError("sdk boom")
    tool = _store_tool(mock_sdk)

    with pytest.raises(SynapIntegrationError):
        await tool._arun("content to store")
