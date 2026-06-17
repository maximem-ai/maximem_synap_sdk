"""Tests for synap_openai_agents.tools — create_search_tool / create_store_tool.

Documented error-handling contract (from tools.py / wrap_sdk_errors_async):
- Both factory functions validate sdk != None and user_id != "".
- The returned async callables wrap SDK errors as SynapIntegrationError.
- Callers (agent frameworks / test harnesses) decide how to handle the error.

Covers:
- Construction validation: sdk=None, empty user_id for both factories
- search_memory happy paths: result forwarding, kwargs forwarded to sdk.fetch,
  None/empty formatted_context fallback sentinel, conversation_id threading,
  customer_id threading
- store_memory happy paths: ingestion_id in return, kwargs forwarded to
  sdk.memories.create, customer_id pass-through
- Failure paths: SynapIntegrationError raised on SDK failure (both tools)
- failing_sdk harness fixture
- Public surface exports
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_openai_agents.tools import create_search_tool, create_store_tool
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Local helper: minimal SDK mock (unit-level, independent of shared harness)
# ---------------------------------------------------------------------------


def _sdk(fetch_return=None, create_return=None):
    sdk = MagicMock()
    sdk.fetch = AsyncMock(return_value=fetch_return or MagicMock(formatted_context="ctx"))
    sdk.memories = MagicMock()
    sdk.memories.create = AsyncMock(
        return_value=create_return or MagicMock(ingestion_id="ing-001")
    )
    return sdk


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_package_exports_create_search_tool():
    from synap_openai_agents import create_search_tool
    assert create_search_tool is not None


def test_package_exports_create_store_tool():
    from synap_openai_agents import create_store_tool
    assert create_store_tool is not None


# ---------------------------------------------------------------------------
# create_search_tool — construction validation
# ---------------------------------------------------------------------------


class TestCreateSearchToolValidation:
    def test_raises_on_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            create_search_tool(None, user_id="u1")  # type: ignore[arg-type]

    def test_raises_on_empty_user_id(self):
        sdk = _sdk()
        with pytest.raises(ValueError, match="non-empty user_id"):
            create_search_tool(sdk, user_id="")

    def test_returns_async_callable(self):
        sdk = _sdk()
        fn = create_search_tool(sdk, user_id="u1")
        assert callable(fn)
        assert inspect.iscoroutinefunction(fn), "search_memory must be async"


# ---------------------------------------------------------------------------
# create_search_tool — happy paths
# ---------------------------------------------------------------------------


class TestSearchToolHappyPath:
    @pytest.mark.asyncio
    async def test_returns_formatted_context(self):
        sdk = _sdk(fetch_return=MagicMock(formatted_context="User likes tea"))
        search = create_search_tool(sdk, user_id="u1")
        result = await search(query="tea")
        assert result == "User likes tea"

    @pytest.mark.asyncio
    async def test_forwards_query_as_list(self):
        sdk = _sdk()
        search = create_search_tool(sdk, user_id="u1")
        await search(query="coffee preferences")
        sdk.fetch.assert_awaited_once()
        assert sdk.fetch.call_args.kwargs["search_query"] == ["coffee preferences"]

    @pytest.mark.asyncio
    async def test_forwards_user_id(self):
        sdk = _sdk()
        search = create_search_tool(sdk, user_id="user-42")
        await search(query="q")
        assert sdk.fetch.call_args.kwargs["user_id"] == "user-42"

    @pytest.mark.asyncio
    async def test_mode_is_accurate(self):
        sdk = _sdk()
        search = create_search_tool(sdk, user_id="u1")
        await search(query="q")
        assert sdk.fetch.call_args.kwargs["mode"] == "accurate"

    @pytest.mark.asyncio
    async def test_include_conversation_context_is_false(self):
        sdk = _sdk()
        search = create_search_tool(sdk, user_id="u1")
        await search(query="q")
        assert sdk.fetch.call_args.kwargs["include_conversation_context"] is False

    @pytest.mark.asyncio
    async def test_none_formatted_context_returns_sentinel(self):
        sdk = _sdk(fetch_return=MagicMock(formatted_context=None))
        search = create_search_tool(sdk, user_id="u1")
        result = await search(query="unknown")
        assert result == "No relevant memories found."

    @pytest.mark.asyncio
    async def test_empty_string_formatted_context_returns_sentinel(self):
        sdk = _sdk(fetch_return=MagicMock(formatted_context=""))
        search = create_search_tool(sdk, user_id="u1")
        result = await search(query="unknown")
        assert result == "No relevant memories found."

    @pytest.mark.asyncio
    async def test_conversation_id_forwarded_when_provided(self):
        sdk = _sdk()
        search = create_search_tool(sdk, user_id="u1", conversation_id="conv-99")
        await search(query="q")
        assert sdk.fetch.call_args.kwargs["conversation_id"] == "conv-99"

    @pytest.mark.asyncio
    async def test_conversation_id_none_when_not_provided(self):
        sdk = _sdk()
        search = create_search_tool(sdk, user_id="u1")
        await search(query="q")
        # None or not present — depends on implementation; either is acceptable
        kwargs = sdk.fetch.call_args.kwargs
        assert kwargs.get("conversation_id") is None

    @pytest.mark.asyncio
    async def test_customer_id_forwarded(self):
        sdk = _sdk()
        search = create_search_tool(sdk, user_id="u1", customer_id="cust-7")
        await search(query="q")
        kwargs = sdk.fetch.call_args.kwargs
        # customer_id='' maps to None internally; non-empty is forwarded as-is
        assert kwargs.get("customer_id") == "cust-7"

    @pytest.mark.asyncio
    async def test_empty_customer_id_coerced_to_none(self):
        """Empty customer_id is coerced to None before sdk.fetch per product code."""
        sdk = _sdk()
        search = create_search_tool(sdk, user_id="u1", customer_id="")
        await search(query="q")
        kwargs = sdk.fetch.call_args.kwargs
        assert kwargs.get("customer_id") is None


# ---------------------------------------------------------------------------
# create_search_tool — failure paths
# ---------------------------------------------------------------------------


class TestSearchToolFailurePath:
    @pytest.mark.asyncio
    async def test_raises_synap_integration_error_on_sdk_failure(self):
        sdk = _sdk()
        sdk.fetch = AsyncMock(side_effect=RuntimeError("sdk boom"))
        search = create_search_tool(sdk, user_id="u1")
        with pytest.raises(SynapIntegrationError):
            await search(query="any query")

    @pytest.mark.asyncio
    async def test_synap_integration_error_chains_original_cause(self):
        original = RuntimeError("original sdk error")
        sdk = _sdk()
        sdk.fetch = AsyncMock(side_effect=original)
        search = create_search_tool(sdk, user_id="u1")
        with pytest.raises(SynapIntegrationError) as exc_info:
            await search(query="q")
        assert exc_info.value.__cause__ is original

    @pytest.mark.asyncio
    async def test_failing_sdk_search_raises(self, failing_sdk):
        """Shared failing_sdk fixture — get_context_for_prompt raises RuntimeError."""
        # failing_sdk wires sdk.fetch to raise; so create_search_tool must propagate
        search = create_search_tool(failing_sdk, user_id="u1")
        with pytest.raises(SynapIntegrationError):
            await search(query="anything")


# ---------------------------------------------------------------------------
# create_store_tool — construction validation
# ---------------------------------------------------------------------------


class TestCreateStoreToolValidation:
    def test_raises_on_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            create_store_tool(None, user_id="u1")  # type: ignore[arg-type]

    def test_raises_on_empty_user_id(self):
        sdk = _sdk()
        with pytest.raises(ValueError, match="non-empty user_id"):
            create_store_tool(sdk, user_id="")

    def test_returns_async_callable(self):
        sdk = _sdk()
        fn = create_store_tool(sdk, user_id="u1")
        assert callable(fn)
        assert inspect.iscoroutinefunction(fn), "store_memory must be async"


# ---------------------------------------------------------------------------
# create_store_tool — happy paths
# ---------------------------------------------------------------------------


class TestStoreToolHappyPath:
    @pytest.mark.asyncio
    async def test_returns_string_with_ingestion_id(self):
        sdk = _sdk(create_return=MagicMock(ingestion_id="ing-123"))
        store = create_store_tool(sdk, user_id="u1")
        result = await store(content="User prefers dark mode")
        assert "ing-123" in result

    @pytest.mark.asyncio
    async def test_returns_string(self):
        sdk = _sdk()
        store = create_store_tool(sdk, user_id="u1")
        result = await store(content="some content")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_forwards_document_to_memories_create(self):
        sdk = _sdk()
        store = create_store_tool(sdk, user_id="u1")
        await store(content="User prefers dark mode")
        sdk.memories.create.assert_awaited_once()
        kwargs = sdk.memories.create.call_args.kwargs
        assert kwargs["document"] == "User prefers dark mode"

    @pytest.mark.asyncio
    async def test_forwards_user_id_to_memories_create(self):
        sdk = _sdk()
        store = create_store_tool(sdk, user_id="user-99")
        await store(content="some info")
        kwargs = sdk.memories.create.call_args.kwargs
        assert kwargs["user_id"] == "user-99"

    @pytest.mark.asyncio
    async def test_forwards_customer_id_to_memories_create(self):
        sdk = _sdk()
        store = create_store_tool(sdk, user_id="u1", customer_id="cust-42")
        await store(content="info")
        kwargs = sdk.memories.create.call_args.kwargs
        assert kwargs["customer_id"] == "cust-42"

    @pytest.mark.asyncio
    async def test_empty_customer_id_forwarded_as_empty_string(self):
        """Default customer_id='' is passed through to memories.create unchanged."""
        sdk = _sdk()
        store = create_store_tool(sdk, user_id="u1")
        await store(content="info")
        kwargs = sdk.memories.create.call_args.kwargs
        assert kwargs["customer_id"] == ""

    @pytest.mark.asyncio
    async def test_mock_sdk_store_happy_path(self, mock_sdk):
        """Shared mock_sdk harness: memories.create pre-wired to return ingestion_id=ing-001."""
        store = create_store_tool(mock_sdk, user_id="u1")
        result = await store(content="remember this")
        assert "ing-001" in result


# ---------------------------------------------------------------------------
# create_store_tool — failure paths
# ---------------------------------------------------------------------------


class TestStoreToolFailurePath:
    @pytest.mark.asyncio
    async def test_raises_synap_integration_error_on_sdk_failure(self):
        sdk = _sdk()
        sdk.memories.create = AsyncMock(side_effect=RuntimeError("sdk boom"))
        store = create_store_tool(sdk, user_id="u1")
        with pytest.raises(SynapIntegrationError):
            await store(content="content to store")

    @pytest.mark.asyncio
    async def test_synap_integration_error_chains_original_cause(self):
        original = RuntimeError("original error")
        sdk = _sdk()
        sdk.memories.create = AsyncMock(side_effect=original)
        store = create_store_tool(sdk, user_id="u1")
        with pytest.raises(SynapIntegrationError) as exc_info:
            await store(content="content")
        assert exc_info.value.__cause__ is original

    @pytest.mark.asyncio
    async def test_failing_sdk_store_raises(self, failing_sdk):
        """Shared failing_sdk fixture — sdk.memories.create raises RuntimeError."""
        store = create_store_tool(failing_sdk, user_id="u1")
        with pytest.raises(SynapIntegrationError):
            await store(content="anything")
