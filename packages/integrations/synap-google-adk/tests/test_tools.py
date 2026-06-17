"""Tests for synap_google_adk.tools — create_synap_tools.

Covers:
- Construction validation (sdk=None, user_id empty)
- Return type: list of exactly two FunctionTool instances
- Tool names: "search_memory" and "store_memory"
- Tool descriptions: populated (non-empty)
- search_memory happy-path: forwards correct kwargs to sdk.fetch
- search_memory: None formatted_context returns fallback sentinel
- search_memory: empty-string formatted_context returns fallback sentinel
- search_memory: with conversation_id — passed through to fetch
- search_memory: without conversation_id — None forwarded
- search_memory: customer_id="" maps to None in fetch call
- search_memory: SDK failure raises SynapIntegrationError
- store_memory happy-path: forwards correct kwargs to sdk.memories.create
- store_memory: return string contains ingestion_id
- store_memory: SDK failure raises SynapIntegrationError
- store_memory: customer_id forwarded unchanged
- Both tools independently usable from same factory call
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.tools import FunctionTool

from synap_google_adk.tools import create_synap_tools
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sdk(
    formatted_context: str | None = "User context here.",
    ingestion_id: str = "ing-001",
) -> MagicMock:
    sdk = MagicMock()
    sdk.fetch = AsyncMock(
        return_value=MagicMock(formatted_context=formatted_context)
    )
    create_result = MagicMock()
    create_result.ingestion_id = ingestion_id
    sdk.memories = MagicMock()
    sdk.memories.create = AsyncMock(return_value=create_result)
    return sdk


def _get_tools(sdk, **kwargs):
    """Convenience: unpack into (search_tool, store_tool)."""
    tools = create_synap_tools(sdk, **kwargs)
    return tools[0], tools[1]


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_requires_non_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            create_synap_tools(None, user_id="u1")  # type: ignore[arg-type]

    def test_requires_non_empty_user_id(self):
        sdk = _make_sdk()
        with pytest.raises(ValueError, match="non-empty user_id"):
            create_synap_tools(sdk, user_id="")

    def test_accepts_optional_customer_id_empty_string(self):
        sdk = _make_sdk()
        tools = create_synap_tools(sdk, user_id="u1", customer_id="")
        assert len(tools) == 2

    def test_accepts_optional_conversation_id_none(self):
        sdk = _make_sdk()
        tools = create_synap_tools(sdk, user_id="u1", conversation_id=None)
        assert len(tools) == 2


# ---------------------------------------------------------------------------
# Return type and metadata
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_exactly_two_tools(self):
        sdk = _make_sdk()
        tools = create_synap_tools(sdk, user_id="u1")
        assert len(tools) == 2

    def test_both_tools_are_function_tool_instances(self):
        sdk = _make_sdk()
        search, store = _get_tools(sdk, user_id="u1")
        assert isinstance(search, FunctionTool)
        assert isinstance(store, FunctionTool)

    def test_first_tool_is_search_memory(self):
        sdk = _make_sdk()
        search, _ = _get_tools(sdk, user_id="u1")
        assert search.name == "search_memory"

    def test_second_tool_is_store_memory(self):
        sdk = _make_sdk()
        _, store = _get_tools(sdk, user_id="u1")
        assert store.name == "store_memory"

    def test_search_tool_has_nonempty_description(self):
        sdk = _make_sdk()
        search, _ = _get_tools(sdk, user_id="u1")
        assert search.description and len(search.description) > 0

    def test_store_tool_has_nonempty_description(self):
        sdk = _make_sdk()
        _, store = _get_tools(sdk, user_id="u1")
        assert store.description and len(store.description) > 0


# ---------------------------------------------------------------------------
# search_memory happy paths
# ---------------------------------------------------------------------------


class TestSearchMemoryHappyPath:
    @pytest.mark.asyncio
    async def test_returns_formatted_context_string(self):
        """Happy path: the tool returns the formatted_context from SDK."""
        sdk = _make_sdk(formatted_context="User likes tea.")
        search, _ = _get_tools(sdk, user_id="u1")
        result = await search.func("tea preferences")
        assert result == "User likes tea."

    @pytest.mark.asyncio
    async def test_forwards_query_as_list_to_fetch(self):
        """search_query is wrapped in a list before calling sdk.fetch."""
        sdk = _make_sdk()
        search, _ = _get_tools(sdk, user_id="u1")
        await search.func("coffee preferences")
        call_kwargs = sdk.fetch.call_args.kwargs
        assert call_kwargs["search_query"] == ["coffee preferences"]

    @pytest.mark.asyncio
    async def test_forwards_user_id_to_fetch(self):
        sdk = _make_sdk()
        search, _ = _get_tools(sdk, user_id="user-42")
        await search.func("query")
        assert sdk.fetch.call_args.kwargs["user_id"] == "user-42"

    @pytest.mark.asyncio
    async def test_mode_is_accurate(self):
        """mode='accurate' must be passed to sdk.fetch."""
        sdk = _make_sdk()
        search, _ = _get_tools(sdk, user_id="u1")
        await search.func("query")
        assert sdk.fetch.call_args.kwargs["mode"] == "accurate"

    @pytest.mark.asyncio
    async def test_include_conversation_context_is_false(self):
        sdk = _make_sdk()
        search, _ = _get_tools(sdk, user_id="u1")
        await search.func("query")
        assert sdk.fetch.call_args.kwargs["include_conversation_context"] is False

    @pytest.mark.asyncio
    async def test_conversation_id_forwarded_when_provided(self):
        sdk = _make_sdk()
        search, _ = _get_tools(sdk, user_id="u1", conversation_id="conv-99")
        await search.func("query")
        assert sdk.fetch.call_args.kwargs["conversation_id"] == "conv-99"

    @pytest.mark.asyncio
    async def test_conversation_id_is_none_when_not_provided(self):
        sdk = _make_sdk()
        search, _ = _get_tools(sdk, user_id="u1")
        await search.func("query")
        assert sdk.fetch.call_args.kwargs["conversation_id"] is None

    @pytest.mark.asyncio
    async def test_customer_id_passed_as_none_when_empty_string(self):
        """Empty customer_id is coerced to None before calling sdk.fetch."""
        sdk = _make_sdk()
        search, _ = _get_tools(sdk, user_id="u1", customer_id="")
        await search.func("query")
        assert sdk.fetch.call_args.kwargs["customer_id"] is None

    @pytest.mark.asyncio
    async def test_customer_id_forwarded_when_provided(self):
        sdk = _make_sdk()
        search, _ = _get_tools(sdk, user_id="u1", customer_id="cust-1")
        await search.func("query")
        assert sdk.fetch.call_args.kwargs["customer_id"] == "cust-1"

    @pytest.mark.asyncio
    async def test_none_formatted_context_returns_fallback_sentinel(self):
        """None from SDK → 'No relevant memories found.' sentinel."""
        sdk = _make_sdk(formatted_context=None)
        search, _ = _get_tools(sdk, user_id="u1")
        result = await search.func("unknown topic")
        assert result == "No relevant memories found."

    @pytest.mark.asyncio
    async def test_empty_string_context_returns_fallback_sentinel(self):
        """Empty string from SDK → 'No relevant memories found.' sentinel."""
        sdk = _make_sdk(formatted_context="")
        search, _ = _get_tools(sdk, user_id="u1")
        result = await search.func("unknown topic")
        assert result == "No relevant memories found."

    @pytest.mark.asyncio
    async def test_sdk_called_exactly_once_per_search(self):
        sdk = _make_sdk()
        search, _ = _get_tools(sdk, user_id="u1")
        await search.func("query")
        sdk.fetch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_using_shared_mock_sdk_fixture(self, mock_sdk):
        """Smoke test with the shared canonical mock_sdk fixture."""
        search, _ = _get_tools(mock_sdk, user_id="u1")
        result = await search.func("something")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# search_memory failure path
# ---------------------------------------------------------------------------


class TestSearchMemoryFailurePath:
    @pytest.mark.asyncio
    async def test_sdk_failure_raises_synap_integration_error(self):
        """RuntimeError from sdk.fetch is wrapped as SynapIntegrationError."""
        sdk = _make_sdk()
        sdk.fetch = AsyncMock(side_effect=RuntimeError("sdk boom"))
        search, _ = _get_tools(sdk, user_id="u1")

        with pytest.raises(SynapIntegrationError):
            await search.func("any query")

    @pytest.mark.asyncio
    async def test_failure_path_with_shared_failing_sdk_fixture(self, failing_sdk):
        """Verify failure path with the canonical shared failing_sdk fixture."""
        search, _ = _get_tools(failing_sdk, user_id="u1")

        with pytest.raises(SynapIntegrationError):
            await search.func("any query")


# ---------------------------------------------------------------------------
# store_memory happy paths
# ---------------------------------------------------------------------------


class TestStoreMemoryHappyPath:
    @pytest.mark.asyncio
    async def test_returns_string_containing_ingestion_id(self):
        """Happy path: return string includes the ingestion_id."""
        sdk = _make_sdk(ingestion_id="ing-123")
        _, store = _get_tools(sdk, user_id="u1")
        result = await store.func("User prefers dark mode")
        assert "ing-123" in result

    @pytest.mark.asyncio
    async def test_return_value_is_formatted_string(self):
        """Return value must match the documented pattern."""
        sdk = _make_sdk(ingestion_id="ing-abc")
        _, store = _get_tools(sdk, user_id="u1")
        result = await store.func("content")
        assert result == "Memory stored (ingestion_id: ing-abc)"

    @pytest.mark.asyncio
    async def test_forwards_content_as_document(self):
        """Content string forwarded as 'document' kwarg to memories.create."""
        sdk = _make_sdk()
        _, store = _get_tools(sdk, user_id="u1")
        await store.func("User likes dark mode")
        assert sdk.memories.create.call_args.kwargs["document"] == "User likes dark mode"

    @pytest.mark.asyncio
    async def test_forwards_user_id_to_memories_create(self):
        sdk = _make_sdk()
        _, store = _get_tools(sdk, user_id="user-55")
        await store.func("content")
        assert sdk.memories.create.call_args.kwargs["user_id"] == "user-55"

    @pytest.mark.asyncio
    async def test_forwards_customer_id_when_provided(self):
        sdk = _make_sdk()
        _, store = _get_tools(sdk, user_id="u1", customer_id="cust-9")
        await store.func("content")
        assert sdk.memories.create.call_args.kwargs["customer_id"] == "cust-9"

    @pytest.mark.asyncio
    async def test_forwards_empty_string_customer_id(self):
        """Empty customer_id is passed directly to memories.create (not coerced)."""
        sdk = _make_sdk()
        _, store = _get_tools(sdk, user_id="u1", customer_id="")
        await store.func("content")
        assert sdk.memories.create.call_args.kwargs["customer_id"] == ""

    @pytest.mark.asyncio
    async def test_full_kwargs_forwarded_together(self):
        """All three kwargs (document, user_id, customer_id) forwarded in one call."""
        sdk = _make_sdk()
        _, store = _get_tools(sdk, user_id="u1", customer_id="c1")
        await store.func("User prefers dark mode")

        sdk.memories.create.assert_awaited_once_with(
            document="User prefers dark mode",
            user_id="u1",
            customer_id="c1",
        )

    @pytest.mark.asyncio
    async def test_sdk_called_exactly_once_per_store(self):
        sdk = _make_sdk()
        _, store = _get_tools(sdk, user_id="u1")
        await store.func("content")
        sdk.memories.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_using_shared_mock_sdk_fixture(self, mock_sdk):
        """Smoke test with the shared canonical mock_sdk fixture."""
        _, store = _get_tools(mock_sdk, user_id="u1")
        result = await store.func("some content")
        assert "ing-001" in result  # shared fixture uses ingestion_id="ing-001"


# ---------------------------------------------------------------------------
# store_memory failure path
# ---------------------------------------------------------------------------


class TestStoreMemoryFailurePath:
    @pytest.mark.asyncio
    async def test_sdk_failure_raises_synap_integration_error(self):
        """RuntimeError from memories.create is wrapped as SynapIntegrationError."""
        sdk = _make_sdk()
        sdk.memories.create = AsyncMock(side_effect=RuntimeError("sdk boom"))
        _, store = _get_tools(sdk, user_id="u1")

        with pytest.raises(SynapIntegrationError):
            await store.func("content to store")

    @pytest.mark.asyncio
    async def test_failure_path_with_shared_failing_sdk_fixture(self, failing_sdk):
        """Verify failure path with the canonical shared failing_sdk fixture."""
        _, store = _get_tools(failing_sdk, user_id="u1")

        with pytest.raises(SynapIntegrationError):
            await store.func("content")


# ---------------------------------------------------------------------------
# Independence: both tools from same factory call
# ---------------------------------------------------------------------------


class TestToolIndependence:
    @pytest.mark.asyncio
    async def test_search_and_store_callable_independently(self):
        """Both tools can be invoked in the same test without interference."""
        sdk = _make_sdk(formatted_context="User info", ingestion_id="ing-joint")
        search, store = _get_tools(sdk, user_id="u1", customer_id="c1")

        search_result = await search.func("user info")
        store_result = await store.func("new memory")

        assert search_result == "User info"
        assert "ing-joint" in store_result

    @pytest.mark.asyncio
    async def test_multiple_calls_accumulate_correctly(self):
        """Multiple sequential search calls each hit sdk.fetch once."""
        sdk = _make_sdk(formatted_context="result")
        search, _ = _get_tools(sdk, user_id="u1")

        await search.func("query 1")
        await search.func("query 2")

        assert sdk.fetch.await_count == 2
