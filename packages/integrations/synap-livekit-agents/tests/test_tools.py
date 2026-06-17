"""Tests for synap_livekit_agents.tools — synap_search_tool, synap_store_tool.

Documented error-handling contract (from tools.py docstring):
- synap_search: read-side failure degrades gracefully — returns a
  natural-language placeholder string, never raises.
- synap_store: write-side failure surfaces as SynapIntegrationError via
  wrap_sdk_errors_async — the LLM sees a proper tool error.
- Both factories validate sdk and user_id at construction time (before any
  async call), raising ValueError immediately.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from livekit.agents import FunctionTool

from synap_integrations_common import SynapIntegrationError
from synap_livekit_agents.tools import synap_search_tool, synap_store_tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _search_sdk(formatted_context: str | None = "User is an engineer."):
    sdk = MagicMock()
    resp = MagicMock()
    resp.formatted_context = formatted_context
    sdk.fetch = AsyncMock(return_value=resp)
    return sdk


def _store_sdk(ingestion_id: str = "ing-001"):
    sdk = MagicMock()
    result = MagicMock()
    result.ingestion_id = ingestion_id
    sdk.memories = MagicMock()
    sdk.memories.create = AsyncMock(return_value=result)
    return sdk


# ---------------------------------------------------------------------------
# synap_search_tool — factory validation
# ---------------------------------------------------------------------------


class TestSearchToolValidation:
    def test_requires_non_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            synap_search_tool(None, user_id="u1")  # type: ignore[arg-type]

    def test_requires_non_empty_user_id(self):
        with pytest.raises(ValueError, match="non-empty user_id"):
            synap_search_tool(MagicMock(), user_id="")


# ---------------------------------------------------------------------------
# synap_search_tool — factory return type
# ---------------------------------------------------------------------------


class TestSearchToolType:
    def test_returns_function_tool_instance(self):
        tool = synap_search_tool(_search_sdk(), user_id="u1")
        assert isinstance(tool, FunctionTool)

    def test_tool_info_name_is_synap_search(self):
        tool = synap_search_tool(_search_sdk(), user_id="u1")
        assert tool.info.name == "synap_search"

    def test_tool_info_description_mentions_memory(self):
        tool = synap_search_tool(_search_sdk(), user_id="u1")
        assert "memory" in tool.info.description.lower()

    def test_tool_is_callable(self):
        tool = synap_search_tool(_search_sdk(), user_id="u1")
        assert callable(tool._func)


# ---------------------------------------------------------------------------
# synap_search_tool — happy paths
# ---------------------------------------------------------------------------


class TestSearchToolHappyPath:
    @pytest.mark.asyncio
    async def test_returns_formatted_context(self):
        sdk = _search_sdk("User is a Python developer.")
        tool = synap_search_tool(sdk, user_id="u1")
        result = await tool._func("programming languages")
        assert result == "User is a Python developer."

    @pytest.mark.asyncio
    async def test_returns_trimmed_context(self):
        sdk = _search_sdk("  trimmed context  ")
        tool = synap_search_tool(sdk, user_id="u1")
        result = await tool._func("query")
        assert result == "trimmed context"

    @pytest.mark.asyncio
    async def test_sdk_called_with_user_id(self):
        sdk = _search_sdk("ctx")
        tool = synap_search_tool(sdk, user_id="user-abc")
        await tool._func("query")
        kw = sdk.fetch.call_args.kwargs
        assert kw["user_id"] == "user-abc"

    @pytest.mark.asyncio
    async def test_sdk_called_with_customer_id(self):
        sdk = _search_sdk("ctx")
        tool = synap_search_tool(sdk, user_id="u1", customer_id="cust-99")
        await tool._func("query")
        kw = sdk.fetch.call_args.kwargs
        assert kw["customer_id"] == "cust-99"

    @pytest.mark.asyncio
    async def test_empty_customer_id_converted_to_none(self):
        sdk = _search_sdk("ctx")
        tool = synap_search_tool(sdk, user_id="u1", customer_id="")
        await tool._func("query")
        kw = sdk.fetch.call_args.kwargs
        assert kw["customer_id"] is None

    @pytest.mark.asyncio
    async def test_sdk_called_with_query_as_list(self):
        sdk = _search_sdk("ctx")
        tool = synap_search_tool(sdk, user_id="u1")
        await tool._func("coffee preferences")
        kw = sdk.fetch.call_args.kwargs
        assert kw["search_query"] == ["coffee preferences"]

    @pytest.mark.asyncio
    async def test_empty_query_passed_as_none(self):
        sdk = _search_sdk("ctx")
        tool = synap_search_tool(sdk, user_id="u1")
        await tool._func("")
        kw = sdk.fetch.call_args.kwargs
        assert kw["search_query"] is None

    @pytest.mark.asyncio
    async def test_sdk_called_with_mode(self):
        sdk = _search_sdk("ctx")
        tool = synap_search_tool(sdk, user_id="u1", mode="fast")
        await tool._func("q")
        kw = sdk.fetch.call_args.kwargs
        assert kw["mode"] == "fast"

    @pytest.mark.asyncio
    async def test_default_mode_is_accurate(self):
        sdk = _search_sdk("ctx")
        tool = synap_search_tool(sdk, user_id="u1")
        await tool._func("q")
        kw = sdk.fetch.call_args.kwargs
        assert kw["mode"] == "accurate"

    @pytest.mark.asyncio
    async def test_sdk_called_with_max_results(self):
        sdk = _search_sdk("ctx")
        tool = synap_search_tool(sdk, user_id="u1", max_results=5)
        await tool._func("q")
        kw = sdk.fetch.call_args.kwargs
        assert kw["max_results"] == 5

    @pytest.mark.asyncio
    async def test_default_max_results_is_10(self):
        sdk = _search_sdk("ctx")
        tool = synap_search_tool(sdk, user_id="u1")
        await tool._func("q")
        kw = sdk.fetch.call_args.kwargs
        assert kw["max_results"] == 10

    @pytest.mark.asyncio
    async def test_none_formatted_context_returns_no_results_placeholder(self):
        sdk = _search_sdk(formatted_context=None)
        tool = synap_search_tool(sdk, user_id="u1")
        result = await tool._func("query")
        assert result == "No relevant long-term memory found for this query."

    @pytest.mark.asyncio
    async def test_empty_formatted_context_returns_no_results_placeholder(self):
        sdk = _search_sdk(formatted_context="")
        tool = synap_search_tool(sdk, user_id="u1")
        result = await tool._func("query")
        assert result == "No relevant long-term memory found for this query."

    @pytest.mark.asyncio
    async def test_whitespace_formatted_context_returns_no_results_placeholder(self):
        sdk = _search_sdk(formatted_context="   ")
        tool = synap_search_tool(sdk, user_id="u1")
        result = await tool._func("query")
        assert result == "No relevant long-term memory found for this query."


# ---------------------------------------------------------------------------
# synap_search_tool — failure paths (graceful degradation)
# ---------------------------------------------------------------------------


class TestSearchToolFailurePath:
    @pytest.mark.asyncio
    async def test_sdk_failure_returns_unavailable_placeholder(self):
        """On any SDK exception, search tool returns the unavailable placeholder."""
        sdk = MagicMock()
        sdk.fetch = AsyncMock(side_effect=RuntimeError("sdk boom"))
        tool = synap_search_tool(sdk, user_id="u1")
        result = await tool._func("anything")
        assert result == "Synap memory is temporarily unavailable."

    @pytest.mark.asyncio
    async def test_sdk_failure_does_not_raise(self):
        sdk = MagicMock()
        sdk.fetch = AsyncMock(side_effect=RuntimeError("network error"))
        tool = synap_search_tool(sdk, user_id="u1")
        # Must not raise
        await tool._func("query")

    @pytest.mark.asyncio
    async def test_sdk_failure_logs_error(self, caplog):
        sdk = MagicMock()
        sdk.fetch = AsyncMock(side_effect=RuntimeError("timeout"))
        tool = synap_search_tool(sdk, user_id="u1")
        with caplog.at_level(logging.ERROR, logger="synap_livekit_agents.tools"):
            await tool._func("query")
        assert any("u1" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_failing_sdk_fixture_returns_placeholder(self, failing_sdk):
        """Shared failing_sdk: sdk.fetch raises — should return placeholder string."""
        # failing_sdk has no .memories but search only uses .fetch
        tool = synap_search_tool(failing_sdk, user_id="u1")
        result = await tool._func("memory query")
        assert result == "Synap memory is temporarily unavailable."


# ---------------------------------------------------------------------------
# synap_store_tool — factory validation
# ---------------------------------------------------------------------------


class TestStoreToolValidation:
    def test_requires_non_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            synap_store_tool(None, user_id="u1")  # type: ignore[arg-type]

    def test_requires_non_empty_user_id(self):
        with pytest.raises(ValueError, match="non-empty user_id"):
            synap_store_tool(MagicMock(), user_id="")


# ---------------------------------------------------------------------------
# synap_store_tool — factory return type
# ---------------------------------------------------------------------------


class TestStoreToolType:
    def test_returns_function_tool_instance(self):
        tool = synap_store_tool(_store_sdk(), user_id="u1")
        assert isinstance(tool, FunctionTool)

    def test_tool_info_name_is_synap_store(self):
        tool = synap_store_tool(_store_sdk(), user_id="u1")
        assert tool.info.name == "synap_store"

    def test_tool_info_description_mentions_memory(self):
        tool = synap_store_tool(_store_sdk(), user_id="u1")
        assert "memory" in tool.info.description.lower()

    def test_tool_is_callable(self):
        tool = synap_store_tool(_store_sdk(), user_id="u1")
        assert callable(tool._func)


# ---------------------------------------------------------------------------
# synap_store_tool — happy paths
# ---------------------------------------------------------------------------


class TestStoreToolHappyPath:
    @pytest.mark.asyncio
    async def test_returns_string_with_ingestion_id(self):
        sdk = _store_sdk("ing-123")
        tool = synap_store_tool(sdk, user_id="u1")
        result = await tool._func("User prefers dark mode")
        assert "ing-123" in result

    @pytest.mark.asyncio
    async def test_result_contains_stored_memory_text(self):
        sdk = _store_sdk("ing-xyz")
        tool = synap_store_tool(sdk, user_id="u1")
        result = await tool._func("some fact")
        assert isinstance(result, str)
        assert "Stored memory" in result

    @pytest.mark.asyncio
    async def test_sdk_create_called_with_content(self):
        sdk = _store_sdk()
        tool = synap_store_tool(sdk, user_id="u1")
        await tool._func("User likes hiking")
        kw = sdk.memories.create.call_args.kwargs
        assert kw["document"] == "User likes hiking"

    @pytest.mark.asyncio
    async def test_sdk_create_called_with_user_id(self):
        sdk = _store_sdk()
        tool = synap_store_tool(sdk, user_id="user-xyz")
        await tool._func("fact")
        kw = sdk.memories.create.call_args.kwargs
        assert kw["user_id"] == "user-xyz"

    @pytest.mark.asyncio
    async def test_sdk_create_called_with_customer_id(self):
        sdk = _store_sdk()
        tool = synap_store_tool(sdk, user_id="u1", customer_id="cust-55")
        await tool._func("fact")
        kw = sdk.memories.create.call_args.kwargs
        assert kw["customer_id"] == "cust-55"

    @pytest.mark.asyncio
    async def test_sdk_create_called_with_document_type(self):
        sdk = _store_sdk()
        tool = synap_store_tool(sdk, user_id="u1", document_type="voice-note")
        await tool._func("fact")
        kw = sdk.memories.create.call_args.kwargs
        assert kw["document_type"] == "voice-note"

    @pytest.mark.asyncio
    async def test_default_document_type_is_ai_chat_conversation(self):
        sdk = _store_sdk()
        tool = synap_store_tool(sdk, user_id="u1")
        await tool._func("fact")
        kw = sdk.memories.create.call_args.kwargs
        assert kw["document_type"] == "ai-chat-conversation"

    @pytest.mark.asyncio
    async def test_category_forwarded_in_metadata(self):
        sdk = _store_sdk()
        tool = synap_store_tool(sdk, user_id="u1")
        await tool._func("User prefers dark mode", category="preference")
        kw = sdk.memories.create.call_args.kwargs
        assert kw["metadata"]["category"] == "preference"

    @pytest.mark.asyncio
    async def test_default_category_is_fact(self):
        sdk = _store_sdk()
        tool = synap_store_tool(sdk, user_id="u1")
        await tool._func("User is tall")
        kw = sdk.memories.create.call_args.kwargs
        assert kw["metadata"]["category"] == "fact"

    @pytest.mark.asyncio
    async def test_none_category_falls_back_to_fact(self):
        sdk = _store_sdk()
        tool = synap_store_tool(sdk, user_id="u1")
        await tool._func("User is tall", category=None)
        kw = sdk.memories.create.call_args.kwargs
        assert kw["metadata"]["category"] == "fact"

    @pytest.mark.asyncio
    async def test_missing_ingestion_id_fallback_to_unknown(self):
        """When result has no ingestion_id attribute, 'unknown' is used."""
        sdk = MagicMock()
        result = MagicMock(spec=[])  # no ingestion_id attr
        sdk.memories = MagicMock()
        sdk.memories.create = AsyncMock(return_value=result)
        tool = synap_store_tool(sdk, user_id="u1")
        result_str = await tool._func("some fact")
        assert "unknown" in result_str

    @pytest.mark.asyncio
    async def test_mock_sdk_fixture_works(self, mock_sdk):
        """Shared mock_sdk fixture: memories.create returns a result with ingestion_id."""
        tool = synap_store_tool(mock_sdk, user_id="u1")
        result = await tool._func("fact via fixture")
        assert "ing-001" in result


# ---------------------------------------------------------------------------
# synap_store_tool — failure paths (SynapIntegrationError raised)
# ---------------------------------------------------------------------------


class TestStoreToolFailurePath:
    @pytest.mark.asyncio
    async def test_sdk_failure_raises_synap_integration_error(self):
        sdk = MagicMock()
        sdk.memories = MagicMock()
        sdk.memories.create = AsyncMock(side_effect=RuntimeError("sdk boom"))
        tool = synap_store_tool(sdk, user_id="u1")
        with pytest.raises(SynapIntegrationError):
            await tool._func("fact to store")

    @pytest.mark.asyncio
    async def test_sdk_failure_preserves_original_cause(self):
        original = RuntimeError("original sdk error")
        sdk = MagicMock()
        sdk.memories = MagicMock()
        sdk.memories.create = AsyncMock(side_effect=original)
        tool = synap_store_tool(sdk, user_id="u1")
        with pytest.raises(SynapIntegrationError) as exc_info:
            await tool._func("fact")
        assert exc_info.value.__cause__ is original

    @pytest.mark.asyncio
    async def test_failing_sdk_fixture_raises_synap_integration_error(self, failing_sdk):
        """Shared failing_sdk: sdk.memories.create raises → SynapIntegrationError."""
        tool = synap_store_tool(failing_sdk, user_id="u1")
        with pytest.raises(SynapIntegrationError):
            await tool._func("content to store")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_surface_exports():
    import synap_livekit_agents
    assert hasattr(synap_livekit_agents, "synap_search_tool")
    assert hasattr(synap_livekit_agents, "synap_store_tool")
    assert "synap_search_tool" in synap_livekit_agents.__all__
    assert "synap_store_tool" in synap_livekit_agents.__all__
