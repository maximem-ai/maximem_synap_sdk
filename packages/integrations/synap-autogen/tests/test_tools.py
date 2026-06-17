"""Bedrock-quality tests for synap_autogen.tools.

Covers:
- SynapSearchTool: construction validation, happy path, failure path,
  cancellation (pre-cancelled token, mid-flight), kwargs forwarding.
- SynapStoreTool: construction validation, happy path, failure path,
  cancellation, kwargs forwarding.
- _await_with_cancellation: unit-tested directly.
- Public surface / module exports.

Documented contracts asserted:
- Both tools wrap SDK errors as SynapIntegrationError (never raw RuntimeError).
- Pre-cancelled CancellationToken raises asyncio.CancelledError immediately
  without calling the SDK.
- tool.name / tool.description are set to canonical values.
- Empty string customer_id is converted to None for SDK.fetch (search);
  stored as-is for memories.create (store).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from autogen_core import CancellationToken
from autogen_core.tools import BaseTool

from synap_autogen.tools import (
    SearchMemoryArgs,
    SearchMemoryResult,
    StoreMemoryArgs,
    StoreMemoryResult,
    SynapSearchTool,
    SynapStoreTool,
    _await_with_cancellation,
)
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sdk_for_search(formatted_context: str | None = "User likes coffee"):
    sdk = MagicMock()
    sdk.fetch = AsyncMock(return_value=MagicMock(formatted_context=formatted_context))
    return sdk


def _make_sdk_for_store(ingestion_id: str = "ing-001"):
    sdk = MagicMock()
    sdk.memories = MagicMock()
    sdk.memories.create = AsyncMock(return_value=MagicMock(ingestion_id=ingestion_id))
    return sdk


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_package_exports_search_tool(self):
        import synap_autogen
        assert hasattr(synap_autogen, "SynapSearchTool")
        assert "SynapSearchTool" in synap_autogen.__all__

    def test_package_exports_store_tool(self):
        import synap_autogen
        assert hasattr(synap_autogen, "SynapStoreTool")
        assert "SynapStoreTool" in synap_autogen.__all__

    def test_search_tool_is_base_tool_subclass(self):
        assert issubclass(SynapSearchTool, BaseTool)

    def test_store_tool_is_base_tool_subclass(self):
        assert issubclass(SynapStoreTool, BaseTool)

    def test_search_tool_canonical_name(self):
        tool = SynapSearchTool(sdk=_make_sdk_for_search(), user_id="u1")
        assert tool.name == "search_memory"

    def test_store_tool_canonical_name(self):
        tool = SynapStoreTool(sdk=_make_sdk_for_store(), user_id="u1")
        assert tool.name == "store_memory"

    def test_search_tool_has_description(self):
        tool = SynapSearchTool(sdk=_make_sdk_for_search(), user_id="u1")
        assert tool.description and len(tool.description) > 10

    def test_store_tool_has_description(self):
        tool = SynapStoreTool(sdk=_make_sdk_for_store(), user_id="u1")
        assert tool.description and len(tool.description) > 10


# ---------------------------------------------------------------------------
# SynapSearchTool — construction validation
# ---------------------------------------------------------------------------


class TestSynapSearchToolValidation:
    def test_requires_non_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapSearchTool(sdk=None, user_id="u1")  # type: ignore[arg-type]

    def test_requires_non_empty_user_id(self):
        with pytest.raises(ValueError, match="non-empty user_id"):
            SynapSearchTool(sdk=_make_sdk_for_search(), user_id="")

    def test_accepts_optional_params_with_defaults(self):
        """No error when customer_id/conversation_id/mode are omitted."""
        tool = SynapSearchTool(sdk=_make_sdk_for_search(), user_id="u1")
        assert tool is not None

    def test_accepts_explicit_optional_params(self):
        tool = SynapSearchTool(
            sdk=_make_sdk_for_search(),
            user_id="u1",
            customer_id="c1",
            conversation_id="conv_xyz",
            mode="fast",
        )
        assert tool is not None


# ---------------------------------------------------------------------------
# SynapSearchTool — happy paths
# ---------------------------------------------------------------------------


class TestSynapSearchToolHappyPath:
    @pytest.mark.asyncio
    async def test_returns_formatted_context(self, ct):
        sdk = _make_sdk_for_search("User likes coffee")
        tool = SynapSearchTool(sdk=sdk, user_id="u1")
        result = await tool.run(SearchMemoryArgs(query="coffee"), ct)
        assert isinstance(result, SearchMemoryResult)
        assert result.context == "User likes coffee"

    @pytest.mark.asyncio
    async def test_none_formatted_context_returns_sentinel(self, ct):
        sdk = _make_sdk_for_search(None)
        tool = SynapSearchTool(sdk=sdk, user_id="u1")
        result = await tool.run(SearchMemoryArgs(query="anything"), ct)
        assert result.context == "No relevant memories found."

    @pytest.mark.asyncio
    async def test_empty_string_formatted_context_returns_sentinel(self, ct):
        """Empty string is falsy — SDK returned nothing useful."""
        sdk = MagicMock()
        sdk.fetch = AsyncMock(return_value=MagicMock(formatted_context=""))
        tool = SynapSearchTool(sdk=sdk, user_id="u1")
        result = await tool.run(SearchMemoryArgs(query="anything"), ct)
        assert result.context == "No relevant memories found."

    @pytest.mark.asyncio
    async def test_forwards_query_as_list(self, ct):
        sdk = _make_sdk_for_search("ctx")
        tool = SynapSearchTool(sdk=sdk, user_id="u1")
        await tool.run(SearchMemoryArgs(query="dark mode"), ct)
        kwargs = sdk.fetch.call_args.kwargs
        assert kwargs["search_query"] == ["dark mode"]

    @pytest.mark.asyncio
    async def test_forwards_user_id(self, ct):
        sdk = _make_sdk_for_search("ctx")
        tool = SynapSearchTool(sdk=sdk, user_id="user_42")
        await tool.run(SearchMemoryArgs(query="q"), ct)
        assert sdk.fetch.call_args.kwargs["user_id"] == "user_42"

    @pytest.mark.asyncio
    async def test_forwards_customer_id(self, ct):
        sdk = _make_sdk_for_search("ctx")
        tool = SynapSearchTool(sdk=sdk, user_id="u1", customer_id="cust_99")
        await tool.run(SearchMemoryArgs(query="q"), ct)
        assert sdk.fetch.call_args.kwargs["customer_id"] == "cust_99"

    @pytest.mark.asyncio
    async def test_empty_customer_id_becomes_none_in_fetch(self, ct):
        """Empty string customer_id is normalized to None for the SDK."""
        sdk = _make_sdk_for_search("ctx")
        tool = SynapSearchTool(sdk=sdk, user_id="u1", customer_id="")
        await tool.run(SearchMemoryArgs(query="q"), ct)
        assert sdk.fetch.call_args.kwargs["customer_id"] is None

    @pytest.mark.asyncio
    async def test_forwards_conversation_id(self, ct):
        sdk = _make_sdk_for_search("ctx")
        tool = SynapSearchTool(sdk=sdk, user_id="u1", conversation_id="conv_xyz")
        await tool.run(SearchMemoryArgs(query="q"), ct)
        assert sdk.fetch.call_args.kwargs["conversation_id"] == "conv_xyz"

    @pytest.mark.asyncio
    async def test_forwards_mode(self, ct):
        sdk = _make_sdk_for_search("ctx")
        tool = SynapSearchTool(sdk=sdk, user_id="u1", mode="fast")
        await tool.run(SearchMemoryArgs(query="q"), ct)
        assert sdk.fetch.call_args.kwargs["mode"] == "fast"

    @pytest.mark.asyncio
    async def test_include_conversation_context_is_false(self, ct):
        """include_conversation_context must always be False."""
        sdk = _make_sdk_for_search("ctx")
        tool = SynapSearchTool(sdk=sdk, user_id="u1")
        await tool.run(SearchMemoryArgs(query="q"), ct)
        assert sdk.fetch.call_args.kwargs["include_conversation_context"] is False

    @pytest.mark.asyncio
    async def test_sdk_called_exactly_once(self, ct):
        sdk = _make_sdk_for_search("ctx")
        tool = SynapSearchTool(sdk=sdk, user_id="u1")
        await tool.run(SearchMemoryArgs(query="q"), ct)
        sdk.fetch.assert_awaited_once()


# ---------------------------------------------------------------------------
# SynapSearchTool — failure path
# ---------------------------------------------------------------------------


class TestSynapSearchToolFailurePath:
    @pytest.mark.asyncio
    async def test_sdk_runtime_error_raises_synap_integration_error(self, ct):
        """SDK RuntimeError must be wrapped as SynapIntegrationError (not leak raw)."""
        sdk = MagicMock()
        sdk.fetch = AsyncMock(side_effect=RuntimeError("sdk boom"))
        tool = SynapSearchTool(sdk=sdk, user_id="u1")
        with pytest.raises(SynapIntegrationError):
            await tool.run(SearchMemoryArgs(query="q"), ct)

    @pytest.mark.asyncio
    async def test_sdk_exception_does_not_leak_raw_runtime_error(self, ct):
        """No bare RuntimeError should propagate — always SynapIntegrationError."""
        sdk = MagicMock()
        sdk.fetch = AsyncMock(side_effect=RuntimeError("raw leak check"))
        tool = SynapSearchTool(sdk=sdk, user_id="u1")
        raised_type = None
        try:
            await tool.run(SearchMemoryArgs(query="q"), ct)
        except Exception as exc:
            raised_type = type(exc)
        assert raised_type is SynapIntegrationError

    @pytest.mark.asyncio
    async def test_failing_sdk_fixture_raises_synap_integration_error(self, failing_sdk, ct):
        """Shared failing_sdk fixture: every SDK call raises."""
        tool = SynapSearchTool(sdk=failing_sdk, user_id="u1")
        with pytest.raises(SynapIntegrationError):
            await tool.run(SearchMemoryArgs(query="q"), ct)


# ---------------------------------------------------------------------------
# SynapSearchTool — cancellation
# ---------------------------------------------------------------------------


class TestSynapSearchToolCancellation:
    @pytest.mark.asyncio
    async def test_pre_cancelled_token_raises_cancelled_error(self, cancelled_ct):
        """Tool must short-circuit with CancelledError when token is already set."""
        sdk = _make_sdk_for_search("ctx")
        tool = SynapSearchTool(sdk=sdk, user_id="u1")
        with pytest.raises(asyncio.CancelledError):
            await tool.run(SearchMemoryArgs(query="q"), cancelled_ct)

    @pytest.mark.asyncio
    async def test_pre_cancelled_token_does_not_call_sdk(self, cancelled_ct):
        """SDK must not be called when the token is pre-cancelled."""
        sdk = _make_sdk_for_search("ctx")
        tool = SynapSearchTool(sdk=sdk, user_id="u1")
        try:
            await tool.run(SearchMemoryArgs(query="q"), cancelled_ct)
        except asyncio.CancelledError:
            pass
        sdk.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_real_cancel_token_not_cancelled_runs_normally(self, ct):
        """A fresh (non-cancelled) real token must NOT raise CancelledError."""
        sdk = _make_sdk_for_search("result")
        tool = SynapSearchTool(sdk=sdk, user_id="u1")
        result = await tool.run(SearchMemoryArgs(query="q"), ct)
        assert result.context == "result"


# ---------------------------------------------------------------------------
# SynapStoreTool — construction validation
# ---------------------------------------------------------------------------


class TestSynapStoreToolValidation:
    def test_requires_non_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapStoreTool(sdk=None, user_id="u1")  # type: ignore[arg-type]

    def test_requires_non_empty_user_id(self):
        with pytest.raises(ValueError, match="non-empty user_id"):
            SynapStoreTool(sdk=_make_sdk_for_store(), user_id="")

    def test_accepts_optional_customer_id_omitted(self):
        tool = SynapStoreTool(sdk=_make_sdk_for_store(), user_id="u1")
        assert tool is not None

    def test_accepts_explicit_customer_id(self):
        tool = SynapStoreTool(sdk=_make_sdk_for_store(), user_id="u1", customer_id="c1")
        assert tool is not None


# ---------------------------------------------------------------------------
# SynapStoreTool — happy paths
# ---------------------------------------------------------------------------


class TestSynapStoreToolHappyPath:
    @pytest.mark.asyncio
    async def test_returns_store_memory_result(self, ct):
        sdk = _make_sdk_for_store("ing-123")
        tool = SynapStoreTool(sdk=sdk, user_id="u1")
        result = await tool.run(StoreMemoryArgs(content="likes dark mode"), ct)
        assert isinstance(result, StoreMemoryResult)

    @pytest.mark.asyncio
    async def test_ingestion_id_matches_sdk_response(self, ct):
        sdk = _make_sdk_for_store("ing-abc")
        tool = SynapStoreTool(sdk=sdk, user_id="u1")
        result = await tool.run(StoreMemoryArgs(content="likes dark mode"), ct)
        assert result.ingestion_id == "ing-abc"

    @pytest.mark.asyncio
    async def test_message_contains_ingestion_id(self, ct):
        sdk = _make_sdk_for_store("ing-999")
        tool = SynapStoreTool(sdk=sdk, user_id="u1")
        result = await tool.run(StoreMemoryArgs(content="x"), ct)
        assert "ing-999" in result.message

    @pytest.mark.asyncio
    async def test_forwards_document_to_sdk(self, ct):
        sdk = _make_sdk_for_store()
        tool = SynapStoreTool(sdk=sdk, user_id="u1")
        await tool.run(StoreMemoryArgs(content="User prefers dark mode"), ct)
        assert sdk.memories.create.call_args.kwargs["document"] == "User prefers dark mode"

    @pytest.mark.asyncio
    async def test_forwards_user_id_to_sdk(self, ct):
        sdk = _make_sdk_for_store()
        tool = SynapStoreTool(sdk=sdk, user_id="user_99")
        await tool.run(StoreMemoryArgs(content="something"), ct)
        assert sdk.memories.create.call_args.kwargs["user_id"] == "user_99"

    @pytest.mark.asyncio
    async def test_forwards_customer_id_to_sdk(self, ct):
        sdk = _make_sdk_for_store()
        tool = SynapStoreTool(sdk=sdk, user_id="u1", customer_id="cust_42")
        await tool.run(StoreMemoryArgs(content="x"), ct)
        assert sdk.memories.create.call_args.kwargs["customer_id"] == "cust_42"

    @pytest.mark.asyncio
    async def test_default_empty_customer_id_passed_to_sdk(self, ct):
        """Default customer_id='' is passed as empty string to memories.create."""
        sdk = _make_sdk_for_store()
        tool = SynapStoreTool(sdk=sdk, user_id="u1")
        await tool.run(StoreMemoryArgs(content="x"), ct)
        assert sdk.memories.create.call_args.kwargs["customer_id"] == ""

    @pytest.mark.asyncio
    async def test_sdk_called_exactly_once(self, ct):
        sdk = _make_sdk_for_store()
        tool = SynapStoreTool(sdk=sdk, user_id="u1")
        await tool.run(StoreMemoryArgs(content="x"), ct)
        sdk.memories.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ingestion_id_is_stringified(self, ct):
        """ingestion_id must be a string even if the SDK returns a non-string."""
        sdk = MagicMock()
        sdk.memories = MagicMock()
        # numeric ingestion_id should be stringified
        sdk.memories.create = AsyncMock(return_value=MagicMock(ingestion_id=12345))
        tool = SynapStoreTool(sdk=sdk, user_id="u1")
        result = await tool.run(StoreMemoryArgs(content="x"), ct)
        assert isinstance(result.ingestion_id, str)
        assert result.ingestion_id == "12345"


# ---------------------------------------------------------------------------
# SynapStoreTool — failure path
# ---------------------------------------------------------------------------


class TestSynapStoreToolFailurePath:
    @pytest.mark.asyncio
    async def test_sdk_runtime_error_raises_synap_integration_error(self, ct):
        """SDK RuntimeError must be wrapped as SynapIntegrationError (not leak raw)."""
        sdk = MagicMock()
        sdk.memories = MagicMock()
        sdk.memories.create = AsyncMock(side_effect=RuntimeError("sdk boom"))
        tool = SynapStoreTool(sdk=sdk, user_id="u1")
        with pytest.raises(SynapIntegrationError):
            await tool.run(StoreMemoryArgs(content="x"), ct)

    @pytest.mark.asyncio
    async def test_sdk_exception_does_not_leak_raw_runtime_error(self, ct):
        sdk = MagicMock()
        sdk.memories = MagicMock()
        sdk.memories.create = AsyncMock(side_effect=RuntimeError("raw leak"))
        tool = SynapStoreTool(sdk=sdk, user_id="u1")
        raised_type = None
        try:
            await tool.run(StoreMemoryArgs(content="x"), ct)
        except Exception as exc:
            raised_type = type(exc)
        assert raised_type is SynapIntegrationError

    @pytest.mark.asyncio
    async def test_failing_sdk_fixture_raises_synap_integration_error(self, failing_sdk, ct):
        """Shared failing_sdk fixture: memories.create raises."""
        tool = SynapStoreTool(sdk=failing_sdk, user_id="u1")
        with pytest.raises(SynapIntegrationError):
            await tool.run(StoreMemoryArgs(content="x"), ct)


# ---------------------------------------------------------------------------
# SynapStoreTool — cancellation
# ---------------------------------------------------------------------------


class TestSynapStoreToolCancellation:
    @pytest.mark.asyncio
    async def test_pre_cancelled_token_raises_cancelled_error(self, cancelled_ct):
        sdk = _make_sdk_for_store()
        tool = SynapStoreTool(sdk=sdk, user_id="u1")
        with pytest.raises(asyncio.CancelledError):
            await tool.run(StoreMemoryArgs(content="x"), cancelled_ct)

    @pytest.mark.asyncio
    async def test_pre_cancelled_token_does_not_call_sdk(self, cancelled_ct):
        sdk = _make_sdk_for_store()
        tool = SynapStoreTool(sdk=sdk, user_id="u1")
        try:
            await tool.run(StoreMemoryArgs(content="x"), cancelled_ct)
        except asyncio.CancelledError:
            pass
        sdk.memories.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_real_cancel_token_not_cancelled_runs_normally(self, ct):
        sdk = _make_sdk_for_store("ing-run")
        tool = SynapStoreTool(sdk=sdk, user_id="u1")
        result = await tool.run(StoreMemoryArgs(content="x"), ct)
        assert result.ingestion_id == "ing-run"


# ---------------------------------------------------------------------------
# _await_with_cancellation — unit tests
# ---------------------------------------------------------------------------


class TestAwaitWithCancellation:
    @pytest.mark.asyncio
    async def test_already_cancelled_raises_immediately(self):
        ct = CancellationToken()
        ct.cancel()

        async def never_run():
            raise AssertionError("should not be called")

        with pytest.raises(asyncio.CancelledError):
            await _await_with_cancellation(never_run(), ct)

    @pytest.mark.asyncio
    async def test_not_cancelled_runs_coroutine(self):
        ct = CancellationToken()

        async def simple_coro():
            return 42

        result = await _await_with_cancellation(simple_coro(), ct)
        assert result == 42

    @pytest.mark.asyncio
    async def test_coroutine_exception_propagates(self):
        ct = CancellationToken()

        async def boom():
            raise ValueError("boom from coro")

        with pytest.raises(ValueError, match="boom from coro"):
            await _await_with_cancellation(boom(), ct)
