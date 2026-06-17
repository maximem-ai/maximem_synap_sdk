"""Tests for SynapDeps and register_synap_tools.

Covers:
- SynapDeps construction (happy paths + validation errors)
- register_synap_tools registration contract (2 tools + 1 system_prompt)
- search_memory: happy path, None context fallback, fetch kwargs contract
- search_memory: failure path (SynapIntegrationError raised, never swallowed)
- store_memory: happy path, ingestion_id in result, create kwargs contract
- store_memory: failure path (SynapIntegrationError raised, never swallowed)
- inject_memory_context: best-effort (returns '' on failure, never crashes)
- inject_memory_context: happy path (returns formatted context)
- inject_memory_context: empty context returns ''
- Public-surface exports from __init__.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_integrations_common import SynapIntegrationError
from synap_pydantic_ai.deps import SynapDeps, register_synap_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeRunContext:
    """Minimal stand-in for pydantic_ai.RunContext. Only .deps is accessed."""
    deps: Any


class _FakeAgent:
    """Captures decorated callables from @agent.tool and @agent.system_prompt."""

    def __init__(self):
        self.tools: List[Callable] = []
        self.system_prompt_callbacks: List[Callable] = []

    def tool(self, fn: Callable) -> Callable:
        self.tools.append(fn)
        return fn

    def system_prompt(self, fn: Callable) -> Callable:
        self.system_prompt_callbacks.append(fn)
        return fn


def _make_sdk(
    *,
    fetch_context: str | None = "User is an engineer",
    ingestion_id: str = "ing-001",
    fetch_error: Exception | None = None,
    create_error: Exception | None = None,
) -> MagicMock:
    sdk = MagicMock()
    resp = MagicMock()
    resp.formatted_context = fetch_context
    sdk.fetch = AsyncMock(
        return_value=resp if fetch_error is None else None,
        side_effect=fetch_error,
    )
    create_resp = MagicMock()
    create_resp.ingestion_id = ingestion_id
    sdk.memories = MagicMock()
    sdk.memories.create = AsyncMock(
        return_value=create_resp if create_error is None else None,
        side_effect=create_error,
    )
    return sdk


def _registered_agent(sdk=None, *, conversation_id="conv-1", user_id="u1", customer_id="c1"):
    """Return (_FakeAgent, search_fn, store_fn, inject_fn, deps)."""
    if sdk is None:
        sdk = _make_sdk()
    agent = _FakeAgent()
    register_synap_tools(agent)
    search_fn = agent.tools[0]
    store_fn = agent.tools[1]
    inject_fn = agent.system_prompt_callbacks[0]
    deps = SynapDeps(
        sdk=sdk,
        user_id=user_id,
        customer_id=customer_id,
        conversation_id=conversation_id,
    )
    return agent, search_fn, store_fn, inject_fn, deps


# ---------------------------------------------------------------------------
# SynapDeps — construction & validation
# ---------------------------------------------------------------------------


class TestSynapDepsConstruction:
    def test_fields_are_stored(self):
        sdk = MagicMock()
        deps = SynapDeps(sdk=sdk, user_id="u1", customer_id="c1", conversation_id="conv-1")
        assert deps.sdk is sdk
        assert deps.user_id == "u1"
        assert deps.customer_id == "c1"
        assert deps.conversation_id == "conv-1"

    def test_customer_id_defaults_to_empty_string(self):
        sdk = MagicMock()
        deps = SynapDeps(sdk=sdk, user_id="u1")
        assert deps.customer_id == ""

    def test_conversation_id_defaults_to_none(self):
        sdk = MagicMock()
        deps = SynapDeps(sdk=sdk, user_id="u1")
        assert deps.conversation_id is None

    def test_raises_on_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapDeps(sdk=None, user_id="u1")  # type: ignore[arg-type]

    def test_raises_on_empty_user_id(self):
        with pytest.raises(ValueError, match="non-empty user_id"):
            SynapDeps(sdk=MagicMock(), user_id="")


# ---------------------------------------------------------------------------
# Public-surface exports
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_all_exports_importable(self):
        import synap_pydantic_ai
        assert hasattr(synap_pydantic_ai, "SynapDeps")
        assert hasattr(synap_pydantic_ai, "register_synap_tools")
        assert hasattr(synap_pydantic_ai, "register_synap_st_system_prompt")

    def test_all_exports_in_dunder_all(self):
        import synap_pydantic_ai
        assert "SynapDeps" in synap_pydantic_ai.__all__
        assert "register_synap_tools" in synap_pydantic_ai.__all__
        assert "register_synap_st_system_prompt" in synap_pydantic_ai.__all__


# ---------------------------------------------------------------------------
# register_synap_tools — registration contract
# ---------------------------------------------------------------------------


class TestRegisterSynapToolsRegistration:
    def test_registers_exactly_two_tools(self):
        agent = _FakeAgent()
        register_synap_tools(agent)
        assert len(agent.tools) == 2

    def test_registers_exactly_one_system_prompt(self):
        agent = _FakeAgent()
        register_synap_tools(agent)
        assert len(agent.system_prompt_callbacks) == 1

    def test_first_tool_is_search_memory(self):
        agent = _FakeAgent()
        register_synap_tools(agent)
        assert agent.tools[0].__name__ == "search_memory"

    def test_second_tool_is_store_memory(self):
        agent = _FakeAgent()
        register_synap_tools(agent)
        assert agent.tools[1].__name__ == "store_memory"

    def test_system_prompt_is_inject_memory_context(self):
        agent = _FakeAgent()
        register_synap_tools(agent)
        assert agent.system_prompt_callbacks[0].__name__ == "inject_memory_context"

    def test_idempotent_registration_on_same_agent_adds_more(self):
        """Calling register twice adds another set — not the integration's job to prevent."""
        agent = _FakeAgent()
        register_synap_tools(agent)
        register_synap_tools(agent)
        assert len(agent.tools) == 4  # 2 x search + 2 x store


# ---------------------------------------------------------------------------
# search_memory — happy paths
# ---------------------------------------------------------------------------


class TestSearchMemoryHappyPath:
    @pytest.mark.asyncio
    async def test_returns_formatted_context(self):
        sdk = _make_sdk(fetch_context="User is a software engineer")
        _, search_fn, _, _, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        result = await search_fn(ctx, "engineering background")
        assert result == "User is a software engineer"

    @pytest.mark.asyncio
    async def test_none_formatted_context_returns_fallback(self):
        sdk = _make_sdk(fetch_context=None)
        _, search_fn, _, _, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        result = await search_fn(ctx, "unknown")
        assert result == "No relevant memories found."

    @pytest.mark.asyncio
    async def test_forwards_query_as_search_query_list(self):
        sdk = _make_sdk()
        _, search_fn, _, _, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        await search_fn(ctx, "coffee preferences")

        sdk.fetch.assert_awaited_once()
        call_kwargs = sdk.fetch.call_args.kwargs
        assert call_kwargs["search_query"] == ["coffee preferences"]

    @pytest.mark.asyncio
    async def test_forwards_user_id(self):
        sdk = _make_sdk()
        _, search_fn, _, _, deps = _registered_agent(sdk, user_id="user-xyz")
        ctx = _FakeRunContext(deps=deps)

        await search_fn(ctx, "query")

        call_kwargs = sdk.fetch.call_args.kwargs
        assert call_kwargs["user_id"] == "user-xyz"

    @pytest.mark.asyncio
    async def test_forwards_customer_id(self):
        sdk = _make_sdk()
        _, search_fn, _, _, deps = _registered_agent(sdk, customer_id="cust-abc")
        ctx = _FakeRunContext(deps=deps)

        await search_fn(ctx, "query")

        call_kwargs = sdk.fetch.call_args.kwargs
        assert call_kwargs["customer_id"] == "cust-abc"

    @pytest.mark.asyncio
    async def test_empty_customer_id_passed_as_none(self):
        """Empty customer_id (default) is normalised to None when calling sdk.fetch."""
        sdk = _make_sdk()
        _, search_fn, _, _, deps = _registered_agent(sdk, customer_id="")
        ctx = _FakeRunContext(deps=deps)

        await search_fn(ctx, "query")

        call_kwargs = sdk.fetch.call_args.kwargs
        assert call_kwargs["customer_id"] is None

    @pytest.mark.asyncio
    async def test_forwards_conversation_id(self):
        sdk = _make_sdk()
        _, search_fn, _, _, deps = _registered_agent(sdk, conversation_id="conv-99")
        ctx = _FakeRunContext(deps=deps)

        await search_fn(ctx, "query")

        call_kwargs = sdk.fetch.call_args.kwargs
        assert call_kwargs["conversation_id"] == "conv-99"

    @pytest.mark.asyncio
    async def test_passes_mode_accurate(self):
        sdk = _make_sdk()
        _, search_fn, _, _, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        await search_fn(ctx, "q")

        call_kwargs = sdk.fetch.call_args.kwargs
        assert call_kwargs["mode"] == "accurate"

    @pytest.mark.asyncio
    async def test_passes_include_conversation_context_false(self):
        sdk = _make_sdk()
        _, search_fn, _, _, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        await search_fn(ctx, "q")

        call_kwargs = sdk.fetch.call_args.kwargs
        assert call_kwargs["include_conversation_context"] is False


# ---------------------------------------------------------------------------
# search_memory — failure path
# ---------------------------------------------------------------------------


class TestSearchMemoryFailurePath:
    @pytest.mark.asyncio
    async def test_raises_synap_integration_error_on_sdk_failure(self):
        sdk = _make_sdk(fetch_error=RuntimeError("sdk boom"))
        _, search_fn, _, _, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        with pytest.raises(SynapIntegrationError):
            await search_fn(ctx, "query")

    @pytest.mark.asyncio
    async def test_error_preserves_original_cause(self):
        original = RuntimeError("original sdk error")
        sdk = _make_sdk(fetch_error=original)
        _, search_fn, _, _, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        with pytest.raises(SynapIntegrationError) as exc_info:
            await search_fn(ctx, "query")

        assert exc_info.value.__cause__ is original

    @pytest.mark.asyncio
    async def test_operation_name_in_error(self):
        sdk = _make_sdk(fetch_error=RuntimeError("boom"))
        _, search_fn, _, _, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        with pytest.raises(SynapIntegrationError) as exc_info:
            await search_fn(ctx, "query")

        assert "pydantic_ai.search_memory" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_sdk_integration_error_not_double_wrapped(self):
        """A SynapIntegrationError from the SDK must propagate as-is, not re-wrapped."""
        inner = SynapIntegrationError("upstream.op", "already wrapped")
        sdk = _make_sdk(fetch_error=inner)
        _, search_fn, _, _, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        with pytest.raises(SynapIntegrationError) as exc_info:
            await search_fn(ctx, "query")

        assert exc_info.value is inner


# ---------------------------------------------------------------------------
# store_memory — happy paths
# ---------------------------------------------------------------------------


class TestStoreMemoryHappyPath:
    @pytest.mark.asyncio
    async def test_returns_ingestion_id_in_result(self):
        sdk = _make_sdk(ingestion_id="ing-777")
        _, _, store_fn, _, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        result = await store_fn(ctx, "User prefers dark mode")

        assert "ing-777" in result

    @pytest.mark.asyncio
    async def test_result_contains_memory_stored_prefix(self):
        sdk = _make_sdk(ingestion_id="ing-abc")
        _, _, store_fn, _, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        result = await store_fn(ctx, "Some content")
        assert "Memory stored" in result

    @pytest.mark.asyncio
    async def test_forwards_document_to_create(self):
        sdk = _make_sdk()
        _, _, store_fn, _, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        await store_fn(ctx, "User prefers vim over emacs")

        sdk.memories.create.assert_awaited_once()
        call_kwargs = sdk.memories.create.call_args.kwargs
        assert call_kwargs["document"] == "User prefers vim over emacs"

    @pytest.mark.asyncio
    async def test_forwards_user_id_to_create(self):
        sdk = _make_sdk()
        _, _, store_fn, _, deps = _registered_agent(sdk, user_id="user-store")
        ctx = _FakeRunContext(deps=deps)

        await store_fn(ctx, "content")

        call_kwargs = sdk.memories.create.call_args.kwargs
        assert call_kwargs["user_id"] == "user-store"

    @pytest.mark.asyncio
    async def test_forwards_customer_id_to_create(self):
        sdk = _make_sdk()
        _, _, store_fn, _, deps = _registered_agent(sdk, customer_id="cust-store")
        ctx = _FakeRunContext(deps=deps)

        await store_fn(ctx, "content")

        call_kwargs = sdk.memories.create.call_args.kwargs
        assert call_kwargs["customer_id"] == "cust-store"

    @pytest.mark.asyncio
    async def test_empty_customer_id_forwarded_as_empty_string(self):
        """Store tool forwards empty customer_id as-is (not coerced to None)."""
        sdk = _make_sdk()
        _, _, store_fn, _, deps = _registered_agent(sdk, customer_id="")
        ctx = _FakeRunContext(deps=deps)

        await store_fn(ctx, "content")

        call_kwargs = sdk.memories.create.call_args.kwargs
        assert call_kwargs["customer_id"] == ""


# ---------------------------------------------------------------------------
# store_memory — failure path
# ---------------------------------------------------------------------------


class TestStoreMemoryFailurePath:
    @pytest.mark.asyncio
    async def test_raises_synap_integration_error_on_sdk_failure(self):
        sdk = _make_sdk(create_error=RuntimeError("sdk create boom"))
        _, _, store_fn, _, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        with pytest.raises(SynapIntegrationError):
            await store_fn(ctx, "content")

    @pytest.mark.asyncio
    async def test_error_preserves_original_cause(self):
        original = RuntimeError("original create error")
        sdk = _make_sdk(create_error=original)
        _, _, store_fn, _, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        with pytest.raises(SynapIntegrationError) as exc_info:
            await store_fn(ctx, "content")

        assert exc_info.value.__cause__ is original

    @pytest.mark.asyncio
    async def test_operation_name_in_error(self):
        sdk = _make_sdk(create_error=RuntimeError("boom"))
        _, _, store_fn, _, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        with pytest.raises(SynapIntegrationError) as exc_info:
            await store_fn(ctx, "content")

        assert "pydantic_ai.store_memory" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_sdk_integration_error_not_double_wrapped(self):
        """A SynapIntegrationError from the SDK must propagate as-is."""
        inner = SynapIntegrationError("upstream.op", "already wrapped")
        sdk = _make_sdk(create_error=inner)
        _, _, store_fn, _, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        with pytest.raises(SynapIntegrationError) as exc_info:
            await store_fn(ctx, "content")

        assert exc_info.value is inner


# ---------------------------------------------------------------------------
# inject_memory_context — best-effort system_prompt (happy + failure)
# ---------------------------------------------------------------------------


class TestInjectMemoryContextHappyPath:
    @pytest.mark.asyncio
    async def test_returns_formatted_context_prefixed(self):
        sdk = _make_sdk(fetch_context="User likes Python")
        _, _, _, inject_fn, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        result = await inject_fn(ctx)

        assert "User likes Python" in result
        assert "Relevant user context:" in result

    @pytest.mark.asyncio
    async def test_empty_formatted_context_returns_empty_string(self):
        sdk = _make_sdk(fetch_context=None)
        _, _, _, inject_fn, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        result = await inject_fn(ctx)

        assert result == ""

    @pytest.mark.asyncio
    async def test_passes_user_id_to_fetch(self):
        sdk = _make_sdk(fetch_context="ctx")
        _, _, _, inject_fn, deps = _registered_agent(sdk, user_id="inject-user")
        ctx = _FakeRunContext(deps=deps)

        await inject_fn(ctx)

        sdk.fetch.assert_awaited_once()
        call_kwargs = sdk.fetch.call_args.kwargs
        assert call_kwargs["user_id"] == "inject-user"

    @pytest.mark.asyncio
    async def test_passes_conversation_id_to_fetch(self):
        sdk = _make_sdk(fetch_context="ctx")
        _, _, _, inject_fn, deps = _registered_agent(sdk, conversation_id="conv-inject")
        ctx = _FakeRunContext(deps=deps)

        await inject_fn(ctx)

        call_kwargs = sdk.fetch.call_args.kwargs
        assert call_kwargs["conversation_id"] == "conv-inject"


class TestInjectMemoryContextFailurePath:
    @pytest.mark.asyncio
    async def test_sdk_failure_returns_empty_string(self):
        """inject_memory_context is best-effort: SDK failures must return '' not crash."""
        sdk = _make_sdk(fetch_error=RuntimeError("boom"))
        _, _, _, inject_fn, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        result = await inject_fn(ctx)

        assert result == ""

    @pytest.mark.asyncio
    async def test_sdk_failure_does_not_raise(self):
        """Best-effort contract: no exception must escape inject_memory_context."""
        sdk = _make_sdk(fetch_error=Exception("catastrophic"))
        _, _, _, inject_fn, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        # Should not raise
        result = await inject_fn(ctx)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_sdk_failure_logged_at_error_level(self, caplog):
        sdk = _make_sdk(fetch_error=RuntimeError("logged error"))
        _, _, _, inject_fn, deps = _registered_agent(sdk)
        ctx = _FakeRunContext(deps=deps)

        with caplog.at_level(logging.ERROR, logger="synap_pydantic_ai.deps"):
            await inject_fn(ctx)

        assert len(caplog.records) >= 1
        assert any(
            deps.user_id in r.message for r in caplog.records
        ), "user_id should appear in the error log"
