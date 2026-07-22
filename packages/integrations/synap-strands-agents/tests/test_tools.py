"""Tests for synap_strands_agents.tools — create_synap_tools.

Contract:
- search_memory / store_memory are Strands @tool functions, directly awaitable.
- reads (search) and writes (store) both surface SDK failures as
  SynapIntegrationError (never a leaked raw SDK exception).
"""

from __future__ import annotations

import pytest

from synap_integrations_common import SynapIntegrationError
from synap_integrations_common.testing import make_unified_response

from synap_strands_agents.tools import create_synap_tools


# ── construction / validation ───────────────────────────────────────────────


def test_returns_two_tools(mock_sdk):
    tools = create_synap_tools(mock_sdk, user_id="alice")
    assert len(tools) == 2
    names = {t.__name__ for t in tools}
    assert names == {"search_memory", "store_memory"}


def test_requires_sdk():
    with pytest.raises(ValueError):
        create_synap_tools(None, user_id="alice")


def test_requires_user_id(mock_sdk):
    with pytest.raises(ValueError):
        create_synap_tools(mock_sdk, user_id="")


# ── search_memory ────────────────────────────────────────────────────────────


async def test_search_returns_formatted_context(mock_sdk):
    search, _ = create_synap_tools(mock_sdk, user_id="alice", customer_id="acme")
    out = await search("what do you know about me")

    assert "User is an engineer" in out
    mock_sdk.fetch.assert_awaited_once()
    kwargs = mock_sdk.fetch.await_args.kwargs
    assert kwargs["user_id"] == "alice"
    assert kwargs["customer_id"] == "acme"
    assert kwargs["search_query"] == ["what do you know about me"]


async def test_search_no_results_message(mock_sdk):
    mock_sdk.fetch.return_value = make_unified_response(formatted_context="")
    search, _ = create_synap_tools(mock_sdk, user_id="alice")
    out = await search("anything")
    assert out == "No relevant memories found."


async def test_search_raises_on_sdk_failure(failing_sdk):
    search, _ = create_synap_tools(failing_sdk, user_id="alice")
    with pytest.raises(SynapIntegrationError):
        await search("boom")


# ── store_memory ─────────────────────────────────────────────────────────────


async def test_store_returns_confirmation(mock_sdk):
    _, store = create_synap_tools(mock_sdk, user_id="alice", customer_id="acme")
    out = await store("User prefers tea")

    assert "ing-001" in out
    mock_sdk.memories.create.assert_awaited_once()
    kwargs = mock_sdk.memories.create.await_args.kwargs
    assert kwargs["document"] == "User prefers tea"
    assert kwargs["user_id"] == "alice"
    assert kwargs["customer_id"] == "acme"


async def test_store_raises_on_sdk_failure(failing_sdk):
    _, store = create_synap_tools(failing_sdk, user_id="alice")
    with pytest.raises(SynapIntegrationError):
        await store("boom")
