"""Tests for create_synap_tools — CAMEL FunctionTools for explicit memory access."""

import pytest

from camel.toolkits import FunctionTool

from synap_camel_ai import create_synap_tools
from synap_integrations_common import SynapIntegrationError


def _by_name(tools):
    return {t.get_openai_tool_schema()["function"]["name"]: t for t in tools}


def test_returns_two_function_tools(mock_sdk):
    tools = create_synap_tools(mock_sdk, "alice")
    assert len(tools) == 2
    assert all(isinstance(t, FunctionTool) for t in tools)
    assert set(_by_name(tools)) == {"search_memory", "store_memory"}


def test_requires_sdk():
    with pytest.raises(ValueError):
        create_synap_tools(None, "alice")


def test_requires_user_id(mock_sdk):
    with pytest.raises(ValueError):
        create_synap_tools(mock_sdk, "")


def test_search_memory_returns_context(mock_sdk):
    search = _by_name(create_synap_tools(mock_sdk, "alice"))["search_memory"]
    out = search.func("what plan am I on")
    assert "User Context" in out
    assert mock_sdk.fetch.call_args.kwargs["search_query"] == ["what plan am I on"]


def test_search_memory_degrades_on_failure(failing_sdk):
    search = _by_name(create_synap_tools(failing_sdk, "alice"))["search_memory"]
    assert search.func("q") == "No relevant memories found."


def test_store_memory_returns_ingestion_id(mock_sdk):
    store = _by_name(create_synap_tools(mock_sdk, "alice"))["store_memory"]
    out = store.func("remember I like tea")
    assert "ing-001" in out
    assert mock_sdk.memories.create.call_args.kwargs["document"] == "remember I like tea"


def test_store_memory_raises_on_failure(failing_sdk):
    store = _by_name(create_synap_tools(failing_sdk, "alice"))["store_memory"]
    with pytest.raises(SynapIntegrationError):
        store.func("boom")


def test_customer_id_flows_through(mock_sdk):
    tools = _by_name(create_synap_tools(mock_sdk, "alice", customer_id="acme"))
    tools["search_memory"].func("q")
    tools["store_memory"].func("c")
    assert mock_sdk.fetch.call_args.kwargs["customer_id"] == "acme"
    assert mock_sdk.memories.create.call_args.kwargs["customer_id"] == "acme"
