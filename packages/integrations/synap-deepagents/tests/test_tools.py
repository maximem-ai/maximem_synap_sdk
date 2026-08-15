"""Tests for :mod:`synap_deepagents.tools`."""

import pytest
from synap_integrations_common import SynapIntegrationError

from synap_deepagents.tools import SynapSearchTool, SynapStoreTool


# ---------------------------------------------------------------------------
# SynapSearchTool
# ---------------------------------------------------------------------------


def test_search_tool_metadata(mock_sdk):
    tool = SynapSearchTool(sdk=mock_sdk, user_id="alice")
    assert tool.name == "search_memory"
    assert "recall" in tool.description.lower()
    assert tool.args_schema is not None


async def test_search_tool_returns_formatted_context(mock_sdk):
    tool = SynapSearchTool(sdk=mock_sdk, user_id="alice")
    result = await tool._arun("what does the user do")
    assert "User is an engineer" in result


async def test_search_tool_passes_query_and_scope(mock_sdk):
    tool = SynapSearchTool(sdk=mock_sdk, user_id="alice", customer_id="acme")
    await tool._arun("preferences")
    kwargs = mock_sdk.fetch.await_args.kwargs
    assert kwargs["search_query"] == ["preferences"]
    assert kwargs["user_id"] == "alice"
    assert kwargs["customer_id"] == "acme"
    assert kwargs["include_conversation_context"] is False


async def test_search_tool_defaults_to_accurate_mode(mock_sdk):
    tool = SynapSearchTool(sdk=mock_sdk, user_id="alice")
    await tool._arun("q")
    assert mock_sdk.fetch.await_args.kwargs["mode"] == "accurate"


async def test_search_tool_honours_max_results(mock_sdk):
    tool = SynapSearchTool(sdk=mock_sdk, user_id="alice", max_results=3)
    await tool._arun("q")
    assert mock_sdk.fetch.await_args.kwargs["max_results"] == 3


async def test_search_tool_empty_result_message(mock_sdk):
    mock_sdk.fetch.return_value.formatted_context = ""
    tool = SynapSearchTool(sdk=mock_sdk, user_id="alice")
    assert await tool._arun("q") == "No relevant memories found."


async def test_search_tool_raises_on_sdk_failure(failing_sdk):
    tool = SynapSearchTool(sdk=failing_sdk, user_id="alice")
    with pytest.raises(SynapIntegrationError):
        await tool._arun("q")


def test_search_tool_sync_invoke(mock_sdk):
    tool = SynapSearchTool(sdk=mock_sdk, user_id="alice")
    assert "engineer" in tool.invoke("what does the user do")


# ---------------------------------------------------------------------------
# SynapStoreTool
# ---------------------------------------------------------------------------


def test_store_tool_metadata(mock_sdk):
    tool = SynapStoreTool(sdk=mock_sdk, user_id="alice")
    assert tool.name == "store_memory"
    assert "store" in tool.description.lower()


async def test_store_tool_creates_memory(mock_sdk):
    tool = SynapStoreTool(sdk=mock_sdk, user_id="alice")
    result = await tool._arun("User prefers TypeScript")
    kwargs = mock_sdk.memories.create.await_args.kwargs
    assert kwargs["document"] == "User prefers TypeScript"
    assert kwargs["user_id"] == "alice"
    assert kwargs["document_type"] == "document"
    assert "ing-001" in result


async def test_store_tool_reports_acceptance_not_completion(mock_sdk):
    """Ingestion is queued, so the message must not claim the write finished."""
    tool = SynapStoreTool(sdk=mock_sdk, user_id="alice")
    result = await tool._arun("fact")
    assert "accepted" in result.lower()


async def test_store_tool_passes_customer_id(mock_sdk):
    tool = SynapStoreTool(sdk=mock_sdk, user_id="alice", customer_id="acme")
    await tool._arun("fact")
    assert mock_sdk.memories.create.await_args.kwargs["customer_id"] == "acme"


async def test_store_tool_custom_document_type(mock_sdk):
    tool = SynapStoreTool(
        sdk=mock_sdk, user_id="alice", document_type="email", ingest_mode="long-range"
    )
    await tool._arun("fact")
    kwargs = mock_sdk.memories.create.await_args.kwargs
    assert kwargs["document_type"] == "email"
    assert kwargs["mode"] == "long-range"


async def test_store_tool_raises_on_sdk_failure(failing_sdk):
    tool = SynapStoreTool(sdk=failing_sdk, user_id="alice")
    with pytest.raises(SynapIntegrationError):
        await tool._arun("fact")


def test_store_tool_sync_invoke(mock_sdk):
    tool = SynapStoreTool(sdk=mock_sdk, user_id="alice")
    assert "ing-001" in tool.invoke("a fact")


# ---------------------------------------------------------------------------
# Both tools are usable as deepagents tools
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [SynapSearchTool, SynapStoreTool])
def test_tools_are_langchain_base_tools(mock_sdk, cls):
    from langchain_core.tools import BaseTool

    assert isinstance(cls(sdk=mock_sdk, user_id="alice"), BaseTool)
