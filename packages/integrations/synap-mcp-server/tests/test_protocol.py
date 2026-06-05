"""MCP protocol-level tests (TC-MCP-*) driven through the FastMCP instance:
tool discovery, tool call round-trip, and the soft/hard error policy."""

import httpx
import pytest
import respx

from synap_mcp_server.server import mcp
from tests.conftest import API_BASE

pytestmark = pytest.mark.asyncio


def _text(result) -> str:
    """Flatten a call_tool result to text.

    FastMCP.call_tool returns a (content_blocks, structured_dict) tuple; it may also
    return just content blocks or a dict depending on version. Handle all shapes."""
    if isinstance(result, tuple):
        content, structured = result
        if isinstance(structured, dict) and "result" in structured:
            return str(structured["result"])
        result = content
    if isinstance(result, dict):
        return str(result.get("result", result))
    parts = []
    for block in result:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)


async def test_tools_list_exposes_three_tools():
    """TC-MCP-02: the three tools are discoverable with descriptions."""
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {"log_exchange", "recall_context", "list_recent_memories"} <= names
    by_name = {t.name: t for t in tools}
    assert "remember" in (by_name["log_exchange"].description or "").lower()
    assert "recall" in (by_name["recall_context"].description or "").lower()


@respx.mock
async def test_call_recall_formats_context(with_token):
    """TC-MCP-03: tools/call happy path; context payload is flattened to text."""
    respx.post(f"{API_BASE}/v1/context/client/fetch").mock(
        return_value=httpx.Response(
            200,
            json={"context": {"facts": [{"content": "favorite color is blue"}]}},
        )
    )
    result = await mcp.call_tool("recall_context", {"query": "favorite color"})
    assert "favorite color is blue" in _text(result)


@respx.mock
async def test_call_log_exchange_returns_ingestion_id(with_token):
    """TC-ADP-01 via protocol: log_exchange returns the ingestion id."""
    respx.post(f"{API_BASE}/api/v1/memories/create").mock(
        return_value=httpx.Response(200, json={"ingestion_id": "ing_42", "status": "completed"})
    )
    result = await mcp.call_tool(
        "log_exchange", {"user_message": "hi", "assistant_message": "hello"}
    )
    assert "ing_42" in _text(result)


@respx.mock
async def test_recall_soft_fails_on_backend_error(with_token):
    """TC-FAIL-03 (read): a backend 500 must not raise — recall degrades gracefully."""
    respx.post(f"{API_BASE}/v1/context/client/fetch").mock(
        return_value=httpx.Response(500, text="boom")
    )
    result = await mcp.call_tool("recall_context", {"query": "anything"})
    assert "No memory available" in _text(result)


@respx.mock
async def test_log_hard_fails_on_backend_error(with_token):
    """TC-FAIL-03 (write): a backend 500 surfaces as an ERROR string to the agent."""
    respx.post(f"{API_BASE}/api/v1/memories/create").mock(
        return_value=httpx.Response(500, text="boom")
    )
    result = await mcp.call_tool("log_exchange", {"user_message": "hi"})
    assert "ERROR" in _text(result)
