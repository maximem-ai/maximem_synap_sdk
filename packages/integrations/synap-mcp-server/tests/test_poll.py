"""Polling-loop tests for _poll_until_terminal and wait_for_processing.

Tests:
  - Poll returns immediately on first terminal response
  - Poll handles multiple non-terminal rounds before terminal
  - Poll returns last-seen payload when cap elapses (no infinite loop)
  - log_exchange wait_for_processing=True with status check failure degrades gracefully
  - check_memory_status with 404 returns benign unknown-id message
  - check_memory_status missing token returns ERROR string
  - Various terminal-status aliases (done, success, partial_success)
  - recall_context with customer_id uses customer scope
  - list_recent_memories happy path (no search_query)
  - list_recent_memories missing token returns ERROR string
  - recall_context with user_id uses user scope
  - log_exchange with conversation_id metadata is forwarded
"""

import httpx
import pytest
import respx

from synap_mcp_server.server import mcp
from tests.conftest import API_BASE

pytestmark = pytest.mark.asyncio


def _text(result) -> str:
    """Flatten a call_tool result to text (same helper as test_protocol.py)."""
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


# ---------------------------------------------------------------------------
# _poll_until_terminal — unit-level tests via patching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_returns_on_first_terminal(monkeypatch):
    """If the first /status call is terminal, the loop exits immediately.

    tools.py imports get_ingestion_status via 'from .client import ... get_ingestion_status',
    so we patch the name inside the tools module (not client).
    """
    from synap_mcp_server import tools
    from unittest.mock import AsyncMock
    from synap_mcp_server.context import set_token

    set_token("synap_testkey")
    try:
        terminal_payload = {"status": "completed", "memories_created": 1}
        sleep_calls: list = []
        import asyncio

        async def noop_sleep(t):
            sleep_calls.append(t)

        monkeypatch.setattr(tools, "get_ingestion_status", AsyncMock(return_value=terminal_payload))
        monkeypatch.setattr(asyncio, "sleep", noop_sleep)

        result = await tools._poll_until_terminal("ing_test")
        assert result == terminal_payload
        assert sleep_calls == [], "sleep should not be called when first status is terminal"
    finally:
        set_token(None)


@pytest.mark.asyncio
async def test_poll_loops_until_terminal(monkeypatch):
    """Loop iterates if the first status is non-terminal."""
    from synap_mcp_server import tools
    from unittest.mock import AsyncMock
    from synap_mcp_server.context import set_token
    import asyncio

    set_token("synap_testkey")
    try:
        responses = [
            {"status": "processing"},
            {"status": "completed", "memories_created": 2},
        ]
        call_count = 0

        async def fake_get_status(ingestion_id):
            nonlocal call_count
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return resp

        monkeypatch.setattr(tools, "get_ingestion_status", fake_get_status)
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        result = await tools._poll_until_terminal("ing_loop")
        assert result["status"] == "completed"
        assert call_count == 2
    finally:
        set_token(None)


@pytest.mark.asyncio
async def test_poll_returns_last_payload_on_cap(monkeypatch):
    """When the cap elapses without a terminal status, returns the last-seen payload."""
    from synap_mcp_server import tools
    from unittest.mock import AsyncMock
    from synap_mcp_server.context import set_token
    import asyncio

    set_token("synap_testkey")
    try:
        stuck_payload = {"status": "processing"}
        monkeypatch.setattr(tools, "get_ingestion_status", AsyncMock(return_value=stuck_payload))

        # Speed up: set POLL_MAX_SECONDS < POLL_INTERVAL_SECONDS so only one iteration runs
        monkeypatch.setattr(tools, "_POLL_MAX_SECONDS", 1.0)
        monkeypatch.setattr(tools, "_POLL_INTERVAL_SECONDS", 2.0)
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        result = await tools._poll_until_terminal("ing_stuck")
        assert result["status"] == "processing"
    finally:
        set_token(None)


@pytest.mark.asyncio
async def test_poll_recognises_done_terminal_status(monkeypatch):
    """'done' is a valid terminal status alias."""
    from synap_mcp_server import tools
    from unittest.mock import AsyncMock
    from synap_mcp_server.context import set_token
    import asyncio

    set_token("synap_testkey")
    try:
        monkeypatch.setattr(tools, "get_ingestion_status", AsyncMock(return_value={"status": "done"}))
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        result = await tools._poll_until_terminal("ing_done")
        assert result["status"] == "done"
    finally:
        set_token(None)


@pytest.mark.asyncio
async def test_poll_recognises_partial_success(monkeypatch):
    from synap_mcp_server import tools
    from unittest.mock import AsyncMock
    from synap_mcp_server.context import set_token
    import asyncio

    set_token("synap_testkey")
    try:
        payload = {"status": "partial_success", "memories_created": 1}
        monkeypatch.setattr(tools, "get_ingestion_status", AsyncMock(return_value=payload))
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        result = await tools._poll_until_terminal("ing_partial")
        assert result["status"] == "partial_success"
    finally:
        set_token(None)


# ---------------------------------------------------------------------------
# log_exchange — extended behavior
# ---------------------------------------------------------------------------


@respx.mock
async def test_log_exchange_wait_for_processing_status_check_fails_gracefully(with_token):
    """If the ingestion succeeds but status check raises SynapAPIError, the tool
    logs a warning and returns the confirmation without crashing."""
    from synap_mcp_server.client import SynapAPIError
    import asyncio

    respx.post(f"{API_BASE}/api/v1/memories/create").mock(
        return_value=httpx.Response(200, json={"ingestion_id": "ing_fail_status"})
    )
    # Status endpoint 503
    respx.get(f"{API_BASE}/api/v1/memories/status/ing_fail_status").mock(
        return_value=httpx.Response(503, text="down")
    )

    result = await mcp.call_tool(
        "log_exchange",
        {"user_message": "hi", "wait_for_processing": True},
    )
    text = _text(result)
    # Must not be an ERROR; ingestion succeeded
    assert "ing_fail_status" in text
    # Either confirmed or says status check failed — no crash
    assert "ERROR" not in text or "Logged" in text


@respx.mock
async def test_log_exchange_conversation_id_in_metadata(with_token):
    """conversation_id is forwarded as metadata in the request body."""
    import json

    route = respx.post(f"{API_BASE}/api/v1/memories/create").mock(
        return_value=httpx.Response(200, json={"ingestion_id": "ing_conv"})
    )
    await mcp.call_tool(
        "log_exchange",
        {"user_message": "hello", "conversation_id": "conv_abc"},
    )
    sent = json.loads(route.calls.last.request.read().decode())
    assert sent.get("metadata", {}).get("conversation_id") == "conv_abc"


@respx.mock
async def test_log_exchange_no_conversation_id_no_metadata(with_token):
    """When no conversation_id is given, metadata does not contain 'conversation_id'."""
    import json

    route = respx.post(f"{API_BASE}/api/v1/memories/create").mock(
        return_value=httpx.Response(200, json={"ingestion_id": "ing_no_conv"})
    )
    await mcp.call_tool("log_exchange", {"user_message": "hello"})
    sent = json.loads(route.calls.last.request.read().decode())
    # metadata always includes 'source', but NOT 'conversation_id'
    assert "conversation_id" not in sent.get("metadata", {})


async def test_log_exchange_missing_token_returns_error_string(no_token):
    """log_exchange without a Bearer token returns an ERROR string (no unhandled exception)."""
    result = await mcp.call_tool("log_exchange", {"user_message": "hi"})
    text = _text(result)
    assert "ERROR" in text


# ---------------------------------------------------------------------------
# check_memory_status — extended behavior
# ---------------------------------------------------------------------------


@respx.mock
async def test_check_memory_status_404_returns_unknown_message(with_token):
    """404 on the status endpoint maps to a benign 'unknown ingestion_id' message."""
    respx.get(f"{API_BASE}/api/v1/memories/status/ing_gone").mock(
        return_value=httpx.Response(404, text="not found")
    )
    result = await mcp.call_tool("check_memory_status", {"ingestion_id": "ing_gone"})
    text = _text(result)
    assert "unknown" in text.lower() or "expired" in text.lower()
    assert "ERROR" not in text


@respx.mock
async def test_check_memory_status_failed_status_summarized(with_token):
    """check_memory_status for a failed ingestion contains 'fail' in the summary."""
    respx.get(f"{API_BASE}/api/v1/memories/status/ing_fail").mock(
        return_value=httpx.Response(
            200, json={"status": "failed", "error_message": "parse error"}
        )
    )
    result = await mcp.call_tool("check_memory_status", {"ingestion_id": "ing_fail"})
    text = _text(result)
    assert "fail" in text.lower()


@respx.mock
async def test_check_memory_status_500_is_error(with_token):
    """A 5xx on the status endpoint surfaces as an ERROR string."""
    respx.get(f"{API_BASE}/api/v1/memories/status/ing_500").mock(
        return_value=httpx.Response(500, text="internal error")
    )
    result = await mcp.call_tool("check_memory_status", {"ingestion_id": "ing_500"})
    text = _text(result)
    assert "ERROR" in text


async def test_check_memory_status_missing_token_returns_error(no_token):
    """check_memory_status without a token returns an ERROR string."""
    result = await mcp.call_tool("check_memory_status", {"ingestion_id": "ing_x"})
    text = _text(result)
    assert "ERROR" in text


# ---------------------------------------------------------------------------
# recall_context — extended behavior
# ---------------------------------------------------------------------------


@respx.mock
async def test_recall_context_customer_scope_routes_to_customer_fetch(with_token):
    """customer_id routes to /v1/context/customer/fetch."""
    import json

    route = respx.post(f"{API_BASE}/v1/context/customer/fetch").mock(
        return_value=httpx.Response(200, json={"context": {}})
    )
    await mcp.call_tool(
        "recall_context", {"query": "prefs", "customer_id": "c1"}
    )
    assert route.called
    sent = json.loads(route.calls.last.request.read().decode())
    assert sent.get("customer_id") == "c1"


@respx.mock
async def test_recall_context_user_scope_scope_label_in_empty_result(with_token):
    """When context is empty, the fallback message mentions the scope."""
    respx.post(f"{API_BASE}/v1/context/user/fetch").mock(
        return_value=httpx.Response(200, json={"context": {}})
    )
    result = await mcp.call_tool(
        "recall_context", {"query": "anything", "user_id": "u2"}
    )
    text = _text(result)
    # Should say "Nothing remembered yet for this user" (or similar)
    assert "user" in text.lower()


@respx.mock
async def test_recall_context_empty_context_returns_nothing_yet_message(with_token):
    """Empty context dict -> 'Nothing remembered yet for ...' message."""
    respx.post(f"{API_BASE}/v1/context/client/fetch").mock(
        return_value=httpx.Response(200, json={"context": {}})
    )
    result = await mcp.call_tool("recall_context", {"query": "x"})
    text = _text(result)
    assert "nothing" in text.lower() or "no memory" in text.lower()


@respx.mock
async def test_recall_context_honors_max_results(with_token):
    """max_results is forwarded to the REST call."""
    import json

    route = respx.post(f"{API_BASE}/v1/context/client/fetch").mock(
        return_value=httpx.Response(200, json={"context": {}})
    )
    await mcp.call_tool("recall_context", {"query": "q", "max_results": 7})
    sent = json.loads(route.calls.last.request.read().decode())
    assert sent["max_results"] == 7


@respx.mock
async def test_recall_context_default_max_results_applied(with_token):
    """When max_results is None, settings.default_max_results is used."""
    import json
    from synap_mcp_server.config import settings

    route = respx.post(f"{API_BASE}/v1/context/client/fetch").mock(
        return_value=httpx.Response(200, json={"context": {}})
    )
    await mcp.call_tool("recall_context", {"query": "q"})
    sent = json.loads(route.calls.last.request.read().decode())
    assert sent["max_results"] == settings.default_max_results


async def test_recall_context_missing_token_returns_error(no_token):
    """recall_context without a token returns an ERROR string."""
    result = await mcp.call_tool("recall_context", {"query": "x"})
    text = _text(result)
    assert "ERROR" in text


@respx.mock
async def test_recall_context_402_soft_fail(with_token):
    """A 402 is soft-failed with credit-mention, not an ERROR."""
    respx.post(f"{API_BASE}/v1/context/client/fetch").mock(
        return_value=httpx.Response(402, text="no credits")
    )
    result = await mcp.call_tool("recall_context", {"query": "x"})
    text = _text(result)
    assert "credit" in text.lower()
    assert "ERROR" not in text


# ---------------------------------------------------------------------------
# list_recent_memories — extended behavior
# ---------------------------------------------------------------------------


@respx.mock
async def test_list_recent_memories_happy_path(with_token):
    """list_recent_memories with a full context response returns formatted lines."""
    respx.post(f"{API_BASE}/v1/context/client/fetch").mock(
        return_value=httpx.Response(
            200,
            json={
                "context": {
                    "facts": [{"content": "Likes Python"}],
                    "preferences": [{"content": "Prefers async"}],
                }
            },
        )
    )
    result = await mcp.call_tool("list_recent_memories", {})
    text = _text(result)
    assert "Likes Python" in text
    assert "Prefers async" in text


@respx.mock
async def test_list_recent_memories_no_context_returns_no_memories(with_token):
    """Empty context -> 'No memories yet.' sentinel."""
    respx.post(f"{API_BASE}/v1/context/client/fetch").mock(
        return_value=httpx.Response(200, json={"context": {}})
    )
    result = await mcp.call_tool("list_recent_memories", {})
    text = _text(result)
    assert "no memories" in text.lower()


@respx.mock
async def test_list_recent_memories_passes_null_search_query(with_token):
    """list_recent_memories sends search_query=null (no specific query)."""
    import json

    route = respx.post(f"{API_BASE}/v1/context/client/fetch").mock(
        return_value=httpx.Response(200, json={"context": {}})
    )
    await mcp.call_tool("list_recent_memories", {})
    sent = json.loads(route.calls.last.request.read().decode())
    assert sent.get("search_query") is None


@respx.mock
async def test_list_recent_memories_respects_max_results(with_token):
    """max_results param is forwarded."""
    import json

    route = respx.post(f"{API_BASE}/v1/context/client/fetch").mock(
        return_value=httpx.Response(200, json={"context": {}})
    )
    await mcp.call_tool("list_recent_memories", {"max_results": 5})
    sent = json.loads(route.calls.last.request.read().decode())
    assert sent["max_results"] == 5


@respx.mock
async def test_list_recent_memories_user_scope(with_token):
    """user_id routes the list call to user scope."""
    import json

    route = respx.post(f"{API_BASE}/v1/context/user/fetch").mock(
        return_value=httpx.Response(200, json={"context": {}})
    )
    await mcp.call_tool("list_recent_memories", {"user_id": "u3"})
    assert route.called
    sent = json.loads(route.calls.last.request.read().decode())
    assert sent.get("user_id") == "u3"


async def test_list_recent_memories_missing_token_returns_error(no_token):
    """list_recent_memories without a token returns an ERROR string."""
    result = await mcp.call_tool("list_recent_memories", {})
    text = _text(result)
    assert "ERROR" in text


@respx.mock
async def test_list_recent_memories_500_soft_fails(with_token):
    """A 5xx on list is soft-failed, not raised."""
    respx.post(f"{API_BASE}/v1/context/client/fetch").mock(
        return_value=httpx.Response(500, text="boom")
    )
    result = await mcp.call_tool("list_recent_memories", {})
    text = _text(result)
    # Soft fail — must not be an unhandled exception; benign message returned
    assert text  # non-empty
    assert "ERROR" not in text  # soft policy


# ---------------------------------------------------------------------------
# get_ingestion_status — client-level happy path
# ---------------------------------------------------------------------------


@respx.mock
async def test_get_ingestion_status_happy_path(with_token):
    """get_ingestion_status GETs /api/v1/memories/status/<id> and returns the payload."""
    respx.get(f"{API_BASE}/api/v1/memories/status/ing_ok").mock(
        return_value=httpx.Response(200, json={"status": "completed", "memories_created": 3})
    )
    from synap_mcp_server import client

    data = await client.get_ingestion_status("ing_ok")
    assert data["status"] == "completed"
    assert data["memories_created"] == 3
