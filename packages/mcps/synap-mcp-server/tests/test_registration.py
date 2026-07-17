"""Tool registration tests — assert all 4 tools are discoverable with correct schemas.

THE BAR: every public MCP tool has a registration test asserting its name,
description keywords, and required/optional parameter schema.
"""

import pytest

from synap_mcp_server.server import mcp
from synap_mcp_server.tools import (
    LOG_DESC,
    STATUS_DESC,
    RECALL_DESC,
    LIST_DESC,
)

pytestmark = pytest.mark.asyncio

EXPECTED_TOOL_NAMES = {"log_exchange", "check_memory_status", "recall_context", "list_recent_memories"}


async def _get_tools_by_name():
    tools = await mcp.list_tools()
    return {t.name: t for t in tools}


# ---------------------------------------------------------------------------
# All four tools are registered
# ---------------------------------------------------------------------------


async def test_all_four_tools_are_registered():
    """All 4 MCP tools must be discoverable; none may be missing."""
    by_name = await _get_tools_by_name()
    assert EXPECTED_TOOL_NAMES <= set(by_name.keys()), (
        f"Missing tools: {EXPECTED_TOOL_NAMES - set(by_name.keys())}"
    )


# ---------------------------------------------------------------------------
# log_exchange schema
# ---------------------------------------------------------------------------


async def test_log_exchange_required_param_user_message():
    """log_exchange requires 'user_message'; everything else is optional."""
    by_name = await _get_tools_by_name()
    schema = by_name["log_exchange"].inputSchema or {}
    assert "user_message" in schema.get("required", [])


async def test_log_exchange_optional_params_present():
    """log_exchange exposes assistant_message, conversation_id, user_id, customer_id, wait_for_processing."""
    by_name = await _get_tools_by_name()
    props = (by_name["log_exchange"].inputSchema or {}).get("properties", {})
    for param in ("assistant_message", "conversation_id", "user_id", "customer_id", "wait_for_processing"):
        assert param in props, f"Expected '{param}' in log_exchange properties"


async def test_log_exchange_description_mentions_remember():
    """Description must guide the model to 'remember' so it calls this tool."""
    by_name = await _get_tools_by_name()
    desc = by_name["log_exchange"].description or ""
    assert "remember" in desc.lower()


async def test_log_exchange_description_is_log_desc_constant():
    """The registered description matches the LOG_DESC constant in tools.py."""
    by_name = await _get_tools_by_name()
    assert by_name["log_exchange"].description == LOG_DESC


# ---------------------------------------------------------------------------
# check_memory_status schema
# ---------------------------------------------------------------------------


async def test_check_memory_status_is_registered():
    """check_memory_status must be registered (was absent from the original 3-tool assertion)."""
    by_name = await _get_tools_by_name()
    assert "check_memory_status" in by_name


async def test_check_memory_status_required_param_ingestion_id():
    """check_memory_status requires 'ingestion_id' and nothing else."""
    by_name = await _get_tools_by_name()
    schema = by_name["check_memory_status"].inputSchema or {}
    required = schema.get("required", [])
    assert "ingestion_id" in required


async def test_check_memory_status_description_mentions_processing():
    """Description mentions processing/status so model knows when to call it."""
    by_name = await _get_tools_by_name()
    desc = by_name["check_memory_status"].description or ""
    assert "status" in desc.lower() or "processing" in desc.lower()


async def test_check_memory_status_description_is_status_desc_constant():
    by_name = await _get_tools_by_name()
    assert by_name["check_memory_status"].description == STATUS_DESC


# ---------------------------------------------------------------------------
# recall_context schema
# ---------------------------------------------------------------------------


async def test_recall_context_required_param_query():
    by_name = await _get_tools_by_name()
    schema = by_name["recall_context"].inputSchema or {}
    assert "query" in schema.get("required", [])


async def test_recall_context_optional_scope_params():
    """user_id, customer_id, max_results are optional on recall_context."""
    by_name = await _get_tools_by_name()
    props = (by_name["recall_context"].inputSchema or {}).get("properties", {})
    for param in ("user_id", "customer_id", "max_results"):
        assert param in props, f"Expected '{param}' in recall_context properties"


async def test_recall_context_description_mentions_recall():
    by_name = await _get_tools_by_name()
    desc = by_name["recall_context"].description or ""
    assert "recall" in desc.lower()


async def test_recall_context_description_is_recall_desc_constant():
    by_name = await _get_tools_by_name()
    assert by_name["recall_context"].description == RECALL_DESC


# ---------------------------------------------------------------------------
# list_recent_memories schema
# ---------------------------------------------------------------------------


async def test_list_recent_memories_has_no_required_params():
    """list_recent_memories takes only optional params (max_results, user_id, customer_id)."""
    by_name = await _get_tools_by_name()
    schema = by_name["list_recent_memories"].inputSchema or {}
    # required list should be absent or empty
    required = schema.get("required", [])
    assert len(required) == 0, f"Expected no required params, got: {required}"


async def test_list_recent_memories_optional_params():
    by_name = await _get_tools_by_name()
    props = (by_name["list_recent_memories"].inputSchema or {}).get("properties", {})
    for param in ("max_results", "user_id", "customer_id"):
        assert param in props


async def test_list_recent_memories_description_is_list_desc_constant():
    by_name = await _get_tools_by_name()
    assert by_name["list_recent_memories"].description == LIST_DESC


# ---------------------------------------------------------------------------
# register() is idempotent on a fresh FastMCP instance
# ---------------------------------------------------------------------------


async def test_register_on_fresh_mcp_instance():
    """tools.register() on a separate FastMCP instance registers the same 4 tools."""
    from mcp.server.fastmcp import FastMCP
    from synap_mcp_server import tools

    fresh_mcp = FastMCP("test-registration", stateless_http=True)
    tools.register(fresh_mcp)
    registered = {t.name for t in await fresh_mcp.list_tools()}
    assert registered == EXPECTED_TOOL_NAMES
