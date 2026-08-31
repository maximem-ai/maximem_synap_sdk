"""Tool registration tests: assert all 4 tools are discoverable with correct schemas.

THE BAR: every public MCP tool has a registration test asserting its name,
description keywords, and required/optional parameter schema.

The descriptions are also the interface a coding agent reads, so the identifier
contract is asserted here as published text, not only as runtime behaviour. A
server that refuses a customer_id correctly while still advertising it as
"optional" has fixed the smaller half of the problem.
"""

import pytest

from synap_mcp_server.server import mcp
from synap_mcp_server.tools import (
    CUSTOMER_ID_HELP,
    LOG_DESC,
    STATUS_DESC,
    RECALL_DESC,
    LIST_DESC,
    SCOPE_CONTRACT,
    USER_ID_HELP,
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
    """log_exchange exposes assistant_message, conversation_id, user_id, customer_id,
    wait_for_processing.

    customer_id stays in the schema on purpose: it is REQUIRED on a B2B instance.
    Removing it to satisfy the B2C rule would break every B2B caller, which is the
    larger of the two failures.
    """
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


# ---------------------------------------------------------------------------
# The identifier contract, as published to the model
# ---------------------------------------------------------------------------

SCOPED_TOOLS = ("log_exchange", "recall_context", "list_recent_memories")


@pytest.mark.parametrize("tool_name", SCOPED_TOOLS)
async def test_every_scoped_tool_publishes_the_contract(tool_name):
    """Each tool that accepts a customer_id states, in its own description, that it
    is B2B only and refused on B2C. Per tool and not once in the server
    instructions, because a model choosing a tool reads that tool."""
    by_name = await _get_tools_by_name()
    desc = by_name[tool_name].description or ""
    assert SCOPE_CONTRACT in desc, f"{tool_name} does not carry the scoping contract"


@pytest.mark.parametrize("tool_name", SCOPED_TOOLS)
async def test_contract_names_both_modes_and_the_rejection(tool_name):
    """The contract text has to be actionable on its own: which mode forbids the
    field, which mode requires it, and what happens if you send it anyway."""
    by_name = await _get_tools_by_name()
    desc = by_name[tool_name].description or ""
    assert "equals_customer" in desc, "the B2C mode value must be named"
    assert "strict" in desc, "the B2B mode value must be named"
    assert "400" in desc, "the consequence of sending customer_id on B2C must be named"
    assert "whoami" in desc, "the model must be told where to read the mode"


@pytest.mark.parametrize("tool_name", SCOPED_TOOLS)
async def test_customer_id_is_never_advertised_as_plainly_optional(tool_name):
    """The old wording was "an organization id as customer_id", with no mode
    attached. "Optional" is wrong in both directions: refused on B2C, mandatory on
    B2B. Any description of the field must qualify it."""
    by_name = await _get_tools_by_name()
    desc = by_name[tool_name].description or ""
    lowered = desc.lower()
    assert "customer_id" in lowered
    assert "b2b only" in lowered, (
        f"{tool_name} mentions customer_id without saying it is B2B only"
    )


@pytest.mark.parametrize("tool_name", SCOPED_TOOLS)
async def test_customer_id_property_carries_its_own_description(tool_name):
    """The inputSchema property description is what a coding agent reads while it
    is deciding what to put in the field, which is later and closer to the mistake
    than the tool description is."""
    by_name = await _get_tools_by_name()
    props = (by_name[tool_name].inputSchema or {}).get("properties", {})
    described = props["customer_id"].get("description", "")
    assert described == CUSTOMER_ID_HELP
    assert "B2B ONLY" in described
    assert "equals_customer" in described


@pytest.mark.parametrize("tool_name", SCOPED_TOOLS)
async def test_user_id_property_carries_its_own_description(tool_name):
    """The mirror half: on B2C, user_id is the whole identity, and the field says so."""
    by_name = await _get_tools_by_name()
    props = (by_name[tool_name].inputSchema or {}).get("properties", {})
    described = props["user_id"].get("description", "")
    assert described == USER_ID_HELP
    assert "whole identity" in described


async def test_check_memory_status_takes_no_scope_ids():
    """The fourth tool is scope-free: an ingestion_id already identifies the write.

    Pinned so nobody "completes the set" by adding a customer_id it has no use for.
    """
    by_name = await _get_tools_by_name()
    props = (by_name["check_memory_status"].inputSchema or {}).get("properties", {})
    assert "customer_id" not in props
    assert "user_id" not in props
    assert "contract does not apply" in (by_name["check_memory_status"].description or "")
