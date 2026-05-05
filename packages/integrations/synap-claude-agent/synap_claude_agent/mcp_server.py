"""Synap MCP server for the Claude Agent SDK.

Exposes two in-process MCP tools that let the agent read/write Synap memory
explicitly as tool calls:

- ``synap_search(query, max_results?)`` — searches Synap via ``sdk.fetch``
  and returns the formatted context as plain text. Read failures degrade to
  an explanatory "no context available" result so the agent loop doesn't
  wedge.

- ``synap_remember(content, metadata?)`` — ingests an explicit fact via
  ``sdk.memories.create``. Write failures raise so ingestion outages are
  observable on the agent side as tool errors.

The wrapping tool names in Claude are ``mcp__synap__synap_search`` and
``mcp__synap__synap_remember`` (per the SDK's ``mcp__<server>__<tool>``
naming). Add them to ``ClaudeAgentOptions(allowed_tools=[...])``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server, tool
from claude_agent_sdk.types import McpSdkServerConfig
from maximem_synap import MaximemSynapSDK

logger = logging.getLogger(__name__)


def _build_synap_tools(
    sdk: MaximemSynapSDK,
    user_id: str,
    customer_id: str,
    conversation_id: Optional[str],
    mode: str,
) -> list[SdkMcpTool]:
    """Build the underlying :class:`SdkMcpTool` list.

    Kept separate from :func:`create_synap_mcp_server` so smoke tests and
    downstream integrations can invoke the tool handlers directly without
    spinning up the MCP server machinery.
    """

    @tool(
        "synap_search",
        "Search the user's Synap memory (facts, preferences, episodes, "
        "emotions, temporal events) for context relevant to a query. Use "
        "this when you need background about the user that isn't in the "
        "current conversation.",
        {"query": str, "max_results": int},
    )
    async def synap_search(args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query", "")
        max_results = int(args.get("max_results") or 10)
        if not query:
            return {
                "content": [
                    {"type": "text", "text": "synap_search: missing `query` argument."}
                ],
                "isError": True,
            }

        try:
            response = await sdk.fetch(
                conversation_id=conversation_id,
                user_id=user_id,
                customer_id=customer_id or None,
                search_query=[query],
                max_results=max_results,
                mode=mode,
                include_conversation_context=False,
            )
        except Exception as exc:  # noqa: BLE001 — tool errors surface as text
            logger.error(
                "synap_search: sdk.fetch failed user_id=%s error=%s",
                user_id, exc, exc_info=True,
            )
            return {
                "content": [
                    {"type": "text", "text": f"synap_search: no context available ({exc.__class__.__name__})."}
                ],
                "isError": False,
            }

        text = (getattr(response, "formatted_context", None) or "").strip()
        if not text:
            text = "synap_search: no relevant context."
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "synap_remember",
        "Persist an explicit fact, preference, or note to the user's "
        "Synap memory for future recall. Call this when the user shares "
        "something worth remembering across sessions.",
        {"content": str, "metadata": dict},
    )
    async def synap_remember(args: dict[str, Any]) -> dict[str, Any]:
        content = args.get("content", "")
        if not content or not str(content).strip():
            return {
                "content": [
                    {"type": "text", "text": "synap_remember: missing `content` argument."}
                ],
                "isError": True,
            }
        metadata = args.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("source", "claude_agent_sdk")

        try:
            result = await sdk.memories.create(
                document=str(content),
                user_id=user_id,
                customer_id=customer_id or None,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 — surface as tool error
            logger.error(
                "synap_remember: sdk.memories.create failed user_id=%s error=%s",
                user_id, exc, exc_info=True,
            )
            return {
                "content": [
                    {"type": "text", "text": f"synap_remember: ingestion failed ({exc})."}
                ],
                "isError": True,
            }

        ingestion_id = getattr(result, "ingestion_id", None) or ""
        return {
            "content": [
                {"type": "text", "text": f"synap_remember: recorded (ingestion_id={ingestion_id})."}
            ]
        }

    return [synap_search, synap_remember]


def create_synap_mcp_server(
    sdk: MaximemSynapSDK,
    user_id: str,
    customer_id: str = "",
    conversation_id: Optional[str] = None,
    *,
    name: str = "synap",
    version: str = "0.1.0",
    mode: str = "accurate",
) -> McpSdkServerConfig:
    """Build an in-process MCP server exposing Synap as Claude tools."""
    if sdk is None:
        raise ValueError("create_synap_mcp_server requires a non-None sdk")
    if not user_id:
        raise ValueError("create_synap_mcp_server requires a non-empty user_id")

    tools = _build_synap_tools(
        sdk=sdk,
        user_id=user_id,
        customer_id=customer_id,
        conversation_id=conversation_id,
        mode=mode,
    )
    return create_sdk_mcp_server(name=name, version=version, tools=tools)
