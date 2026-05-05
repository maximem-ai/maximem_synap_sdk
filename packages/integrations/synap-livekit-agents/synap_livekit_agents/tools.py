"""LLM-callable function tools exposing Synap inside a LiveKit Agent.

Two factories — :func:`synap_search_tool` and :func:`synap_store_tool`
— each close over a configured SDK + scope and return a
:class:`FunctionTool` instance ready for ``Agent(tools=[...])``.

The search tool degrades gracefully on read failures (returns a
natural-language placeholder string the LLM can handle) so a Synap blip
never hard-fails a tool call mid-conversation. The store tool follows
the integration write-policy — :class:`SynapIntegrationError` bubbles
to LiveKit's function-tool runtime so the model sees a proper tool
error.
"""

from __future__ import annotations

import logging
from typing import Optional

from maximem_synap import MaximemSynapSDK

from livekit.agents import FunctionTool, function_tool

from synap_integrations_common import wrap_sdk_errors_async

logger = logging.getLogger(__name__)


def synap_search_tool(
    sdk: MaximemSynapSDK,
    *,
    user_id: str,
    customer_id: str = "",
    mode: str = "accurate",
    max_results: int = 10,
) -> FunctionTool:
    """Factory: returns an LLM-callable ``synap_search(query)`` tool.

    The tool calls ``sdk.fetch`` scoped to ``user_id``/``customer_id`` and
    returns the formatted context string. On SDK failure, returns a
    natural-language placeholder instead of raising — keeps the call
    alive if Synap momentarily misbehaves.
    """
    if sdk is None:
        raise ValueError("synap_search_tool requires a non-None sdk")
    if not user_id:
        raise ValueError("synap_search_tool requires a non-empty user_id")

    @function_tool(
        name="synap_search",
        description=(
            "Search the user's long-term memory (Synap) for facts, "
            "preferences, or prior episodes relevant to the given query. "
            "Returns a formatted context string, or a placeholder if "
            "nothing relevant is found."
        ),
    )
    async def synap_search(query: str) -> str:
        """Search Synap long-term memory.

        Args:
            query: Natural-language description of what to look up.
        """
        try:
            response = await sdk.fetch(
                user_id=user_id,
                customer_id=customer_id or None,
                search_query=[query] if query else None,
                max_results=max_results,
                mode=mode,
            )
        except Exception as exc:  # noqa: BLE001 — read-side graceful degrade
            logger.error(
                "synap_search: sdk.fetch failed user_id=%s error=%s",
                user_id, exc, exc_info=True,
            )
            return "Synap memory is temporarily unavailable."

        formatted = getattr(response, "formatted_context", None) or ""
        if not formatted.strip():
            return "No relevant long-term memory found for this query."
        return formatted.strip()

    return synap_search


def synap_store_tool(
    sdk: MaximemSynapSDK,
    *,
    user_id: str,
    customer_id: str = "",
    document_type: str = "ai-chat-conversation",
) -> FunctionTool:
    """Factory: returns an LLM-callable ``synap_store(content, category)`` tool.

    Write failures surface as :class:`SynapIntegrationError` via
    ``wrap_sdk_errors_async`` — the model sees a tool error rather than
    silent success, consistent with Synap's write policy.
    """
    if sdk is None:
        raise ValueError("synap_store_tool requires a non-None sdk")
    if not user_id:
        raise ValueError("synap_store_tool requires a non-empty user_id")

    @function_tool(
        name="synap_store",
        description=(
            "Persist a new memory about the user to Synap's long-term "
            "store. Use when the user volunteers a durable fact, "
            "preference, or goal worth remembering across sessions."
        ),
    )
    async def synap_store(content: str, category: Optional[str] = "fact") -> str:
        """Store a new memory in Synap.

        Args:
            content: The fact, preference, or detail to remember about the user.
            category: Optional category tag (e.g. "fact", "preference").
        """
        async with wrap_sdk_errors_async(
            "livekit.synap_store",
            logger,
            user_id=user_id,
            category=category or "fact",
        ):
            result = await sdk.memories.create(
                document=content,
                user_id=user_id,
                customer_id=customer_id,
                document_type=document_type,
                metadata={"category": category or "fact"},
            )
        ingestion_id = getattr(result, "ingestion_id", None) or "unknown"
        return f"Stored memory (ingestion_id={ingestion_id})."

    return synap_store


__all__ = ["synap_search_tool", "synap_store_tool"]
