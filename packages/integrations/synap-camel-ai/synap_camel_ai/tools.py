"""Explicit Synap memory tools for CAMEL-AI agents.

For agents that want model-driven memory (the model decides when to search or
store) instead of, or alongside, the always-on :class:`SynapAgentMemory`. Returns
two :class:`camel.toolkits.FunctionTool` objects to pass to ``ChatAgent(tools=...)``.

CAMEL tool bodies are synchronous, so each drives the async SDK through
``run_async``. Reads degrade (return a not-found message); the write raises
``SynapIntegrationError`` so a failed store is observable to the caller.
"""

from __future__ import annotations

import logging
from typing import List

from camel.toolkits import FunctionTool
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import run_async, wrap_sdk_errors

logger = logging.getLogger(__name__)


def create_synap_tools(
    sdk: MaximemSynapSDK,
    user_id: str,
    *,
    customer_id: str = "",
) -> List[FunctionTool]:
    """Build ``search_memory`` and ``store_memory`` tools bound to a Synap scope.

    Args:
        sdk: Configured :class:`MaximemSynapSDK`.
        user_id: Synap user scope. **Required.**
        customer_id: Optional customer/org scope; empty means customer-less.

    Returns:
        A list ``[search_memory, store_memory]`` of :class:`FunctionTool`.
    """
    if sdk is None:
        raise ValueError("create_synap_tools requires a non-None sdk")
    if not user_id or not str(user_id).strip():
        raise ValueError("create_synap_tools requires a non-empty user_id")

    def search_memory(query: str) -> str:
        """Search the user's long-term memory for relevant context.

        Args:
            query (str): What to look up in memory.

        Returns:
            str: Formatted memory context, or a not-found message.
        """
        try:
            response = run_async(
                sdk.fetch(
                    user_id=user_id,
                    customer_id=customer_id or None,
                    search_query=[query],
                    include_conversation_context=False,
                )
            )
        except Exception as exc:  # noqa: BLE001 — read-side graceful degrade
            logger.error(
                "create_synap_tools.search_memory failed user_id=%s error=%s",
                user_id,
                exc,
                exc_info=True,
            )
            return "No relevant memories found."
        return (
            getattr(response, "formatted_context", None)
            or "No relevant memories found."
        )

    def store_memory(content: str) -> str:
        """Store a fact or note in the user's long-term memory for future recall.

        Args:
            content (str): The information to remember.

        Returns:
            str: Confirmation including the Synap ingestion id.
        """
        with wrap_sdk_errors("camel_ai.store_memory", logger, user_id=user_id):
            result = run_async(
                sdk.memories.create(
                    document=content,
                    user_id=user_id,
                    customer_id=customer_id or None,
                )
            )
        return (
            f"Memory stored (ingestion_id: "
            f"{getattr(result, 'ingestion_id', 'unknown')})"
        )

    return [FunctionTool(search_memory), FunctionTool(store_memory)]
