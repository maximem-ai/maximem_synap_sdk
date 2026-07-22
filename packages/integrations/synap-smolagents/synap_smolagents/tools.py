"""Synap memory tools for Smolagents agents.

`create_synap_tools` returns two Smolagents `@tool` functions — `search_memory` and
`store_memory` — that the model can call. Especially idiomatic for `CodeAgent`,
which invokes tools as plain Python calls.

Smolagents tools are synchronous (async `forward` is not supported), so each tool
drives the async Synap SDK through `run_async`. Reads degrade (return a not-found
message); the write raises `SynapIntegrationError` so a failed store is observable.
"""

from __future__ import annotations

import logging
from typing import List

from smolagents import tool
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import run_async, wrap_sdk_errors

logger = logging.getLogger(__name__)


def create_synap_tools(
    sdk: MaximemSynapSDK,
    user_id: str,
    *,
    customer_id: str = "",
) -> List:
    """Build `search_memory` and `store_memory` tools bound to a Synap scope.

    Args:
        sdk: Configured :class:`MaximemSynapSDK`.
        user_id: Synap user scope. **Required.**
        customer_id: Optional customer/org scope; empty means customer-less.

    Returns:
        A list ``[search_memory, store_memory]`` of Smolagents tools.
    """
    if sdk is None:
        raise ValueError("create_synap_tools requires a non-None sdk")
    if not user_id or not str(user_id).strip():
        raise ValueError("create_synap_tools requires a non-empty user_id")

    @tool
    def search_memory(query: str) -> str:
        """Search the user's long-term memory for relevant context.

        Args:
            query: What to look up in the user's memory.
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

    @tool
    def store_memory(content: str) -> str:
        """Store a fact or note in the user's long-term memory for future recall.

        Args:
            content: The information to remember.
        """
        with wrap_sdk_errors("smolagents.store_memory", logger, user_id=user_id):
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

    return [search_memory, store_memory]
