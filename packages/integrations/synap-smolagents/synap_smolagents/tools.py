"""Synap tools for Smolagents.

Provides memory search/store callables for Smolagents agents.
SDK failures are wrapped in :class:`SynapIntegrationError`.
"""

import logging
from typing import Optional

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import wrap_sdk_errors_async

logger = logging.getLogger(__name__)


def create_search_tool(
    sdk: MaximemSynapSDK,
    user_id: str,
    customer_id: str = "",
    conversation_id: Optional[str] = None,
):
    """Create a memory search callable for Smolagents.

    Example::

        search_fn = create_search_tool(sdk, user_id="u1")
        # Wire into your Smolagents agent: Tool(search_memory, name='search_memory')
    """
    if sdk is None:
        raise ValueError("create_search_tool requires a non-None sdk")
    if not user_id:
        raise ValueError("create_search_tool requires a non-empty user_id")

    async def search_memory(query: str) -> str:
        """Search the user's memory for relevant context."""
        async with wrap_sdk_errors_async(
            "smolagents.search_memory",
            logger,
            user_id=user_id,
        ):
            response = await sdk.fetch(
                conversation_id=conversation_id,
                user_id=user_id,
                customer_id=customer_id or None,
                search_query=[query],
                mode="accurate",
                include_conversation_context=False,
            )
        return response.formatted_context or "No relevant memories found."

    return search_memory


def create_store_tool(
    sdk: MaximemSynapSDK,
    user_id: str,
    customer_id: str = "",
):
    """Create a memory store callable for Smolagents."""
    if sdk is None:
        raise ValueError("create_store_tool requires a non-None sdk")
    if not user_id:
        raise ValueError("create_store_tool requires a non-empty user_id")

    async def store_memory(content: str) -> str:
        """Store important information about the user for future reference."""
        async with wrap_sdk_errors_async(
            "smolagents.store_memory",
            logger,
            user_id=user_id,
        ):
            result = await sdk.memories.create(
                document=content,
                user_id=user_id,
                customer_id=customer_id,
            )
        return f"Memory stored (ingestion_id: {result.ingestion_id})"

    return store_memory
