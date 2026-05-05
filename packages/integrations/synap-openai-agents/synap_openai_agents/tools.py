"""Synap tools for OpenAI Agents SDK.

Provides tool functions that can be passed to an OpenAI Agent.
Uses the functional tool pattern since @function_tool requires
Python 3.12+ typing features in newer versions.
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
    """Create a memory search function for OpenAI Agents.

    Returns an async callable that can be wrapped with FunctionTool
    or ``@function_tool``. SDK failures are wrapped in
    :class:`SynapIntegrationError`.

    Example::

        from agents import Agent, FunctionTool

        search_fn = create_search_tool(sdk, user_id="u1")
        agent = Agent(
            name="assistant",
            tools=[FunctionTool(search_fn, name="search_memory",
                                description="Search user memory")],
        )
    """
    if sdk is None:
        raise ValueError("create_search_tool requires a non-None sdk")
    if not user_id:
        raise ValueError("create_search_tool requires a non-empty user_id")

    async def search_memory(query: str) -> str:
        """Search the user's memory for relevant context.

        Args:
            query: Natural language search query.

        Returns:
            Formatted context from memory.
        """
        async with wrap_sdk_errors_async(
            "openai_agents.search_memory", logger, user_id=user_id,
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
    """Create a memory store function for OpenAI Agents.

    Returns an async callable that can be wrapped with FunctionTool.
    SDK failures are wrapped in :class:`SynapIntegrationError`.
    """
    if sdk is None:
        raise ValueError("create_store_tool requires a non-None sdk")
    if not user_id:
        raise ValueError("create_store_tool requires a non-empty user_id")

    async def store_memory(content: str) -> str:
        """Store important information about the user for future reference.

        Args:
            content: Information to remember.

        Returns:
            Confirmation message.
        """
        async with wrap_sdk_errors_async(
            "openai_agents.store_memory", logger, user_id=user_id,
        ):
            result = await sdk.memories.create(
                document=content,
                user_id=user_id,
                customer_id=customer_id,
            )
        return f"Memory stored (ingestion_id: {result.ingestion_id})"

    return store_memory
