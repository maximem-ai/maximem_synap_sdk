"""Synap tools for Google ADK agents.

Provides search_memory and store_memory as ADK FunctionTools.
"""

import logging
from typing import Mapping, Optional

from google.adk.tools import FunctionTool

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import ensure_ids_match_path, wrap_sdk_errors_async, coerce_conversation_id

logger = logging.getLogger(__name__)


def create_synap_tools(
    sdk: MaximemSynapSDK,
    user_id: str,
    customer_id: str = "",
    conversation_id: Optional[str] = None,
    *,
    scope_path: Optional[Mapping[str, str]] = None,
) -> list:
    """Create ADK FunctionTools for Synap memory operations.

    Returns a list of ``[search_memory, store_memory]`` FunctionTool
    instances. SDK failures are wrapped in :class:`SynapIntegrationError`
    so the agent runtime sees a consistent error type rather than
    a leaked raw SDK exception.

    ``scope_path`` names the level on an account with a deeper scope ladder
    (``{level_key: value}``). It is closed over here and sent only when set; it
    is not a parameter of either tool, so the model cannot choose a level.
    """
    if sdk is None:
        raise ValueError("create_synap_tools requires a non-None sdk")
    if not user_id:
        raise ValueError("create_synap_tools requires a non-empty user_id")
    ensure_ids_match_path(
        scope_path, user_id=user_id, customer_id=customer_id or None,
        where="create_synap_tools")
    path = dict(scope_path) if scope_path else None

    async def search_memory(query: str) -> str:
        """Search the user's memory for relevant context.

        Args:
            query: Natural language search query.

        Returns:
            Formatted context from memory.
        """
        async with wrap_sdk_errors_async(
            "google_adk.search_memory", logger, user_id=user_id,
        ):
            response = await sdk.fetch(
                conversation_id=coerce_conversation_id(conversation_id),
                user_id=user_id,
                customer_id=customer_id or None,
                search_query=[query],
                mode="accurate",
                include_conversation_context=False,
                **({"scope_path": path} if path else {}),
            )
        return response.formatted_context or "No relevant memories found."

    async def store_memory(content: str) -> str:
        """Store important information about the user.

        Args:
            content: Information to remember.

        Returns:
            Confirmation message.
        """
        async with wrap_sdk_errors_async(
            "google_adk.store_memory", logger, user_id=user_id,
        ):
            result = await sdk.memories.create(
                document=content,
                user_id=user_id,
                customer_id=customer_id,
                **({"scope": path} if path else {}),
            )
        return f"Memory stored (ingestion_id: {result.ingestion_id})"

    return [FunctionTool(search_memory), FunctionTool(store_memory)]
