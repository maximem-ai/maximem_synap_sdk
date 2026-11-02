"""Synap tools for Strands agents.

Provides ``search_memory`` and ``store_memory`` as Strands ``@tool`` functions
for agents that want explicit memory operations without adopting the full
``MemoryManager`` wiring (see :mod:`synap_strands_agents.store` for that).

SDK failures are wrapped in :class:`SynapIntegrationError` so the agent sees a
consistent error type rather than a leaked raw SDK exception.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

from strands import tool

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import ensure_ids_match_path, wrap_sdk_errors_async, coerce_conversation_id

logger = logging.getLogger(__name__)


def create_synap_tools(
    sdk: MaximemSynapSDK,
    user_id: str,
    customer_id: str = "",
    conversation_id: Optional[str] = None,
    scope_path: Optional[Mapping[str, str]] = None,
) -> list:
    """Create Strands tools for Synap memory operations.

    Returns ``[search_memory, store_memory]`` as Strands ``@tool`` functions
    ready for ``Agent(tools=...)``. The decorated functions are also directly
    awaitable, which is what the tests exercise.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        user_id: External user id these operations act on. **Required.**
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on a
            B2C instance (user_context_isolation=equals_customer): the server
            rejects a call carrying one with HTTP 400. Leave it unset there. Empty means customer-less.
        conversation_id: Optional Synap conversation id threaded into ``fetch``.
        scope_path: For an account with a deeper scope ladder: the level the
            tools read and write at, as ``{level_key: value}``. Closed over
            here and forwarded to ``fetch`` / ``memories.create`` (as ``scope``)
            only when set. It is NOT a parameter of the tools, so the model
            cannot see or change it.

    Example::

        from strands import Agent
        from maximem_synap import MaximemSynapSDK
        from synap_strands_agents import create_synap_tools

        sdk = MaximemSynapSDK(api_key="sk-...")
        agent = Agent(
            system_prompt="You are a helpful assistant.",
            tools=create_synap_tools(sdk, user_id="alice", customer_id="acme"),
        )
    """
    if sdk is None:
        raise ValueError("create_synap_tools requires a non-None sdk")
    if not user_id:
        raise ValueError("create_synap_tools requires a non-empty user_id")
    ensure_ids_match_path(
        scope_path, user_id=user_id, customer_id=customer_id or None,
        where="create_synap_tools")
    path: Optional[Dict[str, str]] = dict(scope_path) if scope_path else None

    def _path(name: str) -> Dict[str, Any]:
        return {name: dict(path)} if path else {}

    @tool
    async def search_memory(query: str) -> str:
        """Search the user's memory for relevant context.

        Args:
            query: Natural language search query.

        Returns:
            Formatted context from memory, or a not-found message.
        """
        async with wrap_sdk_errors_async(
            "strands.search_memory", logger, user_id=user_id,
        ):
            response = await sdk.fetch(
                conversation_id=coerce_conversation_id(conversation_id),
                user_id=user_id,
                customer_id=customer_id or None,
                search_query=[query],
                mode="accurate",
                include_conversation_context=False,
                **_path("scope_path"),
            )
        return response.formatted_context or "No relevant memories found."

    @tool
    async def store_memory(content: str) -> str:
        """Store important information about the user.

        Args:
            content: Information to remember.

        Returns:
            Confirmation message with the ingestion id.
        """
        async with wrap_sdk_errors_async(
            "strands.store_memory", logger, user_id=user_id,
        ):
            result = await sdk.memories.create(
                document=content,
                user_id=user_id,
                customer_id=customer_id,
                **_path("scope"),
            )
        return f"Memory stored (ingestion_id: {result.ingestion_id})"

    return [search_memory, store_memory]


__all__ = ["create_synap_tools"]
