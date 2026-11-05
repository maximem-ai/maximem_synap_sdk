"""Synap tools for HarnessX.

HarnessX registers plain async functions as tools and infers the schema from
the type hints and the ``Args:`` docstring section, so these are built as
closures over the SDK and handed to ``ctx.register_tool``.

Use them when you want the model to decide when to reach for memory. They sit
alongside the automatic recall that :class:`~synap_harnessx.SynapExtension`
does each turn: recall covers what is relevant to the question, search covers
what the model decides it needs to look wider for.

A failed call raises :class:`~synap_integrations_common.SynapIntegrationError`.
HarnessX turns an exception from a tool into an error result the model can
read, so the agent run continues.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import wrap_sdk_errors_async

logger = logging.getLogger(__name__)

SEARCH_DESCRIPTION = (
    "Search the user's long-term memory for relevant context. Use this to "
    "recall past conversations, stated preferences, facts about the user, or "
    "earlier decisions."
)
STORE_DESCRIPTION = (
    "Store an important fact, preference, or decision about the user for "
    "future sessions. Write two or three sentences of plain prose with context, "
    "such as what it is and why or when it matters. Synap extracts and "
    "categorizes it automatically, and a bare one-line statement can yield "
    "nothing to store."
)


def make_synap_tools(
    sdk: MaximemSynapSDK,
    *,
    user_id: str,
    customer_id: Optional[str] = None,
    conversation_id: Optional[Callable[[], Optional[str]]] = None,
    search_mode: str = "accurate",
    search_max_results: int = 10,
    ingest_mode: str = "fast",
    document_type: str = "document",
) -> tuple[Callable[..., Awaitable[str]], Callable[..., Awaitable[str]]]:
    """Return ``(search_memory, store_memory)`` async tool handlers.

    ``conversation_id`` is a zero-argument callable so the tool sees the
    session id the agent has *now*, not the one it had at construction.
    """
    customer = customer_id or None

    async def search_memory(query: str) -> str:
        async with wrap_sdk_errors_async(
            "harnessx.search_memory", logger, user_id=user_id
        ):
            response = await sdk.fetch(
                conversation_id=conversation_id() if conversation_id else None,
                user_id=user_id,
                customer_id=customer,
                search_query=[query],
                mode=search_mode,
                max_results=search_max_results,
                include_conversation_context=False,
            )
        return response.formatted_context or "No relevant memories found."

    async def store_memory(content: str) -> str:
        async with wrap_sdk_errors_async(
            "harnessx.store_memory", logger, user_id=user_id
        ):
            result = await sdk.memories.create(
                document=content,
                user_id=user_id,
                customer_id=customer,
                document_type=document_type,
                mode=ingest_mode,
            )
        return f"Memory accepted for storage (ingestion_id: {result.ingestion_id})"

    search_memory.__doc__ = (
        f"{SEARCH_DESCRIPTION}\n\nArgs:\n    query: Natural language question "
        "to search the user's memory with."
    )
    store_memory.__doc__ = (
        f"{STORE_DESCRIPTION}\n\nArgs:\n    content: The fact, preference, or "
        "decision to remember."
    )
    return search_memory, store_memory


__all__ = ["make_synap_tools"]
