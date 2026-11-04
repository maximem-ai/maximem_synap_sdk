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
from typing import Any, Dict, List, Mapping, Optional

from smolagents import tool
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import ensure_ids_match_path, run_async, wrap_sdk_errors

logger = logging.getLogger(__name__)


def create_synap_tools(
    sdk: MaximemSynapSDK,
    user_id: str,
    *,
    customer_id: str = "",
    scope_path: Optional[Mapping[str, str]] = None,
) -> List:
    """Build `search_memory` and `store_memory` tools bound to a Synap scope.

    Args:
        sdk: Configured :class:`MaximemSynapSDK`.
        user_id: Synap user scope. **Required.**
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on a
            B2C instance (user_context_isolation=equals_customer): the server
            rejects a call carrying one with HTTP 400. Leave it unset there.
        scope_path: For an account with a deeper scope ladder: the level these
            tools read and write at, as ``{level_key: value}`` (for example
            ``ScopeAddress.scope_path``; key order does not matter). Closed over
            here and forwarded to ``fetch`` / ``memories.create`` only when set.
            It is NOT a parameter of the tools, so the model cannot see or
            change it.

    Returns:
        A list ``[search_memory, store_memory]`` of Smolagents tools.
    """
    if sdk is None:
        raise ValueError("create_synap_tools requires a non-None sdk")
    if not user_id or not str(user_id).strip():
        raise ValueError("create_synap_tools requires a non-empty user_id")
    ensure_ids_match_path(
        scope_path, user_id=user_id, customer_id=customer_id or None,
        where="create_synap_tools")
    path: Optional[Dict[str, str]] = dict(scope_path) if scope_path else None

    def _path(name: str) -> Dict[str, Any]:
        # Only when set: a call without a path keeps exactly its old keywords.
        return {name: dict(path)} if path else {}

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
                    **_path("scope_path"),
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
                    **_path("scope"),
                )
            )
        return (
            f"Memory stored (ingestion_id: "
            f"{getattr(result, 'ingestion_id', 'unknown')})"
        )

    return [search_memory, store_memory]
