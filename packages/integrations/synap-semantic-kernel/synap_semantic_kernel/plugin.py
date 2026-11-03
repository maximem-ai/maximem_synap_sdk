"""Synap plugin for Microsoft Semantic Kernel.

Exposes Synap memory as kernel functions (search_memory, store_memory)
that can be registered with a Semantic Kernel instance.
"""

import logging
from typing import Annotated, Any, Dict, Mapping, Optional

from semantic_kernel.functions import kernel_function

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import ensure_ids_match_path, wrap_sdk_errors_async, coerce_conversation_id

logger = logging.getLogger(__name__)


class SynapPlugin:
    """Semantic Kernel plugin for Synap memory operations.

    Example::

        from semantic_kernel import Kernel
        from synap_semantic_kernel import SynapPlugin

        kernel = Kernel()
        kernel.add_plugin(
            SynapPlugin(sdk=sdk, user_id="u1"),
            plugin_name="synap",
        )

    Deeper scope ladders: on an account whose ladder has levels beyond
    customer and user, pass ``scope_path`` (``{level_key: value}``, for example
    ``ScopeAddress.scope_path``; key order does not matter)::

        SynapPlugin(
            sdk=sdk, user_id="dana", customer_id="acme",
            scope_path={"customer": "acme", "team": "payments", "user": "dana"},
        )

    It is closed over at construction and forwarded to ``fetch`` and
    ``memories.create`` only when set. It is NOT a parameter of the kernel
    functions, so the model cannot see or change it.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: str,
        customer_id: str = "",
        conversation_id: Optional[str] = None,
        *,
        scope_path: Optional[Mapping[str, str]] = None,
    ):
        if sdk is None:
            raise ValueError("SynapPlugin requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapPlugin requires a non-empty user_id")
        ensure_ids_match_path(
            scope_path, user_id=user_id, customer_id=customer_id or None,
            where="SynapPlugin")

        self._sdk = sdk
        self._user_id = user_id
        self._customer_id = customer_id
        self._conversation_id = conversation_id
        self._scope_path: Optional[Dict[str, str]] = dict(scope_path) if scope_path else None

    def _path(self, name: str) -> Dict[str, Any]:
        """``{name: path}`` when a scope path is set, else nothing, so a call
        without one has exactly the keyword arguments it always had."""
        return {name: dict(self._scope_path)} if self._scope_path else {}

    @kernel_function(
        name="search_memory",
        description=(
            "Search the user's memory for relevant context. Use when you need "
            "to recall past conversations, preferences, or facts about the user."
        ),
    )
    async def search_memory(
        self,
        query: Annotated[str, "Natural language search query"],
    ) -> str:
        async with wrap_sdk_errors_async(
            "semantic_kernel.search_memory", logger, user_id=self._user_id,
        ):
            response = await self._sdk.fetch(
                conversation_id=coerce_conversation_id(self._conversation_id),
                user_id=self._user_id,
                customer_id=self._customer_id or None,
                search_query=[query],
                mode="accurate",
                include_conversation_context=False,
                **self._path("scope_path"),
            )
        return response.formatted_context or "No relevant memories found."

    @kernel_function(
        name="store_memory",
        description=(
            "Store an important fact, preference, or event about the user "
            "for future reference."
        ),
    )
    async def store_memory(
        self,
        content: Annotated[str, "Information to remember about the user"],
    ) -> str:
        async with wrap_sdk_errors_async(
            "semantic_kernel.store_memory", logger, user_id=self._user_id,
        ):
            result = await self._sdk.memories.create(
                document=content,
                user_id=self._user_id,
                customer_id=self._customer_id,
                **self._path("scope"),
            )
        return f"Memory stored (ingestion_id: {result.ingestion_id})"
