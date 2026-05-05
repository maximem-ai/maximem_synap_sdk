"""Synap plugin for Microsoft Semantic Kernel.

Exposes Synap memory as kernel functions (search_memory, store_memory)
that can be registered with a Semantic Kernel instance.
"""

import logging
from typing import Annotated, Optional

from semantic_kernel.functions import kernel_function

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import wrap_sdk_errors_async

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
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: str,
        customer_id: str = "",
        conversation_id: Optional[str] = None,
    ):
        if sdk is None:
            raise ValueError("SynapPlugin requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapPlugin requires a non-empty user_id")

        self._sdk = sdk
        self._user_id = user_id
        self._customer_id = customer_id
        self._conversation_id = conversation_id

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
                conversation_id=self._conversation_id,
                user_id=self._user_id,
                customer_id=self._customer_id or None,
                search_query=[query],
                mode="accurate",
                include_conversation_context=False,
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
            )
        return f"Memory stored (ingestion_id: {result.ingestion_id})"
