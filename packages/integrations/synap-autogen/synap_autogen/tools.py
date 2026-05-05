"""Synap tools for AutoGen agents.

Provides BaseTool implementations for memory search and storage
that work with AutoGen's tool calling interface.

AutoGen threads a :class:`CancellationToken` through every tool call.
The original implementation accepted the token for signature
compatibility but never consulted it, so cancelling an agent run did
not cancel the in-flight Synap SDK call. This module now:

- Short-circuits with :class:`asyncio.CancelledError` when the token is
  already set at tool entry.
- Wraps the SDK call in a task and links the token so cancellation
  propagates mid-flight.
- Surfaces SDK errors as :class:`SynapIntegrationError` instead of
  leaking raw SDK exceptions into AutoGen.
"""

import asyncio
import logging
from typing import Optional, TypeVar

from autogen_core import CancellationToken
from autogen_core.tools import BaseTool
from pydantic import BaseModel, Field

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import wrap_sdk_errors_async

logger = logging.getLogger(__name__)

_R = TypeVar("_R")


async def _await_with_cancellation(
    coro,
    cancellation_token: CancellationToken,
) -> object:
    """Run ``coro`` as a task linked to ``cancellation_token``.

    If the token is already cancelled, raise immediately without
    starting the task. Otherwise, link the task to the token so that
    ``cancellation_token.cancel()`` cancels the underlying SDK call.
    """
    if cancellation_token.is_cancelled():
        raise asyncio.CancelledError("cancellation_token set before tool ran")

    task = asyncio.ensure_future(coro)
    cancellation_token.link_future(task)
    return await task


class SearchMemoryArgs(BaseModel):
    query: str = Field(description="Natural language search query for user memory")


class SearchMemoryResult(BaseModel):
    context: str = Field(description="Formatted memory context")


class StoreMemoryArgs(BaseModel):
    content: str = Field(description="Information to remember about the user")


class StoreMemoryResult(BaseModel):
    ingestion_id: str = Field(description="ID of the ingestion job")
    message: str = Field(description="Confirmation message")


class SynapSearchTool(BaseTool[SearchMemoryArgs, SearchMemoryResult]):
    """AutoGen tool for searching Synap memory."""

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: str,
        customer_id: str = "",
        conversation_id: Optional[str] = None,
        mode: str = "accurate",
    ):
        if sdk is None:
            raise ValueError("SynapSearchTool requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapSearchTool requires a non-empty user_id")

        super().__init__(
            args_type=SearchMemoryArgs,
            return_type=SearchMemoryResult,
            name="search_memory",
            description=(
                "Search the user's memory for relevant context. Use when you need "
                "to recall past conversations, preferences, or facts about the user."
            ),
        )
        self._sdk = sdk
        self._user_id = user_id
        self._customer_id = customer_id
        self._conversation_id = conversation_id
        self._mode = mode

    async def run(
        self, args: SearchMemoryArgs, cancellation_token: CancellationToken
    ) -> SearchMemoryResult:
        async with wrap_sdk_errors_async(
            "autogen.search_memory",
            logger,
            user_id=self._user_id,
        ):
            response = await _await_with_cancellation(
                self._sdk.fetch(
                    conversation_id=self._conversation_id,
                    user_id=self._user_id,
                    customer_id=self._customer_id or None,
                    search_query=[args.query],
                    mode=self._mode,
                    include_conversation_context=False,
                ),
                cancellation_token,
            )
        return SearchMemoryResult(
            context=response.formatted_context or "No relevant memories found."
        )


class SynapStoreTool(BaseTool[StoreMemoryArgs, StoreMemoryResult]):
    """AutoGen tool for storing information in Synap memory."""

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: str,
        customer_id: str = "",
    ):
        if sdk is None:
            raise ValueError("SynapStoreTool requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapStoreTool requires a non-empty user_id")

        super().__init__(
            args_type=StoreMemoryArgs,
            return_type=StoreMemoryResult,
            name="store_memory",
            description=(
                "Store an important fact, preference, or event about the user "
                "for future reference."
            ),
        )
        self._sdk = sdk
        self._user_id = user_id
        self._customer_id = customer_id

    async def run(
        self, args: StoreMemoryArgs, cancellation_token: CancellationToken
    ) -> StoreMemoryResult:
        async with wrap_sdk_errors_async(
            "autogen.store_memory",
            logger,
            user_id=self._user_id,
        ):
            result = await _await_with_cancellation(
                self._sdk.memories.create(
                    document=args.content,
                    user_id=self._user_id,
                    customer_id=self._customer_id,
                ),
                cancellation_token,
            )
        return StoreMemoryResult(
            ingestion_id=str(result.ingestion_id),
            message=f"Memory stored (ingestion_id: {result.ingestion_id})",
        )
