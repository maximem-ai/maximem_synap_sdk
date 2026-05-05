"""Synap tools for LangChain agents.

Provides tools that agents can explicitly invoke to search or store
memories. Use these when you want the agent to decide when to access
memory, rather than doing it automatically on every turn.
"""

import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import run_async, wrap_sdk_errors_async

logger = logging.getLogger(__name__)


class _SearchInput(BaseModel):
    query: str = Field(description="Natural language search query for user memory")


class _StoreInput(BaseModel):
    content: str = Field(
        description="Important information to remember about the user"
    )


class SynapSearchTool(BaseTool):
    """Search Synap memory for relevant context about the user.

    Example::

        tool = SynapSearchTool(sdk=sdk, user_id="user-456")
        result = tool.invoke("What are the user's preferences?")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "search_memory"
    description: str = (
        "Search the user's memory for relevant context. Use this when you need "
        "to recall past conversations, user preferences, facts about the user, "
        "or previous experiences. Input should be a natural language search query."
    )
    args_schema: Type[BaseModel] = _SearchInput

    sdk: MaximemSynapSDK
    user_id: str
    customer_id: Optional[str] = None
    conversation_id: Optional[str] = None
    mode: str = "accurate"
    max_results: int = 10

    def _run(self, query: str, **kwargs: Any) -> str:
        return run_async(self._arun(query, **kwargs))

    async def _arun(self, query: str, **kwargs: Any) -> str:
        async with wrap_sdk_errors_async(
            "langchain.search_memory", logger, user_id=self.user_id
        ):
            response = await self.sdk.fetch(
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                customer_id=self.customer_id,
                search_query=[query],
                mode=self.mode,
                max_results=self.max_results,
                include_conversation_context=False,
            )
        return response.formatted_context or "No relevant memories found."


class SynapStoreTool(BaseTool):
    """Store important information in Synap memory.

    Example::

        tool = SynapStoreTool(sdk=sdk, user_id="user-456")
        result = tool.invoke("User prefers dark mode and concise responses")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "store_memory"
    description: str = (
        "Store an important fact, preference, or event about the user for "
        "future reference. Input should be a clear statement of what to remember. "
        "The system will automatically extract and categorize the information."
    )
    args_schema: Type[BaseModel] = _StoreInput

    sdk: MaximemSynapSDK
    user_id: str
    customer_id: Optional[str] = None

    def _run(self, content: str, **kwargs: Any) -> str:
        return run_async(self._arun(content, **kwargs))

    async def _arun(self, content: str, **kwargs: Any) -> str:
        async with wrap_sdk_errors_async(
            "langchain.store_memory", logger, user_id=self.user_id
        ):
            result = await self.sdk.memories.create(
                document=content,
                user_id=self.user_id,
                customer_id=self.customer_id or "",
            )
        return f"Memory stored (ingestion_id: {result.ingestion_id})"
