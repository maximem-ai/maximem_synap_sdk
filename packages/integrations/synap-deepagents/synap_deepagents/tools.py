"""Synap tools for deepagents.

``create_deep_agent(tools=[...])`` accepts LangChain ``BaseTool`` instances, so
these are ordinary LangChain tools. Use them when you want the agent to decide
when to reach for memory, rather than mounting :class:`SynapBackend` and having
memory show up as files.

The two surfaces are complementary, not alternatives:

- :class:`SynapBackend` makes memory *look like a filesystem*, so the agent's
  built-in ``read_file`` / ``grep`` tools reach it.
- These tools make memory *look like memory*, with names the model already
  understands.

Mounting both is fine and is often the best setup: the backend covers automatic
recall, the tools cover deliberate lookups.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import run_async, wrap_sdk_errors_async

logger = logging.getLogger(__name__)


class _SearchInput(BaseModel):
    query: str = Field(
        description="Natural language search query for the user's memory"
    )


class _StoreInput(BaseModel):
    content: str = Field(
        description="Important information to remember about the user"
    )


class SynapSearchTool(BaseTool):
    """Search Synap memory for context relevant to a question.

    Example::

        from deepagents import create_deep_agent
        from synap_deepagents import SynapSearchTool

        agent = create_deep_agent(
            model="anthropic:claude-sonnet-5",
            tools=[SynapSearchTool(sdk=sdk, user_id="alice")],
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "search_memory"
    description: str = (
        "Search the user's long-term memory for relevant context. Use this to "
        "recall past conversations, stated preferences, facts about the user, "
        "or earlier decisions. Input should be a natural language question."
    )
    args_schema: Type[BaseModel] = _SearchInput

    sdk: MaximemSynapSDK
    user_id: Optional[str] = None
    customer_id: Optional[str] = None
    conversation_id: Optional[str] = None
    mode: str = "accurate"
    max_results: int = 10

    def _run(self, query: str, **kwargs: Any) -> str:
        return run_async(self._arun(query, **kwargs))

    async def _arun(self, query: str, **kwargs: Any) -> str:
        async with wrap_sdk_errors_async(
            "deepagents.search_memory", logger, user_id=self.user_id
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
    """Store a fact in Synap memory.

    Example::

        from synap_deepagents import SynapStoreTool

        tool = SynapStoreTool(sdk=sdk, user_id="alice")
        tool.invoke("User prefers TypeScript over JavaScript")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "store_memory"
    description: str = (
        "Store an important fact, preference, or decision about the user for "
        "future sessions. Input should be a clear, self-contained statement. "
        "Synap extracts and categorizes it automatically — write plain prose, "
        "not structured data."
    )
    args_schema: Type[BaseModel] = _StoreInput

    sdk: MaximemSynapSDK
    user_id: Optional[str] = None
    customer_id: Optional[str] = None
    document_type: str = "document"
    ingest_mode: str = "fast"

    def _run(self, content: str, **kwargs: Any) -> str:
        return run_async(self._arun(content, **kwargs))

    async def _arun(self, content: str, **kwargs: Any) -> str:
        async with wrap_sdk_errors_async(
            "deepagents.store_memory", logger, user_id=self.user_id
        ):
            result = await self.sdk.memories.create(
                document=content,
                user_id=self.user_id,
                customer_id=self.customer_id,
                document_type=self.document_type,
                mode=self.ingest_mode,
            )
        return f"Memory accepted for storage (ingestion_id: {result.ingestion_id})"


__all__ = ["SynapSearchTool", "SynapStoreTool"]
