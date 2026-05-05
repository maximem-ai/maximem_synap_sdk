"""Synap dependency and tool helpers for Pydantic AI.

Provides SynapDeps (a dependency class) and helper functions for
registering Synap memory tools on a Pydantic AI Agent.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import wrap_sdk_errors_async

logger = logging.getLogger(__name__)


@dataclass
class SynapDeps:
    """Pydantic AI dependency holding Synap SDK and scope identifiers.

    Example::

        from pydantic_ai import Agent
        from synap_pydantic_ai import SynapDeps, register_synap_tools

        agent = Agent('openai:gpt-4o', deps_type=SynapDeps)
        register_synap_tools(agent)
        result = agent.run_sync(
            "What do you know about me?",
            deps=SynapDeps(sdk=sdk, user_id="u1", customer_id="c1"),
        )
    """

    sdk: MaximemSynapSDK
    user_id: str
    customer_id: str = ""
    conversation_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.sdk is None:
            raise ValueError("SynapDeps requires a non-None sdk")
        if not self.user_id:
            raise ValueError("SynapDeps requires a non-empty user_id")


def register_synap_tools(agent) -> None:
    """Register search_memory and store_memory tools on a Pydantic AI Agent.

    Also registers a system prompt that auto-injects memory context.
    The system_prompt path remains best-effort (empty string on failure)
    because a prompt-enrichment outage should not crash the agent, but
    failures are now logged at ``ERROR`` instead of ``DEBUG``.
    Tool-invoked calls (search/store) surface SDK errors as
    :class:`SynapIntegrationError`.
    """
    from pydantic_ai import RunContext

    @agent.tool
    async def search_memory(ctx: RunContext[SynapDeps], query: str) -> str:
        """Search the user's memory for relevant context."""
        deps = ctx.deps
        async with wrap_sdk_errors_async(
            "pydantic_ai.search_memory", logger, user_id=deps.user_id,
        ):
            response = await deps.sdk.fetch(
                conversation_id=deps.conversation_id,
                user_id=deps.user_id,
                customer_id=deps.customer_id or None,
                search_query=[query],
                mode="accurate",
                include_conversation_context=False,
            )
        return response.formatted_context or "No relevant memories found."

    @agent.tool
    async def store_memory(ctx: RunContext[SynapDeps], content: str) -> str:
        """Store important information about the user."""
        deps = ctx.deps
        async with wrap_sdk_errors_async(
            "pydantic_ai.store_memory", logger, user_id=deps.user_id,
        ):
            result = await deps.sdk.memories.create(
                document=content,
                user_id=deps.user_id,
                customer_id=deps.customer_id,
            )
        return f"Memory stored (ingestion_id: {result.ingestion_id})"

    @agent.system_prompt
    async def inject_memory_context(ctx: RunContext[SynapDeps]) -> str:
        """Auto-inject memory context. Best-effort; logs at ERROR on failure."""
        deps = ctx.deps
        try:
            response = await deps.sdk.fetch(
                conversation_id=deps.conversation_id,
                user_id=deps.user_id,
                customer_id=deps.customer_id or None,
            )
            if response.formatted_context:
                return f"Relevant user context:\n{response.formatted_context}"
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "pydantic_ai.inject_memory_context failed "
                "user_id=%s error=%s",
                deps.user_id, exc, exc_info=True,
            )
        return ""
