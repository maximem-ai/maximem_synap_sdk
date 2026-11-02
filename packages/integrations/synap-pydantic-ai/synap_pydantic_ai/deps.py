"""Synap dependency and tool helpers for Pydantic AI.

Provides SynapDeps (a dependency class) and helper functions for
registering Synap memory tools on a Pydantic AI Agent.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import ensure_ids_match_path, wrap_sdk_errors_async, coerce_conversation_id

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

    Deeper scope ladders: on an account whose ladder has levels beyond
    customer and user, pass ``scope_path`` (``{level_key: value}``, for example
    ``ScopeAddress.scope_path``); key order does not matter, the server orders
    by ladder position::

        deps = SynapDeps(
            sdk=sdk, user_id="dana", customer_id="acme",
            scope_path={"customer": "acme", "team": "payments", "user": "dana"},
        )

    It reaches ``fetch``, ``memories.create`` and the turns ``synap_run``
    reports, only when set. It is part of the deps, so the model cannot see or
    change it. The short-term and history helpers read through the conversation
    and take no path.
    """

    sdk: MaximemSynapSDK
    user_id: str
    customer_id: str = ""
    conversation_id: Optional[str] = None
    #: For an account with a deeper scope ladder: ``{level_key: value}`` (see
    #: ``ScopeAddress.scope_path``). Forwarded to every SDK call only when set.
    #: It comes from the developer who builds the deps, never from the model:
    #: the tools below take only ``query`` / ``content``.
    scope_path: Optional[Mapping[str, str]] = None

    def __post_init__(self) -> None:
        if self.sdk is None:
            raise ValueError("SynapDeps requires a non-None sdk")
        if not self.user_id:
            raise ValueError("SynapDeps requires a non-empty user_id")
        self.scope_path = dict(self.scope_path) if self.scope_path else None
        check_ids(self, "SynapDeps")


def check_ids(deps: "SynapDeps", where: str) -> None:
    """Refuse ids that disagree with the deps' own ``scope_path``.

    The server lets a path's ``user`` / ``customer`` level win over the flat ids,
    so deps naming one user next to another user's path would read and write as
    the path's user. Checked when the deps are built and again before each SDK
    call made from them (the fields can be reassigned, and the same agent runs
    with different deps per run).
    """
    ensure_ids_match_path(
        deps.scope_path, user_id=deps.user_id, customer_id=deps.customer_id or None,
        where=where)


def _path(deps: "SynapDeps", name: str) -> Dict[str, Any]:
    """``{name: path}`` when the deps carry a scope path, else nothing, so a
    call without one has exactly the keyword arguments it always had."""
    return {name: dict(deps.scope_path)} if deps.scope_path else {}


def register_synap_tools(agent) -> None:
    """Register search_memory and store_memory tools on a Pydantic AI Agent.

    Also registers a system prompt that auto-injects memory context, and
    installs the history processor that puts Synap's compaction in place of the
    earlier turns (see :func:`synap_history_processor`). The system prompt is
    re-evaluated on every run, including runs given a ``message_history``,
    which would otherwise keep the first run's text. It carries memories only:
    the conversation itself is the processor's job, so it is not sent twice.
    The system_prompt path remains best-effort (empty string on failure)
    because a prompt-enrichment outage should not crash the agent, but
    failures are now logged at ``ERROR`` instead of ``DEBUG``.
    Tool-invoked calls (search/store) surface SDK errors as
    :class:`SynapIntegrationError`.
    """
    from pydantic_ai import RunContext

    from synap_pydantic_ai.history import install_synap_history

    install_synap_history(agent)

    @agent.tool
    async def search_memory(ctx: RunContext[SynapDeps], query: str) -> str:
        """Search the user's memory for relevant context."""
        deps = ctx.deps
        check_ids(deps, "search_memory")
        async with wrap_sdk_errors_async(
            "pydantic_ai.search_memory", logger, user_id=deps.user_id,
        ):
            response = await deps.sdk.fetch(
                conversation_id=coerce_conversation_id(deps.conversation_id),
                user_id=deps.user_id,
                customer_id=deps.customer_id or None,
                search_query=[query],
                mode="accurate",
                include_conversation_context=False,
                **_path(deps, "scope_path"),
            )
        return response.formatted_context or "No relevant memories found."

    @agent.tool
    async def store_memory(ctx: RunContext[SynapDeps], content: str) -> str:
        """Store important information about the user."""
        deps = ctx.deps
        check_ids(deps, "store_memory")
        async with wrap_sdk_errors_async(
            "pydantic_ai.store_memory", logger, user_id=deps.user_id,
        ):
            result = await deps.sdk.memories.create(
                document=content,
                user_id=deps.user_id,
                customer_id=deps.customer_id,
                **_path(deps, "scope"),
            )
        return f"Memory stored (ingestion_id: {result.ingestion_id})"

    @agent.system_prompt(dynamic=True)
    async def inject_memory_context(ctx: RunContext[SynapDeps]) -> str:
        """Auto-inject memory context. Best-effort; logs at ERROR on failure."""
        deps = ctx.deps
        try:
            check_ids(deps, "inject_memory_context")
        except Exception as exc:  # noqa: BLE001 - a prompt callback never raises
            # Only the helper's fingerprints: the generic handler below logs the raw user_id.
            logger.error("pydantic_ai.inject_memory_context skipped: %s", exc)
            return ""
        try:
            response = await deps.sdk.fetch(
                conversation_id=coerce_conversation_id(deps.conversation_id),
                user_id=deps.user_id,
                customer_id=deps.customer_id or None,
                include_conversation_context=False,
                **_path(deps, "scope_path"),
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
