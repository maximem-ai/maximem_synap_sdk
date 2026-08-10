"""Synap integration for Microsoft Agent Framework (MAF).

MAF has two distinct product surfaces, and this package covers both.

The agent SDK
-------------

Context-engineering plug points on a normal MAF agent:

- :class:`SynapContextProvider` — a ``ContextProvider``. Injects Synap-sourced
  context into the agent's instructions each turn via ``before_run``, and
  records the turn back to Synap via ``after_run``. Mirrors
  ``agent_framework.mem0.Mem0ContextProvider``.
- :class:`SynapHistoryProvider` — a ``HistoryProvider``. Persists the
  conversation message log to Synap and loads it back.
- :class:`SynapShortTermContextProvider` — compacted conversation history,
  refreshed each turn.

::

    from agent_framework import InMemoryHistoryProvider
    from synap_microsoft_agent import SynapContextProvider

    agent = client.as_agent(
        name="MemoryAgent",
        instructions="You are a helpful assistant.",
        context_providers=[
            SynapContextProvider(sdk=sdk, user_id="alice", customer_id="acme"),
            InMemoryHistoryProvider(load_messages=True),
        ],
    )

The Agent Harness
-----------------

The harness (``create_harness_agent``) has its own memory subsystem with two
pluggable storage seams, both public constructor arguments:

- :class:`SynapMemoryStore` — a ``MemoryStore``. Backs the harness's topic
  notebook: ``MEMORY.md``, topic records, extraction, consolidation.
- :class:`SynapAgentFileStore` — an ``AgentFileStore``. Backs the
  ``file_memory_*`` tools and the agent's file access.
- :func:`create_synap_harness_memory` — builds the memory provider wired
  correctly. Prefer it over assembling the pieces yourself.

::

    from agent_framework import create_harness_agent
    from synap_microsoft_agent import (
        SynapAgentFileStore,
        create_synap_harness_memory,
    )

    agent = create_harness_agent(
        client,
        history_provider=create_synap_harness_memory(
            sdk, user_id="alice", customer_id="acme",
        ),
        file_memory_store=SynapAgentFileStore(sdk, user_id="alice"),
    )

Two things to know before wiring the harness. ``create_harness_agent`` takes
exactly one ``history_provider``, and both ``SynapHistoryProvider`` and the
harness memory provider are ``HistoryProvider``s — passing both silently drops
one. And every harness API is marked experimental upstream, behind private
module paths, so it will churn; the harness surfaces here need
``agent-framework>=1.13`` and are imported lazily so the SDK surfaces keep
working on ``>=1.0``.
"""

from typing import TYPE_CHECKING, Any

from synap_microsoft_agent.context_provider import SynapContextProvider
from synap_microsoft_agent.history_provider import SynapHistoryProvider
from synap_microsoft_agent.short_term import SynapShortTermContextProvider

# The harness names are resolved on first access rather than at import, so an
# install on agent-framework 1.0 keeps working for the SDK surfaces. Reaching
# for a harness name on such an install raises an ImportError naming the
# version to upgrade to. Same shape as MAF's own optional-connector shims.
_HARNESS_IMPORTS = {
    "SynapMemoryStore": "synap_microsoft_agent.memory_store",
    "TopicRecordStore": "synap_microsoft_agent.memory_store",
    "InMemoryTopicRecordStore": "synap_microsoft_agent.memory_store",
    "SynapMemoryContextProvider": "synap_microsoft_agent.harness_memory",
    "create_synap_harness_memory": "synap_microsoft_agent.harness_memory",
    "session_conversation_id": "synap_microsoft_agent.harness_memory",
    "SynapAgentFileStore": "synap_microsoft_agent.file_store",
}

if TYPE_CHECKING:  # pragma: no cover — import-time cost is the point
    from synap_microsoft_agent.file_store import SynapAgentFileStore
    from synap_microsoft_agent.harness_memory import (
        SynapMemoryContextProvider,
        create_synap_harness_memory,
        session_conversation_id,
    )
    from synap_microsoft_agent.memory_store import (
        InMemoryTopicRecordStore,
        SynapMemoryStore,
        TopicRecordStore,
    )


def __getattr__(name: str) -> Any:
    module_path = _HARNESS_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_path), name)


def __dir__() -> list:
    return sorted(__all__)


__all__ = [
    # SDK surfaces
    "SynapContextProvider",
    "SynapHistoryProvider",
    "SynapShortTermContextProvider",
    # Harness surfaces
    "SynapMemoryStore",
    "SynapMemoryContextProvider",
    "SynapAgentFileStore",
    "create_synap_harness_memory",
    "session_conversation_id",
    "TopicRecordStore",
    "InMemoryTopicRecordStore",
]
