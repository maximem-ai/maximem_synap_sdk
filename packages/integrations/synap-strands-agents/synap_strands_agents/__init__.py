"""Synap memory integration for Strands Agents.

Four surfaces, each mapped onto a native Strands extension point:

- :class:`SynapMemoryStore` — implements Strands' ``MemoryStore`` protocol so
  ``MemoryManager`` uses Synap as the agent's long-term memory backend
  (recall + automatic prompt injection + server-side extraction).
- :class:`SynapShortTermHook` — a ``HookProvider`` that injects Synap
  short-term / working-memory context before each invocation.
- :func:`create_synap_tools` — explicit ``search_memory`` / ``store_memory``
  ``@tool`` functions for agents that don't adopt ``MemoryManager``.
- :class:`SynapStreamHook` — a ``HookProvider`` that feeds the agent's turns
  and tool-call intent onto Synap's gRPC Listen / anticipation stream.

Typical wiring::

    from strands import Agent
    from strands.memory import MemoryManager
    from maximem_synap import MaximemSynapSDK
    from synap_strands_agents import SynapMemoryStore, SynapShortTermHook

    sdk = MaximemSynapSDK(api_key="sk-...")
    store = SynapMemoryStore(sdk, user_id="alice", customer_id="acme")

    agent = Agent(
        system_prompt="You are a helpful assistant.",
        memory_manager=MemoryManager(stores=[store]),
        hooks=[SynapShortTermHook(sdk, conversation_id="conv_abc")],
    )

See each surface's docstring for its error policy and the ``stream.py`` module
for the gRPC Listen participation contract (the app owns ``sdk.instance.listen()``).
"""

from synap_strands_agents.short_term import SynapShortTermHook
from synap_strands_agents.store import SynapMemoryStore
from synap_strands_agents.stream import SynapStreamHook
from synap_strands_agents.tools import create_synap_tools

__all__ = [
    "SynapMemoryStore",
    "SynapShortTermHook",
    "SynapStreamHook",
    "create_synap_tools",
]
