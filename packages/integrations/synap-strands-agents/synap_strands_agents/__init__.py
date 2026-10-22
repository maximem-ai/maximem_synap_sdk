"""Synap memory integration for Strands Agents.

Four surfaces, each mapped onto a native Strands extension point:

- :class:`SynapMemoryStore` — implements Strands' ``MemoryStore`` protocol so
  ``MemoryManager`` uses Synap as the agent's long-term memory backend
  (recall + automatic prompt injection + server-side extraction).
- :class:`SynapShortTermHook` — a ``HookProvider`` that puts Synap's
  compaction in place of the earlier turns the model is sent, once Synap has
  compacted the conversation (model-call middleware; ``agent.messages`` is not
  rewritten).
- :func:`create_synap_tools` — explicit ``search_memory`` / ``store_memory``
  ``@tool`` functions for agents that don't adopt ``MemoryManager``.
- :class:`SynapStreamHook` — a ``HookProvider`` that writes the agent's turns,
  tool calls and results to Synap on its gRPC stream (opened on the first turn;
  ``SYNAP_AUTO_LISTEN=false`` keeps it on REST).

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
for the gRPC stream contract.
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
