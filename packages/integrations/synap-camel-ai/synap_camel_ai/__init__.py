"""Synap memory integration for CAMEL-AI.

CAMEL exposes a first-class pluggable memory interface: ``ChatAgent(memory=...)``
accepts any :class:`camel.memories.AgentMemory`. :class:`SynapAgentMemory` plugs
Synap in there — but as an **augmentation** of CAMEL's own conversation history,
not a replacement for it. CAMEL's ``AgentMemory.get_context()`` is the sole source
of the model's input, so the memory must return the real conversation (system +
user + assistant); :class:`SynapAgentMemory` therefore subclasses
:class:`~camel.memories.ChatHistoryMemory` (which keeps that history) and adds two
things on top:

- **Recall**: ``retrieve()`` prepends Synap's long-term memories (fetched with the
  latest user turn as the query) ahead of the live conversation.
- **Persistence**: completed turns are ingested into Synap for server-side
  extraction, so future sessions remember them.

Typical wiring::

    from camel.agents import ChatAgent
    from camel.models import ModelFactory
    from maximem_synap import MaximemSynapSDK
    from synap_camel_ai import SynapAgentMemory

    sdk = MaximemSynapSDK(api_key="sk-...")
    memory = SynapAgentMemory(sdk, user_id="alice", customer_id="acme")

    agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=ModelFactory.create(...),
        memory=memory,           # constructor only — not agent.memory = ...
    )
    response = agent.step("What did we decide about the rollout?")

Reads degrade (a Synap blip returns no recall rather than crashing the turn);
the long-term persistence write is best-effort by default (``on_error="fallback"``)
because it runs inside the agent loop. The explicit ``store_memory`` tool raises on
failure. See :class:`SynapAgentMemory`, :func:`create_synap_tools`, and
:func:`synap_st_system_message`.
"""

from synap_camel_ai.memory import SynapAgentMemory
from synap_camel_ai.short_term import synap_st_system_message
from synap_camel_ai.tools import create_synap_tools

__all__ = ["SynapAgentMemory", "create_synap_tools", "synap_st_system_message"]
