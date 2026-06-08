"""Synap integration for Microsoft Agent Framework (MAF).

Two plug points into MAF's context-engineering pipeline:

- :class:`SynapContextProvider` — subclass of ``ContextProvider``. Injects
  Synap-sourced context into the agent's instructions on every turn via
  ``before_run``, and records each turn back to Synap via ``after_run``.
  Pattern mirrors ``agent_framework.mem0.Mem0ContextProvider``.

- :class:`SynapHistoryProvider` — subclass of ``HistoryProvider``. Persists
  the conversation message log to Synap via ``sdk.conversation.record_message``
  and loads it back via ``sdk.conversation.context.get_context_for_prompt``.

Typical wiring::

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
"""

from synap_microsoft_agent.context_provider import SynapContextProvider
from synap_microsoft_agent.history_provider import SynapHistoryProvider
from synap_microsoft_agent.short_term import SynapShortTermContextProvider

__all__ = [
    "SynapContextProvider",
    "SynapHistoryProvider",
    "SynapShortTermContextProvider",
]
