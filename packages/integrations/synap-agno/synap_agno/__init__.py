"""Synap integration for Agno.

Agno 2.x unifies every persistence concern — sessions, traces, evals, user
memories, knowledge, metrics, culture — under a single :class:`BaseDb`
interface (46+ abstract methods). Synap only backs user memories natively,
so :class:`SynapDb` extends Agno's :class:`InMemoryDb` and overrides the
user-memory methods to route through Synap. Sessions, traces, evals, and
the rest stay in-process (exactly what InMemoryDb already does).

Typical wiring::

    from agno.agent import Agent
    from maximem_synap import MaximemSynapSDK
    from synap_agno import SynapDb, create_synap_hooks

    sdk = MaximemSynapSDK(api_key="sk-...")
    db = SynapDb(sdk, customer_id="acme")

    agent = Agent(
        db=db,
        update_memory_on_run=True,   # Agno 2.5+; replaces enable_user_memories (removed in 3.0)
        **create_synap_hooks(sdk, customer_id="acme"),
        # ... your model + instructions
    )
    agent.run("Remember I like tea", user_id="alice")

For async agents (``await agent.arun(...)``) use :class:`AsyncSynapDb` instead of
:class:`SynapDb`: Agno awaits it directly instead of calling a sync database from
inside the running event loop.

See :class:`SynapDb` for the full list of user-memory methods and their
behaviour (reads → sdk.fetch, writes → sdk.memories.create, deletes →
sdk.memories.delete for Synap ids, warn + no-op otherwise).

``create_synap_hooks`` is a separate concern from the db: it reports the run
itself (the question, the reply, every tool call and result, and the model's
reasoning) on Synap's live gRPC stream, which is what the anticipation agent
reads. The first run opens that stream itself (``SYNAP_AUTO_LISTEN=false`` stops
it); when the stream cannot carry them the user and assistant turns go over REST
(``conversation.record_message``), never both. The same pre-hook puts Synap's
compaction in place of the history Agno sends once the conversation has been
compacted (:mod:`synap_agno.history`). See :mod:`synap_agno.hooks`.
"""

from synap_agno.async_db import AsyncSynapDb
from synap_agno.db import SynapDb
from synap_agno.hooks import create_synap_async_hooks, create_synap_hooks
from synap_agno.short_term import synap_st_instructions

__all__ = [
    "SynapDb",
    "AsyncSynapDb",
    "create_synap_hooks",
    "create_synap_async_hooks",
    "synap_st_instructions",
]
