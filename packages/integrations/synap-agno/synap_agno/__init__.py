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
        enable_user_memories=True,
        **create_synap_hooks(sdk, customer_id="acme"),
        # ... your model + instructions
    )
    agent.run("Remember I like tea", user_id="alice")

See :class:`SynapDb` for the full list of user-memory methods and their
behaviour (reads → sdk.fetch, writes → sdk.memories.create, deletes →
warn + no-op because Synap has no public delete API).

``create_synap_hooks`` is a separate concern from the db: it reports the run
itself (the question, the reply, every tool call and result, and the model's
reasoning) on Synap's live gRPC stream, which is what the anticipation agent
reads. It is silent unless ``sdk.instance.listen()`` is running, and it never
writes over REST. See :mod:`synap_agno.hooks`.
"""

from synap_agno.db import SynapDb
from synap_agno.hooks import create_synap_async_hooks, create_synap_hooks
from synap_agno.short_term import synap_st_instructions

__all__ = [
    "SynapDb",
    "create_synap_hooks",
    "create_synap_async_hooks",
    "synap_st_instructions",
]
