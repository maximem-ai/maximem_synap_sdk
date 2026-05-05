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
    from synap_agno import SynapDb

    sdk = MaximemSynapSDK(api_key="sk-...")
    db = SynapDb(sdk, customer_id="acme")

    agent = Agent(
        db=db,
        enable_user_memories=True,
        # ... your model + instructions
    )
    agent.run("Remember I like tea", user_id="alice")

See :class:`SynapDb` for the full list of user-memory methods and their
behaviour (reads → sdk.fetch, writes → sdk.memories.create, deletes →
warn + no-op because Synap has no public delete API).
"""

from synap_agno.db import SynapDb

__all__ = ["SynapDb"]
