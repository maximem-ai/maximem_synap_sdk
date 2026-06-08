"""Synap memory integration for CrewAI.

Provides a StorageBackend that plugs into CrewAI's unified Memory class,
delegating persistence and retrieval to Synap's cloud platform.

Example:
    from crewai.memory import Memory
    from synap_crewai import SynapStorageBackend

    backend = SynapStorageBackend(sdk=sdk, user_id="u1", customer_id="c1")
    memory = Memory(storage=backend)
    crew = Crew(agents=agents, tasks=tasks, memory=memory)
"""

from synap_crewai.short_term import build_synap_st_backstory
from synap_crewai.storage import SynapStorageBackend

__all__ = ["SynapStorageBackend", "build_synap_st_backstory"]
