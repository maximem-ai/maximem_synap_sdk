"""Synap memory integration for CrewAI.

Provides a StorageBackend that plugs into CrewAI's unified Memory class,
delegating persistence and retrieval to Synap's cloud platform.

Example:
    from synap_crewai import SynapMemory, SynapStorageBackend

    backend = SynapStorageBackend(sdk=sdk, user_id="u1", customer_id="c1")
    memory = SynapMemory(storage=backend)
    crew = Crew(agents=agents, tasks=tasks, memory=memory)

To report the run itself, build a :class:`SynapCrewAIListener` before kickoff.
It opens the gRPC stream on the first event (``SYNAP_AUTO_LISTEN=false`` turns
that off) and, once Synap has compacted the conversation, puts that compaction
in front of the model in place of the history CrewAI built.
"""

from synap_crewai.memory import HashingEmbedder, SynapMemory
from synap_crewai.short_term import build_synap_st_backstory
from synap_crewai.storage import SynapStorageBackend
from synap_crewai.stream import SynapCrewAIListener, report_user_turn

__all__ = [
    "SynapMemory",
    "HashingEmbedder",
    "SynapStorageBackend",
    "build_synap_st_backstory",
    "SynapCrewAIListener",
    "report_user_turn",
]
