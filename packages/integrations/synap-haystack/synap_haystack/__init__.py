"""Synap memory integration for Haystack.

Store-centric, matching the official ``mem0-haystack`` shape:

- :class:`SynapMemoryStore` — the memory store (a plain object, not a component)
  that owns all Synap SDK interaction. ``add_memories`` / ``search_memories`` /
  ``search_memories_as_single_message`` / ``delete_*`` + ``to_dict`` / ``from_dict``.
- :class:`SynapMemoryRetriever` — ``@component`` returning ``ChatMessage``s
  (Mem0-shaped chat read path).
- :class:`SynapRetriever` — ``@component`` returning ``Document``s (RAG read path).
- :class:`SynapMemoryWriter` — ``@component`` that records conversation turns.

The components accept a ready ``store=...`` or an ``sdk=...`` + scope kwargs.
"""

from synap_haystack.store import SynapMemoryStore
from synap_haystack.retriever import SynapMemoryRetriever, SynapRetriever
from synap_haystack.writer import SynapMemoryWriter

__all__ = [
    "SynapMemoryStore",
    "SynapMemoryRetriever",
    "SynapRetriever",
    "SynapMemoryWriter",
]
