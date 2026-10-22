"""Synap memory integration for Haystack.

Store-centric, matching the official ``mem0-haystack`` shape:

- :class:`SynapMemoryStore` — the memory store (a plain object, not a component)
  that owns all Synap SDK interaction. ``add_memories`` / ``search_memories`` /
  ``search_memories_as_single_message`` / ``delete_*`` + ``to_dict`` / ``from_dict``.
  It is also a Haystack ``ChatMessageStore`` (``write_messages`` /
  ``retrieve_messages`` / ``count_messages``), so an experimental ``Agent`` takes it
  as ``chat_message_store=`` and loads Synap's compaction and the turns after it
  instead of a ten-message window. Turns are written on the gRPC stream, which
  the first one opens (``SYNAP_AUTO_LISTEN=false`` turns that off).
- :class:`SynapMemoryRetriever` — ``@component`` returning ``ChatMessage``s
  (Mem0-shaped chat read path).
- :class:`SynapRetriever` — ``@component`` returning ``Document``s (RAG read path).
- :class:`SynapMemoryWriter` — ``@component`` that records conversation turns.
- :class:`SynapHistoryHook` / :class:`SynapRecordHook` — hooks for haystack-ai 3's
  ``Agent(hooks=...)``: Synap's compaction in place of the earlier turns
  (``before_llm``) and the run recorded on the stream (``after_run``).
- :class:`SynapShortTermContext` — ``@component`` emitting Synap's compaction of
  the conversation for a prompt builder.

The components accept a ready ``store=...`` or an ``sdk=...`` + scope kwargs.
"""

from synap_haystack.store import SynapMemoryStore
from synap_haystack.retriever import SynapMemoryRetriever, SynapRetriever
from synap_haystack.writer import SynapMemoryWriter
from synap_haystack.short_term import SynapShortTermContext

try:  # needs haystack-ai 3's Agent hooks only at run time; importing is safe
    from synap_haystack.hooks import SynapHistoryHook, SynapRecordHook
except ImportError:  # pragma: no cover
    SynapHistoryHook = SynapRecordHook = None  # type: ignore[assignment,misc]

__all__ = [
    "SynapMemoryStore",
    "SynapMemoryRetriever",
    "SynapRetriever",
    "SynapMemoryWriter",
    "SynapShortTermContext",
    "SynapHistoryHook",
    "SynapRecordHook",
]
