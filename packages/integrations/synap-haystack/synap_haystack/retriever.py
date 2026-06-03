"""Synap retriever components for Haystack pipelines.

Two read components, both thin wrappers over :class:`SynapMemoryStore` (the store
owns all SDK interaction — see ``store.py``):

- :class:`SynapRetriever` — RAG-shaped: ``run(query)`` → ``documents`` as a list
  of Haystack ``Document``s. Use when memory feeds a document-oriented pipeline.
- :class:`SynapMemoryRetriever` — Mem0-shaped: ``run(query)`` → ``messages`` as a
  list of ``ChatMessage``s, matching ``mem0_haystack.Mem0MemoryRetriever``. Use
  when memory feeds a chat/agent pipeline.

Either accept a ready ``store=SynapMemoryStore(...)`` or the ``sdk=...`` +
scope kwargs (a store is then built internally for convenience).
"""

import logging
from typing import Dict, List, Optional

from haystack import Document, component
from haystack.dataclasses import ChatMessage

from maximem_synap import MaximemSynapSDK
from synap_haystack.store import SynapMemoryStore

logger = logging.getLogger(__name__)


def _resolve_store(
    store: Optional[SynapMemoryStore],
    sdk: Optional[MaximemSynapSDK],
    user_id: str,
    customer_id: str,
    conversation_id: Optional[str],
    mode: str,
    max_results: int,
    *,
    component_name: str,
) -> SynapMemoryStore:
    if store is not None:
        return store
    if sdk is None:
        raise ValueError(f"{component_name} requires either a store or an sdk")
    if not user_id and not customer_id:
        raise ValueError(
            f"{component_name} requires a non-empty user_id or customer_id"
        )
    return SynapMemoryStore(
        sdk=sdk,
        user_id=user_id or None,
        customer_id=customer_id,
        conversation_id=conversation_id,
        mode=mode,
        max_results=max_results,
    )


@component
class SynapRetriever:
    """Haystack component that retrieves memory from Synap as ``Document``s.

    Example::

        retriever = SynapRetriever(sdk=sdk, user_id="u1")
        pipeline.add_component("memory", retriever)
    """

    def __init__(
        self,
        sdk: Optional[MaximemSynapSDK] = None,
        user_id: str = "",
        customer_id: str = "",
        conversation_id: Optional[str] = None,
        mode: str = "accurate",
        max_results: int = 20,
        *,
        store: Optional[SynapMemoryStore] = None,
    ):
        self.store = _resolve_store(
            store, sdk, user_id, customer_id, conversation_id, mode, max_results,
            component_name="SynapRetriever",
        )

    @component.output_types(documents=List[Document])
    def run(self, query: str) -> Dict[str, List[Document]]:
        return {"documents": self.store.search_documents(query=query)}


@component
class SynapMemoryRetriever:
    """Haystack component that retrieves memory from Synap as ``ChatMessage``s.

    Mirrors ``mem0_haystack.Mem0MemoryRetriever``: drop it into a chat pipeline
    to surface long-term memory as messages.

    Example::

        retriever = SynapMemoryRetriever(store=store)
        pipeline.add_component("memory", retriever)
    """

    def __init__(
        self,
        store: Optional[SynapMemoryStore] = None,
        *,
        sdk: Optional[MaximemSynapSDK] = None,
        user_id: str = "",
        customer_id: str = "",
        conversation_id: Optional[str] = None,
        mode: str = "accurate",
        max_results: int = 20,
    ):
        self.store = _resolve_store(
            store, sdk, user_id, customer_id, conversation_id, mode, max_results,
            component_name="SynapMemoryRetriever",
        )

    @component.output_types(messages=List[ChatMessage])
    def run(self, query: str) -> Dict[str, List[ChatMessage]]:
        return {"messages": self.store.search_memories(query=query)}
