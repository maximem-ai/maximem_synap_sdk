"""Synap retriever components for Haystack pipelines.

Two read components, both thin wrappers over :class:`SynapMemoryStore` (the store
owns all SDK interaction — see ``store.py``):

- :class:`SynapRetriever` — RAG-shaped: ``run(query)`` → ``documents`` as a list
  of Haystack ``Document``s. Use when memory feeds a document-oriented pipeline.
- :class:`SynapMemoryRetriever` — Mem0-shaped: ``run(query)`` → ``messages`` as a
  list of ``ChatMessage``s, matching ``mem0_haystack.Mem0MemoryRetriever``. Use
  when memory feeds a chat/agent pipeline.

Either accept a ready ``store=SynapMemoryStore(...)`` or the ``sdk=...`` +
scope kwargs (a store is then built internally for convenience). ``scope_path``
(a deeper scope ladder's level path) is passed to that internal store; with a
ready ``store`` set it on the store.
"""

import inspect
import logging
from typing import Any, Dict, List, Mapping, Optional

from haystack import Document, component
from haystack.core.serialization import default_to_dict
from haystack.dataclasses import ChatMessage

from maximem_synap import MaximemSynapSDK
from synap_haystack.store import SynapMemoryStore

logger = logging.getLogger(__name__)


def _component_to_dict(obj: Any) -> Dict[str, Any]:
    """Haystack's default component serialisation, minus ``scope_path``.

    The default reads every init parameter off the instance, so a new
    ``scope_path`` parameter would appear as ``"scope_path": null`` in every
    existing pipeline's dict. The level path lives on the store (which is
    serialised, with its ``scope_path`` only when set), so it is left out here:
    a dict without a path is exactly what it was before the parameter existed.
    """
    init: Dict[str, Any] = {}
    for name, param in inspect.signature(obj.__init__).parameters.items():
        if name in ("args", "kwargs", "scope_path"):
            continue
        init[name] = getattr(obj, name, param.default)
    return default_to_dict(obj, **init)


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
    scope_path: Optional[Mapping[str, str]] = None,
) -> SynapMemoryStore:
    if store is not None:
        if scope_path:
            raise ValueError(
                f"{component_name}: scope_path with a ready store is ambiguous; "
                "set scope_path on the store"
            )
        return store
    if sdk is None:
        raise ValueError(f"{component_name} requires either a store or an sdk")
    if not user_id and not customer_id and not scope_path:
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
        **({"scope_path": scope_path} if scope_path else {}),
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
        scope_path: Optional[Mapping[str, str]] = None,
    ):
        self.store = _resolve_store(
            store, sdk, user_id, customer_id, conversation_id, mode, max_results,
            component_name="SynapRetriever", scope_path=scope_path,
        )

    def to_dict(self) -> Dict[str, Any]:
        return _component_to_dict(self)

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
        scope_path: Optional[Mapping[str, str]] = None,
    ):
        self.store = _resolve_store(
            store, sdk, user_id, customer_id, conversation_id, mode, max_results,
            component_name="SynapMemoryRetriever", scope_path=scope_path,
        )

    def to_dict(self) -> Dict[str, Any]:
        return _component_to_dict(self)

    @component.output_types(messages=List[ChatMessage])
    def run(self, query: str) -> Dict[str, List[ChatMessage]]:
        return {"messages": self.store.search_memories(query=query)}
