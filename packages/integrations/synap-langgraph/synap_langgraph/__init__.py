"""Synap integration for LangGraph.

Exposes:

- :class:`SynapStore` — a ``BaseStore`` implementation backed by Synap's
  semantic memory (``sdk.memories.create`` + ``sdk.fetch``). Drop it into
  ``StateGraph.compile(store=SynapStore(...))`` for cross-thread long-term
  memory. Semantic search via ``store.asearch`` is the natural fit for our
  retrieval model.

- :class:`SynapCheckpointSaver` — a ``BaseCheckpointSaver`` that persists
  thread execution state to Synap. **Best-effort fuzzy retrieval**: reads
  use ``sdk.fetch`` with metadata filters, which is semantic-search-shaped
  rather than exact KV. Pair with a real KV saver (SqliteSaver, PostgresSaver)
  for production-grade checkpoint fidelity; use ``SynapCheckpointSaver`` for
  observability/audit and demo flows.

- :func:`create_synap_node` — re-exported from ``synap_langchain.graph`` for
  continuity with pre-0.1.0 users who discovered our LangGraph support via the
  LangChain package. The canonical home is now this package.
"""

from synap_langgraph.store import SynapStore
from synap_langgraph.checkpointer import SynapCheckpointSaver
from synap_langchain.graph import create_synap_node

__all__ = ["SynapStore", "SynapCheckpointSaver", "create_synap_node"]
