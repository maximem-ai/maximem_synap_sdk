"""Synap integration for LangGraph.

Exposes:

- :class:`SynapStore` — a ``BaseStore`` implementation backed by Synap's
  semantic memory (``sdk.memories.create`` + ``sdk.fetch``). Drop it into
  ``StateGraph.compile(store=SynapStore(...))`` for cross-thread long-term
  memory. Semantic search via ``store.asearch`` is the natural fit for our
  retrieval model.

- :class:`SynapCheckpointSaver` — **deprecated.** It cannot restore what it
  saves: Synap extracts facts from the checkpoint and ``fetch`` returns those,
  so a thread always resumes from scratch. Use ``SqliteSaver`` /
  ``PostgresSaver`` for checkpoints and :class:`SynapStore` for long-term
  memory.

- :func:`create_synap_node` — re-exported from ``synap_langchain.graph`` for
  continuity with pre-0.1.0 users who discovered our LangGraph support via the
  LangChain package. The canonical home is now this package.

- :func:`synap_st_prompt` — Synap's compaction as the prompt callable for
  ``create_react_agent(prompt=...)``. Once the conversation has been compacted,
  the model sees the compaction above the user's system prompt in place of the
  turns before the current one (cache-first via
  ``sdk.conversation.context.get_context_for_prompt``).

- :func:`create_synap_st_node` — same short-term context, exposed as a
  state-mutating node for custom ``StateGraph`` flows. Writes the ST string
  into ``state[state_key]`` for downstream LLM nodes to assemble.

- :class:`SynapLangGraphCallbackHandler` — reports a run on Synap's live
  gRPC stream so the anticipation agent sees the whole turn: the user turn,
  the reasoning, every tool call and result, and the assistant turn. Pass it
  in ``config={"callbacks": [...]}``. LangGraph runs on LangChain's callback
  machinery, so this extends the LangChain handler rather than repeating it.
  The stream opens on the first turn; no ``listen()`` call is needed
  (``SYNAP_AUTO_LISTEN=false`` turns that off).
"""

from synap_langgraph.store import SynapStore
from synap_langgraph.checkpointer import SynapCheckpointSaver
from synap_langgraph.callbacks import SynapLangGraphCallbackHandler
from synap_langgraph.short_term import create_synap_st_node, synap_st_prompt
from synap_langchain.graph import create_synap_node

__all__ = [
    "SynapStore",
    "SynapCheckpointSaver",
    "SynapLangGraphCallbackHandler",
    "create_synap_node",
    "create_synap_st_node",
    "synap_st_prompt",
]
