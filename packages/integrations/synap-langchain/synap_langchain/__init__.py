"""Synap memory integration for LangChain and LangGraph.

Integration surfaces:
- SynapChatMessageHistory: BaseChatMessageHistory for use with RunnableWithMessageHistory
- SynapRetriever: BaseRetriever for RAG pipelines with typed memory items
- SynapSearchTool / SynapStoreTool: Agent tools for explicit memory access
- SynapCallbackHandler: Zero-config auto-recording of conversation turns
- SynapHistoryMiddleware: Synap's compaction in place of a ``create_agent``
  agent's earlier turns (needs ``langchain>=1``)
- create_synap_node: LangGraph node for state injection
- synap_st_runnable: LCEL Runnable emitting Synap short-term context as a string
- synap_st_message: async factory producing a combined SystemMessage

Turns are written on the gRPC stream, which the first turn opens for you; there
is no ``listen()`` to call. Without a stream (``SYNAP_AUTO_LISTEN=false``, or
``synap_integrations_common.set_auto_listen(sdk, False)``, or a stream that will
not open) they go over REST.
"""

from synap_langchain.memory import SynapChatMessageHistory, SynapMemory
from synap_langchain.retriever import SynapRetriever
from synap_langchain.tools import SynapSearchTool, SynapStoreTool
from synap_langchain.callbacks import SynapCallbackHandler
from synap_langchain.graph import create_synap_node
from synap_langchain.short_term import synap_st_message, synap_st_runnable

try:  # needs langchain>=1; langchain-core alone has no agent middleware
    from synap_langchain.middleware import SynapHistoryMiddleware
except ImportError:  # pragma: no cover
    SynapHistoryMiddleware = None  # type: ignore[assignment,misc]

__all__ = [
    "SynapChatMessageHistory",
    "SynapMemory",  # backward-compatible alias
    "SynapRetriever",
    "SynapSearchTool",
    "SynapStoreTool",
    "SynapCallbackHandler",
    "SynapHistoryMiddleware",
    "create_synap_node",
    "synap_st_runnable",
    "synap_st_message",
]
