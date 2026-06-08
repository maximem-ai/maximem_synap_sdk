"""Synap memory integration for LangChain and LangGraph.

Integration surfaces:
- SynapChatMessageHistory: BaseChatMessageHistory for use with RunnableWithMessageHistory
- SynapRetriever: BaseRetriever for RAG pipelines with typed memory items
- SynapSearchTool / SynapStoreTool: Agent tools for explicit memory access
- SynapCallbackHandler: Zero-config auto-recording of conversation turns
- create_synap_node: LangGraph node for state injection
- synap_st_runnable: LCEL Runnable emitting Synap short-term context as a string
- synap_st_message: async factory producing a combined SystemMessage
"""

from synap_langchain.memory import SynapChatMessageHistory, SynapMemory
from synap_langchain.retriever import SynapRetriever
from synap_langchain.tools import SynapSearchTool, SynapStoreTool
from synap_langchain.callbacks import SynapCallbackHandler
from synap_langchain.graph import create_synap_node
from synap_langchain.short_term import synap_st_message, synap_st_runnable

__all__ = [
    "SynapChatMessageHistory",
    "SynapMemory",  # backward-compatible alias
    "SynapRetriever",
    "SynapSearchTool",
    "SynapStoreTool",
    "SynapCallbackHandler",
    "create_synap_node",
    "synap_st_runnable",
    "synap_st_message",
]
