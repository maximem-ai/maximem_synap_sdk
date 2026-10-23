"""Synap memory integration for LlamaIndex.

Provides:
- SynapRetriever: Memory-augmented retriever returning NodeWithScore
- SynapChatMemory: Chat memory backed by Synap conversation context. Pass it as
  an agent's ``memory=``: turns, tool calls, results and reasoning go on the gRPC
  stream (the first one opens it; no ``listen()`` call is needed, and
  ``SYNAP_AUTO_LISTEN=false`` turns that off), and once Synap has compacted the
  conversation the agent is sent the compaction in place of the turns before the
  current one.
- synap_st_chat_message: Synap's compaction as a system message, for a chat
  engine you assemble by hand
"""

from synap_llamaindex.retriever import SynapRetriever
from synap_llamaindex.memory import SynapChatMemory
from synap_llamaindex.short_term import synap_st_chat_message

__all__ = ["SynapRetriever", "SynapChatMemory", "synap_st_chat_message"]
