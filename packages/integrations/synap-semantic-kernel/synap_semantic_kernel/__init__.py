"""Synap memory integration for Microsoft Semantic Kernel.

Three plug-in points:

1. **Plugin** — register :class:`SynapPlugin` so the model can search and
   store memory itself.
2. **Short-term context** — :func:`synap_st_chat_message` builds the system
   message that carries Synap's recent-conversation block.
3. **Report the turn** — :class:`SynapAgentThread` reports user and
   assistant turns, and :func:`attach_synap_filters` reports the tool call
   and its result. The first turn opens the Synap gRPC stream, and turns go
   out on it; they fall back to ``sdk.conversation.record_message`` only if
   the stream cannot open, never both. ``SYNAP_AUTO_LISTEN=false`` keeps
   everything on REST.

:class:`SynapAgentThread` also sends the model Synap's compaction in place of
the earlier turns once Synap has compacted the conversation
(:class:`SynapChatHistoryReducer`). Semantic Kernel's own reducers stay in
place and are the fallback when Synap has nothing.
"""

from synap_semantic_kernel.history import SynapChatHistoryReducer
from synap_semantic_kernel.plugin import SynapPlugin
from synap_semantic_kernel.short_term import synap_st_chat_message
from synap_semantic_kernel.stream import (
    SynapAgentThread,
    attach_synap_filters,
    synap_auto_function_filter,
)

__all__ = [
    "SynapPlugin",
    "synap_st_chat_message",
    "SynapAgentThread",
    "SynapChatHistoryReducer",
    "attach_synap_filters",
    "synap_auto_function_filter",
]
