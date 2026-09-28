"""Synap memory integration for Microsoft Semantic Kernel.

Three plug-in points:

1. **Plugin** — register :class:`SynapPlugin` so the model can search and
   store memory itself.
2. **Short-term context** — :func:`synap_st_chat_message` builds the system
   message that carries Synap's recent-conversation block.
3. **Report the turn** — :class:`SynapAgentThread` reports user and
   assistant turns, and :func:`attach_synap_filters` reports the tool call
   and its result. Turns go out on the Synap stream when one is open and
   fall back to ``sdk.conversation.record_message`` when it is not; never
   both.
"""

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
    "attach_synap_filters",
    "synap_auto_function_filter",
]
