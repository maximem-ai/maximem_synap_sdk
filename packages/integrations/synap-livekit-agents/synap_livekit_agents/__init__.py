"""Synap integration for LiveKit Agents (realtime voice).

Three plug-in points for a LiveKit Agent:

1. **Preload** — before `session.start(...)`, call :func:`preload_synap_context`
   on the agent's :class:`ChatContext` so long-term memory is in the prompt
   window from the very first turn.
2. **Record** — before `session.start(...)`, call
   :func:`attach_synap_recording` on the :class:`AgentSession`. It reports
   the whole turn, not just its two ends: the committed user and assistant
   turns (`conversation_item_added`), the tool call at the moment it is
   dispatched (`tool_execution_updated`), the tool result
   (`function_tools_executed`), and the end of the call (`close`).
   Turns go out on the Synap gRPC stream, opened on the first one if you did
   not open it (``SYNAP_AUTO_LISTEN=false`` turns that off), and fall back to
   ``sdk.conversation.record_message`` when the stream cannot carry them,
   never both. Once Synap has compacted the conversation, the agent's LLM is
   sent Synap's compaction instead of the agent's earlier turns.
3. **Tools** — register :func:`synap_search_tool` and/or
   :func:`synap_store_tool` on your :class:`Agent` so the LLM can
   retrieve or persist memory mid-call.

See each helper's docstring for the exact frame / event contract and
error policy (reads degrade silently — a Synap blip must never break a
live call; writes surface as ``SynapIntegrationError``, except in the
recording callback where LiveKit's sync event contract forces us to
log-and-swallow).
"""

from synap_livekit_agents.compaction import install_synap_history, splice_synap_history
from synap_livekit_agents.recording import attach_synap_recording
from synap_livekit_agents.context import preload_synap_context
from synap_livekit_agents.short_term import preload_synap_st, refresh_synap_st
from synap_livekit_agents.tools import synap_search_tool, synap_store_tool

__all__ = [
    "preload_synap_context",
    "preload_synap_st",
    "refresh_synap_st",
    "attach_synap_recording",
    "install_synap_history",
    "splice_synap_history",
    "synap_search_tool",
    "synap_store_tool",
]
