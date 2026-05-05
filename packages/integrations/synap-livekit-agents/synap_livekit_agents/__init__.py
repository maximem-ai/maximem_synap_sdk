"""Synap integration for LiveKit Agents (realtime voice).

Three plug-in points for a LiveKit Agent:

1. **Preload** — before `session.start(...)`, call :func:`preload_synap_context`
   on the agent's :class:`ChatContext` so long-term memory is in the prompt
   window from the very first turn.
2. **Record** — before `session.start(...)`, call
   :func:`attach_synap_recording` on the :class:`AgentSession`; it wires
   `on("conversation_item_added", ...)` to
   ``sdk.conversation.record_message`` for both user and assistant turns.
3. **Tools** — register :func:`synap_search_tool` and/or
   :func:`synap_store_tool` on your :class:`Agent` so the LLM can
   retrieve or persist memory mid-call.

See each helper's docstring for the exact frame / event contract and
error policy (reads degrade silently — a Synap blip must never break a
live call; writes surface as ``SynapIntegrationError``, except in the
recording callback where LiveKit's sync event contract forces us to
log-and-swallow).
"""

from synap_livekit_agents.recording import attach_synap_recording
from synap_livekit_agents.context import preload_synap_context
from synap_livekit_agents.tools import synap_search_tool, synap_store_tool

__all__ = [
    "preload_synap_context",
    "attach_synap_recording",
    "synap_search_tool",
    "synap_store_tool",
]
