"""Synap gRPC Listen-stream participation for Strands Agents.

Makes a Strands agent a first-class participant in Synap's real-time
anticipation pipeline. A :class:`~strands.hooks.HookProvider` maps Strands
lifecycle events onto the SDK's outbound stream sends so the Synap Anticipation
Agent sees the agent's turns and planned tool calls *as they happen* and can
pre-fetch context before the agent asks.

The **read** side needs no help — when the stream is live the SDK's own
``fetch()`` already emits ``context_fetch`` / ``context_used`` /
``context_assembled`` events. This hook only feeds the **activity** the SDK
can't infer: conversation turns and tool-call intent.

Lifecycle is the app's job, not the hook's:

    await sdk.instance.listen()                     # app opens the stream
    hook = SynapStreamHook(sdk, conversation_id="c1", user_id="alice")
    agent = Agent(hooks=[hook])
    # ... run the agent on the SAME asyncio event loop ...
    await sdk.instance.stop_listening()             # app closes it on shutdown

Design rules (these differ from the other surfaces):
- **Gate every send on ``sdk.instance.is_listening``**; no-op silently when the
  app never called ``listen()`` — never raise ``ListeningNotActiveError``.
- **Log, never raise.** These run inside Strands' event loop; a raising hook can
  abort the agent turn. This mirrors the SDK's own fire-and-forget stream sends.
- This feeds *signal*, it does not durably ingest. Long-term memory still comes
  only from :class:`~synap_strands_agents.store.SynapMemoryStore` /
  :func:`~synap_strands_agents.tools.create_synap_tools`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from strands.hooks import (
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)

from maximem_synap import MaximemSynapSDK

from synap_strands_agents._util import message_text

logger = logging.getLogger(__name__)


class SynapStreamHook(HookProvider):
    """Feed a Strands agent's turns and tool intent onto Synap's Listen stream.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`. The app owns its
            ``listen()`` / ``stop_listening()`` lifecycle; this hook only feeds
            an already-open stream and no-ops when none is active.
        conversation_id: Synap conversation id for the emitted events. **Required.**
        user_id: External user id for the conversation. **Required.**
        customer_id: Optional customer/org scope.
        session_id: Optional session identifier attached to emitted events.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        conversation_id: str,
        user_id: str,
        *,
        customer_id: str = "",
        session_id: str = "",
    ) -> None:
        if sdk is None:
            raise ValueError("SynapStreamHook requires a non-None sdk")
        if not conversation_id or not str(conversation_id).strip():
            raise ValueError("SynapStreamHook requires a non-empty conversation_id")
        if not user_id:
            raise ValueError("SynapStreamHook requires a non-empty user_id")
        self._sdk = sdk
        self._conversation_id = conversation_id
        self._user_id = user_id
        self._customer_id = customer_id
        self._session_id = session_id

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(MessageAddedEvent, self._on_message)
        registry.add_callback(BeforeToolCallEvent, self._on_tool_call)

    async def _on_message(self, event: MessageAddedEvent) -> None:
        try:
            if not self._sdk.instance.is_listening:
                return
            message = event.message
            text = message_text(message)
            if not text:
                return  # tool-result / tool-use-only turns carry no text
            role = message.get("role", "user") if isinstance(message, dict) else "user"
            event_type = "assistant_message" if role == "assistant" else "user_message"
            await self._sdk.instance.send_message(
                content=text,
                role=role,
                event_type=event_type,
                conversation_id=self._conversation_id,
                user_id=self._user_id,
                customer_id=self._customer_id or None,
                session_id=self._session_id or None,
            )
        except Exception as exc:  # noqa: BLE001 — stream feed must never abort the turn
            logger.warning(
                "SynapStreamHook: message feed failed: %s", exc, exc_info=True
            )

    async def _on_tool_call(self, event: BeforeToolCallEvent) -> None:
        try:
            if not self._sdk.instance.is_listening:
                return
            tool_use = event.tool_use or {}
            tool_name = tool_use.get("name", "") if isinstance(tool_use, dict) else ""
            if not tool_name:
                return
            tool_args = tool_use.get("input") if isinstance(tool_use, dict) else None
            await self._sdk.instance.send_message(
                content=tool_name,
                role="assistant",
                event_type="tool_call",
                tool_name=tool_name,
                tool_args=tool_args,
                conversation_id=self._conversation_id,
                user_id=self._user_id,
                customer_id=self._customer_id or None,
                session_id=self._session_id or None,
            )
        except Exception as exc:  # noqa: BLE001 — stream feed must never abort the turn
            logger.warning(
                "SynapStreamHook: tool-call feed failed: %s", exc, exc_info=True
            )


__all__ = ["SynapStreamHook"]
