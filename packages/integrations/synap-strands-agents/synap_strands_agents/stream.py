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

Nothing to set up: the first turn opens the stream (``sdk.instance.listen()``),
and the turns, tool calls and tool results that follow go out on it::

    hook = SynapStreamHook(sdk, conversation_id="c1", user_id="alice")
    agent = Agent(hooks=[hook])
    # ... run the agent on one asyncio event loop ...

``SYNAP_AUTO_LISTEN=false`` (or ``set_auto_listen(sdk, False)``) keeps the hook
on REST and never opens a stream; a stream the app opened itself is used as is.

Design rules (these differ from the other surfaces):
- **Turns are always written.** A user or assistant turn goes on the stream,
  and falls back to REST (``conversation.record_message``) only when the stream
  cannot carry it; never both (``record_turn``). Without it Synap keeps no
  conversation to compact or search.
- **Tool events are stream-only**: a silent no-op when the stream cannot carry
  them, never a ``ListeningNotActiveError``.
- **Log, never raise.** These run inside Strands' event loop; a raising hook can
  abort the agent turn. This mirrors the SDK's own fire-and-forget stream sends.
- **Synap's compaction replaces Strands' for the model.** The first invocation
  installs the model-call middleware of :mod:`synap_strands_agents.history`.
- Apart from the turns, this feeds *signal*, it does not ingest. Long-term memory still comes
  only from :class:`~synap_strands_agents.store.SynapMemoryStore` /
  :func:`~synap_strands_agents.tools.create_synap_tools`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from strands.hooks import (
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)

from maximem_synap import MaximemSynapSDK

from synap_integrations_common.stream_events import (
    record_turn,
    report_tool_call,
    report_tool_result,
)

from synap_strands_agents._util import message_text, note_written
from synap_strands_agents.history import install_synap_history

logger = logging.getLogger(__name__)


def _tool_use_id(tool_use: Any) -> str:
    """Strands' id for one tool invocation.

    Bedrock spells it `toolUseId`; other providers have used `tool_use_id` and
    plain `id`. Checked in that order rather than assuming one, because an
    empty id silently unlinks a call from its result and nothing complains.
    """
    if not isinstance(tool_use, dict):
        return ""
    for key in ("toolUseId", "tool_use_id", "id"):
        value = tool_use.get(key)
        if value:
            return str(value)
    return ""


class SynapStreamHook(HookProvider):
    """Feed a Strands agent's turns and tool intent onto Synap's Listen stream.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`. The app owns its
            ``listen()`` / ``stop_listening()`` lifecycle, or leaves it to the
            hook: the first turn opens the stream. If it cannot open, only the
            user and assistant turns are written, over REST.
        conversation_id: Synap conversation id for the emitted events. **Required.**
        user_id: External user id for the conversation. **Required.**
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on a
            B2C instance (user_context_isolation=equals_customer): the server
            rejects a call carrying one with HTTP 400. Leave it unset there.
        session_id: Optional session identifier, sent as turn metadata. (The
            stream attaches its own session to each event.)
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
        registry.add_callback(BeforeInvocationEvent, self._on_invocation)
        registry.add_callback(MessageAddedEvent, self._on_message)
        registry.add_callback(BeforeToolCallEvent, self._on_tool_call)
        # ⚠ The result was never reported. Strands fires this and nothing
        # listened, so anticipation saw every tool call go out and no answer
        # come back: it knew the agent had asked something and never what it
        # learned, which is the half that says what to prefetch next.
        registry.add_callback(AfterToolCallEvent, self._on_tool_result)

    async def _on_invocation(self, event: BeforeInvocationEvent) -> None:
        install_synap_history(event.agent, self._sdk, self._conversation_id)

    async def _on_message(self, event: MessageAddedEvent) -> None:
        try:
            message = event.message
            text = message_text(message)
            if not text:
                return  # tool-result / tool-use-only turns carry no text
            role = "assistant" if isinstance(message, dict) and message.get("role") == "assistant" else "user"
            # Stream first (opened if need be), REST only if the stream cannot
            # carry it, never both: the server persists a streamed turn itself.
            # record_turn never raises.
            if await record_turn(
                self._sdk,
                role=role,
                content=text,
                conversation_id=self._conversation_id,
                user_id=self._user_id,
                customer_id=self._customer_id or "",
                metadata={"session_id": self._session_id} if self._session_id else None,
            ):
                note_written(self._user_id, role, text)
        except Exception as exc:  # noqa: BLE001 — stream feed must never abort the turn
            logger.warning(
                "SynapStreamHook: message feed failed: %s", exc, exc_info=True
            )

    async def _on_tool_call(self, event: BeforeToolCallEvent) -> None:
        try:
            tool_use = event.tool_use or {}
            tool_name = tool_use.get("name", "") if isinstance(tool_use, dict) else ""
            if not tool_name:
                return
            tool_args = tool_use.get("input") if isinstance(tool_use, dict) else None
            # `toolUseId` is Strands' own id for this invocation and the same
            # value arrives on AfterToolCallEvent, so it ties the call to its
            # result. Without it the anticipation agent sees a call and a
            # result and cannot tell they are the same thing.
            #
            # Through the typed helper rather than a hand-rolled
            # `send_message(event_type="tool_call", role="assistant")`: the
            # role on a tool event is a thing the server reads, and getting it
            # wrong fails silently.
            await report_tool_call(
                self._sdk,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=str(_tool_use_id(tool_use)),
                conversation_id=self._conversation_id,
                user_id=self._user_id,
                customer_id=self._customer_id or "",
            )
        except Exception as exc:  # noqa: BLE001 — stream feed must never abort the turn
            logger.warning(
                "SynapStreamHook: tool-call feed failed: %s", exc, exc_info=True
            )

    async def _on_tool_result(self, event: AfterToolCallEvent) -> None:
        """What the tool returned, tied to the call by its id.

        Reports the failure too. A tool that raised is a fact about the turn:
        the agent is about to apologise or retry, and either is worth
        anticipating. `exception` is preferred over `result` when both are
        present, because a partial result beside an exception is not an answer.
        """
        try:
            tool_use = event.tool_use or {}
            tool_name = tool_use.get("name", "") if isinstance(tool_use, dict) else ""
            exception = getattr(event, "exception", None)
            result = (
                {"error": str(exception)} if exception is not None
                else getattr(event, "result", None)
            )
            if result is None:
                return
            await report_tool_result(
                self._sdk,
                result=result,
                tool_name=tool_name,
                tool_call_id=_tool_use_id(tool_use),
                conversation_id=self._conversation_id,
                user_id=self._user_id,
                customer_id=self._customer_id or "",
            )
        except Exception as exc:  # noqa: BLE001 — stream feed must never abort the turn
            logger.warning(
                "SynapStreamHook: tool-result feed failed: %s", exc, exc_info=True
            )


__all__ = ["SynapStreamHook"]
