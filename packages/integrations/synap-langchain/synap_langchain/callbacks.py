"""Synap callback handler for zero-config auto-memory.

Add SynapCallbackHandler to any LangChain chain or agent. Every
conversation turn is automatically recorded to Synap for automatic
memory extraction.

No explicit save_context() calls needed.

## Design notes

LangChain callbacks **must not raise** — a raising callback aborts the
whole chain, which is almost never what the user wants for a
best-effort side-channel like memory recording. So this handler
catches every exception. The *original* implementation demoted those
failures to ``logger.debug``, which meant ingestion outages were
effectively invisible: DEBUG is off in virtually every production
logger configuration. We now log at ``ERROR`` with structured context
instead, preserving the "don't break the chain" contract while making
failures actually observable.

Extracting the LLM response text previously chained two attribute
lookups with a silent fallback (``generation.text`` -> ``generation.
message.content``). That's fragile in the presence of custom LLM
wrappers. Centralised in ``_extract_text`` so the dispatch is explicit
and unit-tested.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import Generation, LLMResult

from maximem_synap import MaximemSynapSDK

from synap_integrations_common.stream_events import (
    report_reasoning,
    report_tool_call,
    report_tool_result,
    report_turn,
)

logger = logging.getLogger(__name__)


def _extract_text(generation: Generation) -> str:
    """Return the text payload from a LangChain Generation.

    Ordering matters. ``ChatGeneration`` subclasses populate both
    ``.text`` and ``.message.content``, but when the message content
    is a structured list (e.g., multiple text blocks plus tool calls),
    LangChain derives ``.text`` from only the **first** text block and
    drops the rest. Reading ``.text`` naively would therefore lose
    content. So:

    1. If the generation has a message with **list** content, rebuild
       the string by concatenating every text part. This covers
       multi-block chat outputs.
    2. Otherwise prefer ``.text`` (string LLMs and chat models whose
       content is already a plain string).
    3. Fall back to ``message.content`` as a string.
    """
    message = getattr(generation, "message", None)
    if message is not None:
        content = getattr(message, "content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
            return "".join(parts)

    text = getattr(generation, "text", "") or ""
    if text:
        return text

    if message is not None:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        return str(content) if content else ""
    return ""


class SynapCallbackHandler(AsyncCallbackHandler):
    """Auto-records all conversation turns to Synap.

    Listens for chat model start (to capture user messages) and
    LLM end (to capture assistant responses). Record failures are
    logged at ``ERROR`` but never raised — breaking the chain over
    a memory-sidechannel failure would do more harm than good.

    Example::

        handler = SynapCallbackHandler(
            sdk=sdk,
            conversation_id="conv-123",
            user_id="user-456",
            customer_id="cust-789",
        )
        chain = ConversationChain(llm=ChatOpenAI(), callbacks=[handler])
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        conversation_id: str,
        user_id: str,
        customer_id: str = "",
    ):
        if sdk is None:
            raise ValueError("SynapCallbackHandler requires a non-None sdk")
        if not conversation_id:
            raise ValueError(
                "SynapCallbackHandler requires a non-empty conversation_id"
            )
        if not user_id:
            raise ValueError("SynapCallbackHandler requires a non-empty user_id")

        self.sdk = sdk
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.customer_id = customer_id

    async def _record(self, role: str, content: str) -> None:
        # Stream first, and fall back to REST only if there was no stream.
        #
        # ⚠ Never both. The server persists `user_message` and
        # `assistant_message` from the stream itself
        # (`grpc/servicer.py`), so a `record_message` on top of a delivered
        # stream event writes the same turn twice and extracts it twice.
        # `report_turn` returns whether it actually went out, which is what
        # makes this decidable rather than guessed.
        if await report_turn(
            self.sdk, role=role, content=content,
            conversation_id=self.conversation_id,
            user_id=self.user_id, customer_id=self.customer_id,
        ):
            return
        try:
            await self.sdk.conversation.record_message(
                conversation_id=self.conversation_id,
                role=role,
                content=content,
                user_id=self.user_id,
                customer_id=self.customer_id,
            )
        except Exception as exc:  # noqa: BLE001 — callbacks must not raise
            logger.error(
                "SynapCallbackHandler: record_message failed "
                "role=%s conversation_id=%s error=%s",
                role,
                self.conversation_id,
                exc,
                exc_info=True,
            )

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Record the user message when a chat model starts."""
        if not messages:
            return
        last_batch = messages[-1]
        for msg in reversed(last_batch):
            if msg.type == "human":
                await self._record("user", str(msg.content))
                break

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Record the assistant response when the LLM finishes."""
        if not response.generations or not response.generations[0]:
            return
        text = _extract_text(response.generations[0][0])
        if text:
            await self._record("assistant", text)

    # ─── the three events this handler never reported ───────────────────
    #
    # It recorded the user turn and the assistant turn and stopped there, so
    # anticipation saw a conversation with questions and answers and nothing
    # in between: no tool calls, no results, no reasoning. Those are the
    # events that say what the agent is ABOUT to need, which is the whole
    # point of anticipating.
    #
    # All three are silent when no stream is running, and none of them can
    # raise into the chain: `stream_events` swallows everything by design,
    # and a callback that throws aborts the whole LangChain run.

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Report a tool the agent is about to call.

        `run_id` is LangChain's own id for this tool invocation and it is the
        same value `on_tool_end` receives, so it ties the call to its result.
        """
        await report_tool_call(
            self.sdk,
            tool_name=str((serialized or {}).get("name") or "tool"),
            tool_args={"input": input_str} if input_str else {},
            tool_call_id=str(run_id),
            conversation_id=self.conversation_id,
            user_id=self.user_id, customer_id=self.customer_id,
        )

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Report what the tool returned, tied to the call by `run_id`."""
        await report_tool_result(
            self.sdk,
            result=output,
            tool_call_id=str(run_id),
            conversation_id=self.conversation_id,
            user_id=self.user_id, customer_id=self.customer_id,
        )

    async def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Report the agent's reasoning for choosing a tool.

        `AgentAction.log` is where LangChain puts the model's own words about
        why it picked this action, which is the closest thing the framework
        exposes to a reasoning step.
        """
        text = getattr(action, "log", None)
        if not isinstance(text, str) or not text.strip():
            return
        await report_reasoning(
            self.sdk, content=text.strip(),
            conversation_id=self.conversation_id,
            user_id=self.user_id, customer_id=self.customer_id,
        )
