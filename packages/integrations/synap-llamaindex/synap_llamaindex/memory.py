"""Synap chat memory for LlamaIndex.

Implements LlamaIndex's BaseMemory interface backed by Synap's
conversation context.

Error-handling split follows the LangChain analogue:

- Read paths (:meth:`aget`) degrade gracefully — a memory lookup
  failure should not crash the chain — but they now log at ``ERROR``
  (previously ``DEBUG``, which meant outages were invisible).
- Write paths (:meth:`aput`) raise :class:`SynapIntegrationError`:
  the caller explicitly wrote data and has a right to know if it
  didn't land.

## The live stream

``aput`` is the one place every message an agent produces passes through.
LlamaIndex's ``FunctionAgent`` builds a scratchpad of ``ChatMessage``s and
hands the whole thing to ``memory.aput_messages`` when a step finishes, so
this method sees the user turn, the assistant turn, the model's reasoning,
the tools it asked for and what each one returned. Before, it forwarded the
two turns and dropped the other three on the floor, which left the
anticipation agent watching a conversation of questions and answers with
nothing in between.

Everything but the turns is stream-only: there is no REST equivalent for a
tool call, and none of it is memory. The turns follow the one rule that
matters, **stream first, REST only as a fallback, never both**, because
the server persists ``user_message`` and ``assistant_message`` straight off
the stream (``grpc/servicer.py``), so a ``record_message`` on top of a
delivered stream event writes the turn twice and extracts it twice.

Reasoning and tool calls ride on an assistant message's ``blocks`` as
``ThinkingBlock`` and ``ToolCallBlock``; a tool result arrives as its own
``ChatMessage`` with ``role="tool"`` carrying ``tool_call_id`` in
``additional_kwargs``. That id is LlamaIndex's own id for the invocation
(``ToolCall.tool_id``), and it is what ties a call to its result. Blocks are
matched on ``block_type`` rather than by class, so an older
``llama-index-core`` that predates those block types reports nothing extra
instead of failing to import.
"""

import logging
from typing import Any, List, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory.types import BaseMemory

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import run_async, wrap_sdk_errors_async
from synap_integrations_common.stream_events import (
    report_reasoning,
    report_tool_call,
    report_tool_result,
    report_turn,
)

logger = logging.getLogger(__name__)

# A tool result is not a turn. ``FUNCTION`` is the older spelling of ``TOOL``
# and both are still produced depending on which LLM package is in use.
_TOOL_ROLES = (MessageRole.TOOL, MessageRole.FUNCTION)


def _text_of(message: Any) -> str:
    """The message's text, and an empty string when it has none.

    ``ChatMessage.content`` returns ``None`` for a message built only from
    non-text blocks, which is the tool-call-only assistant message a
    function-calling agent emits on every step. ``str()`` on that produced
    the four-character
    string ``"None"``, which then went to Synap as an assistant turn and got
    extracted as a memory.
    """
    content = getattr(message, "content", None)
    return "" if content is None else str(content)


def _blocks(message: Any) -> List[Any]:
    blocks = getattr(message, "blocks", None)
    return list(blocks) if isinstance(blocks, (list, tuple)) else []


def _block_type(block: Any) -> str:
    return str(getattr(block, "block_type", "") or "")


class SynapChatMemory(BaseMemory):
    """LlamaIndex chat memory backed by Synap."""

    _sdk: MaximemSynapSDK
    _conversation_id: str
    _user_id: str
    _customer_id: str
    _messages: List[ChatMessage]

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        conversation_id: str,
        user_id: str,
        customer_id: str = "",
    ):
        if sdk is None:
            raise ValueError("SynapChatMemory requires a non-None sdk")
        if not conversation_id:
            raise ValueError(
                "SynapChatMemory requires a non-empty conversation_id"
            )
        if not user_id:
            raise ValueError("SynapChatMemory requires a non-empty user_id")

        self._sdk = sdk
        self._conversation_id = conversation_id
        self._user_id = user_id
        self._customer_id = customer_id
        self._messages = []

    @classmethod
    def from_defaults(
        cls,
        sdk: Optional[MaximemSynapSDK] = None,
        conversation_id: str = "",
        user_id: str = "",
        customer_id: str = "",
        **kwargs: Any,
    ) -> "SynapChatMemory":
        if sdk is None:
            raise ValueError(
                "SynapChatMemory.from_defaults requires sdk — "
                "construct a MaximemSynapSDK(instance_id=...) first"
            )
        return cls(
            sdk=sdk,
            conversation_id=conversation_id,
            user_id=user_id,
            customer_id=customer_id,
        )

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        return run_async(self.aget(input, **kwargs))

    async def aget(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        """Get conversation history from Synap (best-effort).

        A memory lookup feeding into an LLM prompt should not abort the
        chain on failure. Partial failures are logged at ERROR.
        """
        messages: List[ChatMessage] = []

        if input:
            try:
                response = await self._sdk.fetch(
                    conversation_id=self._conversation_id,
                    user_id=self._user_id,
                    customer_id=self._customer_id or None,
                    search_query=[input],
                    include_conversation_context=False,
                )
                if response.formatted_context:
                    messages.append(ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=f"Relevant user context:\n{response.formatted_context}",
                    ))
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "SynapChatMemory.aget: fetch failed "
                    "conversation_id=%s error=%s",
                    self._conversation_id, exc, exc_info=True,
                )

        try:
            prompt_ctx = await self._sdk.conversation.context.get_context_for_prompt(
                conversation_id=self._conversation_id,
            )
            if prompt_ctx.formatted_context:
                messages.append(ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=f"Conversation history:\n{prompt_ctx.formatted_context}",
                ))
            for msg in prompt_ctx.recent_messages:
                role = MessageRole.USER if msg.role == "user" else MessageRole.ASSISTANT
                messages.append(ChatMessage(role=role, content=msg.content))
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "SynapChatMemory.aget: get_context_for_prompt failed "
                "conversation_id=%s error=%s",
                self._conversation_id, exc, exc_info=True,
            )

        messages.extend(self._messages)
        return messages

    def get_all(self) -> List[ChatMessage]:
        return self.get()

    async def aget_all(self) -> List[ChatMessage]:
        return await self.aget()

    def put(self, message: ChatMessage) -> None:
        run_async(self.aput(message))

    async def aput(self, message: ChatMessage) -> None:
        """Record a message. Surfaces SDK errors (explicit write path).

        Reports the whole turn on the live stream when one is open: the
        model's reasoning, the turn itself, the tools it asked for, and what
        a tool returned. Only the turn has a REST fallback, and it is taken
        only when the stream did not carry it.
        """
        self._messages.append(message)

        role = getattr(message, "role", None)
        if role in _TOOL_ROLES:
            # A tool message is a result, never a turn: report it and stop.
            await self._report_tool_result(message)
            return
        if role not in (MessageRole.USER, MessageRole.ASSISTANT):
            return

        is_assistant = role == MessageRole.ASSISTANT
        if is_assistant:
            await self._report_reasoning(message)

        text = _text_of(message)
        if text:
            await self._record("assistant" if is_assistant else "user", text)

        if is_assistant:
            # After the turn: the model says its piece, then asks for tools.
            await self._report_tool_calls(message)

    async def _record(self, role: str, content: str) -> None:
        """Stream first, and fall back to REST only if there was no stream.

        ⚠ Never both. ``report_turn`` returns whether the event actually went
        out, which is what makes this decidable rather than guessed.
        """
        if await report_turn(
            self._sdk, role=role, content=content,
            conversation_id=self._conversation_id,
            user_id=self._user_id, customer_id=self._customer_id,
        ):
            return
        async with wrap_sdk_errors_async(
            "llamaindex.aput", logger,
            role=role, conversation_id=self._conversation_id,
        ):
            await self._sdk.conversation.record_message(
                conversation_id=self._conversation_id,
                role=role,
                content=content,
                user_id=self._user_id,
                customer_id=self._customer_id,
            )

    # ─── the three events this memory never reported ────────────────────
    #
    # None of these can raise into the agent loop: `stream_events` swallows
    # everything by design, and all three are silent when no stream is open.

    async def _report_reasoning(self, message: Any) -> None:
        """Report each ``ThinkingBlock`` on an assistant message."""
        step = 0
        for block in _blocks(message):
            if _block_type(block) != "thinking":
                continue
            content = str(getattr(block, "content", "") or "").strip()
            if not content:
                continue
            await report_reasoning(
                self._sdk, content=content,
                step_index=step, thought_type="thinking",
                conversation_id=self._conversation_id,
                user_id=self._user_id, customer_id=self._customer_id,
            )
            step += 1

    async def _report_tool_calls(self, message: Any) -> None:
        """Report each ``ToolCallBlock`` on an assistant message.

        ``tool_call_id`` is LlamaIndex's own id for the invocation, and the
        tool message that carries the result repeats it. Without it the
        anticipation agent cannot tell a call and a result belong together.
        """
        for block in _blocks(message):
            if _block_type(block) != "tool_call":
                continue
            args = getattr(block, "tool_kwargs", None)
            await report_tool_call(
                self._sdk,
                tool_name=str(getattr(block, "tool_name", "") or "tool"),
                tool_args=args if isinstance(args, dict) else None,
                tool_call_id=str(getattr(block, "tool_call_id", "") or ""),
                conversation_id=self._conversation_id,
                user_id=self._user_id, customer_id=self._customer_id,
            )

    async def _report_tool_result(self, message: Any) -> None:
        """Report a ``role="tool"`` message, tied to its call by id."""
        extra = getattr(message, "additional_kwargs", None)
        extra = extra if isinstance(extra, dict) else {}
        await report_tool_result(
            self._sdk,
            result=_text_of(message),
            tool_name=str(extra.get("name") or extra.get("tool_name") or ""),
            tool_call_id=str(extra.get("tool_call_id") or ""),
            conversation_id=self._conversation_id,
            user_id=self._user_id, customer_id=self._customer_id,
        )

    def set(self, messages: List[ChatMessage]) -> None:
        self._messages = list(messages)

    async def aset(self, messages: List[ChatMessage]) -> None:
        self._messages = list(messages)

    def reset(self) -> None:
        self._messages.clear()

    async def areset(self) -> None:
        self._messages.clear()
