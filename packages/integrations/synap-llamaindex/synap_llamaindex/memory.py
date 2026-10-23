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

## What :meth:`aget` returns

This memory *replaces* the agent's chat buffer, so :meth:`aget` is the one place
that decides what history the model is sent. It keeps the messages put on it in
this process (full fidelity: tool calls, results, reasoning) and, once Synap has
compacted the conversation, replaces every turn before the current one with
Synap's compaction and the turns after it. Until then the local messages are
returned as they are. It used to return Synap's formatted context (which already
contains the recent turns), then the recent turns again, then every local
message: each turn two or three times, and unbounded.

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
matters: on the gRPC stream, which the first event opens (there is no
``listen()`` to call; ``SYNAP_AUTO_LISTEN=false`` turns that off), REST only
as a fallback, **never both**, because the server persists ``user_message`` and
``assistant_message`` straight off the stream (``grpc/servicer.py``), so a REST
write on top of a delivered stream event writes the turn twice and extracts it
twice.

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
from collections import Counter
from typing import Any, List, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory.types import BaseMemory

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    CompactedHistory,
    SynapIntegrationError,
    is_synap_history,
    record_turn,
    run_async,
    splice_history,
)
from synap_integrations_common.history import HISTORY_CLOSE, HISTORY_OPEN
from synap_integrations_common.stream_events import (
    report_reasoning,
    report_tool_call,
    report_tool_result,
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


_ROLES = {
    MessageRole.USER: "user",
    MessageRole.ASSISTANT: "assistant",
    MessageRole.SYSTEM: "system",
}
_ROLE_OF_NAME = {"user": MessageRole.USER, "assistant": MessageRole.ASSISTANT,
                 "system": MessageRole.SYSTEM}


def _role_of(message: Any) -> str:
    return _ROLES.get(getattr(message, "role", None), "other")


def _make_message(role: str, text: str) -> ChatMessage:
    return ChatMessage(role=_ROLE_OF_NAME[role], content=text)


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

        local = list(self._messages)
        history, recent = await self._load_history()
        if history is not None:
            # Synap's compaction in place of every turn before the current one.
            # Only text is swapped; the current turn (and the tool calls and
            # results after it) stay as the agent built them.
            try:
                messages.extend(splice_history(
                    local, history,
                    role_of=_role_of, text_of=_text_of, make_message=_make_message,
                ))
                return messages
            except Exception:  # noqa: BLE001 — the local messages still stand
                logger.warning("SynapChatMemory.aget: could not splice the compacted "
                               "history", exc_info=True)
        if local:
            messages.extend(local)
        else:
            # A fresh process picking a conversation up: nothing put here yet,
            # so Synap's recent turns are the history.
            messages.extend(_make_message(role, text) for role, text in recent)
        return messages

    async def _load_history(self) -> tuple:
        """``(compaction or None, recent turns)`` as Synap holds them.

        The compaction is None until the conversation has been compacted, and
        on any SDK failure (logged at ERROR; a lookup miss must not stop the
        agent).
        """
        try:
            ctx = await self._sdk.conversation.context.get_context_for_prompt(
                conversation_id=self._conversation_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "SynapChatMemory.aget: get_context_for_prompt failed "
                "conversation_id=%s error=%s",
                self._conversation_id, exc, exc_info=True,
            )
            return None, []
        recent = [
            (m.role, str(m.content))
            for m in (getattr(ctx, "recent_messages", None) or [])
            if getattr(m, "role", None) in ("user", "assistant")
            and isinstance(getattr(m, "content", None), str) and m.content.strip()
        ]
        compaction = getattr(ctx, "context_block", None)
        if getattr(ctx, "available", True) is False or not isinstance(compaction, str) \
                or not compaction.strip():
            return None, recent
        parts = [compaction.strip()]
        found = getattr(ctx, "history_block", None)
        if isinstance(found, str) and found.strip():
            parts.append(found.strip())
        return CompactedHistory(
            preamble=f"{HISTORY_OPEN}\n" + "\n\n".join(parts) + f"\n{HISTORY_CLOSE}",
            turns=tuple(recent),
        ), recent

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

    async def aput_messages(self, messages: List[ChatMessage]) -> None:
        """Put each message with :meth:`aput`, on the caller's event loop.

        ⚠ ``BaseMemory.aput_messages`` is ``asyncio.to_thread(self.put_messages)``,
        and that calls the sync ``put``, which runs :meth:`aput` on the
        integrations' background loop. The gRPC stream lives on the loop that
        opened it, so from there nothing could use it: every turn fell back to
        REST and every tool call, result and reasoning step was dropped. This is
        the method ``FunctionAgent`` calls to hand over its whole scratchpad.
        """
        for message in messages:
            await self.aput(message)

    async def _record(self, role: str, content: str) -> None:
        """On the stream (opened here if need be), REST only if the stream could
        not carry it, never both. Raises if neither took the turn."""
        if is_synap_history(content):
            return  # the compaction aget returned, not something anybody said
        if not await record_turn(
            self._sdk, role=role, content=content,
            conversation_id=self._conversation_id,
            user_id=self._user_id, customer_id=self._customer_id,
        ):
            raise SynapIntegrationError(
                "llamaindex.aput",
                "the turn was not recorded: neither the gRPC stream nor REST accepted it",
                {"role": role, "conversation_id": self._conversation_id},
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
        run_async(self.aset(messages))

    async def aset(self, messages: List[ChatMessage]) -> None:
        """Replace the local messages, and record the turns Synap does not have.

        An agent calls this with the ``chat_history`` it was handed. Turns that
        were already put on this memory, or that Synap already holds after its
        last compaction, are not recorded again; the rest are, so a history
        that arrives this way reaches Synap like any other.
        """
        previous = Counter(
            (_role_of(m), _text_of(m).strip()) for m in self._messages
        )
        self._messages = list(messages)
        known = previous + await self._synap_turns()
        for message in messages:
            role = _role_of(message)
            text = _text_of(message).strip()
            if role not in ("user", "assistant") or not text:
                continue
            key = (role, text)
            if known[key] > 0:
                known[key] -= 1
                continue
            await self._record(role, text)

    async def _synap_turns(self) -> Counter:
        """The turns Synap holds after its last compaction, by (role, text)."""
        _, recent = await self._load_history()  # logs, and answers [], on failure
        return Counter((role, text.strip()) for role, text in recent)

    def reset(self) -> None:
        self._messages.clear()

    async def areset(self) -> None:
        self._messages.clear()
