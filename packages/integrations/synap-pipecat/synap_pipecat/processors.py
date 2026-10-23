"""Synap memory processor + recorder for Pipecat pipelines.

Two :class:`FrameProcessor` subclasses:

- :class:`SynapMemoryProcessor` — READ path. Inserted BEFORE the user
  context aggregator. On each :class:`TranscriptionFrame`, fetches Synap
  context via ``sdk.fetch`` and inlines it as a system message into the
  shared ``LLMContext`` so the LLM sees it before replying. The frame
  itself is passed through unchanged (the user aggregator still needs to
  see it to record the user turn).

- :class:`SynapRecorder` — WRITE path. Inserted AFTER the assistant
  context aggregator. Buffers the most recent user transcription and the
  streamed assistant text; on :class:`LLMFullResponseEndFrame` it records
  both turns: on the gRPC stream (opened on the first one if the caller did
  not open it; ``SYNAP_AUTO_LISTEN=false`` turns that off), over
  ``sdk.conversation.record_message`` when the stream cannot carry them,
  never both. With a ``context``, it then swaps the context's earlier turns
  for Synap's compaction once Synap has made one
  (:mod:`synap_pipecat.compaction`).

Error policy (see README):
- Read failures degrade gracefully — log at ERROR and skip context
  injection. A Synap blip must NEVER break a live voice call.
- Write failures surface as :class:`SynapIntegrationError` via
  ``wrap_sdk_errors_async``. The processor pushes an ``ErrorFrame``
  upstream so observability hooks fire, but swallows the exception
  locally — Pipecat's own error contract is frames-not-raises.
- Non-matching frames pass through untouched.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import List, Optional

from maximem_synap import MaximemSynapSDK

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMThoughtTextFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from synap_integrations_common import SynapIntegrationError, is_synap_history
from synap_integrations_common.stream_events import (
    record_turn,
    report_reasoning,
    report_tool_call,
    report_tool_result,
)

from synap_pipecat.compaction import HistorySplicer, _text_of

try:  # pipecat-ai >= 1.7: the one end-of-turn frame the assistant aggregator lets through
    from pipecat.frames.frames import LLMContextAssistantTurnFrame
except ImportError:  # older pipecat: its aggregator passes LLMFullResponseEndFrame on
    LLMContextAssistantTurnFrame = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Tool calls already reported, so a call seen as a frame and again in the
# context is reported once. An hours-long call makes tens of them.
_SEEN_LIMIT = 512
_IN_PROGRESS = "IN_PROGRESS"  # what the aggregator writes as a result until the real one lands


_SYSTEM_MEMORY_PREAMBLE = (
    "Relevant long-term memory about this user (from Synap):\n{body}\n"
    "Use this context only when it helps answer the user's current turn."
)


class SynapMemoryProcessor(FrameProcessor):
    """Injects Synap context into an ``LLMContext`` on each transcription.

    Placement: BEFORE the user context aggregator in the pipeline. On each
    :class:`TranscriptionFrame`, fetches user-scoped context from Synap
    and appends it as a system message to the shared ``LLMContext``
    before forwarding the frame downstream. If no ``LLMContext`` is
    supplied at construction, the processor becomes a read-only tap —
    frames still pass through, but no context is injected (useful if you
    want to plug in a recorder without changing the prompt).

    Args:
        sdk: Configured :class:`MaximemSynapSDK`.
        user_id: Required — Synap memory is user-scoped.
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on a
            B2C instance (user_context_isolation=equals_customer): the server
            rejects a call carrying one with HTTP 400. Leave it unset there. Empty string means
            customer-less.
        context: Shared :class:`LLMContext` used by the user/assistant
            aggregators. When ``None``, the processor is inert (no
            injection) but still passes frames through.
        mode: Synap fetch mode (``"accurate"`` or ``"fast"``).
        max_results: Cap on ``sdk.fetch`` per turn.
        include_conversation_context: Whether to request Synap's recent
            conversation block alongside the semantic memory.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        *,
        user_id: str,
        customer_id: str = "",
        context: Optional[LLMContext] = None,
        mode: str = "accurate",
        max_results: int = 10,
        include_conversation_context: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapMemoryProcessor requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapMemoryProcessor requires a non-empty user_id")
        super().__init__(name=name, **kwargs)
        self.sdk = sdk
        self.user_id = user_id
        self.customer_id = customer_id
        self._context = context
        self.mode = mode
        self.max_results = max_results
        self.include_conversation_context = include_conversation_context
        self._last_injected_message: Optional[dict] = None

    def set_context(self, context: LLMContext) -> None:
        """Attach or swap the shared :class:`LLMContext` at runtime."""
        self._context = context
        self._last_injected_message = None

    def inject_memory_message(self, formatted_context: str) -> None:
        """Replace the previously injected Synap memory block with a fresh
        one. Appending a new system message every turn would make the
        context grow with stale (and potentially contradictory) memory
        blocks over a long call — each turn should see exactly one,
        current, memory block."""
        if self._context is None:
            return
        if self._last_injected_message is not None:
            current = self._context.messages
            if self._last_injected_message in current:
                self._context.set_messages(
                    [m for m in current if m is not self._last_injected_message]
                )
        message = {
            "role": "system",
            "content": _SYSTEM_MEMORY_PREAMBLE.format(body=formatted_context),
        }
        self._context.add_message(message)
        self._last_injected_message = message

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text:
            await self._inject_context(frame.text)

        await self.push_frame(frame, direction)

    async def _inject_context(self, query: str) -> None:
        if self._context is None:
            return

        try:
            response = await self.sdk.fetch(
                user_id=self.user_id,
                customer_id=self.customer_id or None,
                search_query=[query],
                max_results=self.max_results,
                mode=self.mode,
                include_conversation_context=self.include_conversation_context,
            )
        except Exception as exc:  # noqa: BLE001 — read-side graceful degrade
            logger.error(
                "SynapMemoryProcessor: sdk.fetch failed user_id=%s error=%s",
                self.user_id, exc, exc_info=True,
            )
            return

        formatted = getattr(response, "formatted_context", None) or ""
        if not formatted.strip():
            return  # nothing to inject — pass through silently

        self.inject_memory_message(formatted.strip())


class SynapRecorder(FrameProcessor):
    """Records user + assistant turns to Synap at end-of-response.

    Placement: AFTER the assistant context aggregator in the pipeline.
    On :class:`LLMFullResponseEndFrame`, flushes the user and assistant
    turns (on the gRPC stream, REST only when the stream cannot carry them)
    and resets its buffers. When a ``context`` is given and Synap has
    compacted the conversation, the context's earlier turns are then
    replaced by the compaction and the turns after it, so the next LLM
    request carries that instead of every turn so far.

    **Where it sits decides what it sees.** Pipecat's universal
    ``LLMAssistantAggregator`` (pipecat-ai 1.x) consumes
    ``LLMFullResponseStartFrame``/``EndFrame``, every ``TextFrame`` (so
    ``TTSTextFrame`` too), ``LLMThoughtTextFrame`` and the function-call
    frames, and pushes none of them on. After it, the recorder gets one
    end-of-turn frame, ``LLMContextAssistantTurnFrame`` (the text of the
    assistant turn as stored in the context), and records from that: both
    turns, plus the tool calls and results it reads back out of the shared
    ``context``. Reasoning is not recoverable there. To also get tool calls
    at the moment they are dispatched, and reasoning, put the recorder
    BEFORE the assistant aggregator (for example right after the LLM
    service); the turn then ends on ``LLMFullResponseEndFrame``.

    Turn capture works across both pipeline generations:

    - Direct frames — :class:`TranscriptionFrame` (user) and
      :class:`LLMTextFrame` (assistant) are buffered when they reach this
      processor. This is how pre-universal-aggregator pipelines behave.
    - Pipecat ≥1.x universal aggregators + TTS — ``LLMUserAggregator``
      consumes ``TranscriptionFrame`` and TTS services consume
      ``LLMTextFrame``, so NEITHER reaches a recorder placed at the end of
      a standard voice pipeline. For those, the assistant text is rebuilt
      from the :class:`TTSTextFrame` stream (what was actually spoken —
      correct under interruption), and the user turn is recovered from the
      shared ``context`` at flush time. Pass ``context=`` whenever your
      pipeline uses the universal aggregators.

    Args:
        sdk: Configured :class:`MaximemSynapSDK`.
        user_id: Required — Synap conversations are user-scoped.
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on a
            B2C instance (user_context_isolation=equals_customer): the server
            rejects a call carrying one with HTTP 400. Leave it unset there. Empty string means
            customer-less.
        conversation_id: Stable id for this call. Auto-generated per
            processor lifetime when absent.
        context: Optional shared :class:`LLMContext` (the one given to the
            user/assistant aggregators). Used to recover the user turn when
            ``TranscriptionFrame`` never reaches this processor. It is also
            the context Synap's compaction is spliced into.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        *,
        user_id: str,
        customer_id: str = "",
        conversation_id: Optional[str] = None,
        context: Optional[LLMContext] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapRecorder requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapRecorder requires a non-empty user_id")
        super().__init__(name=name, **kwargs)
        self.sdk = sdk
        self.user_id = user_id
        self.customer_id = customer_id
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self._context = context

        self._user_buffer: Optional[str] = None
        self._assistant_parts: List[str] = []
        self._assistant_tts_parts: List[str] = []
        # Index of the last context message we recorded as a user turn, so
        # a turn that ends twice (e.g. function-call follow-ups) doesn't
        # record the same user message twice.
        self._last_user_msg_index: int = -1
        self._splicer = HistorySplicer()
        self._last_recorded_user = ""
        self._saw_response_end = False
        self._reported_calls: dict = {}

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text:
            # Latest transcription wins — voice pipelines can emit several
            # refinements before the LLM responds; only the last is the
            # committed user turn.
            self._user_buffer = frame.text
        elif isinstance(frame, TTSTextFrame) and frame.text:
            # Spoken-text stream from the TTS service (universal-aggregator
            # pipelines). Kept separate from _assistant_parts so pipelines
            # where both frame types arrive don't double-count.
            self._assistant_tts_parts.append(frame.text)
        elif isinstance(frame, LLMTextFrame) and frame.text:
            self._assistant_parts.append(frame.text)
        elif isinstance(frame, FunctionCallInProgressFrame):
            # ⚠ Voice is where this matters most. A tool call on a phone call
            # is dead air for the caller, and it is the clearest signal of
            # what the agent is about to need. Pipecat carries `tool_call_id`
            # on both frames, so the call and its result tie themselves
            # together without us inventing an id.
            call_id = str(getattr(frame, "tool_call_id", "") or "")
            if self._claim_call(call_id):
                await report_tool_call(
                    self.sdk,
                    tool_name=getattr(frame, "function_name", "") or "tool",
                    tool_args=getattr(frame, "arguments", None),
                    tool_call_id=call_id,
                    conversation_id=self.conversation_id,
                    user_id=self.user_id, customer_id=self.customer_id,
                )
        elif isinstance(frame, FunctionCallResultFrame):
            call_id = str(getattr(frame, "tool_call_id", "") or "")
            if self._claim_call("result:" + call_id if call_id else ""):
                await report_tool_result(
                    self.sdk,
                    result=getattr(frame, "result", None),
                    tool_name=getattr(frame, "function_name", "") or "",
                    tool_call_id=call_id,
                    conversation_id=self.conversation_id,
                    user_id=self.user_id, customer_id=self.customer_id,
                )
        elif isinstance(frame, LLMThoughtTextFrame) and getattr(frame, "text", ""):
            # Only reasoning models emit these, so most pipelines will never
            # see one. That is fine: reporting nothing is correct when the
            # model thought nothing out loud.
            await report_reasoning(
                self.sdk, content=frame.text,
                conversation_id=self.conversation_id,
                user_id=self.user_id, customer_id=self.customer_id,
            )
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._saw_response_end = True
            await self._flush()
        elif (
            LLMContextAssistantTurnFrame is not None
            and isinstance(frame, LLMContextAssistantTurnFrame)
            and not self._saw_response_end
        ):
            # Behind the universal assistant aggregator this is the only
            # end-of-turn signal. Where LLMFullResponseEndFrame does arrive the
            # turn is already recorded, and this frame (broadcast upstream
            # too) is ignored.
            await self._flush(spoken=getattr(frame, "text", "") or "", from_context=True)

        await self.push_frame(frame, direction)

    def _user_text_from_context(self) -> Optional[str]:
        """Recover the latest not-yet-recorded user message from the shared
        LLM context (used when TranscriptionFrames never reach us)."""
        if self._context is None:
            return None
        try:
            messages = self._context.get_messages()
        except Exception:  # noqa: BLE001 — context shape is LLM-specific
            return None
        for idx in range(len(messages) - 1, -1, -1):
            message = messages[idx]
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            if idx <= self._last_user_msg_index:
                return None  # already recorded on a previous flush
            content = message.get("content")
            if isinstance(content, list):
                text = " ".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ).strip()
            else:
                text = (content or "").strip() if isinstance(content, str) else ""
            if text and not is_synap_history(text):
                self._last_user_msg_index = idx
                return text
            return None
        return None

    async def _compact_context(self) -> None:
        """Swap the context's earlier turns for Synap's compaction, if it has one."""
        if not await self._splicer.apply(self._context, self.sdk, self.conversation_id):
            return
        # The splice shortened the list, so an index into the old one points at
        # the wrong message. Point it at the last user message that was
        # recorded, not at the last one: a caller who spoke over the reply has
        # a newer user message here that has not been recorded yet.
        messages = self._context.get_messages()
        users = [i for i, m in enumerate(messages)
                 if isinstance(m, dict) and m.get("role") == "user"]
        recorded = [i for i in users
                    if " ".join(_text_of(messages[i]).split()) == self._last_recorded_user]
        self._last_user_msg_index = (recorded or users or [-1])[-1]

    def _claim_call(self, call_id: str) -> bool:
        """Say whether this tool call is new. A call with no id is always new."""
        if not call_id:
            return True
        if call_id in self._reported_calls:
            return False
        self._reported_calls[call_id] = None
        while len(self._reported_calls) > _SEEN_LIMIT:
            self._reported_calls.pop(next(iter(self._reported_calls)))
        return True

    async def _report_context_tools(self) -> None:
        """Report the tool calls and results the context holds that no frame
        reported: the only trace of them behind the assistant aggregator."""
        if self._context is None:
            return
        try:
            messages = list(self._context.get_messages())
        except Exception:  # noqa: BLE001 — context shape is LLM-specific
            return
        names: dict = {}
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    call_id = str(call.get("id") or "")
                    names[call_id] = fn.get("name") or "tool"
                    if self._claim_call(call_id):
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except (TypeError, ValueError):
                            args = {"input": str(fn.get("arguments"))}
                        await report_tool_call(
                            self.sdk, tool_name=names[call_id],
                            tool_args=args if isinstance(args, dict) else {"input": str(args)},
                            tool_call_id=call_id,
                            conversation_id=self.conversation_id,
                            user_id=self.user_id, customer_id=self.customer_id,
                        )
            elif message.get("role") == "tool":
                call_id = str(message.get("tool_call_id") or "")
                result = message.get("content")
                if not call_id or result == _IN_PROGRESS or not self._claim_call("result:" + call_id):
                    continue
                await report_tool_result(
                    self.sdk, result=result, tool_name=names.get(call_id, ""),
                    tool_call_id=call_id,
                    conversation_id=self.conversation_id,
                    user_id=self.user_id, customer_id=self.customer_id,
                )

    async def _flush(self, spoken: str = "", from_context: bool = False) -> None:
        user_text = self._user_buffer or self._user_text_from_context()
        assistant_text = "".join(self._assistant_parts).strip()
        if not assistant_text:
            assistant_text = " ".join(
                part.strip() for part in self._assistant_tts_parts if part.strip()
            ).strip()
        if not assistant_text:
            assistant_text = spoken.strip()
        # Reset buffers before we await — the next turn may start flowing
        # while we're still writing to Synap.
        self._user_buffer = None
        self._assistant_parts = []
        self._assistant_tts_parts = []

        if not user_text and not assistant_text:
            return

        # Stream first (opened here if the caller did not), REST only for what
        # the stream could not take, never both: the server persists the turn
        # from the stream itself, so a `record_message` on top of a delivered
        # event stores it twice and extracts it twice. `record_turn` decides.
        if user_text:
            self._last_recorded_user = " ".join(user_text.split())
        failed = False
        for role, text in (("user", user_text), ("assistant", assistant_text)):
            if role == "assistant" and from_context:
                await self._report_context_tools()  # before the reply they led to
            if text and not await record_turn(
                self.sdk, role=role, content=text,
                conversation_id=self.conversation_id,
                user_id=self.user_id, customer_id=self.customer_id,
            ):
                failed = True
        if failed:
            # Write failures surface as ErrorFrame (Pipecat's contract:
            # exceptions inside process_frame would tear down the whole
            # pipeline). `record_turn` has already logged the cause.
            exc = SynapIntegrationError(
                "pipecat.record_turn", "turn write failed",
                {"conversation_id": self.conversation_id, "user_id": self.user_id},
            )
            await self.push_frame(
                ErrorFrame(error=str(exc), exception=exc),
                FrameDirection.UPSTREAM,
            )
        await self._compact_context()
