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
  both turns via ``sdk.conversation.record_message``.

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

import logging
import uuid
from typing import List, Optional

from maximem_synap import MaximemSynapSDK

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from synap_integrations_common import (
    SynapIntegrationError,
    wrap_sdk_errors_async,
)

logger = logging.getLogger(__name__)


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
        customer_id: Optional customer/org scope. Empty string means
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
    turns via ``sdk.conversation.record_message`` and resets its buffers.

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
        customer_id: Optional customer/org scope. Empty string means
            customer-less.
        conversation_id: Stable id for this call. Auto-generated per
            processor lifetime when absent.
        context: Optional shared :class:`LLMContext` (the one given to the
            user/assistant aggregators). Used to recover the user turn when
            ``TranscriptionFrame`` never reaches this processor.
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
        self.conversation_id = conversation_id or f"pipecat-{uuid.uuid4().hex[:12]}"
        self._context = context

        self._user_buffer: Optional[str] = None
        self._assistant_parts: List[str] = []
        self._assistant_tts_parts: List[str] = []
        # Index of the last context message we recorded as a user turn, so
        # a turn that ends twice (e.g. function-call follow-ups) doesn't
        # record the same user message twice.
        self._last_user_msg_index: int = -1

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
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._flush()

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
            if text:
                self._last_user_msg_index = idx
                return text
            return None
        return None

    async def _flush(self) -> None:
        user_text = self._user_buffer or self._user_text_from_context()
        assistant_text = "".join(self._assistant_parts).strip()
        if not assistant_text:
            assistant_text = " ".join(
                part.strip() for part in self._assistant_tts_parts if part.strip()
            ).strip()
        # Reset buffers before we await — the next turn may start flowing
        # while we're still writing to Synap.
        self._user_buffer = None
        self._assistant_parts = []
        self._assistant_tts_parts = []

        if not user_text and not assistant_text:
            return

        try:
            async with wrap_sdk_errors_async(
                "pipecat.record_turn",
                logger,
                conversation_id=self.conversation_id,
                user_id=self.user_id,
            ):
                if user_text:
                    await self.sdk.conversation.record_message(
                        conversation_id=self.conversation_id,
                        role="user",
                        content=user_text,
                        user_id=self.user_id,
                        customer_id=self.customer_id,
                    )
                if assistant_text:
                    await self.sdk.conversation.record_message(
                        conversation_id=self.conversation_id,
                        role="assistant",
                        content=assistant_text,
                        user_id=self.user_id,
                        customer_id=self.customer_id,
                    )
        except SynapIntegrationError as exc:
            # Write failures surface as ErrorFrame (Pipecat's contract —
            # exceptions inside process_frame would tear down the whole
            # pipeline). The wrap_sdk_errors_async call has already
            # logged the underlying cause.
            await self.push_frame(
                ErrorFrame(error=str(exc), exception=exc),
                FrameDirection.UPSTREAM,
            )
