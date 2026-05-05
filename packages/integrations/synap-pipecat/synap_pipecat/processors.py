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

    def set_context(self, context: LLMContext) -> None:
        """Attach or swap the shared :class:`LLMContext` at runtime."""
        self._context = context

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

        self._context.add_message({
            "role": "system",
            "content": _SYSTEM_MEMORY_PREAMBLE.format(body=formatted.strip()),
        })


class SynapRecorder(FrameProcessor):
    """Records user + assistant turns to Synap at end-of-response.

    Placement: AFTER the assistant context aggregator in the pipeline.
    Buffers the most recent user transcription (from
    :class:`TranscriptionFrame`) and the streamed assistant tokens (from
    :class:`LLMTextFrame`); on :class:`LLMFullResponseEndFrame`, flushes
    both turns via ``sdk.conversation.record_message`` and resets the
    buffer.

    Args:
        sdk: Configured :class:`MaximemSynapSDK`.
        user_id: Required — Synap conversations are user-scoped.
        customer_id: Optional customer/org scope. Empty string means
            customer-less.
        conversation_id: Stable id for this call. Auto-generated per
            processor lifetime when absent.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        *,
        user_id: str,
        customer_id: str = "",
        conversation_id: Optional[str] = None,
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

        self._user_buffer: Optional[str] = None
        self._assistant_parts: List[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text:
            # Latest transcription wins — voice pipelines can emit several
            # refinements before the LLM responds; only the last is the
            # committed user turn.
            self._user_buffer = frame.text
        elif isinstance(frame, LLMTextFrame) and frame.text:
            self._assistant_parts.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._flush()

        await self.push_frame(frame, direction)

    async def _flush(self) -> None:
        user_text = self._user_buffer
        assistant_text = "".join(self._assistant_parts).strip()
        # Reset buffers before we await — the next turn may start flowing
        # while we're still writing to Synap.
        self._user_buffer = None
        self._assistant_parts = []

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
