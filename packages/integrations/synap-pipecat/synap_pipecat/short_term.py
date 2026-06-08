"""Synap short-term context for Pipecat voice pipelines.

Mirrors the existing :class:`SynapMemoryProcessor` pattern but injects
**short-term** conversation context (compacted summary + recent turns)
into the shared ``LLMContext`` on each :class:`TranscriptionFrame`.

Place BEFORE the user context aggregator in your pipeline. The shared
``LLMContext`` is mutated in place: a single tagged system message is
appended on every user turn. The next turn's ST replaces the previous
one (we look up the tagged message and overwrite its content) so the
prompt window doesn't grow unbounded.

Quality contract identical to the LangGraph adapter:

- ``conversation_id`` required + explicit at construction.
- Read failures degrade silently by default
  (``on_error="fallback"``): a Synap blip must never break a live call.
  ``on_error="raise"`` available for tests.
- Empty ST is a no-op — never injects a blank system message.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from maximem_synap import MaximemSynapSDK

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from synap_integrations_common import (
    SynapIntegrationError,
    wrap_sdk_errors_async,
)

logger = logging.getLogger(__name__)

_SUPPORTED_STYLES = ("structured", "narrative", "bullet_points")
_DEFAULT_OPEN = "<synap_short_term_context>"
_DEFAULT_CLOSE = "</synap_short_term_context>"
_ST_MARKER = "<!-- synap_st -->"

_OnError = Literal["fallback", "raise"]


class SynapShortTermProcessor(FrameProcessor):
    """Injects Synap short-term context into an ``LLMContext`` per user turn.

    Placement: BEFORE the user context aggregator. On each
    :class:`TranscriptionFrame` we call
    ``sdk.conversation.context.get_context_for_prompt`` (cache-first),
    locate or insert a tagged system message in the shared
    ``LLMContext``, and refresh its content with the latest ST.

    Frames pass through unchanged.

    Args:
        sdk: Configured :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation ID. **Required.**
        context: Shared :class:`LLMContext` used by the aggregators.
            When ``None``, the processor is inert (frames still pass
            through, no injection happens).
        style: One of ``"structured" | "narrative" | "bullet_points"``.
        preamble_open / preamble_close: ST block wrappers; pass ``None``
            for both to drop the tags.
        on_error: ``"fallback"`` (default) or ``"raise"``.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        *,
        conversation_id: str,
        context: Optional[LLMContext] = None,
        style: str = "narrative",
        preamble_open: Optional[str] = _DEFAULT_OPEN,
        preamble_close: Optional[str] = _DEFAULT_CLOSE,
        on_error: _OnError = "fallback",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapShortTermProcessor requires a non-None sdk")
        if not conversation_id or not str(conversation_id).strip():
            raise ValueError(
                "SynapShortTermProcessor requires a non-empty conversation_id"
            )
        if style not in _SUPPORTED_STYLES:
            raise ValueError(
                f"SynapShortTermProcessor: unsupported style={style!r}; "
                f"expected one of {_SUPPORTED_STYLES}"
            )
        if on_error not in ("fallback", "raise"):
            raise ValueError(
                "SynapShortTermProcessor: on_error must be 'fallback' or "
                f"'raise', got {on_error!r}"
            )

        super().__init__(name=name, **kwargs)
        self.sdk = sdk
        self.conversation_id = conversation_id
        self._context = context
        self.style = style
        self.preamble_open = preamble_open
        self.preamble_close = preamble_close
        self.on_error: _OnError = on_error

    def set_context(self, context: LLMContext) -> None:
        """Attach or swap the shared :class:`LLMContext` at runtime."""
        self._context = context

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text:
            await self._refresh_st()
        await self.push_frame(frame, direction)

    async def _refresh_st(self) -> None:
        if self._context is None:
            return

        st_block = ""
        try:
            async with wrap_sdk_errors_async(
                "synap_pipecat.SynapShortTermProcessor._refresh_st",
                logger,
                conversation_id=self.conversation_id,
                style=self.style,
            ):
                response = await self.sdk.conversation.context.get_context_for_prompt(
                    conversation_id=self.conversation_id,
                    style=self.style,
                )
            if getattr(response, "available", False):
                st_block = (getattr(response, "formatted_context", None) or "").strip()
        except SynapIntegrationError:
            if self.on_error == "raise":
                raise
            return

        if not st_block:
            self._drop_existing_st()
            return

        if self.preamble_open and self.preamble_close:
            wrapped = f"{self.preamble_open}\n{st_block}\n{self.preamble_close}"
        else:
            wrapped = st_block

        marker_content = f"{_ST_MARKER}\n{wrapped}"

        # Replace existing tagged system message or append a new one.
        messages = self._context.get_messages()
        for i, msg in enumerate(messages):
            if (
                isinstance(msg, dict)
                and msg.get("role") == "system"
                and isinstance(msg.get("content"), str)
                and msg["content"].startswith(_ST_MARKER)
            ):
                messages[i] = {"role": "system", "content": marker_content}
                # Reset the context's messages list. LLMContext exposes a
                # ``set_messages`` method on most versions.
                if hasattr(self._context, "set_messages"):
                    self._context.set_messages(messages)
                return

        # Not found — append.
        self._context.add_message(
            {"role": "system", "content": marker_content}
        )

    def _drop_existing_st(self) -> None:
        if self._context is None or not hasattr(self._context, "set_messages"):
            return
        messages = self._context.get_messages()
        filtered = [
            m for m in messages
            if not (
                isinstance(m, dict)
                and m.get("role") == "system"
                and isinstance(m.get("content"), str)
                and m["content"].startswith(_ST_MARKER)
            )
        ]
        if len(filtered) != len(messages):
            self._context.set_messages(filtered)


__all__ = ["SynapShortTermProcessor"]
