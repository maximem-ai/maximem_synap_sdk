"""Synap short-term context for Pipecat voice pipelines.

Mirrors the existing :class:`SynapMemoryProcessor` pattern but injects
**short-term** conversation context (compacted summary + recent turns)
into the shared ``LLMContext`` on each :class:`TranscriptionFrame`.

Place BEFORE the user context aggregator in your pipeline. The shared
``LLMContext`` is mutated in place: a single tagged system message sits at
the head of the history (after the leading system prompt). The next turn's
ST replaces the previous one (we look up the tagged message and overwrite
its content) so the prompt window doesn't grow unbounded.

The block never sends a turn the context already holds a second time. Once
Synap has compacted the conversation, the context's earlier turns are
replaced by the compaction and the turns after it
(:func:`synap_pipecat.compaction.splice_synap_history`) and no block is
added. Before that, the block carries only the turns the context lacks (all
of them for a resumed conversation, none for a live one).

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

from synap_pipecat.compaction import _text_of, splice_synap_history

logger = logging.getLogger(__name__)

_SUPPORTED_STYLES = ("structured", "narrative", "bullet_points")
_DEFAULT_OPEN = "<synap_short_term_context>"
_DEFAULT_CLOSE = "</synap_short_term_context>"
_ST_MARKER = "<!-- synap_st -->"

_OnError = Literal["fallback", "raise"]


def _norm(text: object) -> str:
    return " ".join(str(text or "").split())


def _is_system(message: Any) -> bool:
    return isinstance(message, dict) and message.get("role") == "system"


def _is_st_message(message: Any) -> bool:
    return (
        _is_system(message)
        and isinstance(message.get("content"), str)
        and message["content"].startswith(_ST_MARKER)
    )


def _without_held_turns(response: Any, held: frozenset) -> str:
    """The ST text, minus the turns the context already holds.

    ``formatted_context`` carries the recent turns, so beside a context that
    holds them it sends each one twice. Nothing is dropped from a context that
    holds none of them (a resumed conversation), and the block is left whole
    when it has no turns to compare (a summary).
    """
    st_block = (getattr(response, "formatted_context", None) or "").strip()
    recent = getattr(response, "recent_messages", None)
    if not st_block or not held or not isinstance(recent, list):
        return st_block
    turns = [(getattr(m, "role", None), getattr(m, "content", None)) for m in recent]
    turns = [(r, t) for r, t in turns if r in ("user", "assistant") and isinstance(t, str)]
    if not turns:
        return st_block
    missing = [(r, t) for r, t in turns if (r, _norm(t)) not in held]
    if not missing:
        return ""
    if len(missing) == len(turns):
        return st_block
    return "\n".join(f"{r}: {t}" for r, t in missing)


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
                # The same read answers both: has Synap compacted this
                # conversation, and what does the block say.
                if await splice_synap_history(
                    self._context, self.sdk, conversation_id=self.conversation_id,
                    timeout=None, response=response,
                ):
                    self._drop_existing_st()  # the compaction replaced it
                    return
                st_block = _without_held_turns(response, self._held_turns())
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
        new_message = {"role": "system", "content": marker_content}

        # Replace the existing tagged system message, or insert one at the
        # head: after the leading system prompt, ahead of the history.
        messages = list(self._context.get_messages())
        for i, msg in enumerate(messages):
            if _is_st_message(msg):
                messages[i] = new_message
                self._context.set_messages(messages)
                return

        head = 0
        while head < len(messages) and _is_system(messages[head]):
            head += 1
        messages.insert(head, new_message)
        self._context.set_messages(messages)

    def _held_turns(self) -> frozenset:
        """The ``(role, text)`` turns the context already sends to the model."""
        return frozenset(
            (m["role"], _norm(_text_of(m))) for m in self._context.get_messages()
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
        )

    def _drop_existing_st(self) -> None:
        if self._context is None or not hasattr(self._context, "set_messages"):
            return
        messages = self._context.get_messages()
        filtered = [m for m in messages if not _is_st_message(m)]
        if len(filtered) != len(messages):
            self._context.set_messages(filtered)


__all__ = ["SynapShortTermProcessor"]
