"""Let Synap's compaction decide what history the model sees in a Pipecat call.

Pipecat's ``LLMContext`` only grows unless ``enable_auto_context_summarization``
is on (it is off by default). The framework's own compaction works by rewriting
the shared context in place: ``LLMContextSummarizer._apply_summary`` ends in
``context.set_messages(new_messages)``. This does the same with Synap's
compaction (:func:`synap_integrations_common.splice_history`): once Synap has
compacted the conversation, every turn before the current one in the shared
context is replaced by Synap's compaction and the turns after it.

Before the first compaction, or when Synap cannot answer, the context is left
exactly as the aggregators built it, and ``enable_auto_context_summarization``
(if the caller turned it on) stays the fallback.

:class:`~synap_pipecat.SynapRecorder` applies it after each turn it records,
when the caller has nothing to wait for; :class:`~synap_pipecat.SynapShortTermProcessor`
applies it before each user turn.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from typing import Any, Optional

from pipecat.processors.aggregators.llm_context import LLMContext
from synap_integrations_common import fetch_compacted_history, splice_history

logger = logging.getLogger(__name__)

# A voice call cannot wait on a slow read. Past this the context goes through
# unchanged, and the read is not tried again for _COOLDOWN_S.
_FETCH_TIMEOUT_S = 1.5
_COOLDOWN_S = 30.0


def _role_of(message: Any) -> str:
    if not isinstance(message, dict):
        return "tool"  # an LLMSpecificMessage: not a turn we can rewrite
    role = message.get("role")
    return "system" if role == "developer" else str(role)


def _text_of(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text")
        ).strip()
    return ""


def _make_message(role: str, text: str) -> dict:
    return {"role": role, "content": text}


async def _within(awaitable: Any, timeout: float) -> Any:
    task = asyncio.ensure_future(awaitable)
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if not done:
        task.cancel()
        raise TimeoutError(f"no answer within {timeout:g}s")
    return task.result()


class _Replay:
    """An SDK stand-in that answers ``get_context_for_prompt`` with a response
    the caller already has, so one read can serve both the splice and the
    short-term block."""

    def __init__(self, response: Any) -> None:
        async def get_context_for_prompt(**_kwargs: Any) -> Any:
            return response

        self.conversation = SimpleNamespace(
            context=SimpleNamespace(get_context_for_prompt=get_context_for_prompt))


async def splice_synap_history(
    context: LLMContext,
    sdk: Any,
    *,
    conversation_id: str,
    timeout: Optional[float] = _FETCH_TIMEOUT_S,
    response: Any = None,
) -> bool:
    """Replace the turns before the current one in ``context`` with Synap's
    compaction, in place. Returns whether it did.

    ``response`` is a ``get_context_for_prompt`` result already in hand, used
    instead of reading again.

    False, with ``context`` untouched, when the conversation has no compaction
    yet or the read fails or times out. Never raises.
    """
    try:
        coro = fetch_compacted_history(
            _Replay(response) if response is not None else sdk, conversation_id)
        history = await (_within(coro, timeout) if timeout else coro)
        if history is None:
            return False
        # No await from here to set_messages: the aggregators append to this
        # list between awaits, and a snapshot taken before one would drop it.
        context.set_messages(splice_history(
            list(context.get_messages()), history,
            role_of=_role_of, text_of=_text_of, make_message=_make_message,
        ))
        return True
    except Exception:  # noqa: BLE001 — the aggregators' own context still stands
        logger.warning("synap: history splice skipped (conversation_id=%s)",
                       conversation_id, exc_info=True)
        return False


class HistorySplicer:
    """The splice for one pipeline processor, with a back-off after a slow read."""

    def __init__(self) -> None:
        self._skip_until = 0.0

    async def apply(
        self, context: Optional[LLMContext], sdk: Any, conversation_id: str,
    ) -> bool:
        if context is None or time.monotonic() < self._skip_until:
            return False
        started = time.monotonic()
        spliced = await splice_synap_history(
            context, sdk, conversation_id=conversation_id, timeout=_FETCH_TIMEOUT_S)
        if not spliced and time.monotonic() - started >= _FETCH_TIMEOUT_S:
            self._skip_until = time.monotonic() + _COOLDOWN_S
        return spliced


__all__ = ["HistorySplicer", "splice_synap_history"]
