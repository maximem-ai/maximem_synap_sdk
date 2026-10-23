"""Let Synap's compaction decide what history the model sees on a LiveKit call.

A LiveKit ``ChatContext`` only grows: nothing in ``livekit-agents`` compacts it
(``ChatContext.truncate`` is a manual call). Left alone, a long call re-sends
every turn on every LLM request.

The seam is ``Agent.llm_node(chat_ctx, tools, model_settings)``. The framework
hands it a **copy** of the agent's history for one generation, so rewriting it
changes what the model reads and nothing the agent keeps. :func:`install_synap_history`
wraps the agent's ``llm_node`` so that, once Synap has compacted the
conversation, every turn before the current one is replaced by Synap's
compaction and the turns after it (:func:`synap_integrations_common.splice_history`).
Before the first compaction, or when Synap cannot answer, the history is left
exactly as LiveKit built it.

Not reachable: a ``RealtimeModel`` session. There the provider holds the
history and ``llm_node`` is never called (``AgentActivity._realtime_reply_task``
does not use it).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from types import SimpleNamespace
from collections.abc import AsyncIterable
from typing import Any, Optional

from livekit.agents import ChatContext, ChatMessage
from synap_integrations_common import fetch_compacted_history, splice_history

logger = logging.getLogger(__name__)

# A voice turn cannot wait on a slow read. Past this the history goes through
# unchanged, and the read is not tried again for _COOLDOWN_S.
_FETCH_TIMEOUT_S = 1.5
_COOLDOWN_S = 30.0

_INSTALLED_ATTR = "_synap_history_splice"


def _role_of(item: Any) -> str:
    if getattr(item, "type", None) != "message":
        return "tool"  # function_call, function_call_output, handoff, config update
    role = getattr(item, "role", None)
    return "system" if role == "developer" else str(role)


def _text_of(item: Any) -> str:
    return getattr(item, "text_content", None) or ""


def _make_message(role: str, text: str) -> ChatMessage:
    return ChatMessage(role=role, content=[text])  # type: ignore[arg-type]


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
    chat_ctx: ChatContext,
    sdk: Any,
    *,
    conversation_id: str,
    timeout: Optional[float] = _FETCH_TIMEOUT_S,
    response: Any = None,
) -> bool:
    """Replace the turns before the current one in ``chat_ctx`` with Synap's
    compaction, in place. Returns whether it did.

    ``response`` is a ``get_context_for_prompt`` result already in hand, used
    instead of reading again.

    False, with ``chat_ctx`` untouched, when the conversation has no compaction
    yet, when the read fails or times out. Never raises.
    """
    try:
        coro = fetch_compacted_history(
            _Replay(response) if response is not None else sdk, conversation_id)
        history = await (_within(coro, timeout) if timeout else coro)
        if history is None:
            return False
        chat_ctx.items[:] = splice_history(
            list(chat_ctx.items), history,
            role_of=_role_of, text_of=_text_of, make_message=_make_message,
        )
        return True
    except Exception:  # noqa: BLE001 — the framework's own history still stands
        logger.warning("synap: history splice skipped (conversation_id=%s)",
                       conversation_id, exc_info=True)
        return False


class _HistorySplice:
    """One conversation's splice, shared by every agent it is installed on."""

    def __init__(self, sdk: Any, conversation_id: str) -> None:
        self.sdk = sdk
        self.conversation_id = conversation_id
        self._skip_until = 0.0

    async def apply(self, chat_ctx: ChatContext) -> None:
        if time.monotonic() < self._skip_until:
            return
        started = time.monotonic()
        spliced = await splice_synap_history(
            chat_ctx, self.sdk, conversation_id=self.conversation_id,
            timeout=_FETCH_TIMEOUT_S)
        if not spliced and time.monotonic() - started >= _FETCH_TIMEOUT_S:
            self._skip_until = time.monotonic() + _COOLDOWN_S

    def install(self, agent: Any) -> bool:
        """Wrap ``agent.llm_node``. Idempotent; returns whether this call wrapped it."""
        existing = getattr(agent, _INSTALLED_ATTR, None)
        if existing is not None:
            existing.rebind(self)
            return False
        original = agent.llm_node
        holder = _Binding(self)

        async def llm_node(chat_ctx: ChatContext, tools: Any, model_settings: Any):
            await holder.splice.apply(chat_ctx)
            result = original(chat_ctx, tools, model_settings)
            if inspect.isawaitable(result):
                result = await result
            if result is None:
                return
            if not isinstance(result, AsyncIterable):
                yield result  # a whole reply as a str or one ChatChunk
                return
            iterator = result.__aiter__()
            try:
                async for chunk in iterator:
                    yield chunk
            finally:
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    await aclose()

        try:
            agent.llm_node = llm_node
            setattr(agent, _INSTALLED_ATTR, holder)
        except Exception:  # noqa: BLE001 — an agent that refuses attributes keeps its own history
            logger.warning("synap: could not install the history splice", exc_info=True)
            return False
        return True


class _Binding:
    """Lets a second attach to the same agent point the wrapper at its conversation."""

    def __init__(self, splice: _HistorySplice) -> None:
        self.splice = splice

    def rebind(self, splice: _HistorySplice) -> None:
        self.splice = splice


def install_synap_history(agent: Any, sdk: Any, *, conversation_id: str) -> bool:
    """Make ``agent`` send Synap's compaction instead of its own earlier turns.

    Wraps ``agent.llm_node`` (a subclass override of it still runs, after the
    splice). :func:`attach_synap_recording` calls this for the session's agent
    by default. Returns whether it wrapped the agent just now.
    """
    return _HistorySplice(sdk, conversation_id).install(agent)


__all__ = ["install_synap_history", "splice_synap_history"]
