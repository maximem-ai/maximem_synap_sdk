"""Synap short-term context for LiveKit Agents (realtime voice).

Two helpers mirroring the existing :func:`preload_synap_context` pattern
but for **short-term** context (compacted summary + recent turns per
conversation):

- :func:`preload_synap_st` — async helper called **before**
  ``session.start(...)`` to prepend a Synap ST ``system`` message to
  the agent's ``ChatContext``. One-shot — captures ST as of session
  start.

- :func:`refresh_synap_st` — async helper safe to call inside
  ``Agent.on_user_turn_completed`` (or any per-turn hook) to keep the
  ST system message **refreshed** on every turn. It locates and
  replaces an existing Synap-tagged ST message in the ``ChatContext``,
  or inserts a new one at the head if absent.

Both wrap ``sdk.conversation.context.get_context_for_prompt`` (cache-
first behind ``SYNAP_SDK_ST_AUTHORITATIVE``).

Neither helper sends a turn the agent's ``ChatContext`` already holds a
second time. Once Synap has compacted the conversation they replace the
context's earlier turns with the compaction and the turns after it
(:func:`synap_livekit_agents.compaction.splice_synap_history`). Before that,
the ST block carries only the turns the context does not already hold (all of
them for a resumed conversation, none for a live one).

Quality contract identical to the LangGraph adapter:

- ``conversation_id`` required + explicit.
- Read failures degrade silently by default (``on_error="fallback"``):
  logged via :class:`SynapIntegrationError`, ChatContext untouched —
  a Synap blip must never break a live call. ``on_error="raise"``
  available for tests.
- Empty ST is a no-op — never prepends a blank system message.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from livekit.agents import ChatContext, ChatMessage
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    SynapIntegrationError,
    is_synap_history,
    wrap_sdk_errors_async,
)

from synap_livekit_agents.compaction import splice_synap_history

logger = logging.getLogger(__name__)

_SUPPORTED_STYLES = ("structured", "narrative", "bullet_points")
_DEFAULT_OPEN = "<synap_short_term_context>"
_DEFAULT_CLOSE = "</synap_short_term_context>"

# Tag the system message content with this marker so refresh_synap_st can
# locate and replace it in subsequent turns without disturbing the user's
# other system messages.
_ST_MARKER = "<!-- synap_st -->"

_OnError = Literal["fallback", "raise"]


def _validate(
    sdk: Optional[MaximemSynapSDK],
    conversation_id: str,
    style: str,
    on_error: str,
    site: str,
) -> None:
    if sdk is None:
        raise ValueError(f"{site} requires a non-None sdk")
    if not conversation_id or not str(conversation_id).strip():
        raise ValueError(f"{site} requires a non-empty conversation_id")
    if style not in _SUPPORTED_STYLES:
        raise ValueError(
            f"{site}: unsupported style={style!r}; "
            f"expected one of {_SUPPORTED_STYLES}"
        )
    if on_error not in ("fallback", "raise"):
        raise ValueError(
            f"{site}: on_error must be 'fallback' or 'raise', got {on_error!r}"
        )


async def _fetch_st(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    style: str,
    on_error: _OnError,
    site: str,
) -> Optional[object]:
    """The ``get_context_for_prompt`` response, or None when there is no ST."""
    try:
        async with wrap_sdk_errors_async(
            site,
            logger,
            conversation_id=conversation_id,
            style=style,
        ):
            response = await sdk.conversation.context.get_context_for_prompt(
                conversation_id=conversation_id,
                style=style,
            )
    except SynapIntegrationError:
        if on_error == "raise":
            raise
        return None
    return response if getattr(response, "available", False) else None


def _wrap_block(
    response: object,
    held: frozenset,
    preamble_open: Optional[str],
    preamble_close: Optional[str],
) -> str:
    st_block = _without_held_turns(response, held)
    if not st_block:
        return ""
    if preamble_open and preamble_close:
        return f"{preamble_open}\n{st_block}\n{preamble_close}"
    return st_block


def _norm(text: object) -> str:
    return " ".join(str(text or "").split())


def _held_turns(chat_ctx: ChatContext) -> frozenset:
    """The ``(role, text)`` turns the context already sends to the model."""
    return frozenset(
        (m.role, _norm(m.text_content)) for m in chat_ctx.items
        if getattr(m, "type", None) == "message" and m.role in ("user", "assistant")
    )


def _without_held_turns(response: object, held: frozenset) -> str:
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


def _wrap_for_chat_message(content: str) -> str:
    """Prefix the marker so refresh_synap_st can find this message later."""
    return f"{_ST_MARKER}\n{content}"


def _is_synap_st_message(msg: ChatMessage) -> bool:
    content = getattr(msg, "content", None)
    role = getattr(msg, "role", None)
    if role != "system":
        return False
    if isinstance(content, str):
        return content.startswith(_ST_MARKER)
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str) and part.startswith(_ST_MARKER):
                return True
    return False


def _head_index(chat_ctx: ChatContext) -> int:
    """Where the ST block goes: after the leading system messages (the agent's
    instructions), ahead of the history."""
    i = 0
    while i < len(chat_ctx.items) and getattr(chat_ctx.items[i], "type", None) == "message" \
            and chat_ctx.items[i].role in ("system", "developer"):
        i += 1
    return i


def _drop_st_messages(chat_ctx: ChatContext) -> None:
    chat_ctx.items[:] = [m for m in chat_ctx.items if not _is_synap_st_message(m)]


async def _splice_compaction(
    chat_ctx: ChatContext, sdk: MaximemSynapSDK, conversation_id: str, response: object,
) -> Optional[ChatMessage]:
    """Swap the context's earlier turns for Synap's compaction, if it has one.

    Returns the compaction message now in the context, or None when there was
    nothing to splice (no compaction yet).
    """
    if not await splice_synap_history(
        chat_ctx, sdk, conversation_id=conversation_id, timeout=None, response=response,
    ):
        return None
    _drop_st_messages(chat_ctx)
    for m in chat_ctx.items:
        if getattr(m, "type", None) == "message" and is_synap_history(m.text_content):
            return m
    return None


async def preload_synap_st(
    chat_ctx: ChatContext,
    sdk: MaximemSynapSDK,
    *,
    conversation_id: str,
    style: str = "narrative",
    preamble_open: Optional[str] = _DEFAULT_OPEN,
    preamble_close: Optional[str] = _DEFAULT_CLOSE,
    on_error: _OnError = "fallback",
) -> Optional[ChatMessage]:
    """Prepend Synap short-term context to ``chat_ctx`` before session start.

    Mutates ``chat_ctx`` in place: inserts a single system message at
    the head whose content is the wrapped Synap ST block (with an
    internal marker so :func:`refresh_synap_st` can locate it later).
    No-op when no ST is available yet.

    Returns the inserted :class:`ChatMessage`, or ``None`` if nothing
    was injected.
    """
    _validate(sdk, conversation_id, style, on_error, "preload_synap_st")

    response = await _fetch_st(
        sdk, conversation_id, style, on_error,
        site="synap_livekit_agents.preload_synap_st",
    )
    if response is None:
        return None
    spliced = await _splice_compaction(chat_ctx, sdk, conversation_id, response)
    if spliced is not None:
        return spliced
    wrapped = _wrap_block(
        response, _held_turns(chat_ctx), preamble_open, preamble_close)
    if not wrapped:
        return None

    msg = ChatMessage(role="system", content=[_wrap_for_chat_message(wrapped)])
    chat_ctx.items.insert(_head_index(chat_ctx), msg)
    return msg


async def refresh_synap_st(
    chat_ctx: ChatContext,
    sdk: MaximemSynapSDK,
    *,
    conversation_id: str,
    style: str = "narrative",
    preamble_open: Optional[str] = _DEFAULT_OPEN,
    preamble_close: Optional[str] = _DEFAULT_CLOSE,
    on_error: _OnError = "fallback",
) -> Optional[ChatMessage]:
    """Replace the Synap ST system message in ``chat_ctx`` with a fresh one.

    Safe to call inside per-turn hooks (e.g.
    ``Agent.on_user_turn_completed``). Once Synap has compacted the
    conversation, the context's earlier turns are replaced by the
    compaction and the turns after it, and the returned message is the
    compaction. Before that, it looks for a system message tagged with the
    Synap ST marker and replaces it; if none exists, inserts one at the
    head. If the SDK returns no ST and a previous tagged message exists,
    removes it (so the prompt window doesn't keep stale context). Turns the
    context already holds are left out of the block.
    """
    _validate(sdk, conversation_id, style, on_error, "refresh_synap_st")

    response = await _fetch_st(
        sdk, conversation_id, style, on_error,
        site="synap_livekit_agents.refresh_synap_st",
    )
    wrapped = ""
    if response is not None:
        spliced = await _splice_compaction(chat_ctx, sdk, conversation_id, response)
        if spliced is not None:
            return spliced
        wrapped = _wrap_block(
            response, _held_turns(chat_ctx), preamble_open, preamble_close)

    # Find existing tagged message (if any)
    existing_idx = None
    for i, msg in enumerate(chat_ctx.items):
        if _is_synap_st_message(msg):
            existing_idx = i
            break

    if not wrapped:
        # No ST available now — drop any stale tagged message
        if existing_idx is not None:
            chat_ctx.items.pop(existing_idx)
        return None

    new_msg = ChatMessage(role="system", content=[_wrap_for_chat_message(wrapped)])
    if existing_idx is not None:
        chat_ctx.items[existing_idx] = new_msg
    else:
        chat_ctx.items.insert(_head_index(chat_ctx), new_msg)
    return new_msg


__all__ = ["preload_synap_st", "refresh_synap_st"]
