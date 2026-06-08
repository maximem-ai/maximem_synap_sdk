"""Synap short-term context for LlamaIndex.

Mirrors the LangGraph template, adapted to LlamaIndex's ``ChatMessage``
list shape. Wraps ``sdk.conversation.context.get_context_for_prompt``
(cache-first behind ``SYNAP_SDK_ST_AUTHORITATIVE``).

LlamaIndex's chat engines take a message list (``chat_history``) and a
static ``system_prompt`` string. There's no per-step prompt-prep
callable, so we expose :func:`synap_st_chat_message` — an async factory
that returns a ``ChatMessage`` with ``MessageRole.SYSTEM`` whose content
is ``<ST block in preamble tags>\\n\\n<user system>``. Prepend it to
``chat_history`` on each chat engine invocation for refreshed ST.

Quality contract identical to the LangGraph adapter:

- ``conversation_id`` required + explicit at construction.
- SDK failures never crash the chain by default
  (``on_error="fallback"``): logs via :class:`SynapIntegrationError`,
  returns the bare user system content (or ``None`` if both empty).
- Empty ST is a no-op — never wipes the user's system text.
- ``on_error="raise"`` available for strict environments.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Literal, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    SynapIntegrationError,
    wrap_sdk_errors_async,
)

logger = logging.getLogger(__name__)

_SUPPORTED_STYLES = ("structured", "narrative", "bullet_points")
_DEFAULT_OPEN = "<synap_short_term_context>"
_DEFAULT_CLOSE = "</synap_short_term_context>"

_OnError = Literal["fallback", "raise"]


def _validate_args(
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


async def _fetch_st_block(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    style: str,
    on_error: _OnError,
    site: str,
) -> str:
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
        return ""
    if not getattr(response, "available", False):
        return ""
    formatted = getattr(response, "formatted_context", None)
    return (formatted or "").strip()


def _compose(
    st_block: str,
    user_system: str,
    preamble_open: Optional[str],
    preamble_close: Optional[str],
) -> str:
    parts = []
    st_block = (st_block or "").strip()
    user_system = (user_system or "").strip()
    if st_block:
        if preamble_open and preamble_close:
            parts.append(f"{preamble_open}\n{st_block}\n{preamble_close}")
        else:
            parts.append(st_block)
    if user_system:
        parts.append(user_system)
    return "\n\n".join(parts)


def synap_st_chat_message(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    *,
    system: str = "",
    style: str = "narrative",
    preamble_open: Optional[str] = _DEFAULT_OPEN,
    preamble_close: Optional[str] = _DEFAULT_CLOSE,
    on_error: _OnError = "fallback",
) -> Callable[[], Awaitable[Optional[ChatMessage]]]:
    """Return an async factory producing a combined system ``ChatMessage``.

    Call the returned coroutine to get a ``ChatMessage`` with role
    ``MessageRole.SYSTEM`` whose content is the ST block (cache-first
    fetched from the SDK helper) wrapped in the configured preamble
    tags, followed by the user's static ``system`` text. Returns
    ``None`` when **both** the ST block and ``system`` are empty so
    callers can skip the system message cleanly rather than emitting a
    blank one.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation ID. **Required.**
        system: Your own system prompt; ST is prepended above.
        style: One of ``"structured" | "narrative" | "bullet_points"``.
        preamble_open / preamble_close: ST block wrappers; pass ``None``
            for both to drop the tags.
        on_error: ``"fallback"`` (default) returns a SystemMessage
            holding just ``system`` on SDK failure; ``"raise"``
            propagates :class:`SynapIntegrationError`.

    Example::

        from llama_index.core.chat_engine import SimpleChatEngine
        from synap_llamaindex import synap_st_chat_message

        get_system_message = synap_st_chat_message(
            sdk, conversation_id="conv_abc",
            system="You are a helpful agent.",
        )

        async def chat(user_msg: str):
            sys_msg = await get_system_message()
            history = [sys_msg] if sys_msg else []
            return await chat_engine.achat(user_msg, chat_history=history)
    """
    _validate_args(
        sdk, conversation_id, style, on_error, "synap_st_chat_message"
    )

    async def _factory() -> Optional[ChatMessage]:
        st_block = await _fetch_st_block(
            sdk,
            conversation_id,
            style,
            on_error,
            site="synap_llamaindex.synap_st_chat_message",
        )
        combined = _compose(st_block, system, preamble_open, preamble_close)
        if not combined:
            return None
        return ChatMessage(role=MessageRole.SYSTEM, content=combined)

    _factory.__name__ = "synap_st_chat_message_factory"
    return _factory


__all__ = ["synap_st_chat_message"]
