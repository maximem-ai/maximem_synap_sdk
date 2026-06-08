"""Synap short-term context for LangChain.

Two composable surfaces, both wrapping
``sdk.conversation.context.get_context_for_prompt`` (cache-first via the
``SYNAP_SDK_ST_AUTHORITATIVE`` flag — see
``docs/internal/sdk_authoritative_short_term_context_plan.md``):

- :func:`synap_st_runnable` — returns an LCEL ``Runnable`` that emits the
  Synap short-term context **string**. Compose into a prompt template
  with ``RunnablePassthrough.assign(synap_st=...)`` and reference
  ``{synap_st}`` in your system template. The Runnable supports both
  ``.invoke`` (sync) and ``.ainvoke`` (async) automatically.

- :func:`synap_st_message` — returns an **async factory** that produces a
  ``SystemMessage`` whose content is ``<ST block>\\n\\n<user system>``,
  ready to drop into a message list. Use when you assemble messages by
  hand instead of via a prompt template (mirrors the LangGraph
  ``synap_st_prompt`` shape).

Quality contract (matches the LangGraph adapter):

- ``conversation_id`` is **required + explicit** at construction. We
  deliberately don't infer from any RunnableConfig thread_id — those
  namespaces can diverge.
- SDK failures **never crash the chain** by default
  (``on_error="fallback"``): logged via :class:`SynapIntegrationError`,
  Runnable returns ``""`` and the SystemMessage factory returns the
  bare user system content. ``on_error="raise"`` available for strict
  use.
- An empty short-term result (no compaction yet **and** no recent turns)
  is a no-op — must not wipe the user's system prompt.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Literal, Optional

from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
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
        raise ValueError(
            f"{site} requires a non-empty conversation_id "
            f"(pass it explicitly per-run for multi-conversation agents)"
        )
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


def _compose_system_prompt(
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


# ---------------------------------------------------------------------------
# Public surface — LCEL Runnable
# ---------------------------------------------------------------------------


def synap_st_runnable(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    *,
    style: str = "narrative",
    on_error: _OnError = "fallback",
) -> Runnable[Any, str]:
    """Return an LCEL :class:`Runnable` that emits the Synap ST string.

    The Runnable ignores its input and produces the formatted short-term
    context string from the SDK helper (cache-first when warm). Use with
    ``RunnablePassthrough.assign(synap_st=...)`` to wire it into an LCEL
    chain that references ``{synap_st}`` in the system template.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation ID. **Required.**
        style: One of ``"structured" | "narrative" | "bullet_points"``.
            Defaults to ``"narrative"``.
        on_error: ``"fallback"`` (default) returns ``""`` on SDK failure;
            ``"raise"`` propagates :class:`SynapIntegrationError`.

    Returns:
        ``Runnable[Any, str]`` — async-native via ``RunnableLambda``.

    Example::

        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.runnables import RunnablePassthrough
        from synap_langchain import synap_st_runnable

        prompt = ChatPromptTemplate.from_messages([
            ("system", "{synap_st}\\n\\nYou are a helpful agent."),
            MessagesPlaceholder("messages"),
        ])

        chain = (
            RunnablePassthrough.assign(
                synap_st=synap_st_runnable(sdk, "conv_abc"),
            )
            | prompt
            | llm
        )
    """
    _validate_args(
        sdk, conversation_id, style, on_error, "synap_st_runnable"
    )

    async def _fetch(_: Any) -> str:
        return await _fetch_st_block(
            sdk,
            conversation_id,
            style,
            on_error,
            site="synap_langchain.synap_st_runnable",
        )

    return RunnableLambda(_fetch, name="synap_st").with_config(
        run_name="synap_st",
    )


# ---------------------------------------------------------------------------
# Public surface — async SystemMessage factory
# ---------------------------------------------------------------------------


def synap_st_message(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    *,
    system: str = "",
    style: str = "narrative",
    preamble_open: Optional[str] = _DEFAULT_OPEN,
    preamble_close: Optional[str] = _DEFAULT_CLOSE,
    on_error: _OnError = "fallback",
) -> Callable[[], Awaitable[Optional[SystemMessage]]]:
    """Return an async factory producing a combined :class:`SystemMessage`.

    Call the returned coroutine to get a :class:`SystemMessage` whose
    ``content`` is ``<ST block in preamble tags>\\n\\n<user system>``.
    Returns ``None`` when both the ST block and ``system`` are empty —
    callers should skip the SystemMessage in that case rather than
    emitting a blank one.

    Use when you build the message list by hand instead of going through
    an LCEL prompt template. For prompt-template flows prefer
    :func:`synap_st_runnable`.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation ID. **Required.**
        system: Your own system prompt. Stays authoritative for behaviour.
        style: One of ``"structured" | "narrative" | "bullet_points"``.
        preamble_open / preamble_close: Wrapping tags. Pass ``None`` for
            both to drop tags and prepend raw text.
        on_error: ``"fallback"`` (default) | ``"raise"``.

    Example::

        from synap_langchain import synap_st_message

        get_system_message = synap_st_message(
            sdk, "conv_abc", system="You are a helpful agent."
        )

        # Inside your request handler:
        sys_msg = await get_system_message()
        messages = [sys_msg] if sys_msg else []
        messages += incoming_messages
        response = await llm.ainvoke(messages)
    """
    _validate_args(
        sdk, conversation_id, style, on_error, "synap_st_message"
    )

    async def _factory() -> Optional[SystemMessage]:
        st_block = await _fetch_st_block(
            sdk,
            conversation_id,
            style,
            on_error,
            site="synap_langchain.synap_st_message",
        )
        combined = _compose_system_prompt(
            st_block, system, preamble_open, preamble_close
        )
        if not combined:
            return None
        return SystemMessage(content=combined)

    _factory.__name__ = "synap_st_message_factory"
    return _factory


__all__ = [
    "synap_st_runnable",
    "synap_st_message",
]
