"""Synap short-term context for OpenAI Agents SDK.

Mirrors the LangGraph template: a single thin wrapper around
``sdk.conversation.context.get_context_for_prompt``. Quality contract is
identical to the LangGraph adapter (see synap_langgraph.short_term).

OpenAI Agents accepts ``Agent(instructions=...)`` where ``instructions``
can be a string OR an async callable receiving ``(context, agent)`` and
returning a string. :func:`synap_st_instructions` returns the second
form: each agent step calls into the SDK helper (cache-first), prepends
the ST block above your own system text inside the configured preamble
tags, and returns the combined instructions string.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Literal, Optional

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
            f"{site} requires a non-empty conversation_id"
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


def synap_st_instructions(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    *,
    system: str = "",
    style: str = "narrative",
    preamble_open: Optional[str] = _DEFAULT_OPEN,
    preamble_close: Optional[str] = _DEFAULT_CLOSE,
    on_error: _OnError = "fallback",
) -> Callable[[Any, Any], Awaitable[str]]:
    """Return an async ``instructions`` callable for OpenAI Agents.

    The callable matches OpenAI Agents' expected signature
    ``async (context, agent) -> str``. Both arguments are ignored — we
    use the explicitly-bound ``conversation_id`` rather than inferring
    from any framework-internal session.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation ID. **Required.**
        system: Your own agent instructions; ST is prepended above.
        style: One of ``"structured" | "narrative" | "bullet_points"``.
        preamble_open / preamble_close: Wrapping tags; pass ``None`` for
            both to drop the tags.
        on_error: ``"fallback"`` (default) returns bare ``system`` on
            SDK failure; ``"raise"`` propagates :class:`SynapIntegrationError`.

    Example::

        from agents import Agent
        from maximem_synap import MaximemSynapSDK
        from synap_openai_agents import synap_st_instructions

        sdk = MaximemSynapSDK(api_key="sk-...")
        agent = Agent(
            name="support",
            instructions=synap_st_instructions(
                sdk,
                conversation_id="conv_abc",
                system="You are a polite support agent.",
            ),
            tools=[...],
        )
    """
    _validate_args(
        sdk, conversation_id, style, on_error, "synap_st_instructions"
    )

    async def _instructions(context: Any, agent: Any) -> str:  # noqa: ARG001 — framework signature
        st_block = await _fetch_st_block(
            sdk,
            conversation_id,
            style,
            on_error,
            site="synap_openai_agents.synap_st_instructions",
        )
        return _compose(st_block, system, preamble_open, preamble_close)

    _instructions.__name__ = "synap_st_instructions"
    return _instructions


__all__ = ["synap_st_instructions"]
