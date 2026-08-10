"""Synap short-term context for deepagents.

Short-term context is the compacted history of the *current* conversation —
distinct from the long-term memory :class:`~synap_deepagents.backend.SynapBackend`
serves. It comes from ``sdk.conversation.context.get_context_for_prompt``.

``create_deep_agent`` takes a static ``system_prompt``, so this module gives you
the string to build it from::

    from deepagents import create_deep_agent
    from synap_deepagents import synap_st_instructions

    system_prompt = await synap_st_instructions(
        sdk, "conv_abc", system="You are a helpful coding agent."
    )
    agent = create_deep_agent(model=..., system_prompt=system_prompt)

That snapshot is taken once, at construction. For a long-running agent that
should see the context refresh each turn, use
:class:`~synap_deepagents.middleware.SynapShortTermMiddleware` instead.

Failures never crash the agent by default (``on_error="fallback"``): the SDK
error is logged and an empty short-term block is used, so you are left with
your own system prompt rather than no prompt at all. Pass ``on_error="raise"``
if you would rather fail loudly.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

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


async def fetch_st_block(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    *,
    style: str = "narrative",
    on_error: _OnError = "fallback",
    site: str = "synap_deepagents.fetch_st_block",
) -> str:
    """Return the short-term context string, or ``""`` when unavailable.

    Shared by :func:`synap_st_instructions` and the middleware so both behave
    identically on failure.
    """
    _validate_args(sdk, conversation_id, style, on_error, site)
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
    return (getattr(response, "formatted_context", None) or "").strip()


def compose_system_prompt(
    st_block: str,
    system: str,
    preamble_open: Optional[str] = _DEFAULT_OPEN,
    preamble_close: Optional[str] = _DEFAULT_CLOSE,
) -> str:
    """Join a short-term block and a system prompt, skipping empty parts.

    An empty short-term block must never wipe the caller's system prompt, and
    an empty system prompt must not leave dangling preamble tags.
    """
    parts = []
    st_block = (st_block or "").strip()
    system = (system or "").strip()
    if st_block:
        if preamble_open and preamble_close:
            parts.append(f"{preamble_open}\n{st_block}\n{preamble_close}")
        else:
            parts.append(st_block)
    if system:
        parts.append(system)
    return "\n\n".join(parts)


async def synap_st_instructions(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    *,
    system: str = "",
    style: str = "narrative",
    preamble_open: Optional[str] = _DEFAULT_OPEN,
    preamble_close: Optional[str] = _DEFAULT_CLOSE,
    on_error: _OnError = "fallback",
) -> str:
    """Build a system prompt with Synap short-term context prepended.

    Args:
        sdk: An initialised :class:`~maximem_synap.MaximemSynapSDK`.
        conversation_id: Synap conversation ID. **Required** — deepagents thread
            IDs and Synap conversation IDs are separate namespaces and must not
            be inferred from one another.
        system: Your own system prompt. Stays authoritative for behaviour.
        style: One of ``"structured"``, ``"narrative"``, ``"bullet_points"``.
        preamble_open / preamble_close: Wrapping tags. Pass ``None`` for both to
            prepend the raw text with no tags.
        on_error: ``"fallback"`` (default) returns just ``system`` on SDK
            failure; ``"raise"`` propagates ``SynapIntegrationError``.

    Returns:
        The composed system prompt. Returns ``""`` only when both the
        short-term block and ``system`` are empty.

    Example::

        system_prompt = await synap_st_instructions(
            sdk, "conv_abc", system="You are a helpful coding agent."
        )
        agent = create_deep_agent(model=..., system_prompt=system_prompt)
    """
    st_block = await fetch_st_block(
        sdk,
        conversation_id,
        style=style,
        on_error=on_error,
        site="synap_deepagents.synap_st_instructions",
    )
    return compose_system_prompt(st_block, system, preamble_open, preamble_close)


__all__ = [
    "synap_st_instructions",
    "fetch_st_block",
    "compose_system_prompt",
]
