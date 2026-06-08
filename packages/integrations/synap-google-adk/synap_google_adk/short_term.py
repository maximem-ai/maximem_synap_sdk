"""Synap short-term context for Google ADK.

Mirrors the LangGraph template, adapted to Google ADK's
``LlmAgent.instruction`` mechanism (which accepts a string OR a callable
``(ReadonlyContext) -> str | Awaitable[str]`` that the runtime invokes
before every model call).

Wraps ``sdk.conversation.context.get_context_for_prompt`` (cache-first
behind ``SYNAP_SDK_ST_AUTHORITATIVE``).

Quality contract identical to the LangGraph adapter:

- ``conversation_id`` required + explicit at construction (no inference
  from any ADK session id).
- SDK failures never crash the agent by default
  (``on_error="fallback"``): logs via :class:`SynapIntegrationError`,
  returns the bare static ``instruction`` text.
- Empty ST is a no-op — never wipes the user's static instruction.
- ``on_error="raise"`` available for strict environments.
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


def synap_st_instruction(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    *,
    instruction: str = "",
    style: str = "narrative",
    preamble_open: Optional[str] = _DEFAULT_OPEN,
    preamble_close: Optional[str] = _DEFAULT_CLOSE,
    on_error: _OnError = "fallback",
) -> Callable[[Any], Awaitable[str]]:
    """Return an async ``instruction`` callable for Google ADK ``LlmAgent``.

    The callable matches Google ADK's expected
    ``Callable[[ReadonlyContext], Awaitable[str]]`` shape. ADK invokes
    it before every model call; we fetch ST via the SDK helper
    (cache-first), prepend it above the static ``instruction`` string
    inside the configured preamble tags, and return the combined text.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation ID. **Required.**
        instruction: Your own static agent instruction; ST is prepended
            above. Returned unchanged when ST is unavailable.
        style: One of ``"structured" | "narrative" | "bullet_points"``.
        preamble_open / preamble_close: Wrapping tags; pass ``None`` for
            both to drop the tags.
        on_error: ``"fallback"`` (default) returns the bare ``instruction``
            on SDK failure; ``"raise"`` propagates
            :class:`SynapIntegrationError`.

    Example::

        from google.adk.agents import LlmAgent
        from maximem_synap import MaximemSynapSDK
        from synap_google_adk import synap_st_instruction

        sdk = MaximemSynapSDK(api_key="sk-...")
        agent = LlmAgent(
            name="support",
            model="gemini-2.0-flash",
            instruction=synap_st_instruction(
                sdk,
                conversation_id="conv_abc",
                instruction="You are a polite support agent.",
            ),
        )
    """
    _validate_args(
        sdk, conversation_id, style, on_error, "synap_st_instruction"
    )

    async def _instruction(_ctx: Any) -> str:  # noqa: ARG001 — ADK signature
        st_block = await _fetch_st_block(
            sdk,
            conversation_id,
            style,
            on_error,
            site="synap_google_adk.synap_st_instruction",
        )
        return _compose(st_block, instruction, preamble_open, preamble_close)

    _instruction.__name__ = "synap_st_instruction"
    return _instruction


__all__ = ["synap_st_instruction"]
