"""Synap short-term context for CAMEL-AI.

CAMEL's ``ChatAgent(system_message=...)`` takes a static string, so short-term
context is folded into the system message once, at construction, via
``sdk.conversation.context.get_context_for_prompt``. Returns the composed system
message string.

Quality contract (mirrors the other Synap integrations):

- ``conversation_id`` required + explicit.
- SDK failures never crash construction by default (``on_error="fallback"``):
  the bare ``system_message`` is returned.
- Empty short-term context is a no-op — the user's ``system_message`` is untouched.
- ``on_error="raise"`` propagates :class:`SynapIntegrationError` for strict setups.

Because the context is fetched once at construction, it is a snapshot; rebuild the
agent (or re-call this) to refresh it.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    SynapIntegrationError,
    run_async,
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
) -> None:
    if sdk is None:
        raise ValueError("synap_st_system_message requires a non-None sdk")
    if not conversation_id or not str(conversation_id).strip():
        raise ValueError(
            "synap_st_system_message requires a non-empty conversation_id"
        )
    if style not in _SUPPORTED_STYLES:
        raise ValueError(
            f"synap_st_system_message: unsupported style={style!r}; "
            f"expected one of {_SUPPORTED_STYLES}"
        )
    if on_error not in ("fallback", "raise"):
        raise ValueError(
            f"synap_st_system_message: on_error must be 'fallback' or 'raise', "
            f"got {on_error!r}"
        )


async def _fetch_st_block(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    style: str,
    on_error: _OnError,
) -> str:
    try:
        async with wrap_sdk_errors_async(
            "camel_ai.synap_st_system_message",
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


def _compose(
    st_block: str,
    system_message: str,
    preamble_open: Optional[str],
    preamble_close: Optional[str],
) -> str:
    parts = []
    st_block = (st_block or "").strip()
    system_message = (system_message or "").strip()
    if st_block:
        if preamble_open and preamble_close:
            parts.append(f"{preamble_open}\n{st_block}\n{preamble_close}")
        else:
            parts.append(st_block)
    if system_message:
        parts.append(system_message)
    return "\n\n".join(parts)


def synap_st_system_message(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    *,
    system_message: str = "",
    style: str = "narrative",
    preamble_open: Optional[str] = _DEFAULT_OPEN,
    preamble_close: Optional[str] = _DEFAULT_CLOSE,
    on_error: _OnError = "fallback",
) -> str:
    """Compose a CAMEL system message with Synap short-term context prepended.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation id. **Required.**
        system_message: Your own static system message; ST is prepended above it.
        style: One of ``"structured" | "narrative" | "bullet_points"``.
        preamble_open / preamble_close: ST block wrappers; pass ``None`` for both
            to drop the tags.
        on_error: ``"fallback"`` (default) returns the bare ``system_message`` on
            SDK failure; ``"raise"`` propagates :class:`SynapIntegrationError`.

    Returns:
        The composed system message string for ``ChatAgent(system_message=...)``.

    Example::

        from camel.agents import ChatAgent
        from synap_camel_ai import SynapAgentMemory, synap_st_system_message

        agent = ChatAgent(
            system_message=synap_st_system_message(
                sdk, conversation_id="conv_abc",
                system_message="You are a helpful support agent.",
            ),
            memory=SynapAgentMemory(sdk, user_id="alice"),
        )
    """
    _validate_args(sdk, conversation_id, style, on_error)
    st_block = run_async(
        _fetch_st_block(sdk, conversation_id, style, on_error)
    )
    return _compose(st_block, system_message, preamble_open, preamble_close)
