"""Synap short-term context for Smolagents.

Smolagents `CodeAgent(instructions=...)` / `ToolCallingAgent(instructions=...)` take
a static string, so short-term context is folded into `instructions` once, at
construction, via `sdk.conversation.context.get_context_for_prompt`. Returns the
composed instructions string.

Quality contract (mirrors the other Synap integrations):

- ``conversation_id`` required + explicit.
- SDK failures never crash construction by default (``on_error="fallback"``): the
  bare ``instructions`` are returned.
- Empty short-term context is a no-op — the user's ``instructions`` are untouched.
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
        raise ValueError("synap_st_instructions requires a non-None sdk")
    if not conversation_id or not str(conversation_id).strip():
        raise ValueError("synap_st_instructions requires a non-empty conversation_id")
    if style not in _SUPPORTED_STYLES:
        raise ValueError(
            f"synap_st_instructions: unsupported style={style!r}; "
            f"expected one of {_SUPPORTED_STYLES}"
        )
    if on_error not in ("fallback", "raise"):
        raise ValueError(
            f"synap_st_instructions: on_error must be 'fallback' or 'raise', "
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
            "smolagents.synap_st_instructions",
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
    instructions: str,
    preamble_open: Optional[str],
    preamble_close: Optional[str],
) -> str:
    parts = []
    st_block = (st_block or "").strip()
    instructions = (instructions or "").strip()
    if st_block:
        if preamble_open and preamble_close:
            parts.append(f"{preamble_open}\n{st_block}\n{preamble_close}")
        else:
            parts.append(st_block)
    if instructions:
        parts.append(instructions)
    return "\n\n".join(parts)


def synap_st_instructions(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    *,
    instructions: str = "",
    style: str = "narrative",
    preamble_open: Optional[str] = _DEFAULT_OPEN,
    preamble_close: Optional[str] = _DEFAULT_CLOSE,
    on_error: _OnError = "fallback",
) -> str:
    """Compose Smolagents ``instructions`` with Synap short-term context prepended.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation id. **Required.**
        instructions: Your own static instructions; ST is prepended above them.
        style: One of ``"structured" | "narrative" | "bullet_points"``.
        preamble_open / preamble_close: ST block wrappers; pass ``None`` for both to
            drop the tags.
        on_error: ``"fallback"`` (default) returns the bare ``instructions`` on SDK
            failure; ``"raise"`` propagates :class:`SynapIntegrationError`.

    Returns:
        The composed instructions string for ``CodeAgent(instructions=...)``.

    Example::

        from smolagents import CodeAgent, InferenceClientModel
        from synap_smolagents import synap_st_instructions

        agent = CodeAgent(
            model=InferenceClientModel(),
            tools=[],
            instructions=synap_st_instructions(
                sdk, conversation_id="conv_abc",
                instructions="You are a helpful assistant.",
            ),
        )
    """
    _validate_args(sdk, conversation_id, style, on_error)
    st_block = run_async(_fetch_st_block(sdk, conversation_id, style, on_error))
    return _compose(st_block, instructions, preamble_open, preamble_close)
