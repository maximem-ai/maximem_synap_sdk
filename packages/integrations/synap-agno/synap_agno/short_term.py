"""Synap short-term context for Agno.

Mirrors the LangGraph template, adapted to Agno's
``Agent(instructions=callable)`` mechanism. Wraps
``sdk.conversation.context.get_context_for_prompt`` (cache-first behind
``SYNAP_SDK_ST_AUTHORITATIVE``).

Agno accepts a ``Callable`` for ``instructions``. It introspects the
callable's signature and passes optional kwargs (``agent``, ``team``,
``session_state``, ``run_context``). Our adapter returns an async
callable; users invoke ``agent.arun(...)`` (Agno raises if you call
``agent.run(...)`` with an async instructions callable).

Quality contract identical to the LangGraph adapter:

- ``conversation_id`` required + explicit at construction.
- SDK failures never crash the agent by default
  (``on_error="fallback"``): returns the bare ``instructions`` string.
- Empty ST is a no-op — never wipes the user's instructions.
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
    user_instructions: str,
    preamble_open: Optional[str],
    preamble_close: Optional[str],
) -> str:
    parts = []
    st_block = (st_block or "").strip()
    user_instructions = (user_instructions or "").strip()
    if st_block:
        if preamble_open and preamble_close:
            parts.append(f"{preamble_open}\n{st_block}\n{preamble_close}")
        else:
            parts.append(st_block)
    if user_instructions:
        parts.append(user_instructions)
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
) -> Callable[..., Awaitable[str]]:
    """Return an async callable suitable for Agno ``Agent(instructions=...)``.

    Agno introspects the callable's signature for known kwargs
    (``agent``, ``team``, ``session_state``, ``run_context``); we accept
    them as ``**_ignored`` since this callable's behaviour doesn't
    depend on any of them.

    Because the callable is async, run the agent via ``agent.arun(...)``
    (Agno raises if you call ``agent.run(...)`` with an async
    instructions callable).

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation ID. **Required.**
        instructions: Your own static instructions; ST is prepended above.
        style: One of ``"structured" | "narrative" | "bullet_points"``.
        preamble_open / preamble_close: ST block wrappers; pass ``None``
            for both to drop the tags.
        on_error: ``"fallback"`` (default) returns the bare
            ``instructions`` on SDK failure; ``"raise"`` propagates
            :class:`SynapIntegrationError`.

    Example::

        from agno.agent import Agent
        from synap_agno import SynapDb, synap_st_instructions

        agent = Agent(
            db=SynapDb(sdk, customer_id="acme"),
            instructions=synap_st_instructions(
                sdk,
                conversation_id="conv_abc",
                instructions="You are a helpful agent.",
            ),
        )

        result = await agent.arun("What's my next step?", user_id="alice")
    """
    _validate_args(
        sdk, conversation_id, style, on_error, "synap_st_instructions"
    )

    async def _instructions(**_ignored: Any) -> str:
        st_block = await _fetch_st_block(
            sdk,
            conversation_id,
            style,
            on_error,
            site="synap_agno.synap_st_instructions",
        )
        return _compose(st_block, instructions, preamble_open, preamble_close)

    _instructions.__name__ = "synap_st_instructions"
    return _instructions


__all__ = ["synap_st_instructions"]
