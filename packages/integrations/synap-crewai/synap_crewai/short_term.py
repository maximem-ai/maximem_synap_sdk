"""Synap short-term context for CrewAI.

CrewAI doesn't expose a per-step prompt-prep hook the way LangGraph,
OpenAI Agents, or Pydantic AI do. ``Agent`` takes a static
``backstory`` (and ``system_template``) at construction; there's no
callable evaluated before each LLM step.

So this adapter is **best-effort, one-shot**: :func:`build_synap_st_backstory`
fetches Synap short-term context **once** and returns a combined
backstory string. The user passes the result to
``Agent(backstory=...)``. The ST is a snapshot at agent-creation time —
to refresh it, re-create the agent.

For multi-turn use cases that need refreshed ST, prefer one of the
framework adapters that supports per-step injection (LangGraph,
LangChain, OpenAI Agents, Pydantic AI, Google ADK, Microsoft Agent
Framework, AutoGen, …) or re-instantiate the CrewAI ``Agent`` per
conversation turn.

Wraps ``sdk.conversation.context.get_context_for_prompt`` (cache-first
behind ``SYNAP_SDK_ST_AUTHORITATIVE``).

Quality contract identical to the LangGraph adapter:

- ``conversation_id`` required + explicit.
- Read failures degrade gracefully (``on_error="fallback"``): returns
  the bare ``base_backstory``. ``on_error="raise"`` available for tests.
- Empty ST is a no-op — never wipes the user's backstory.
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
    base_backstory: str,
    preamble_open: Optional[str],
    preamble_close: Optional[str],
) -> str:
    parts = []
    st_block = (st_block or "").strip()
    base_backstory = (base_backstory or "").strip()
    if st_block:
        if preamble_open and preamble_close:
            parts.append(f"{preamble_open}\n{st_block}\n{preamble_close}")
        else:
            parts.append(st_block)
    if base_backstory:
        parts.append(base_backstory)
    return "\n\n".join(parts)


async def build_synap_st_backstory(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    *,
    base_backstory: str = "",
    style: str = "narrative",
    preamble_open: Optional[str] = _DEFAULT_OPEN,
    preamble_close: Optional[str] = _DEFAULT_CLOSE,
    on_error: _OnError = "fallback",
) -> str:
    """Build a one-shot Synap-ST-enriched backstory string for a CrewAI Agent.

    Awaits the SDK helper, prepends the formatted ST block above the
    caller's ``base_backstory`` inside the configured preamble tags, and
    returns the combined string. Snapshot at call time — re-invoke and
    re-create the agent to refresh ST.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation ID. **Required.**
        base_backstory: The agent's existing backstory text. Returned
            unchanged when ST is unavailable.
        style: One of ``"structured" | "narrative" | "bullet_points"``.
        preamble_open / preamble_close: ST block wrappers. Pass ``None``
            for both to drop the tags.
        on_error: ``"fallback"`` (default) returns ``base_backstory`` on
            SDK failure; ``"raise"`` propagates
            :class:`SynapIntegrationError`.

    Returns:
        The combined backstory string. Empty when both inputs are empty.

    Example::

        from crewai import Agent
        from synap_crewai import build_synap_st_backstory

        backstory = await build_synap_st_backstory(
            sdk,
            conversation_id="conv_abc",
            base_backstory=(
                "You are a customer-support agent who loves to help "
                "customers solve their problems."
            ),
        )
        agent = Agent(role="Support", backstory=backstory, ...)

    Note:
        CrewAI agents do not support per-step prompt callables, so ST
        injected here is a snapshot at agent-creation time. For
        per-turn refreshed ST, use one of the per-step-hooked framework
        integrations (LangGraph, OpenAI Agents, Pydantic AI, etc.) or
        re-instantiate the CrewAI Agent per conversation turn.
    """
    _validate_args(
        sdk, conversation_id, style, on_error, "build_synap_st_backstory"
    )
    st_block = await _fetch_st_block(
        sdk,
        conversation_id,
        style,
        on_error,
        site="synap_crewai.build_synap_st_backstory",
    )
    return _compose(st_block, base_backstory, preamble_open, preamble_close)


__all__ = ["build_synap_st_backstory"]
