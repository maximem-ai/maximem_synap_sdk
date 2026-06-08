"""Synap short-term context for Pydantic AI.

Mirrors the LangGraph template, adapted to Pydantic AI's
``@agent.system_prompt`` mechanism. Wraps
``sdk.conversation.context.get_context_for_prompt`` (cache-first via the
``SYNAP_SDK_ST_AUTHORITATIVE`` flag).

Pydantic AI lets you register dynamic system prompts via the
``@agent.system_prompt`` decorator. The callable receives the
``RunContext`` (carrying your ``deps``) and returns a string that
Pydantic AI prepends to the model's input on every run.

This module exposes :func:`register_synap_st_system_prompt`, which
registers an async system_prompt callable on the agent. The callable
reads ``sdk`` and ``conversation_id`` from :class:`SynapDeps` at run
time — Pydantic AI's idiomatic "explicit per-run" mechanism. If
``conversation_id`` is unset on the deps, ST injection is skipped (the
user's static ``system`` string is returned, never wiped).

Quality contract matches the LangGraph adapter:

- ``conversation_id`` required per run (via ``SynapDeps``) — never
  inferred from framework session state.
- SDK failures never crash the agent by default — logged via
  :class:`SynapIntegrationError`, the user's static ``system`` string
  is returned.
- An empty ST result is a no-op — never wipes the user's system text.
- ``on_error="raise"`` available for strict environments.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    SynapIntegrationError,
    wrap_sdk_errors_async,
)

from synap_pydantic_ai.deps import SynapDeps

logger = logging.getLogger(__name__)

_SUPPORTED_STYLES = ("structured", "narrative", "bullet_points")
_DEFAULT_OPEN = "<synap_short_term_context>"
_DEFAULT_CLOSE = "</synap_short_term_context>"

_OnError = Literal["fallback", "raise"]


def _validate(style: str, on_error: str, site: str) -> None:
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


def register_synap_st_system_prompt(
    agent,
    *,
    system: str = "",
    style: str = "narrative",
    preamble_open: Optional[str] = _DEFAULT_OPEN,
    preamble_close: Optional[str] = _DEFAULT_CLOSE,
    on_error: _OnError = "fallback",
) -> None:
    """Register a ``@agent.system_prompt`` that injects Synap ST.

    Requires :class:`SynapDeps` (or a subclass) as the agent's
    ``deps_type``. The callback pulls ``sdk`` and ``conversation_id``
    from ``ctx.deps`` at run time. Set ``conversation_id`` on
    ``SynapDeps`` per ``agent.run(...)`` / ``agent.run_sync(...)``
    invocation — that's the explicit per-run binding.

    When ``deps.conversation_id`` is unset, ST injection is skipped and
    the static ``system`` argument is returned (never wiped).

    Args:
        agent: A ``pydantic_ai.Agent`` instance with
            ``deps_type=SynapDeps`` (or a subclass).
        system: Static framing prepended below the ST block. Returned
            unchanged when ST is unavailable.
        style: One of ``"structured" | "narrative" | "bullet_points"``.
        preamble_open / preamble_close: ST block wrappers. Pass ``None``
            for both to drop the tags.
        on_error: ``"fallback"`` (default) returns ``system`` on SDK
            failure; ``"raise"`` propagates
            :class:`SynapIntegrationError`.

    Example::

        from pydantic_ai import Agent
        from synap_pydantic_ai import SynapDeps, register_synap_st_system_prompt

        agent = Agent('openai:gpt-4o', deps_type=SynapDeps)
        register_synap_st_system_prompt(
            agent, system="You are a helpful agent."
        )

        result = await agent.run(
            "What's my next step?",
            deps=SynapDeps(sdk=sdk, user_id="u1", conversation_id="conv_abc"),
        )
    """
    _validate(style, on_error, "register_synap_st_system_prompt")

    # Late import: avoid pulling pydantic_ai into module-level import
    # unless the caller actually exercises this code path. Matches the
    # pattern in synap_pydantic_ai.deps.register_synap_tools.
    from pydantic_ai import RunContext

    @agent.system_prompt
    async def _synap_st(ctx: RunContext[SynapDeps]) -> str:
        deps = ctx.deps
        if deps is None or not deps.conversation_id:
            return system.strip()
        st_block = await _fetch_st_block(
            deps.sdk,
            deps.conversation_id,
            style,
            on_error,
            site="synap_pydantic_ai.register_synap_st_system_prompt",
        )
        return _compose(st_block, system, preamble_open, preamble_close)


__all__ = ["register_synap_st_system_prompt"]
