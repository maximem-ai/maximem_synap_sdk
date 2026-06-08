"""Synap short-term context for Anthropic's Claude Agent SDK.

Mirrors the LangGraph template: a thin wrapper around
``sdk.conversation.context.get_context_for_prompt`` that exposes Synap
short-term context to the agent via a ``UserPromptSubmit`` hook.

Claude Agent's ``UserPromptSubmit`` hook lets us return
``hookSpecificOutput.additionalContext`` — a string the agent SDK
splices into the model's input as supplementary context. We use it to
deliver Synap's compacted summary + recent turns at every turn.

This module **composes** with :func:`create_synap_hooks` from
``synap_claude_agent.hooks`` rather than replacing it: that one injects
**long-term** context (semantic memory via ``sdk.fetch``), this one
injects **short-term** context (per-conversation compacted history). A
user wanting both stacks them under the ``UserPromptSubmit`` key.

Quality contract identical to the LangGraph adapter:

- ``conversation_id`` required + explicit
- Hook NEVER raises — failure surfaces as an empty additionalContext
  (or the bare user system text)
- Empty ST is a no-op, never wipes the user's system framing
- ``on_error="raise"`` available for tests/strict environments
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from claude_agent_sdk import HookMatcher
from claude_agent_sdk.types import HookContext, UserPromptSubmitHookInput
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


def _wrap(
    st_block: str,
    preamble_open: Optional[str],
    preamble_close: Optional[str],
) -> str:
    st_block = (st_block or "").strip()
    if not st_block:
        return ""
    if preamble_open and preamble_close:
        return f"{preamble_open}\n{st_block}\n{preamble_close}"
    return st_block


def create_synap_st_hook(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    *,
    style: str = "narrative",
    preamble_open: Optional[str] = _DEFAULT_OPEN,
    preamble_close: Optional[str] = _DEFAULT_CLOSE,
    on_error: _OnError = "fallback",
) -> dict[str, list[HookMatcher]]:
    """Return a ``hooks`` dict that injects Synap short-term context.

    Drop the return value into ``ClaudeAgentOptions(hooks=...)``. The
    registered ``UserPromptSubmit`` hook is invoked on every user
    message and returns the formatted short-term context inside
    ``hookSpecificOutput.additionalContext`` — Claude Agent SDK stitches
    that into the model's input automatically.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation ID. **Required.**
        style: One of ``"structured" | "narrative" | "bullet_points"``.
        preamble_open / preamble_close: Wrapping tags. Pass ``None`` for
            both to skip the wrapper.
        on_error: ``"fallback"`` (default) returns no additional context
            on SDK failure; ``"raise"`` propagates
            :class:`SynapIntegrationError`.

    Composing with the LT hook::

        from synap_claude_agent import create_synap_hooks, create_synap_st_hook

        lt_hooks = create_synap_hooks(sdk, user_id="alice")
        st_hooks = create_synap_st_hook(sdk, conversation_id="conv_abc")

        # Stack both under UserPromptSubmit:
        hooks = {
            "UserPromptSubmit": lt_hooks["UserPromptSubmit"]
                               + st_hooks["UserPromptSubmit"],
        }
        options = ClaudeAgentOptions(hooks=hooks)
    """
    _validate_args(
        sdk, conversation_id, style, on_error, "create_synap_st_hook"
    )

    async def _on_user_prompt_submit(
        input_data: Any,
        tool_use_id: Optional[str],
        context: HookContext,
    ) -> dict[str, Any]:
        # We deliberately do NOT pull conv_id from session_id here — the
        # caller bound it explicitly. Different from the LT hook because
        # ST is per-conversation, not per-prompt-session.
        try:
            st_block = await _fetch_st_block(
                sdk,
                conversation_id,
                style,
                on_error,
                site="synap_claude_agent.create_synap_st_hook",
            )
        except SynapIntegrationError:
            # on_error="raise": surface the failure to the caller. The
            # Claude Agent SDK doesn't crash on hook exceptions — it
            # logs and skips — but the contract here is "strict mode
            # propagates", same as every other ST adapter.
            raise

        wrapped = _wrap(st_block, preamble_open, preamble_close)
        if not wrapped:
            return {}

        return {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": wrapped,
            }
        }

    return {
        "UserPromptSubmit": [HookMatcher(hooks=[_on_user_prompt_submit])],
    }


# Keep the type import alive so users discover it via IDE hovers.
_ = UserPromptSubmitHookInput


__all__ = ["create_synap_st_hook"]
