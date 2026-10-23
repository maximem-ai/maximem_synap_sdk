"""Synap short-term context for Anthropic's Claude Agent SDK.

A thin wrapper around ``sdk.conversation.context.get_context_for_prompt`` that
exposes Synap's view of the conversation to the agent through a
``UserPromptSubmit`` hook, as ``hookSpecificOutput.additionalContext``.

What that context holds depends on who has the turns. Claude Code keeps its own
transcript and resumes from it, so when the transcript already holds earlier
turns the hook adds only Synap's compaction (and the earlier turns history
search matched), and only when the transcript does not already contain it:
``formatted_context`` ends with the recent turns, which Claude Code would
otherwise see twice. When the transcript is empty (a fresh ``query()`` per turn,
which is how many apps use the SDK) Synap is the only record, and the hook adds
``formatted_context`` whole, recent turns included.

This module **composes** with :func:`create_synap_hooks` from
``synap_claude_agent.hooks``: that one injects **long-term** context (semantic
memory via ``sdk.fetch``) and, through the same helper, this short-term context;
this one is the short-term context alone.

- ``conversation_id`` required + explicit
- Hook NEVER raises: failure surfaces as an empty additionalContext
- Empty ST is a no-op, never wipes the user's system framing
- ``on_error="raise"`` available for tests/strict environments
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, Optional, Tuple

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


async def _fetch_response(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    style: str,
    on_error: _OnError,
    site: str,
) -> Any:
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
        return None
    return response if getattr(response, "available", False) else None


def _compaction_block(response: Any) -> str:
    """The compaction and the earlier turns history search matched: no recent turns."""
    parts = [getattr(response, "context_block", None), getattr(response, "history_block", None)]
    return "\n\n".join(p.strip() for p in parts if isinstance(p, str) and p.strip())


def _scan_transcript(path: Any, probe: str) -> Tuple[bool, bool]:
    """``(has_turns, has_block)`` for Claude Code's transcript at ``path``.

    ``has_turns``: an earlier user entry is there. ``has_block``: ``probe`` is
    there after the last compaction Claude Code made (what it compacted left the
    model's context, so a block from before it is gone). An unreadable or
    missing transcript is ``(False, False)``.
    """
    if not isinstance(path, str) or not path:
        return False, False
    needle = json.dumps(probe, ensure_ascii=False)[1:-1] if probe else None
    has_turns = has_block = False
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if '"compact_boundary"' in line and _entry_subtype(line) == "compact_boundary":
                    has_block = False
                    continue
                if needle and needle in line:
                    has_block = True
                if not has_turns and '"type":"user"' in line:
                    has_turns = True
    except OSError:
        return False, False
    return has_turns, has_block


def _entry_subtype(line: str) -> Optional[str]:
    try:
        entry = json.loads(line)
    except ValueError:
        return None
    return entry.get("subtype") if isinstance(entry, dict) else None


async def history_context(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    input_data: Any,
    *,
    style: str = "narrative",
    on_error: _OnError = "fallback",
    site: str = "synap_claude_agent.history_context",
) -> str:
    """What to add to this prompt so the model has the conversation, or ``""``."""
    response = await _fetch_response(sdk, conversation_id, style, on_error, site)
    if response is None:
        return ""
    block = _compaction_block(response)
    path = input_data.get("transcript_path") if isinstance(input_data, dict) else getattr(
        input_data, "transcript_path", None)
    has_turns, has_block = _scan_transcript(path, block[:300])
    if has_turns:
        return "" if has_block else block
    return (getattr(response, "formatted_context", None) or "").strip()


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
            st_block = await history_context(
                sdk,
                conversation_id,
                input_data,
                style=style,
                on_error=on_error,
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
