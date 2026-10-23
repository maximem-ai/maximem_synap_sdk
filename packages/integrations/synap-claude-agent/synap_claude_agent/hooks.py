"""Synap hooks for the Claude Agent SDK.

Hooks that:

1. ``UserPromptSubmit``: fetch Synap context for the incoming prompt and return
   it via ``hookSpecificOutput.additionalContext``, which the SDK stitches into
   Claude's prompt. That is the user's long-term memory and, through
   :func:`synap_claude_agent.short_term.history_context`, Synap's compaction of
   this conversation when Claude Code's transcript does not already hold it.
   The user's prompt is recorded back to Synap too.
2. ``PreToolUse`` / ``PostToolUse``: report tool calls and results.
3. ``Stop``: record the agent's reply (``last_assistant_message``) and end the
   session.
4. ``PreCompact``: when Claude Code is about to compact the transcript, hand its
   summarizer Synap's compaction to build on.

The hooks NEVER raise. A failing Synap call returns ``{}`` (no additional
context, no block) so agent runs continue uninterrupted. This mirrors the
policy we use for LangChain's SynapCallbackHandler and MAF's
SynapContextProvider.after_run: context providers must not crash the agent.
"""

from __future__ import annotations

import logging
import weakref
from typing import Any, Dict, Optional

from claude_agent_sdk import HookMatcher
from claude_agent_sdk.types import (
    HookContext,
    UserPromptSubmitHookInput,
)
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    end_session,
    fetch_compacted_history,
    is_synap_history,
    record_turn,
    report_tool_call,
    report_tool_result,
)

from synap_claude_agent.short_term import history_context

logger = logging.getLogger(__name__)

# What `report_assistant_turn` wrote last, per SDK and conversation, so the Stop
# hook that reads the same reply does not write it a second time.
_manual_turns: "weakref.WeakKeyDictionary[Any, Dict[str, str]]" = weakref.WeakKeyDictionary()


_DEFAULT_CONTEXT_PREAMBLE = (
    "<synap_memory>\n"
    "Relevant context from the user's long-term memory:\n\n"
    "{body}\n"
    "</synap_memory>"
)


def create_synap_hooks(
    sdk: MaximemSynapSDK,
    user_id: str,
    customer_id: str = "",
    conversation_id: Optional[str] = None,
    *,
    mode: str = "accurate",
    max_results: int = 20,
    context_preamble: Optional[str] = None,
    record_user_prompts: bool = True,
) -> dict[str, list[HookMatcher]]:
    """Return a ``hooks`` dict suitable for ``ClaudeAgentOptions(hooks=...)``.

    Args:
        sdk: Configured :class:`MaximemSynapSDK` instance.
        user_id: Synap user scope. Required.
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on a
            B2C instance (user_context_isolation=equals_customer): the server
            rejects a call carrying one with HTTP 400. Leave it unset there.
        conversation_id: Optional static conversation id. When ``None`` the
            SDK's per-session ``session_id`` is used instead.
        mode: Synap fetch mode; ``"accurate"`` (default) or ``"fast"``.
        max_results: Cap on Synap fetch results.
        context_preamble: Optional format string with one ``{body}``
            placeholder. If the fetched Synap context is empty, no additional
            context is injected.
        record_user_prompts: When True (default), record the user's prompt
            to Synap conversation history via
            ``sdk.conversation.record_message``. Disable if you want
            injection-only semantics.

    Stream events
    -------------
    The first hook to report opens ``sdk.instance.listen()`` if you have not
    (``SYNAP_AUTO_LISTEN=false`` stops that), and the user's turn, every tool
    call, every tool result, the agent's reply and the end of the session go on
    it. That is what anticipation reads, and it is what makes the NEXT turn's
    fetch a cache hit instead of a cold retrieval. When the stream cannot carry
    a turn, the user's and the assistant's turns go over REST instead; tool
    calls and results exist only on the stream.

    The reply comes from the ``Stop`` hook's ``last_assistant_message``. Claude
    Code sends it on versions that carry the field (2.1.308 does); on one that
    does not, call :func:`report_assistant_turn` yourself once your run returns.

    Claude Code's transcript
    ------------------------
    Claude Code owns the transcript, and these hooks cannot replace it: a hook
    can add context to a prompt (``additionalContext``) but not remove what the
    transcript already holds. What they do is add Synap's compaction when the
    transcript lacks it, and feed it to Claude Code's own summarizer when it
    compacts. To make a resumed session start from Synap's compaction instead of
    its own stored transcript, also pass
    :func:`synap_claude_agent.create_synap_session_store` as
    ``ClaudeAgentOptions(session_store=...)``.
    """
    if sdk is None:
        raise ValueError("create_synap_hooks requires a non-None sdk")
    if not user_id:
        raise ValueError("create_synap_hooks requires a non-empty user_id")

    preamble = context_preamble or _DEFAULT_CONTEXT_PREAMBLE

    async def on_user_prompt_submit(
        input_data: Any,
        tool_use_id: Optional[str],
        context: HookContext,
    ) -> dict[str, Any]:
        # The SDK may pass the TypedDict as a plain dict — support both.
        prompt = _field(input_data, "prompt", "")
        if not prompt or not str(prompt).strip():
            return {}

        conv_id = conversation_id or _field(input_data, "session_id", None) or None

        formatted = ""
        try:
            response = await sdk.fetch(
                conversation_id=conv_id,
                user_id=user_id,
                customer_id=customer_id or None,
                search_query=[str(prompt)],
                max_results=max_results,
                mode=mode,
                include_conversation_context=False,
            )
            formatted = (getattr(response, "formatted_context", None) or "").strip()
        except Exception as exc:  # noqa: BLE001 — read degrades gracefully
            logger.error(
                "synap_claude_agent.UserPromptSubmit: sdk.fetch failed "
                "user_id=%s conversation_id=%s error=%s",
                user_id, conv_id, exc, exc_info=True,
            )

        # Before the prompt is recorded: Synap's answer would otherwise end with
        # the prompt itself.
        history = ""
        if conv_id:
            history = await history_context(
                sdk, str(conv_id), input_data,
                site="synap_claude_agent.UserPromptSubmit",
            )

        if record_user_prompts and conv_id:
            # The stream, opened if need be; REST only when it cannot carry the
            # turn. Never both: the server persists a streamed turn itself.
            await record_turn(
                sdk, role="user", content=str(prompt),
                conversation_id=str(conv_id), user_id=user_id,
                customer_id=customer_id,
            )

        parts = [preamble.format(body=formatted)] if formatted else []
        if history:
            parts.append(history)
        if not parts:
            return {}

        return {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": "\n\n".join(parts),
            }
        }

    async def on_pre_tool_use(
        input_data: Any,
        tool_use_id: Optional[str],
        context: HookContext,
    ) -> dict[str, Any]:
        """Report the tool the agent is about to run."""
        conv_id = conversation_id or _field(input_data, "session_id", None) or ""
        await report_tool_call(
            sdk,
            tool_name=str(_field(input_data, "tool_name", "") or ""),
            tool_args=_field(input_data, "tool_input", None),
            tool_call_id=str(
                _field(input_data, "tool_use_id", None) or tool_use_id or ""
            ),
            conversation_id=str(conv_id),
            user_id=user_id,
            customer_id=customer_id,
        )
        return {}

    async def on_post_tool_use(
        input_data: Any,
        tool_use_id: Optional[str],
        context: HookContext,
    ) -> dict[str, Any]:
        """Report what the tool returned.

        ⚠ A tool response is usually the caller's own customer data. It is an
        anticipation hint and never becomes a long-term memory, but it does
        leave their process. Whatever the framework hands us is what travels,
        and this hook does not go looking for more.
        """
        conv_id = conversation_id or _field(input_data, "session_id", None) or ""
        await report_tool_result(
            sdk,
            result=_field(input_data, "tool_response", None),
            tool_name=str(_field(input_data, "tool_name", "") or ""),
            tool_call_id=str(
                _field(input_data, "tool_use_id", None) or tool_use_id or ""
            ),
            conversation_id=str(conv_id),
            user_id=user_id,
            customer_id=customer_id,
        )
        return {}

    async def on_stop(
        input_data: Any,
        tool_use_id: Optional[str],
        context: HookContext,
    ) -> dict[str, Any]:
        """Record the reply, then close the session so the turn is finalised now
        rather than on a timer."""
        conv_id = str(conversation_id or _field(input_data, "session_id", None) or "")
        reply = _field(input_data, "last_assistant_message", None)
        if conv_id and isinstance(reply, str) and reply.strip() and not is_synap_history(reply):
            if _manual_turns.get(sdk, {}).pop(conv_id, None) != reply:
                await record_turn(
                    sdk, role="assistant", content=reply,
                    conversation_id=conv_id, user_id=user_id, customer_id=customer_id,
                )
        await end_session(sdk, conv_id)
        return {}

    async def on_pre_compact(
        input_data: Any,
        tool_use_id: Optional[str],
        context: HookContext,
    ) -> dict[str, Any]:
        """Give Claude Code's summarizer Synap's compaction to build on.

        Claude Code appends ``systemMessage`` to its summarization prompt as
        "Additional Instructions" (checked on Claude Code 2.1.308). Its summary
        replaces its transcript, so this is the only point where Synap's view can
        reach what Claude Code keeps.
        """
        conv_id = conversation_id or _field(input_data, "session_id", None)
        if not conv_id:
            return {}
        history = await fetch_compacted_history(sdk, str(conv_id))
        if history is None:
            return {}
        return {
            "systemMessage": (
                "Synap holds this conversation's own compaction, given below. Build "
                "your summary on it and keep its names, figures and decisions "
                "exactly.\n\n" + history.preamble
            )
        }

    return {
        "UserPromptSubmit": [HookMatcher(hooks=[on_user_prompt_submit])],
        "PreToolUse": [HookMatcher(hooks=[on_pre_tool_use])],
        "PostToolUse": [HookMatcher(hooks=[on_post_tool_use])],
        "Stop": [HookMatcher(hooks=[on_stop])],
        "PreCompact": [HookMatcher(hooks=[on_pre_compact])],
    }


async def report_assistant_turn(
    sdk: MaximemSynapSDK,
    content: str,
    *,
    conversation_id: str,
    user_id: str,
    customer_id: str = "",
) -> bool:
    """Write the agent's reply yourself. Returns whether it was written.

    The ``Stop`` hook already writes it, from ``last_assistant_message``, so most
    callers never need this. It is for a Claude Code that does not send that
    field, or a run without the hooks; if you call it and also use them, the
    reply is written once.
    """
    written = await record_turn(
        sdk, role="assistant", content=content or "",
        conversation_id=conversation_id, user_id=user_id,
        customer_id=customer_id,
    )
    if written:
        try:
            _manual_turns.setdefault(sdk, {})[conversation_id] = content
        except TypeError:  # an SDK that cannot be weakly referenced
            pass
    return written


def _field(input_data: Any, name: str, default: Any) -> Any:
    """Pull a field from a TypedDict/dict-like hook input."""
    if isinstance(input_data, dict):
        return input_data.get(name, default)
    return getattr(input_data, name, default)


# Keep an import alive for documentation/type-checking users even though we
# don't access it directly here.
_ = UserPromptSubmitHookInput
