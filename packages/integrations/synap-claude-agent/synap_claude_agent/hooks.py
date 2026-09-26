"""Synap hooks for the Claude Agent SDK.

Wires a ``UserPromptSubmit`` hook that:

1. Fetches Synap context for the incoming prompt and returns it via
   ``hookSpecificOutput.additionalContext`` — the SDK stitches that into
   Claude's prompt automatically.
2. Records the user's prompt back to Synap conversation history so future
   turns can recall it.

The hook NEVER raises. A failing Synap call returns ``{}`` (no additional
context, no block) so agent runs continue uninterrupted. This mirrors the
policy we use for LangChain's SynapCallbackHandler and MAF's
SynapContextProvider.after_run — context providers must not crash the agent.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from claude_agent_sdk import HookMatcher
from claude_agent_sdk.types import (
    HookContext,
    UserPromptSubmitHookInput,
)
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    end_session,
    report_tool_call,
    report_tool_result,
    report_turn,
)

logger = logging.getLogger(__name__)


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
    When the application has an active ``sdk.instance.listen()`` stream, these
    hooks also report the run on it: the user's turn, every tool call, every
    tool result, and the end of the session. That is what anticipation reads,
    and it is what makes the NEXT turn's fetch a cache hit instead of a cold
    retrieval.

    Nothing here requires a stream. With no ``listen()`` running, every report
    is a no-op and these hooks behave exactly as they did before.

    ⚠ One event cannot come from a hook: ``assistant_message``, which is the
    event anticipation actually acts on. The Claude Agent SDK's hook inputs do
    not carry the final assistant text — ``StopHookInput`` has the session id
    and nothing else. Call :func:`report_assistant_turn` yourself once your run
    returns, or anticipation never fires for this agent.
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

        if record_user_prompts and conv_id:
            try:
                await sdk.conversation.record_message(
                    conversation_id=conv_id,
                    role="user",
                    content=str(prompt),
                    user_id=user_id,
                    customer_id=customer_id,
                )
            except Exception as exc:  # noqa: BLE001 — must not raise
                logger.error(
                    "synap_claude_agent.UserPromptSubmit: record_message failed "
                    "conversation_id=%s error=%s",
                    conv_id, exc, exc_info=True,
                )

        # The same turn on the stream, when there is one. `record_message`
        # above is the durable write; this is what anticipation reads, and the
        # two are not interchangeable.
        if conv_id:
            await report_turn(
                sdk, role="user", content=str(prompt),
                conversation_id=str(conv_id), user_id=user_id,
                customer_id=customer_id,
            )

        if not formatted:
            return {}

        return {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": preamble.format(body=formatted),
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
        """Close the session so the turn is finalised now rather than on a timer."""
        conv_id = conversation_id or _field(input_data, "session_id", None) or ""
        await end_session(sdk, str(conv_id))
        return {}

    return {
        "UserPromptSubmit": [HookMatcher(hooks=[on_user_prompt_submit])],
        "PreToolUse": [HookMatcher(hooks=[on_pre_tool_use])],
        "PostToolUse": [HookMatcher(hooks=[on_post_tool_use])],
        "Stop": [HookMatcher(hooks=[on_stop])],
    }


async def report_assistant_turn(
    sdk: MaximemSynapSDK,
    content: str,
    *,
    conversation_id: str,
    user_id: str,
    customer_id: str = "",
) -> bool:
    """Report the agent's reply. Call this once your run returns.

    This cannot be a hook. The Claude Agent SDK's hook inputs do not carry the
    final assistant text: ``StopHookInput`` has a session id and a flag, and
    nothing else. So the one event anticipation actually acts on is the one
    event the hooks cannot produce.

    Without it, Synap sees the question and the tools and never learns what was
    answered, and the next turn's fetch is a cold retrieval. One line at the
    end of your run is the whole cost.

    Returns whether it was sent: False when no ``listen()`` stream is running,
    which is not an error.
    """
    return await report_turn(
        sdk, role="assistant", content=content or "",
        conversation_id=conversation_id, user_id=user_id,
        customer_id=customer_id,
    )


def _field(input_data: Any, name: str, default: Any) -> Any:
    """Pull a field from a TypedDict/dict-like hook input."""
    if isinstance(input_data, dict):
        return input_data.get(name, default)
    return getattr(input_data, name, default)


# Keep an import alive for documentation/type-checking users even though we
# don't access it directly here.
_ = UserPromptSubmitHookInput
