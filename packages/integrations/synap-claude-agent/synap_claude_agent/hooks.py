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
        customer_id: Optional customer/org scope.
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

        if not formatted:
            return {}

        return {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": preamble.format(body=formatted),
            }
        }

    return {
        "UserPromptSubmit": [HookMatcher(hooks=[on_user_prompt_submit])],
    }


def _field(input_data: Any, name: str, default: Any) -> Any:
    """Pull a field from a TypedDict/dict-like hook input."""
    if isinstance(input_data, dict):
        return input_data.get(name, default)
    return getattr(input_data, name, default)


# Keep an import alive for documentation/type-checking users even though we
# don't access it directly here.
_ = UserPromptSubmitHookInput
