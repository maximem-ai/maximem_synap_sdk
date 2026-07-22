"""Synap short-term context injection for Strands Agents.

Strands' ``system_prompt`` is a static string, so short-term / working-memory
context is injected via a :class:`~strands.hooks.HookProvider` on
``BeforeInvocationEvent`` — the one before-event whose ``.messages`` list is
writable (``BeforeModelCallEvent`` only permits cancelling).

**On persistence.** Strands has no ephemeral per-call injection hook for user
code (only ``MemoryManager`` uses ``strands.injection`` internally). Whatever
you put in ``event.messages`` is appended to the agent's durable history. To
avoid accumulating throwaway context turns, this hook **folds the short-term
block into the current turn's first user message** rather than adding a
separate turn — each user turn simply carries its contemporaneous context, and
there is an idempotency guard so a re-entered invocation never double-folds.
Folding (vs. a standalone ``user`` turn) also avoids consecutive same-role
messages, which some model providers reject.

Wraps ``sdk.conversation.context.get_context_for_prompt`` (cache-first behind
``SYNAP_SDK_ST_AUTHORITATIVE``). Quality contract, identical to the LangGraph /
Agno adapters:

- ``conversation_id`` required + explicit at construction.
- SDK failures never crash the agent by default (``on_error="fallback"``).
- Empty short-term context is a no-op — never mutates the user's input.
- ``on_error="raise"`` available for strict environments.
"""

from __future__ import annotations

import logging
from typing import Any, List, Literal, Optional

from strands.hooks import BeforeInvocationEvent, HookProvider, HookRegistry

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import SynapIntegrationError, wrap_sdk_errors_async

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


class SynapShortTermHook(HookProvider):
    """Inject Synap short-term context before each Strands invocation.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation ID. **Required and explicit** —
            never inferred from a Strands session id.
        style: One of ``"structured" | "narrative" | "bullet_points"``.
        preamble_open / preamble_close: Tags wrapping the injected block; pass
            ``None`` for both to drop them (which also disables the idempotency
            guard, so keep the defaults unless you have a reason not to).
        on_error: ``"fallback"`` (default) swallows SDK failures and leaves the
            input untouched; ``"raise"`` propagates :class:`SynapIntegrationError`.

    Example::

        from strands import Agent
        from maximem_synap import MaximemSynapSDK
        from synap_strands_agents import SynapShortTermHook

        sdk = MaximemSynapSDK(api_key="sk-...")
        agent = Agent(
            system_prompt="You are a support agent.",
            hooks=[SynapShortTermHook(sdk, conversation_id="conv_abc")],
        )
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        conversation_id: str,
        *,
        style: str = "narrative",
        preamble_open: Optional[str] = _DEFAULT_OPEN,
        preamble_close: Optional[str] = _DEFAULT_CLOSE,
        on_error: _OnError = "fallback",
    ) -> None:
        _validate_args(sdk, conversation_id, style, on_error, "SynapShortTermHook")
        self._sdk = sdk
        self._conversation_id = conversation_id
        self._style = style
        self._preamble_open = preamble_open
        self._preamble_close = preamble_close
        self._on_error = on_error

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeInvocationEvent, self._inject)

    async def _inject(self, event: BeforeInvocationEvent) -> None:
        st_block = await _fetch_st_block(
            self._sdk,
            self._conversation_id,
            self._style,
            self._on_error,
            "synap_strands_agents.SynapShortTermHook",
        )
        if not st_block:
            return  # no short-term context -> never touch the input
        messages = event.messages
        if not messages:
            return  # nothing to fold into (e.g. the structured_output path)
        self._fold_into_first_user(messages, st_block)

    def _wrap(self, st_block: str) -> str:
        if self._preamble_open and self._preamble_close:
            return f"{self._preamble_open}\n{st_block}\n{self._preamble_close}"
        return st_block

    def _already_injected(self, content: List[Any]) -> bool:
        if not self._preamble_open:
            return False
        return any(
            isinstance(b, dict)
            and isinstance(b.get("text"), str)
            and b["text"].lstrip().startswith(self._preamble_open)
            for b in content
        )

    def _fold_into_first_user(self, messages: List[Any], st_block: str) -> None:
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            if self._already_injected(content):
                return  # re-entered invocation — don't double-fold
            content.insert(0, {"text": self._wrap(st_block)})
            return


__all__ = ["SynapShortTermHook"]
