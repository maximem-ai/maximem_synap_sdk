"""Synap short-term context for AutoGen / AG2.

``AssistantAgent`` builds every model call from ``model_context.get_messages()``,
so that is where Synap's view of the conversation goes. This wraps the agent's
``ChatCompletionContext``: once Synap has compacted the conversation, the list it
returns is a system message with the compaction block, the turns Synap holds after
the compaction, and the current turn. Everything the framework would have sent
from before that is left out, so no turn goes to the model twice. Until Synap
has a compaction, and whenever it cannot answer, the inner context's messages
are returned unchanged.

Pass the resulting context into the agent via
``AssistantAgent(model_context=...)``.

Quality contract identical to the LangGraph adapter:

- ``conversation_id`` required + explicit at construction.
- SDK failures never crash the agent by default
  (``on_error="fallback"``): we log via :class:`SynapIntegrationError`
  and return the wrapped context's messages unchanged. ``on_error="raise"``
  available.
- No compaction is a no-op: nothing is added to the inner messages.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, List, Literal, Optional

from autogen_core.model_context import (
    ChatCompletionContext,
    UnboundedChatCompletionContext,
)
from autogen_core.models import AssistantMessage, LLMMessage
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    SynapIntegrationError,
    fetch_compacted_history,
    wrap_sdk_errors_async,
)

from synap_autogen.history import splice_messages

logger = logging.getLogger(__name__)

_SUPPORTED_STYLES = ("structured", "narrative", "bullet_points")
_DEFAULT_OPEN = "<synap_short_term_context>"
_DEFAULT_CLOSE = "</synap_short_term_context>"

_OnError = Literal["fallback", "raise"]


def _answered(response: Any) -> Any:
    """An SDK stand-in that returns a context response already fetched, so
    ``fetch_compacted_history`` reads it without a second request and this
    class keeps its own error handling."""

    async def get_context_for_prompt(**_: Any) -> Any:
        return response

    return SimpleNamespace(
        conversation=SimpleNamespace(
            context=SimpleNamespace(get_context_for_prompt=get_context_for_prompt)
        )
    )


def _agent_name(messages: List[LLMMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AssistantMessage) and message.source:
            return message.source
    return "assistant"


class SynapShortTermChatContext(ChatCompletionContext):
    """AG2 ``ChatCompletionContext`` that puts Synap's compaction in place of
    the framework's earlier turns.

    Wraps an inner context (defaults to
    :class:`UnboundedChatCompletionContext`, which is also the fallback when
    Synap has nothing to add). On every ``get_messages()`` invocation:

    1. Awaits ``sdk.conversation.context.get_context_for_prompt`` (cache-
       first via the SDK helper).
    2. If the conversation has been compacted, returns the inner messages with
       every turn before the current one replaced by a
       :class:`autogen_core.models.SystemMessage` (the compaction block, wrapped
       in the configured preamble tags) and the turns Synap holds after it.
    3. Otherwise returns the inner messages unchanged.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation ID. **Required.**
        inner: Optional inner context to wrap. Defaults to a fresh
            :class:`UnboundedChatCompletionContext`.
        style: One of ``"structured" | "narrative" | "bullet_points"``.
            Forwarded to the SDK; the compaction block does not depend on it.
        preamble_open / preamble_close: wrappers around the compaction block; pass ``None``
            for both to skip wrapping.
        on_error: ``"fallback"`` (default) returns inner messages
            unchanged on SDK failure; ``"raise"`` propagates
            :class:`SynapIntegrationError`.

    Example::

        from autogen_agentchat.agents import AssistantAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from synap_autogen import SynapShortTermChatContext

        agent = AssistantAgent(
            name="support",
            model_client=OpenAIChatCompletionClient(model="gpt-4o"),
            system_message="You are a polite support agent.",
            model_context=SynapShortTermChatContext(
                sdk, conversation_id="conv_abc"
            ),
        )
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        conversation_id: str,
        *,
        inner: Optional[ChatCompletionContext] = None,
        style: str = "narrative",
        preamble_open: Optional[str] = _DEFAULT_OPEN,
        preamble_close: Optional[str] = _DEFAULT_CLOSE,
        on_error: _OnError = "fallback",
    ) -> None:
        if sdk is None:
            raise ValueError(
                "SynapShortTermChatContext requires a non-None sdk"
            )
        if not conversation_id or not str(conversation_id).strip():
            raise ValueError(
                "SynapShortTermChatContext requires a non-empty conversation_id"
            )
        if style not in _SUPPORTED_STYLES:
            raise ValueError(
                f"SynapShortTermChatContext: unsupported style={style!r}; "
                f"expected one of {_SUPPORTED_STYLES}"
            )
        if on_error not in ("fallback", "raise"):
            raise ValueError(
                "SynapShortTermChatContext: on_error must be 'fallback' "
                f"or 'raise', got {on_error!r}"
            )

        super().__init__()
        self._sdk = sdk
        self._conversation_id = conversation_id
        self._inner = inner or UnboundedChatCompletionContext()
        self._style = style
        self._preamble_open = preamble_open
        self._preamble_close = preamble_close
        self._on_error: _OnError = on_error

    async def add_message(self, message: LLMMessage) -> None:
        await self._inner.add_message(message)

    async def clear(self) -> None:
        await self._inner.clear()

    async def get_messages(self) -> List[LLMMessage]:
        inner_messages = await self._inner.get_messages()
        history = None
        try:
            async with wrap_sdk_errors_async(
                "synap_autogen.SynapShortTermChatContext.get_messages",
                logger,
                conversation_id=self._conversation_id,
                style=self._style,
            ):
                response = await self._sdk.conversation.context.get_context_for_prompt(
                    conversation_id=self._conversation_id,
                    style=self._style,
                )
            history = await fetch_compacted_history(
                _answered(response), self._conversation_id
            )
        except SynapIntegrationError:
            if self._on_error == "raise":
                raise
            return list(inner_messages)

        if history is None:
            return list(inner_messages)

        if self._preamble_open and self._preamble_close:
            # The preamble already carries the SDK's own tags; the configured
            # wrapper goes around them as before.
            history = replace(
                history,
                preamble=f"{self._preamble_open}\n{history.preamble}\n{self._preamble_close}",
            )
        return splice_messages(
            list(inner_messages), history, _agent_name(inner_messages)
        )

    async def save_state(self) -> dict:
        # Delegate to the inner context's state.
        inner_state = await self._inner.save_state()
        return {"inner": inner_state}

    async def load_state(self, state: dict) -> None:
        await self._inner.load_state(state.get("inner", {}))


__all__ = ["SynapShortTermChatContext"]
