"""Synap short-term context for AutoGen / AG2.

AG2's ``AssistantAgent.system_message`` is a static string, so we can't
update the system prompt per LLM call directly. Instead we wrap the
agent's ``ChatCompletionContext`` so every ``get_messages()`` call
prepends a fresh Synap short-term context message (cache-first via the
SDK helper) ahead of the existing message list.

Pass the resulting context into the agent via
``AssistantAgent(model_context=...)``.

Quality contract identical to the LangGraph adapter:

- ``conversation_id`` required + explicit at construction.
- SDK failures never crash the agent by default
  (``on_error="fallback"``): we log via :class:`SynapIntegrationError`
  and return the wrapped context's messages unchanged. ``on_error="raise"``
  available.
- Empty ST is a no-op — never prepends a blank message.
"""

from __future__ import annotations

import inspect
import logging
from typing import List, Literal, Optional

from autogen_core.model_context import (
    ChatCompletionContext,
    UnboundedChatCompletionContext,
)
from autogen_core.models import LLMMessage, SystemMessage
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


class SynapShortTermChatContext(ChatCompletionContext):
    """AG2 ``ChatCompletionContext`` that prepends Synap ST every call.

    Wraps an inner context (defaults to
    :class:`UnboundedChatCompletionContext`). On every
    ``get_messages()`` invocation:

    1. Awaits ``sdk.conversation.context.get_context_for_prompt`` (cache-
       first via the SDK helper).
    2. If a non-empty ST block comes back, prepends a
       :class:`autogen_core.models.SystemMessage` (wrapped in the
       configured preamble tags) before the inner messages.
    3. Returns the combined list.

    Args:
        sdk: Initialised :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation ID. **Required.**
        inner: Optional inner context to wrap. Defaults to a fresh
            :class:`UnboundedChatCompletionContext`.
        style: One of ``"structured" | "narrative" | "bullet_points"``.
        preamble_open / preamble_close: ST block wrappers; pass ``None``
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
        st_block = ""
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
            if getattr(response, "available", False):
                st_block = (getattr(response, "formatted_context", None) or "").strip()
        except SynapIntegrationError:
            if self._on_error == "raise":
                raise
            return list(inner_messages)

        if not st_block:
            return list(inner_messages)

        if self._preamble_open and self._preamble_close:
            wrapped = f"{self._preamble_open}\n{st_block}\n{self._preamble_close}"
        else:
            wrapped = st_block

        return [SystemMessage(content=wrapped), *inner_messages]

    async def save_state(self) -> dict:
        # Delegate to the inner context's state.
        inner_state = await self._inner.save_state()
        return {"inner": inner_state}

    async def load_state(self, state: dict) -> None:
        await self._inner.load_state(state.get("inner", {}))


__all__ = ["SynapShortTermChatContext"]
