"""Synap chat memory for LlamaIndex.

Implements LlamaIndex's BaseMemory interface backed by Synap's
conversation context.

Error-handling split follows the LangChain analogue:

- Read paths (:meth:`aget`) degrade gracefully — a memory lookup
  failure should not crash the chain — but they now log at ``ERROR``
  (previously ``DEBUG``, which meant outages were invisible).
- Write paths (:meth:`aput`) raise :class:`SynapIntegrationError`:
  the caller explicitly wrote data and has a right to know if it
  didn't land.
"""

import logging
from typing import Any, List, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory.types import BaseMemory

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import run_async, wrap_sdk_errors_async

logger = logging.getLogger(__name__)


class SynapChatMemory(BaseMemory):
    """LlamaIndex chat memory backed by Synap."""

    _sdk: MaximemSynapSDK
    _conversation_id: str
    _user_id: str
    _customer_id: str
    _messages: List[ChatMessage]

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        conversation_id: str,
        user_id: str,
        customer_id: str = "",
    ):
        if sdk is None:
            raise ValueError("SynapChatMemory requires a non-None sdk")
        if not conversation_id:
            raise ValueError(
                "SynapChatMemory requires a non-empty conversation_id"
            )
        if not user_id:
            raise ValueError("SynapChatMemory requires a non-empty user_id")

        self._sdk = sdk
        self._conversation_id = conversation_id
        self._user_id = user_id
        self._customer_id = customer_id
        self._messages = []

    @classmethod
    def from_defaults(
        cls,
        sdk: Optional[MaximemSynapSDK] = None,
        conversation_id: str = "",
        user_id: str = "",
        customer_id: str = "",
        **kwargs: Any,
    ) -> "SynapChatMemory":
        if sdk is None:
            raise ValueError(
                "SynapChatMemory.from_defaults requires sdk — "
                "construct a MaximemSynapSDK(instance_id=...) first"
            )
        return cls(
            sdk=sdk,
            conversation_id=conversation_id,
            user_id=user_id,
            customer_id=customer_id,
        )

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        return run_async(self.aget(input, **kwargs))

    async def aget(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        """Get conversation history from Synap (best-effort).

        A memory lookup feeding into an LLM prompt should not abort the
        chain on failure. Partial failures are logged at ERROR.
        """
        messages: List[ChatMessage] = []

        if input:
            try:
                response = await self._sdk.fetch(
                    conversation_id=self._conversation_id,
                    user_id=self._user_id,
                    customer_id=self._customer_id or None,
                    search_query=[input],
                    include_conversation_context=False,
                )
                if response.formatted_context:
                    messages.append(ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=f"Relevant user context:\n{response.formatted_context}",
                    ))
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "SynapChatMemory.aget: fetch failed "
                    "conversation_id=%s error=%s",
                    self._conversation_id, exc, exc_info=True,
                )

        try:
            prompt_ctx = await self._sdk.conversation.context.get_context_for_prompt(
                conversation_id=self._conversation_id,
            )
            if prompt_ctx.formatted_context:
                messages.append(ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=f"Conversation history:\n{prompt_ctx.formatted_context}",
                ))
            for msg in prompt_ctx.recent_messages:
                role = MessageRole.USER if msg.role == "user" else MessageRole.ASSISTANT
                messages.append(ChatMessage(role=role, content=msg.content))
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "SynapChatMemory.aget: get_context_for_prompt failed "
                "conversation_id=%s error=%s",
                self._conversation_id, exc, exc_info=True,
            )

        messages.extend(self._messages)
        return messages

    def get_all(self) -> List[ChatMessage]:
        return self.get()

    async def aget_all(self) -> List[ChatMessage]:
        return await self.aget()

    def put(self, message: ChatMessage) -> None:
        run_async(self.aput(message))

    async def aput(self, message: ChatMessage) -> None:
        """Record a message. Surfaces SDK errors (explicit write path)."""
        self._messages.append(message)
        if message.role not in (MessageRole.USER, MessageRole.ASSISTANT):
            return

        role = "user" if message.role == MessageRole.USER else "assistant"
        async with wrap_sdk_errors_async(
            "llamaindex.aput", logger,
            role=role, conversation_id=self._conversation_id,
        ):
            await self._sdk.conversation.record_message(
                conversation_id=self._conversation_id,
                role=role,
                content=str(message.content),
                user_id=self._user_id,
                customer_id=self._customer_id,
            )

    def set(self, messages: List[ChatMessage]) -> None:
        self._messages = list(messages)

    async def aset(self, messages: List[ChatMessage]) -> None:
        self._messages = list(messages)

    def reset(self) -> None:
        self._messages.clear()

    async def areset(self) -> None:
        self._messages.clear()
