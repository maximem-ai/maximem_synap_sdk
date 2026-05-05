"""Synap chat message history for LangChain.

Implements LangChain's BaseChatMessageHistory interface backed by Synap.
Use with RunnableWithMessageHistory to add memory to any chain or agent.

Error-handling split:

- :meth:`aget_messages` does **not** raise on SDK failure — a memory
  lookup miss should not crash the chain's main response path. Failures
  are logged at ``ERROR`` (previously ``DEBUG``, which was invisible in
  practice) and an empty list is returned.
- :meth:`aadd_messages` **does** raise. The caller explicitly invoked
  ``add``; silently dropping messages masks ingestion outages.
"""

import logging
from typing import List, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import run_async, wrap_sdk_errors_async

logger = logging.getLogger(__name__)


class SynapChatMessageHistory(BaseChatMessageHistory):
    """LangChain chat message history backed by Synap.

    Records conversation messages via the SDK and retrieves them
    using get_context_for_prompt(). Use with RunnableWithMessageHistory
    for automatic memory on every turn.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        conversation_id: str,
        user_id: str,
        customer_id: str = "",
    ):
        if sdk is None:
            raise ValueError("SynapChatMessageHistory requires a non-None sdk")
        if not conversation_id:
            raise ValueError(
                "SynapChatMessageHistory requires a non-empty conversation_id"
            )
        if not user_id:
            raise ValueError(
                "SynapChatMessageHistory requires a non-empty user_id"
            )

        self.sdk = sdk
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.customer_id = customer_id

    @classmethod
    def from_instance(
        cls,
        instance_id: str,
        conversation_id: str,
        user_id: str,
        customer_id: str = "",
    ) -> "SynapChatMessageHistory":
        """Create from an instance ID. Initializes the SDK automatically."""
        sdk = MaximemSynapSDK(instance_id=instance_id)
        return cls(
            sdk=sdk,
            conversation_id=conversation_id,
            user_id=user_id,
            customer_id=customer_id,
        )

    @property
    def messages(self) -> List[BaseMessage]:
        return run_async(self.aget_messages())

    async def aget_messages(self) -> List[BaseMessage]:
        """Best-effort retrieve. Failures return an empty list + ERROR log."""
        try:
            prompt_ctx = await self.sdk.conversation.context.get_context_for_prompt(
                conversation_id=self.conversation_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "SynapChatMessageHistory.aget_messages failed "
                "conversation_id=%s error=%s",
                self.conversation_id, exc, exc_info=True,
            )
            return []

        if not prompt_ctx or not prompt_ctx.recent_messages:
            return []

        msgs: List[BaseMessage] = []
        for rm in prompt_ctx.recent_messages:
            role = getattr(rm, "role", None) or "user"
            content = getattr(rm, "content", "") or ""
            if role == "assistant":
                msgs.append(AIMessage(content=str(content)))
            else:
                msgs.append(HumanMessage(content=str(content)))
        return msgs

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        run_async(self.aadd_messages(messages))

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Write messages. Surfaces SDK errors (caller invoked this explicitly)."""
        for msg in messages:
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            async with wrap_sdk_errors_async(
                "langchain.aadd_messages",
                logger,
                role=role,
                conversation_id=self.conversation_id,
            ):
                await self.sdk.conversation.record_message(
                    conversation_id=self.conversation_id,
                    role=role,
                    content=str(msg.content),
                    user_id=self.user_id,
                    customer_id=self.customer_id,
                )

    def clear(self) -> None:
        self.sdk.cache.clear()

    async def aclear(self) -> None:
        self.sdk.cache.clear()


SynapMemory = SynapChatMessageHistory
