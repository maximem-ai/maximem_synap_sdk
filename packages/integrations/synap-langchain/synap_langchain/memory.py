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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    SynapIntegrationError,
    record_turn,
    run_async,
)

from synap_langchain._messages import message_text, turn_role

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

        if not prompt_ctx:
            return []

        msgs: List[BaseMessage] = []
        # The compaction block (and the earlier turns the SDK found for the
        # latest message) stands in for the turns it covers: recent_messages
        # holds only the turns after it. Older SDKs send neither.
        preamble = "\n\n".join(
            part for part in (getattr(prompt_ctx, "context_block", None),
                              getattr(prompt_ctx, "history_block", None)) if isinstance(part, str) and part
        )
        if preamble:
            msgs.append(SystemMessage(content=preamble))
        for rm in prompt_ctx.recent_messages or []:
            role = getattr(rm, "role", None) or "user"
            content = str(getattr(rm, "content", "") or "")
            if role == "assistant":
                msgs.append(AIMessage(content=content))
            elif role == "user":
                msgs.append(HumanMessage(content=content))
            elif role == "system":
                msgs.append(SystemMessage(content=content))
            # Anything else (a tool turn) is not something either side said.
        return msgs

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        run_async(self.aadd_messages(messages))

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Write messages. Surfaces SDK errors (caller invoked this explicitly).

        Only what a user or the assistant said is written, on the gRPC stream
        (opened on the first write) and over REST when the stream cannot carry
        it. A system message, including the compaction this store returned, is
        not a turn; a tool result goes to the SDK's local archive instead.
        """
        for msg in messages:
            if isinstance(msg, ToolMessage):
                # What a tool returned, not something the user said: kept in
                # the SDK's local archive for history search, never sent as a turn.
                archive = getattr(self.sdk, "archive_tool_result", None)
                if callable(archive):
                    archive(self.conversation_id, msg.content, tool_name=getattr(msg, "name", None))
                continue
            role = turn_role(msg)
            if role is None:
                continue
            text = message_text(msg)
            if not text.strip():
                continue  # tool calls only: no words to record
            if not await record_turn(
                self.sdk,
                role=role,
                content=text,
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                customer_id=self.customer_id,
            ):
                raise SynapIntegrationError(
                    "langchain.aadd_messages",
                    "the turn was not recorded: neither the gRPC stream nor REST accepted it",
                    {"role": role, "conversation_id": self.conversation_id},
                )

    def clear(self) -> None:
        self.sdk.cache.clear()

    async def aclear(self) -> None:
        self.sdk.cache.clear()


SynapMemory = SynapChatMessageHistory
