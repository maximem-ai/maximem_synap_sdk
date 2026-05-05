"""SynapHistoryProvider — MAF HistoryProvider backed by Synap.

Persists conversation messages to Synap and loads them back into the MAF
session. Wrap one or more of these in ``context_providers`` to get durable,
cross-process conversation memory.

Load path (``get_messages``) reads via ``sdk.conversation.context.get_context_for_prompt``
and degrades gracefully on failure (returns ``[]``) — a history read outage
shouldn't abort the agent turn.

Save path (``save_messages``) iterates ``sdk.conversation.record_message``
and surfaces SDK errors as :class:`SynapIntegrationError` because silent drops
would hide ingestion problems from the caller.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional, Sequence

from agent_framework import Message
from agent_framework._sessions import HistoryProvider
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import wrap_sdk_errors_async

logger = logging.getLogger(__name__)


class SynapHistoryProvider(HistoryProvider):
    """Store + load MAF conversation history via Synap."""

    DEFAULT_SOURCE_ID: ClassVar[str] = "synap_history"

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: str,
        customer_id: str = "",
        conversation_id: Optional[str] = None,
        *,
        source_id: str = DEFAULT_SOURCE_ID,
        load_messages: bool = True,
        store_inputs: bool = True,
        store_context_messages: bool = False,
        store_context_from: Optional[set[str]] = None,
        store_outputs: bool = True,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapHistoryProvider requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapHistoryProvider requires a non-empty user_id")

        super().__init__(
            source_id,
            load_messages=load_messages,
            store_inputs=store_inputs,
            store_context_messages=store_context_messages,
            store_context_from=store_context_from,
            store_outputs=store_outputs,
        )
        self.sdk = sdk
        self.user_id = user_id
        self.customer_id = customer_id
        self.conversation_id = conversation_id

    async def get_messages(
        self,
        session_id: Optional[str],
        *,
        state: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[Message]:
        conv_id = self.conversation_id or session_id
        if not conv_id:
            return []

        try:
            response = await self.sdk.conversation.context.get_context_for_prompt(
                conversation_id=conv_id,
            )
        except Exception as exc:  # noqa: BLE001 — read-side degrades gracefully
            logger.error(
                "SynapHistoryProvider.get_messages: get_context_for_prompt failed "
                "conversation_id=%s error=%s",
                conv_id,
                exc,
                exc_info=True,
            )
            return []

        recent = getattr(response, "recent_messages", None) or []
        messages: list[Message] = []
        for rm in recent:
            role = getattr(rm, "role", None)
            content = getattr(rm, "content", None)
            if not role or not content:
                continue
            messages.append(Message(role=role, contents=[content]))
        return messages

    async def save_messages(
        self,
        session_id: Optional[str],
        messages: Sequence[Message],
        *,
        state: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        conv_id = self.conversation_id or session_id
        if not conv_id or not messages:
            return

        async with wrap_sdk_errors_async(
            "microsoft_agent.save_messages",
            logger,
            user_id=self.user_id,
            conversation_id=conv_id,
        ):
            for message in messages:
                role_raw = getattr(message, "role", None)
                role = role_raw.value if hasattr(role_raw, "value") else str(role_raw)
                if role not in {"user", "assistant", "system"}:
                    continue
                text = getattr(message, "text", "") or ""
                if not text.strip():
                    continue
                await self.sdk.conversation.record_message(
                    conversation_id=conv_id,
                    role=role,
                    content=text,
                    user_id=self.user_id,
                    customer_id=self.customer_id,
                )
