"""SynapHistoryProvider — MAF HistoryProvider backed by Synap.

Persists conversation messages to Synap and loads them back into the MAF
session. Wrap one or more of these in ``context_providers`` to get durable,
cross-process conversation memory.

Load path (``get_messages``) reads via ``sdk.conversation.context.get_context_for_prompt``
and degrades gracefully on failure (returns ``[]``) — a history read outage
shouldn't abort the agent turn. What it returns is Synap's view of the
conversation: the compaction block (when the server has compacted) as one system
message, then the turns after it. That is the history, whole, so the agent never
holds the turns the compaction replaced.

Save path (``save_messages``) puts each user and assistant turn on the gRPC
stream (opened if need be) and falls back to ``sdk.conversation.record_message``
when the stream cannot carry it. **Never both**: the server writes the
conversation row from the stream event itself, so doing both stores the turn
twice and extracts it twice. System messages are not turns and are not written.
SDK errors on the REST fallback surface as :class:`SynapIntegrationError`
because silent drops would hide ingestion problems from the caller; every
message is still tried before the first error is raised.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional, Sequence

from agent_framework import HistoryProvider, Message
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import SynapIntegrationError, wrap_sdk_errors_async

from synap_microsoft_agent.stream import reported_ids, report_message_turn

# How the server's compaction block starts (v3/render.py).
COMPACTION_PREFIX = "## Conversation so far (compacted"

logger = logging.getLogger(__name__)


class SynapHistoryProvider(HistoryProvider):
    """Store + load MAF conversation history via Synap."""

    DEFAULT_SOURCE_ID: ClassVar[str] = "synap_history"
    #: `SynapContextProvider` reads these to avoid loading Synap's history and
    #: writing the turns a second time beside this provider.
    SYNAP_HISTORY: ClassVar[bool] = True

    @property
    def SYNAP_LOADS_HISTORY(self) -> bool:  # noqa: N802 — read like the class flag above
        return self.load_messages

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
        # The compaction block (and the earlier turns the SDK found for the
        # latest message) stands in for the turns it covers: recent_messages
        # holds only the turns after it. Older SDKs send neither.
        preamble = "\n\n".join(
            part for part in (getattr(response, "context_block", None),
                              getattr(response, "history_block", None)) if isinstance(part, str) and part
        )
        if preamble:
            messages.append(Message(role="system", contents=[preamble]))
        for rm in recent:
            role = getattr(rm, "role", None)
            content = getattr(rm, "content", None)
            if not role or not content:
                continue
            messages.append(Message(role=role, contents=[content]))
        return messages

    async def after_run(
        self,
        *,
        agent: Any,
        session: Any,
        context: Any,
        state: dict[str, Any],
    ) -> None:
        """Store the turn, minus the input another Synap provider already put on the stream."""
        reported = reported_ids(context)
        original = context.input_messages
        if reported and original:
            context.input_messages = [m for m in original if id(m) not in reported]
        try:
            await super().after_run(agent=agent, session=session, context=context, state=state)
        finally:
            context.input_messages = original

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

        first_error: Optional[SynapIntegrationError] = None
        for message in messages:
            role_raw = getattr(message, "role", None)
            role = role_raw.value if hasattr(role_raw, "value") else str(role_raw)
            # `record_message` takes "user" or "assistant" only, and a system
            # message is an instruction, not something somebody said.
            if role not in {"user", "assistant"}:
                continue
            text = getattr(message, "text", "") or ""
            if not text.strip():
                continue
            try:
                async with wrap_sdk_errors_async(
                    "microsoft_agent.save_messages",
                    logger,
                    user_id=self.user_id,
                    conversation_id=conv_id,
                ):
                    # Stream first, REST only as a fallback, never both.
                    if await report_message_turn(
                        self.sdk,
                        role=role,
                        content=text,
                        conversation_id=conv_id,
                        user_id=self.user_id,
                        customer_id=self.customer_id,
                    ):
                        continue
                    await self.sdk.conversation.record_message(
                        conversation_id=conv_id,
                        role=role,
                        content=text,
                        user_id=self.user_id,
                        customer_id=self.customer_id,
                    )
            except SynapIntegrationError as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error
