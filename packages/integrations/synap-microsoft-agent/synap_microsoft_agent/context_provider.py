"""SynapContextProvider — MAF ContextProvider backed by Synap.

Pattern mirrors ``agent_framework.mem0.Mem0ContextProvider``:

- ``before_run`` builds a query from ``context.input_messages`` text, calls
  ``sdk.fetch(...)``, and appends the formatted result as instructions via
  ``context.extend_instructions(source_id, ...)``. Read failures degrade
  gracefully — we log + skip rather than breaking the agent turn.

- ``after_run`` records each input + response message to Synap via
  ``sdk.conversation.record_message(...)``. Write failures are logged but
  never re-raised, following MAF's "context providers should not crash the
  agent" contract (same as LangChain's ``SynapCallbackHandler``).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional

from agent_framework import ContextProvider
from maximem_synap import MaximemSynapSDK

logger = logging.getLogger(__name__)


class SynapContextProvider(ContextProvider):
    """Inject Synap context + record turns back to Synap."""

    DEFAULT_SOURCE_ID: ClassVar[str] = "synap"
    DEFAULT_CONTEXT_PROMPT: ClassVar[str] = (
        "## User Memory Context\n"
        "Consider the following context about the user and their history "
        "when answering. Treat it as background knowledge, not direct input."
    )

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: str,
        customer_id: str = "",
        conversation_id: Optional[str] = None,
        *,
        source_id: str = DEFAULT_SOURCE_ID,
        mode: str = "accurate",
        max_results: int = 20,
        context_prompt: Optional[str] = None,
        include_scope_labels: bool = False,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapContextProvider requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapContextProvider requires a non-empty user_id")

        super().__init__(source_id)
        self.sdk = sdk
        self.user_id = user_id
        self.customer_id = customer_id
        self.conversation_id = conversation_id
        self.mode = mode
        self.max_results = max_results
        self.context_prompt = context_prompt or self.DEFAULT_CONTEXT_PROMPT
        self.include_scope_labels = include_scope_labels

    async def before_run(
        self,
        *,
        agent: Any,
        session: Any,
        context: Any,
        state: dict[str, Any],
    ) -> None:
        """Fetch Synap context and append it to the agent's instructions."""
        input_text = self._concat_text(context.input_messages)
        if not input_text:
            return

        conv_id = self._resolve_conversation_id(context)

        try:
            response = await self.sdk.fetch(
                conversation_id=conv_id,
                user_id=self.user_id,
                customer_id=self.customer_id or None,
                search_query=[input_text],
                max_results=self.max_results,
                mode=self.mode,
                include_conversation_context=False,
                include_scope_labels=self.include_scope_labels,
            )
        except Exception as exc:  # noqa: BLE001 — read-side degrades gracefully
            logger.error(
                "SynapContextProvider.before_run: sdk.fetch failed "
                "user_id=%s conversation_id=%s error=%s",
                self.user_id,
                conv_id,
                exc,
                exc_info=True,
            )
            return

        formatted = (response.formatted_context or "").strip()
        if not formatted:
            return

        context.extend_instructions(
            self.source_id,
            f"{self.context_prompt}\n{formatted}",
        )

    async def after_run(
        self,
        *,
        agent: Any,
        session: Any,
        context: Any,
        state: dict[str, Any],
    ) -> None:
        """Record input + response messages to Synap conversation history."""
        conv_id = self._resolve_conversation_id(context)
        if not conv_id:
            # Without a conversation id we can't attribute the turn — skip.
            return

        to_record: list[tuple[str, str]] = []
        for message in context.input_messages or []:
            role, text = self._message_to_role_text(message)
            if role and text:
                to_record.append((role, text))

        response = context.response
        if response is not None and getattr(response, "messages", None):
            for message in response.messages:
                role, text = self._message_to_role_text(message)
                if role and text:
                    to_record.append((role, text))

        for role, text in to_record:
            try:
                await self.sdk.conversation.record_message(
                    conversation_id=conv_id,
                    role=role,
                    content=text,
                    user_id=self.user_id,
                    customer_id=self.customer_id,
                )
            except Exception as exc:  # noqa: BLE001 — must not crash the agent
                logger.error(
                    "SynapContextProvider.after_run: record_message failed "
                    "role=%s conversation_id=%s error=%s",
                    role,
                    conv_id,
                    exc,
                    exc_info=True,
                )

    # -- helpers --------------------------------------------------------------

    def _resolve_conversation_id(self, context: Any) -> Optional[str]:
        if self.conversation_id:
            return self.conversation_id
        sid = getattr(context, "session_id", None)
        return sid if sid else None

    @staticmethod
    def _concat_text(messages: Any) -> str:
        if not messages:
            return ""
        parts: list[str] = []
        for msg in messages:
            text = getattr(msg, "text", None)
            if text and text.strip():
                parts.append(text.strip())
        return "\n".join(parts)

    @staticmethod
    def _message_to_role_text(message: Any) -> tuple[Optional[str], str]:
        role_raw = getattr(message, "role", None)
        if role_raw is None:
            return None, ""
        role = role_raw.value if hasattr(role_raw, "value") else str(role_raw)
        if role not in {"user", "assistant", "system"}:
            return None, ""
        text = getattr(message, "text", "") or ""
        if not text.strip():
            return None, ""
        return role, text
