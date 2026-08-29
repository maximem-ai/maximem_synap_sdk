"""Conversation recording controller for SDK.

Provides methods for recording conversation messages to the cloud,
which enables automatic compaction when thresholds are exceeded.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..transport.http_client import HTTPTransport
    from ..auth.models import AuthContext

logger = logging.getLogger("synap.sdk.facade.conversation")


class ConversationController:
    """Controller for conversation recording operations."""

    def __init__(self, transport: "HTTPTransport", auth_provider):
        """Initialize conversation controller.

        Args:
            transport: HTTP transport for API calls
            auth_provider: Async callable returning AuthContext
        """
        self._transport = transport
        self._auth_provider = auth_provider

    def _ensure_transport(self):
        if not self._transport:
            raise RuntimeError("Transport not initialized. Call SDK.initialize() first.")

    async def record_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        user_id: str,
        # Optional, to match ConversationInterface.record_message above it. A
        # B2C caller sends no customer_id, and a controller that still demands
        # one positionally is the same wall one layer down.
        customer_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a single conversation message.

        Args:
            conversation_id: Unique conversation identifier
            role: Message role ("user" or "assistant")
            content: Message content text
            user_id: User identifier (required)
            customer_id: Customer identifier (required)
            session_id: Session identifier (optional, auto-generated if not provided)
            metadata: Additional metadata (optional)
            correlation_id: Optional correlation ID for tracing

        Returns:
            Dict with message_id, conversation_id, session_id, recorded_at

        Raises:
            RuntimeError: If transport not initialized
            ValueError: If role is invalid
        """
        self._ensure_transport()

        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role '{role}'. Must be 'user' or 'assistant'.")

        auth_context = await self._auth_provider(correlation_id)

        payload = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "user_id": user_id,
            "customer_id": customer_id,
            "metadata": metadata or {},
        }
        if session_id:
            payload["session_id"] = session_id

        result = await self._transport.post(
            "/v1/conversations/messages",
            auth_context=auth_context,
            json=payload,
            correlation_id=correlation_id,
        )

        return result

    async def record_messages_batch(
        self,
        messages: List[Dict[str, Any]],
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record multiple conversation messages in a batch.

        Args:
            messages: List of message dicts, each with:
                - conversation_id: str (required)
                - role: str ("user" or "assistant") (required)
                - content: str (required)
                - user_id: Optional[str]
                - customer_id: Optional[str]
                - session_id: Optional[str]
                - metadata: Optional[Dict]
            correlation_id: Optional correlation ID for tracing

        Returns:
            Dict with total, succeeded, failed, results[]

        Raises:
            RuntimeError: If transport not initialized
            ValueError: If any message is missing required fields
        """
        self._ensure_transport()

        for i, msg in enumerate(messages):
            if "conversation_id" not in msg:
                raise ValueError(f"Message {i} missing required field 'conversation_id'")
            if "role" not in msg:
                raise ValueError(f"Message {i} missing required field 'role'")
            if "content" not in msg:
                raise ValueError(f"Message {i} missing required field 'content'")
            if msg["role"] not in ("user", "assistant"):
                raise ValueError(f"Message {i} has invalid role '{msg['role']}'")

        auth_context = await self._auth_provider(correlation_id)

        result = await self._transport.post(
            "/v1/conversations/messages/batch",
            auth_context=auth_context,
            json={"messages": messages},
            correlation_id=correlation_id,
        )

        return result
