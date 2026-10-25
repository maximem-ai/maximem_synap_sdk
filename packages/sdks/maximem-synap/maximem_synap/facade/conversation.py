"""Conversation recording controller for SDK.

Provides methods for recording conversation messages to the cloud,
which enables automatic compaction when thresholds are exceeded.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from ..memories.parts import (
    FilePart,
    TextPart,
    has_files,
    inline_wire_parts,
    is_parts,
    join_text,
    resolve_files,
    to_parts,
)

if TYPE_CHECKING:
    from ..transport.http_client import HTTPTransport
    from ..auth.models import AuthContext

logger = logging.getLogger("synap.sdk.facade.conversation")


async def _wire_content(content: Any, *, what: str) -> Any:
    """`content` as it goes on the wire.

    Only parts (a list or a single part) are unpacked. A string, and
    anything else a caller passed before parts existed, goes out exactly as
    it did. A list holding only
    text becomes the newline-joined string, so it is that string's bytes; a
    list holding files becomes text and image parts with the bytes inline.
    """
    if not is_parts(content):
        return content
    parts = to_parts(content, what=what)
    if not has_files(parts):
        return join_text(parts)
    return inline_wire_parts(await resolve_files(parts))


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
        content: Union[str, List[Union[str, TextPart, FilePart]]],
        user_id: str,
        # Optional, to match ConversationInterface.record_message above it. A
        # B2C caller sends no customer_id, and a controller that still demands
        # one positionally is the same wall one layer down.
        customer_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        scope_path: Optional[Dict[str, str]] = None,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a single conversation message.

        Args:
            conversation_id: Unique conversation identifier
            role: Message role ("user" or "assistant")
            content: Message content text, or a list of strings, TextPart and
                FilePart. Text only is joined with newlines; files are read
                and sent inline (images only, 10 MB each).
            user_id: User identifier (required)
            customer_id: Customer identifier (required)
            session_id: Session identifier (optional, auto-generated if not provided)
            metadata: Additional metadata (optional)
            scope_path: The rung this conversation belongs to, named in full.
                Sent to the server as ``scope``. Only meaningful on a client
                with a ladder deeper than the default three rungs.
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

        content = await _wire_content(content, what="content")

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
        # Named `scope` on the wire and `scope_path` in the SDK, matching
        # `memories.create` and `ingest_transcript`, which made the same split.
        if scope_path:
            payload["scope"] = scope_path

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
                - content: str, or a list of parts as in ``record_message``
                  (required)
                - user_id: Optional[str]
                - customer_id: Optional[str]
                - session_id: Optional[str]
                - metadata: Optional[Dict]
                - scope_path: Optional[Dict[str, str]] — the rung, same as the
                  single-message method. Normalised to the wire name `scope`
                  below so a batch caller and a single caller spell it the
                  same way.
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

        # Copy rather than mutate: the caller's list is theirs, and the batch
        # mirror in sdk.py reads the same dicts afterwards.
        wire = []
        for i, msg in enumerate(messages):
            if "scope_path" in msg:
                m = {k: v for k, v in msg.items() if k != "scope_path"}
                if msg["scope_path"]:
                    m["scope"] = msg["scope_path"]
            else:
                m = msg
            if is_parts(msg["content"]):
                if m is msg:
                    m = dict(msg)
                m["content"] = await _wire_content(
                    msg["content"], what=f"messages[{i}].content"
                )
            wire.append(m)

        auth_context = await self._auth_provider(correlation_id)

        result = await self._transport.post(
            "/v1/conversations/messages/batch",
            auth_context=auth_context,
            json={"messages": wire},
            correlation_id=correlation_id,
        )

        return result
