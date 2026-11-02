"""Conversation ids: any string for a developer, a UUID for the SDK's REST calls.

Developers name conversations as they like (``pgv2_<hex>``, ``chat-42``, a
thread id). The gRPC stream accepts any string and the server derives the UUID
it files the conversation under. The SDK's REST-backed calls
(``conversation.record_message``, ``conversation.context.get_context_for_prompt``,
the ``conversation_id`` of ``fetch``) refuse anything that is not a UUID with
``InvalidConversationIdError``.

:func:`coerce_conversation_id` is that same derivation, so a REST call made with
the derived UUID reads and writes the conversation the stream wrote under the
caller's own id: a UUID is returned unchanged, any other string becomes
``uuid5(NAMESPACE_URL, value)`` (the server's ``coerce_conversation_id``).
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


def coerce_conversation_id(value: str) -> str:
    """``value`` if it is a UUID, else the UUID the server derives from it.

    Idempotent: ``coerce(coerce(x)) == coerce(x)``. An empty or ``None`` value
    is returned as is (callers decide what a missing id means).
    """
    if not value:
        return value
    try:
        uuid.UUID(str(value))
        return value
    except (ValueError, AttributeError, TypeError):
        derived = str(uuid.uuid5(uuid.NAMESPACE_URL, str(value)))
        logger.debug("synap: conversation id %r is not a UUID; REST calls use %s", value, derived)
        return derived


def rest_conversation_id(value: Optional[str]) -> str:
    """The id to send on a REST-backed SDK call (``""`` for none)."""
    return coerce_conversation_id(value) if value else ""


__all__ = ["coerce_conversation_id", "rest_conversation_id"]
