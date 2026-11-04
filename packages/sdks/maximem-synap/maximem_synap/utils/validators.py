"""Client-side validation for identifier formats.

These helpers raise the *specific* ``InvalidInputError`` subtypes
(``InvalidConversationIdError`` / ``InvalidInstanceIdError``) so callers can
catch them directly (e.g. ``except InvalidConversationIdError``) instead of the
generic parent. Validation happens before any network call, so malformed ids
fail fast.
"""

import re
import uuid as _uuid
from typing import Optional

from ..models.errors import InvalidConversationIdError, InvalidInstanceIdError

# Instance ids are issued as ``inst_`` followed by 16 hex characters.
_INSTANCE_ID_RE = re.compile(r"^inst_[0-9a-fA-F]{16}$")


def normalize_conversation_id(conversation_id):
    """The one id every local cache is keyed by for a conversation.

    The stream and the server accept any string as a conversation id and turn
    one that is not a UUID into ``uuid5(NAMESPACE_URL, id)`` (the server's
    ``cloud/shared/conversation_id_norm.coerce_conversation_id``). The REST
    reads need that UUID, so a caller that sends a turn on the stream under
    ``pgv2_abc`` and reads under the UUID it derived is talking about ONE
    conversation. A cache that keys the two differently drops the entry under
    one name and serves the other for its whole lifetime.

    A string that parses as a UUID, in any spelling, is returned exactly as it
    came (so a real UUID keys what it always keyed). Empty and ``None`` pass
    through. Idempotent. Must stay byte-identical to the server's function
    (``tests/test_conversation_id_normalisation.py`` compares them).
    """
    if not conversation_id:
        return conversation_id
    text = conversation_id if isinstance(conversation_id, str) else str(conversation_id)
    try:
        _uuid.UUID(text)
        return text
    except (ValueError, AttributeError, TypeError):
        return str(_uuid.uuid5(_uuid.NAMESPACE_URL, text))


def validate_conversation_id(conversation_id: Optional[str]) -> None:
    """Raise ``InvalidConversationIdError`` if a non-empty ``conversation_id``
    is not a valid UUID string.

    Empty/``None`` values are left untouched (callers that require a
    conversation id surface that separately), so this only rejects clearly
    malformed values such as ``"conv_123"``.
    """
    if not conversation_id:
        return
    try:
        _uuid.UUID(str(conversation_id))
    except (ValueError, AttributeError, TypeError):
        raise InvalidConversationIdError(conversation_id) from None


def validate_instance_id(instance_id: Optional[str]) -> None:
    """Raise ``InvalidInstanceIdError`` if a non-empty ``instance_id`` is not in
    the ``inst_<hex16>`` format.

    Empty/``None`` values are left untouched (the instance id is normally
    resolved from the API key at ``initialize()``), so this only rejects a
    malformed id that was explicitly provided.
    """
    if not instance_id:
        return
    if not _INSTANCE_ID_RE.match(str(instance_id)):
        raise InvalidInstanceIdError(instance_id)
