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
