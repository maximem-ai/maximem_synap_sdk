"""Internal helpers shared across surfaces. Not part of the public API."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from synap_integrations_common import is_synap_history

logger = logging.getLogger(__name__)

#: What :class:`SynapShortTermHook` wraps the block it folds into a user message in.
ST_OPEN = "<synap_short_term_context>"


def is_synap_block(text: Any) -> bool:
    """Whether a text block is Synap's own context, not something somebody said.

    A turn recorded with the block in it would be compacted by Synap and
    extracted from, and Synap would be reading its own output back as the user.
    """
    return isinstance(text, str) and (
        text.lstrip().startswith(ST_OPEN) or is_synap_history(text)
    )


def message_text(message: Any) -> str:
    """Join the text blocks of a Strands ``Message`` into one string.

    A Strands ``Message`` is ``{"role": str, "content": list[ContentBlock]}``.
    Only ``{"text": ...}`` blocks contribute; ``toolUse`` / ``toolResult`` /
    image blocks yield no text. Callers treat an empty result as "skip this
    message" (e.g. tool-result turns carry no natural-language text). Blocks
    that are Synap's own context (:func:`is_synap_block`) do not contribute
    either.
    """
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not content:
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str) and text and not is_synap_block(text):
                parts.append(text)
    return "\n".join(parts).strip()


# Turns the stream hook has written that the memory store has not yet been
# handed. Strands' MemoryManager sends the same turns to ``add_messages`` for
# extraction; Synap already has them, and ingesting them twice extracts them
# twice. Counted, not a set: two "ok" turns are two writes.
#
# A write is the same write only if it went to the same place: the key holds
# the user, the conversation and the scope path (canonical form: sorted items)
# as well as the role and text. Keyed on text alone, a short reply such as "ok"
# written by one store was taken for the copy of another store's, in another
# conversation or at another level, and that second write never happened.
# It stays a module-level ledger because the hook and the store are separate
# objects that share no state but this; narrowing it further would need a new
# argument on one of them.
_written: Dict[Tuple[Any, ...], int] = {}
_LEDGER_LIMIT = 4096


def _key(user_id: str, conversation_id: str, scope_path: Any, role: str, text: str) -> Tuple[Any, ...]:
    path = tuple(sorted((str(k), str(v)) for k, v in scope_path.items())) if scope_path else ()
    return (user_id, conversation_id or "", path, role, text)


def note_written(user_id: str, conversation_id: str, scope_path: Any, role: str, text: str) -> None:
    if len(_written) >= _LEDGER_LIMIT:
        _written.clear()  # a hook with no store would otherwise grow this forever
    key = _key(user_id, conversation_id, scope_path, role, text)
    _written[key] = _written.get(key, 0) + 1


def consume_written(user_id: str, conversation_id: str, scope_path: Any, role: str, text: str) -> bool:
    """Whether this turn was already written (same user, conversation, path,
    role and text); each write is consumed once."""
    key = _key(user_id, conversation_id, scope_path, role, text)
    count = _written.get(key, 0)
    if not count:
        return False
    if count == 1:
        del _written[key]
    else:
        _written[key] = count - 1
    return True


# A store with no ``conversation_id`` writes its turns under its own fresh id,
# and the ledger above only matches writes to the same conversation, so next to
# a SynapStreamHook every turn is written twice, once per conversation. We
# cannot fix that for the caller (which conversation is "the" one is theirs to
# say), so say it once, loudly, when both exist.
_hook_active = False
_warned_unshared = False


def mark_hook_active() -> None:
    """Called when a :class:`SynapStreamHook` is built in this process."""
    global _hook_active
    _hook_active = True


def warn_if_unshared_conversation(conversation_id_given: bool) -> None:
    """Warn once per process: a stream hook exists and a store got no ``conversation_id``."""
    global _warned_unshared
    if conversation_id_given or _warned_unshared or not _hook_active:
        return
    _warned_unshared = True
    logger.warning(
        "SynapMemoryStore was created without a conversation_id while a SynapStreamHook "
        "is active in this process. The store writes its turns under its own conversation, "
        "so every turn is recorded twice (once by the hook, once by the store). Pass the "
        "hook's conversation_id to SynapMemoryStore(conversation_id=...) so the two share "
        "one conversation and each turn is written once."
    )
