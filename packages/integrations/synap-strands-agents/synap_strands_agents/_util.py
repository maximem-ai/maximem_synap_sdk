"""Internal helpers shared across surfaces. Not part of the public API."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from synap_integrations_common import is_synap_history

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
_written: Dict[Tuple[str, str, str], int] = {}
_LEDGER_LIMIT = 4096


def note_written(user_id: str, role: str, text: str) -> None:
    if len(_written) >= _LEDGER_LIMIT:
        _written.clear()  # a hook with no store would otherwise grow this forever
    key = (user_id, role, text)
    _written[key] = _written.get(key, 0) + 1


def consume_written(user_id: str, role: str, text: str) -> bool:
    """Whether this turn was already written; each write is consumed once."""
    key = (user_id, role, text)
    count = _written.get(key, 0)
    if not count:
        return False
    if count == 1:
        del _written[key]
    else:
        _written[key] = count - 1
    return True
