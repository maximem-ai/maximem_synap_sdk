"""Which conversation turns have already been written to Synap.

The exporter records a run's user and assistant turns. A NAT ``auto_memory_agent``
also hands the same turns to the memory editor, as ``MemoryItem(conversation=...)``.
Synap extracts from the conversation it keeps, so writing them again as documents
extracts them twice. The exporter notes each turn it wrote; the editor skips one
turn per note. Counted, not a set: two "ok" turns are two writes.
"""

from __future__ import annotations

from typing import Dict, Tuple

_written: Dict[Tuple[str, str, str], int] = {}
_LIMIT = 4096


def note_written(user_id: str, role: str, text: str) -> None:
    if len(_written) >= _LIMIT:
        _written.clear()  # an exporter with no editor would otherwise grow this forever
    key = (user_id, role, text.strip())
    _written[key] = _written.get(key, 0) + 1


def consume_written(user_id: str, role: str, text: str) -> bool:
    """Whether this turn was already written; each write is consumed once."""
    key = (user_id, role, text.strip())
    count = _written.get(key, 0)
    if not count:
        return False
    if count == 1:
        del _written[key]
    else:
        _written[key] = count - 1
    return True
