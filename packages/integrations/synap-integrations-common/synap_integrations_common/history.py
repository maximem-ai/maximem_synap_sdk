"""Put Synap's compaction where the framework's own history would go.

Every framework keeps the conversation as a list of messages and has some way
to shrink it when it grows: a summarizer, a sliding window, a token limit. An
integration that only adds Synap context beside that list leaves the
framework's compaction in charge, and sends the recent turns twice: once in
the list and once inside Synap's block.

This module is the one place that decides what replaces what:

- :func:`fetch_compacted_history` asks the SDK for the conversation as Synap
  holds it: the compaction block (plus the earlier turns the SDK's local search
  found for the latest message) and the turns after the compaction.
- :func:`splice_history` takes the framework's message list and returns it with
  every turn before the current one replaced by that.

The framework's own compaction is left installed. It now sees a short history
and does not fire, and if Synap cannot answer it is still there.

Both functions are framework-agnostic: an integration passes small callables
that read a role and a text from its message type and build a new message.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar

logger = logging.getLogger(__name__)

M = TypeVar("M")

#: What the SDK's compaction block starts with.
COMPACTION_PREFIX = "## Conversation so far (compacted"
HISTORY_OPEN = "<synap_conversation_history>"
HISTORY_CLOSE = "</synap_conversation_history>"


@dataclass(frozen=True)
class CompactedHistory:
    """The conversation before the current turn, as Synap holds it."""

    #: The compaction block and any earlier turns history search found,
    #: wrapped in ``<synap_conversation_history>`` tags.
    preamble: str
    #: The turns after the compaction, oldest first, as ``(role, text)`` with
    #: role ``"user"`` or ``"assistant"``.
    turns: Tuple[Tuple[str, str], ...]
    compaction_id: Optional[str] = None


def is_synap_history(text: Any) -> bool:
    """Whether a message is the history this module put there, not a turn.

    An integration that writes messages back to Synap skips these, or the
    compaction would be recorded as something somebody said.
    """
    if not isinstance(text, str):
        return False
    stripped = text.lstrip()
    return stripped.startswith(HISTORY_OPEN) or stripped.startswith(COMPACTION_PREFIX)


async def fetch_compacted_history(
    sdk: Any,
    conversation_id: str,
) -> Optional[CompactedHistory]:
    """The compacted conversation, or None when there is nothing to replace.

    None when the conversation has no compaction yet (the framework's list is
    still short and is the better record), when the SDK predates compaction
    blocks, and on any error. Never raises.
    """
    if not conversation_id:
        return None
    try:
        ctx = await sdk.conversation.context.get_context_for_prompt(
            conversation_id=conversation_id,
        )
    except Exception:  # noqa: BLE001 — the framework's own history still stands
        logger.warning("synap: compacted history unavailable (conversation_id=%s)",
                       conversation_id, exc_info=True)
        return None
    context_block = getattr(ctx, "context_block", None)
    if not isinstance(context_block, str) or not context_block.strip():
        return None
    parts = [context_block.strip()]
    history_block = getattr(ctx, "history_block", None)
    if isinstance(history_block, str) and history_block.strip():
        parts.append(history_block.strip())
    turns: List[Tuple[str, str]] = []
    for rm in getattr(ctx, "recent_messages", None) or []:
        role = getattr(rm, "role", None)
        content = getattr(rm, "content", None)
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            turns.append((role, content))
    compaction_id = getattr(ctx, "compaction_id", None)
    return CompactedHistory(
        preamble=f"{HISTORY_OPEN}\n" + "\n\n".join(parts) + f"\n{HISTORY_CLOSE}",
        turns=tuple(turns),
        compaction_id=compaction_id if isinstance(compaction_id, str) else None,
    )


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split())


def splice_history(
    messages: Sequence[M],
    history: CompactedHistory,
    *,
    role_of: Callable[[M], str],
    text_of: Callable[[M], str],
    make_message: Callable[[str, str], M],
) -> List[M]:
    """Replace every turn before the current one with Synap's history.

    ``role_of`` returns ``"system"``, ``"user"``, ``"assistant"`` or anything
    else (a tool message, say). ``make_message(role, text)`` builds a message of
    the framework's type; it is called with ``"system"`` once, for the
    preamble, and with ``"user"`` or ``"assistant"`` for turns.

    The result, in order:

    1. the framework's own system messages from before the current turn;
    2. the preamble, as one system message;
    3. Synap's turns after the compaction;
    4. any earlier turns the framework has and Synap does not yet (a write
       still in flight), rebuilt as plain text;
    5. the current turn untouched: the last user message and everything after
       it, so a tool call and its result stay paired.

    Tool messages from earlier turns are dropped with the turns they belonged
    to: the compaction covers them, and history search finds their results.
    """
    current_start = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        if role_of(messages[i]) == "user":
            current_start = i
            break
    earlier, current = list(messages[:current_start]), list(messages[current_start:])

    systems = [m for m in earlier
               if role_of(m) == "system" and not is_synap_history(text_of(m))]
    framework_turns = [(role_of(m), text_of(m)) for m in earlier
                       if role_of(m) in ("user", "assistant") and _norm(text_of(m))]

    turns = list(history.turns)
    # The current user message is usually recorded before the model is called,
    # so Synap already has it, and maybe a reply from an earlier lap of this
    # turn. Both belong to the current turn, which the framework still holds.
    if current and role_of(current[0]) == "user":
        now = _norm(text_of(current[0]))
        for i in range(len(turns) - 1, -1, -1):
            if turns[i][0] == "user" and _norm(turns[i][1]) == now:
                del turns[i:]
                break

    # Turns the framework has after the last one Synap has.
    pending: List[Tuple[str, str]] = []
    if turns:
        last_role, last_text = turns[-1][0], _norm(turns[-1][1])
        for i in range(len(framework_turns) - 1, -1, -1):
            role, text = framework_turns[i]
            if role == last_role and _norm(text) == last_text:
                pending = framework_turns[i + 1:]
                break

    out: List[M] = list(systems)
    out.append(make_message("system", history.preamble))
    out.extend(make_message(role, text) for role, text in turns)
    out.extend(make_message(role, str(text)) for role, text in pending)
    out.extend(current)
    return out
