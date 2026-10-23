"""Put Synap's compaction where Agno reads its history.

Agno builds each run's history in one place: ``AgentSession.get_messages`` over
the runs it holds. Its own compaction (``Agent(compaction=...)``) shortens what
that returns; nothing else sits between the stored runs and the model.

``create_synap_hooks``' pre-hook runs after Agno has read the session and before
it builds the run's messages, and is handed that same session. It asks Synap for
the conversation as Synap holds it and, when a compaction exists, wraps *that
session object's* ``get_messages`` so the history Agno reads is Synap's: the
compaction block, the turns after it, and nothing from before. The stored runs
are never touched, so what Agno persists is unchanged, and the wrapper is taken
off in the post-hook.

When Synap has no compaction yet, or cannot answer, nothing is wrapped and
Agno's history (and its own compaction, if configured) stands.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from agno.models.message import Message
from synap_integrations_common import (
    CompactedHistory,
    fetch_compacted_history,
    splice_history,
)

logger = logging.getLogger(__name__)

#: Set on the run's ``run_context`` once Synap's history is in the session, so
#: ``synap_st_instructions`` does not add the same compaction block again.
SPLICED_MARKER = "_synap_history_spliced"


def _role(message: Any) -> str:
    return str(getattr(message, "role", "") or "")


def _text(message: Any) -> str:
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""


def _make_message(role: str, text: str) -> Message:
    # Agno asks for history without system messages (it skips its own
    # ``system_message_role`` when reading a session, and sends its own system
    # prompt first), so a system-role preamble would be dropped before the
    # model saw it. It goes in as a user message, which is what survives.
    return Message(role="user" if role == "system" else role, content=text)


def _splice(messages: List[Message], history: CompactedHistory, current: str) -> List[Message]:
    # Agno's history holds the turns before this run; the run's own user message
    # is added after it. splice_history keeps "the current turn" untouched, so
    # it is handed a stand-in for it and the stand-in is dropped again.
    stand_in = Message(role="user", content=current)
    out = splice_history(
        [*messages, stand_in],
        history,
        role_of=_role,
        text_of=_text,
        make_message=_make_message,
    )
    return out[:-1]


def install_history(session: Any, history: CompactedHistory, current_input: str) -> None:
    """Make ``session.get_messages`` return Synap's history for this run."""
    remove_history(session)
    original: Callable[..., List[Message]] = session.get_messages

    def get_messages(*args: Any, **kwargs: Any) -> List[Message]:
        framework = original(*args, **kwargs)
        spliced = _splice(framework, history, current_input)
        skip_roles = kwargs.get("skip_roles")
        if skip_roles:
            spliced = [m for m in spliced if _role(m) not in skip_roles]
        return spliced

    session.get_messages = get_messages


def remove_history(session: Any) -> None:
    """Take the wrapper off a session. Safe on one that never had it."""
    try:
        vars(session).pop("get_messages", None)
    except TypeError:  # no instance dict: nothing was installed
        pass


async def prepare_history(
    sdk: Any,
    session: Any,
    run_context: Any,
    conversation_id: str,
    current_input: str,
) -> None:
    """Splice Synap's compaction into ``session`` for this run, if it has one.

    Never raises: a run that cannot be spliced runs on Agno's own history.
    """
    if session is None or not callable(getattr(session, "get_messages", None)):
        return
    try:
        history: Optional[CompactedHistory] = await fetch_compacted_history(sdk, conversation_id)
        if history is None:
            remove_history(session)
            return
        install_history(session, history, current_input)
        if run_context is not None:
            setattr(run_context, SPLICED_MARKER, True)
    except Exception:  # noqa: BLE001: telemetry never breaks the run
        logger.debug("synap history splice failed", exc_info=True)
