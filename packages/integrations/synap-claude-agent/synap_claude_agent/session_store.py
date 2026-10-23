"""Resume a Claude Code session from Synap's compaction.

Claude Code owns its transcript and, when it resumes a session, replays it. The
SDK has one place where a caller can decide what that transcript is: with
``ClaudeAgentOptions(resume=..., session_store=...)`` the SDK calls
``session_store.load()`` before it starts the CLI, writes whatever comes back to
a temporary config directory, and the CLI resumes from that file (see
``claude_agent_sdk/_internal/session_resume.py``).

:class:`SynapSessionStore` answers that call with Synap's view once the
conversation has been compacted: the compaction, then the turns after it, in
place of every turn the compaction covers. Until then, and whenever Synap
cannot answer, it returns the transcript it mirrored unchanged.

This only decides what a *resumed* session starts from. Inside a running
session Claude Code keeps appending to its own transcript and compacts it on its
own terms; nothing in the SDK lets a caller replace that (see
:func:`synap_claude_agent.create_synap_hooks` for what a hook can do instead).
The transcript is Claude Code's internal format, which the SDK documents as
opaque: this builds the plain user and assistant entries a resume needs and
nothing else, and falls back to the mirrored transcript if it cannot.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any, List, Optional, Tuple

from claude_agent_sdk.types import (
    SessionKey,
    SessionListSubkeysKey,
    SessionStoreEntry,
    SessionStoreListEntry,
    SessionSummaryEntry,
)
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import CompactedHistory, fetch_compacted_history

logger = logging.getLogger(__name__)


class SynapSessionStore:
    """A ``SessionStore`` that resumes from Synap's compaction when it has one.

    Every method but :meth:`load` passes straight through to ``inner``, which
    keeps the raw transcript so a conversation Synap has not compacted resumes
    exactly as it would without this store.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        *,
        conversation_id: Optional[str] = None,
        inner: Any = None,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapSessionStore requires a non-None sdk")
        if inner is None:
            from claude_agent_sdk._internal.session_store import InMemorySessionStore

            inner = InMemorySessionStore()
        self._sdk = sdk
        self._conversation_id = conversation_id
        self._inner = inner

    async def append(self, key: SessionKey, entries: List[SessionStoreEntry]) -> None:
        await self._inner.append(key, entries)

    async def load(self, key: SessionKey) -> Optional[List[SessionStoreEntry]]:
        entries = await self._inner.load(key)
        if key.get("subpath"):
            return entries
        try:
            history = await fetch_compacted_history(
                self._sdk, self._conversation_id or key["session_id"])
            if history is None:
                return entries
            return _transcript(key["session_id"], history, entries or [])
        except Exception:  # noqa: BLE001 — a load that raises fails the whole resume
            logger.warning("synap: resume from Synap's history failed; using the mirrored "
                           "transcript", exc_info=True)
            return entries

    async def list_sessions(self, project_key: str) -> List[SessionStoreListEntry]:
        return await self._pass("list_sessions", project_key)

    async def list_session_summaries(self, project_key: str) -> List[SessionSummaryEntry]:
        return await self._pass("list_session_summaries", project_key)

    async def delete(self, key: SessionKey) -> None:
        await self._pass("delete", key)

    async def list_subkeys(self, key: SessionListSubkeysKey) -> List[str]:
        return await self._pass("list_subkeys", key)

    async def _pass(self, method: str, arg: Any) -> Any:
        # An optional method the inner store lacks is absent here too.
        impl = getattr(self._inner, method, None)
        if impl is None:
            raise NotImplementedError(method)
        return await impl(arg)


def create_synap_session_store(
    sdk: MaximemSynapSDK,
    *,
    conversation_id: Optional[str] = None,
    inner: Any = None,
) -> SynapSessionStore:
    """``ClaudeAgentOptions(session_store=...)`` that resumes from Synap's compaction.

    ``conversation_id`` is the Synap conversation; leave it unset to use the
    Claude Code session id, as :func:`create_synap_hooks` does. ``inner`` is
    where the raw transcript is mirrored (default: in memory, lost on exit; pass
    a durable ``SessionStore`` to keep it).
    """
    return SynapSessionStore(sdk, conversation_id=conversation_id, inner=inner)


def _transcript(
    session_id: str, history: CompactedHistory, mirrored: List[Any],
) -> List[SessionStoreEntry]:
    """Plain user and assistant entries: the compaction, then the turns after it.

    The compaction rides on the first user turn (or stands alone as one), so no
    turn is invented to hold it.
    """
    turns: List[Tuple[str, str]] = list(history.turns)
    if turns and turns[0][0] == "user":
        turns[0] = ("user", f"{history.preamble}\n\n{turns[0][1]}")
    else:
        turns.insert(0, ("user", history.preamble))

    shared = {}
    for entry in mirrored:
        if isinstance(entry, dict) and entry.get("type") == "user":
            shared = {k: entry[k] for k in ("cwd", "version", "userType", "entrypoint", "gitBranch")
                      if k in entry}
            break

    now = datetime.datetime.now(datetime.timezone.utc)
    out: List[SessionStoreEntry] = []
    parent: Optional[str] = None
    for i, (role, text) in enumerate(turns):
        entry_id = str(uuid.uuid4())
        message: dict = {"role": role, "content": text}
        if role == "assistant":
            message = {
                "id": f"msg_{entry_id[:12]}", "type": "message", "role": "assistant",
                "model": "synap", "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        out.append({
            "parentUuid": parent, "isSidechain": False, "type": role, "uuid": entry_id,
            "timestamp": (now + datetime.timedelta(milliseconds=i)).isoformat(
                timespec="milliseconds").replace("+00:00", "Z"),
            "sessionId": session_id, "message": message, **shared,
        })
        parent = entry_id
    return out


__all__ = ["SynapSessionStore", "create_synap_session_store"]
