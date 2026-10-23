"""Put Synap's compaction where MAF builds the model's messages.

MAF keeps a conversation in whatever ``HistoryProvider`` the agent has
(``InMemoryHistoryProvider``, Redis, Cosmos...) and sends all of it to the
model on every call, shrinking it with a ``CompactionStrategy`` when it grows.
Synap compacts the same conversation on the server. Left alone the two do not
meet: the model sees MAF's turns, and Synap's compaction is only ever added
beside them.

:class:`SynapCompactionMiddleware` is a ``ChatMiddleware``, which runs just
before each model call with the final message list (history from every
provider, the new input, and the tool exchanges of the turn in flight). Once
Synap has compacted the conversation it replaces every turn before the current
one with the compaction and the turns after it. Before that, and whenever
Synap cannot answer, the list is untouched. MAF's own compaction stays
installed: it sees a short history afterwards and has nothing to do, and it is
what runs when Synap has nothing to give.

``SynapContextProvider`` adds this middleware to every run by default. It is a
chat middleware rather than a ``before_run`` rewrite because ``before_run``
runs in provider order, so a history provider listed after the Synap provider
would not have loaded its messages yet; the middleware sees them all.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from agent_framework import ChatMiddleware, Message
from synap_integrations_common import (
    COMPACTION_PREFIX,
    fetch_compacted_history,
    splice_history,
)

logger = logging.getLogger(__name__)


def message_role(message: Any) -> str:
    role = getattr(message, "role", None)
    return str(getattr(role, "value", role) or "")


class SynapCompactionMiddleware(ChatMiddleware):
    """Send the model Synap's compaction in place of the earlier turns.

    ``conversation_id`` is a callable so it can follow the session: it is called
    with no arguments on every model call and may return ``None``.
    """

    def __init__(self, sdk: Any, conversation_id: Callable[[], Optional[str]]) -> None:
        self.sdk = sdk
        self._conversation_id = conversation_id

    async def process(self, context: Any, call_next: Any) -> None:
        try:
            conversation_id = self._conversation_id()
            history = (await fetch_compacted_history(self.sdk, conversation_id)
                       if conversation_id else None)
            if history is not None:
                spliced = splice_history(
                    list(context.messages), history,
                    role_of=message_role,
                    text_of=lambda m: getattr(m, "text", "") or "",
                    make_message=lambda role, text: Message(role=role, contents=[text]),
                )
                # `SynapShortTermContextProvider` already put the compaction in
                # the instructions: do not send it twice.
                instructions = (getattr(context, "options", None) or {}).get("instructions")
                if isinstance(instructions, str) and COMPACTION_PREFIX in instructions:
                    spliced = [m for m in spliced if not (
                        message_role(m) == "system" and getattr(m, "text", "") == history.preamble)]
                context.messages = spliced
        except Exception:  # noqa: BLE001 — the framework's own history still stands
            logger.warning("synap: could not apply Synap's compaction to the model input",
                           exc_info=True)
        await call_next()


__all__ = ["SynapCompactionMiddleware"]
