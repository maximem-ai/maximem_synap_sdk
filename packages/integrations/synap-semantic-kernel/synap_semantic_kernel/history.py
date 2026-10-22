"""Put Synap's compaction where Semantic Kernel's chat history goes.

Semantic Kernel's answer to a long chat is a :class:`ChatHistoryReducer`: a
``ChatHistory`` with a ``reduce()`` that truncates or summarizes it in place.
:class:`SynapChatHistoryReducer` is one whose ``reduce()`` asks Synap for the
conversation as Synap holds it and, once a compaction exists, replaces every
turn before the current one with the compaction block and the turns after it
(:func:`synap_integrations_common.splice_history`). With no compaction yet, or
on any error, it changes nothing: the history stays as the framework has it.

A reducer only runs when somebody calls ``reduce()``. Semantic Kernel's agents
never do: ``ChatCompletionAgent`` builds the model's input from
``thread.get_messages()`` and does not reduce, and ``ChatHistoryAgentThread``
appends with the synchronous ``add_message``, which skips ``auto_reduce``.
:class:`~synap_semantic_kernel.SynapAgentThread` therefore calls ``reduce()``
itself, at the top of ``get_messages()``, which is the one place every agent
call reads the history.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from pydantic import Field
from semantic_kernel.contents import AuthorRole, ChatMessageContent, TextContent
from semantic_kernel.contents.history_reducer.chat_history_reducer import ChatHistoryReducer

from synap_integrations_common import fetch_compacted_history, splice_history

logger = logging.getLogger(__name__)

_SYSTEM_ROLES = (AuthorRole.SYSTEM, AuthorRole.DEVELOPER)


def _role(message: ChatMessageContent) -> str:
    if message.role in _SYSTEM_ROLES:
        return "system"
    if message.role == AuthorRole.USER:
        return "user"
    if message.role == AuthorRole.ASSISTANT:
        return "assistant"
    return "other"


def _text(message: ChatMessageContent) -> str:
    """Every text block, not just the first (``.content`` drops the rest)."""
    parts = [item.text for item in message.items if isinstance(item, TextContent) and item.text]
    return "".join(parts) if parts else (message.content or "")


def _make(role: str, text: str) -> ChatMessageContent:
    author = {"system": AuthorRole.SYSTEM, "assistant": AuthorRole.ASSISTANT}.get(role, AuthorRole.USER)
    return ChatMessageContent(role=author, content=text)


async def splice_messages(
    sdk: Any, conversation_id: str, messages: Sequence[ChatMessageContent],
) -> Optional[List[ChatMessageContent]]:
    """The messages with Synap's compaction in place of the earlier turns, or
    None when Synap has nothing to replace them with. Never raises."""
    history = await fetch_compacted_history(sdk, conversation_id)
    if history is None:
        return None
    try:
        return splice_history(
            messages, history, role_of=_role, text_of=_text, make_message=_make,
        )
    except Exception:  # noqa: BLE001 — the framework's own history still stands
        logger.warning("synap semantic-kernel: splice failed (conversation_id=%s)",
                       conversation_id, exc_info=True)
        return None


class SynapChatHistoryReducer(ChatHistoryReducer):
    """A ``ChatHistory`` that swaps in Synap's compaction when reduced.

    Use it anywhere Semantic Kernel takes a ``ChatHistory``; call
    ``await history.reduce()`` before a model call. :class:`SynapAgentThread`
    does that for you.

    ``reduce()`` returns the history when it changed it and ``None`` when it
    did not, as Semantic Kernel's own reducers do.
    """

    sdk: Any = Field(exclude=True)
    conversation_id: str
    target_count: int = Field(default=1, gt=0)

    async def reduce(self) -> Optional["SynapChatHistoryReducer"]:
        spliced = await splice_messages(self.sdk, self.conversation_id, self.messages)
        if spliced is None:
            return None
        self.messages = spliced
        return self


__all__ = ["SynapChatHistoryReducer", "splice_messages"]
