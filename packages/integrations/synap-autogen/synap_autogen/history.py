"""Put Synap's compaction where AutoGen reads its history.

An ``AssistantAgent`` builds every model call from ``model_context.get_messages()``
and from nothing else, so that is the one seam. A context that returns the
conversation as Synap holds it, the compaction block and the turns after it,
replaces whatever the framework would have sent for the turns before the current
one. ``BufferedChatCompletionContext`` and friends stay installed as the
fallback: with no compaction yet, or when Synap cannot answer, the list is
returned unchanged.
"""

from __future__ import annotations

from typing import Any, List, Optional

from autogen_core.models import (
    AssistantMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from synap_integrations_common import (
    CompactedHistory,
    fetch_compacted_history,
    splice_history,
)


def _role(message: LLMMessage) -> str:
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, UserMessage):
        return "user"
    if isinstance(message, AssistantMessage) and isinstance(message.content, str):
        return "assistant"
    return "tool"  # tool calls and their results: dropped with their turn


def _text(message: LLMMessage) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(message, UserMessage) and isinstance(content, list):
        return " ".join(part for part in content if isinstance(part, str))
    return ""


def splice_messages(
    messages: List[LLMMessage], history: CompactedHistory, agent_name: str = "assistant"
) -> List[LLMMessage]:
    """``messages`` with every turn before the current one replaced by ``history``."""

    def make(role: str, text: str) -> LLMMessage:
        if role == "system":
            return SystemMessage(content=text)
        if role == "assistant":
            return AssistantMessage(content=text, source=agent_name)
        return UserMessage(content=text, source="user")

    return splice_history(messages, history, role_of=_role, text_of=_text, make_message=make)


async def compacted_messages(
    sdk: Any,
    conversation_id: str,
    messages: List[LLMMessage],
    agent_name: str = "assistant",
    history: Optional[CompactedHistory] = None,
) -> List[LLMMessage]:
    """``messages`` as the model should see them: Synap's compaction spliced in
    once there is one, otherwise the framework's own list. Never raises."""
    if history is None:
        history = await fetch_compacted_history(sdk, conversation_id)
    if history is None:
        return list(messages)
    return splice_messages(list(messages), history, agent_name)
