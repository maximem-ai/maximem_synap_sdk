"""Reading a LangChain message without caring which shape it came in."""

from typing import Any, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, ChatMessage, HumanMessage, SystemMessage

_USER_ROLES = ("user", "human")
_ASSISTANT_ROLES = ("assistant", "ai")


def message_text(message: Any) -> str:
    """The text of a message: a string, or every text block of a block list.

    ``.text`` is a property in langchain-core 1.x and was a method before; the
    1.x value is also callable, so it is tested for ``str`` first.
    """
    if message is None:
        return ""
    text = getattr(message, "text", None)
    if text is not None and not isinstance(text, str) and callable(text):
        try:
            text = text()
        except Exception:  # noqa: BLE001 — a wrapper, not a failure
            text = None
    if isinstance(text, str) and text.strip():
        return text
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""


def turn_role(message: BaseMessage) -> Optional[str]:
    """``"user"`` or ``"assistant"`` for a message that is a conversation turn.

    None for everything else: a system prompt, a tool result and a function
    message are not something the user or the assistant said.
    """
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, ChatMessage):
        role = str(message.role or "").lower()
        if role in _USER_ROLES:
            return "user"
        if role in _ASSISTANT_ROLES:
            return "assistant"
    return None


def role_of(message: BaseMessage) -> str:
    """The role :func:`synap_integrations_common.splice_history` reads."""
    if isinstance(message, SystemMessage):
        return "system"
    return turn_role(message) or "other"
