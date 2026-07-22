"""Internal helpers shared across surfaces. Not part of the public API."""

from __future__ import annotations

from typing import Any


def message_text(message: Any) -> str:
    """Join the text blocks of a Strands ``Message`` into one string.

    A Strands ``Message`` is ``{"role": str, "content": list[ContentBlock]}``.
    Only ``{"text": ...}`` blocks contribute; ``toolUse`` / ``toolResult`` /
    image blocks yield no text. Callers treat an empty result as "skip this
    message" (e.g. tool-result turns carry no natural-language text).
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
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts).strip()
