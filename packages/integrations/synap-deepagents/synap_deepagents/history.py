"""Synap's compaction in place of a deepagents run's earlier turns.

deepagents keeps the conversation in ``state["messages"]`` and shrinks it with
its own ``SummarizationMiddleware``. This module is what the middleware in this
package call on ``wrap_model_call``, the hook that rewrites what the model is
sent without rewriting what the graph stores: once Synap has compacted the
conversation, every turn before the current one is replaced by Synap's
compaction and the turns after it. Until then, and whenever Synap cannot
answer, the request goes through untouched.

The compaction goes into the request's system message, not the message list: a
provider that wants one system message first (Anthropic) rejects the other
shape.

**Where this sits.** ``create_deep_agent`` puts its summarizer in the base stack
and appends caller middleware after it (``_apply_custom_middleware`` in
``deepagents/graph.py``), and the first middleware in the list wraps the rest.
So the summarizer sees the request first and this runs inside it. The splice
still decides what the model is sent; what it cannot do is stop the summarizer
from spending a model call on a long history first. Nothing a caller can pass to
``create_deep_agent`` runs ahead of it.
"""

from __future__ import annotations

import logging
from typing import Any, List, Literal, Optional

from deepagents.middleware._utils import append_to_system_message
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    CompactedHistory,
    SynapIntegrationError,
    splice_history,
    wrap_sdk_errors_async,
)
from synap_integrations_common.history import COMPACTION_PREFIX, HISTORY_CLOSE, HISTORY_OPEN

logger = logging.getLogger(__name__)

_OnError = Literal["fallback", "raise"]
_ROLES = {"human": "user", "user": "user", "ai": "assistant", "assistant": "assistant",
          "system": "system"}


def history_from_context(response: Any) -> Optional[CompactedHistory]:
    """Synap's compaction of the conversation out of an SDK response, or None."""
    if getattr(response, "available", True) is False:
        return None
    compaction = getattr(response, "context_block", None)
    if not isinstance(compaction, str) or not compaction.strip():
        return None  # not compacted yet: the run's own messages are the record
    parts = [compaction.strip()]
    found = getattr(response, "history_block", None)
    if isinstance(found, str) and found.strip():
        parts.append(found.strip())
    turns = []
    for rm in getattr(response, "recent_messages", None) or []:
        role, content = getattr(rm, "role", None), getattr(rm, "content", None)
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            turns.append((role, content))
    return CompactedHistory(
        preamble=f"{HISTORY_OPEN}\n" + "\n\n".join(parts) + f"\n{HISTORY_CLOSE}",
        turns=tuple(turns),
    )


async def fetch_history(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    *,
    style: str = "narrative",
    on_error: _OnError = "fallback",
    site: str = "synap_deepagents.fetch_history",
) -> Optional[CompactedHistory]:
    """The compacted conversation, or None when there is nothing to replace.

    None also on an SDK failure under ``on_error="fallback"``; under ``"raise"``
    the failure propagates as :class:`SynapIntegrationError`.
    """
    try:
        async with wrap_sdk_errors_async(
            site, logger, conversation_id=conversation_id, style=style,
        ):
            response = await sdk.conversation.context.get_context_for_prompt(
                conversation_id=conversation_id,
                style=style,
            )
    except SynapIntegrationError:
        if on_error == "raise":
            raise
        return None
    return history_from_context(response)


def inner(preamble: str) -> str:
    """The compaction text without the tags that mark it as Synap's."""
    text = preamble.strip()
    if text.startswith(HISTORY_OPEN) and text.endswith(HISTORY_CLOSE):
        text = text[len(HISTORY_OPEN):-len(HISTORY_CLOSE)]
    return text.strip()


def _text(message: Any) -> str:
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
        return "".join(
            b if isinstance(b, str) else str(b.get("text", ""))
            for b in content if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text")
        )
    return ""


def _role(message: Any) -> str:
    kind = getattr(message, "type", None)
    if kind is None and isinstance(message, dict):
        kind = message.get("type") or message.get("role")
    return _ROLES.get(str(kind or ""), "other")


def _make(role: str, text: str) -> Any:
    if role == "system":
        return SystemMessage(content=text)
    return AIMessage(content=text) if role == "assistant" else HumanMessage(content=text)


def is_spliced(request: Any) -> bool:
    """Whether a layer outside this one already put Synap's history in the request."""
    system = getattr(request, "system_message", None)
    text = _text(system) if system is not None else ""
    return HISTORY_OPEN in text or COMPACTION_PREFIX in text


def splice_request(request: Any, history: CompactedHistory, block: Optional[str] = None) -> Any:
    """``request`` with the turns before the current one replaced by Synap's.

    ``block`` is the text for the system message; it defaults to the tagged
    compaction. The request comes back unchanged if it holds no user message to
    anchor the current turn on, or if the splice fails.
    """
    messages: List[Any] = list(request.messages)
    if not any(_role(m) == "user" for m in messages):
        return request
    try:
        spliced = splice_history(messages, history, role_of=_role, text_of=_text, make_message=_make)
    except Exception:  # noqa: BLE001 — the run's own messages still stand
        logger.warning("synap: could not splice the compacted history", exc_info=True)
        return request
    spliced = [m for m in spliced
               if not (_role(m) == "system" and _text(m) == history.preamble)]
    system = append_to_system_message(request.system_message, block or history.preamble)
    return request.override(messages=spliced, system_message=system)
