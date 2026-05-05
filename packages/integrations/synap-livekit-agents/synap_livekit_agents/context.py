"""Preload Synap context into a LiveKit ``ChatContext`` before a call starts.

:func:`preload_synap_context` is an async helper: call it BEFORE
``session.start(agent=..., room=...)`` so the agent's first LLM turn sees
long-term memory. It calls ``sdk.fetch`` for the user/customer scope and,
if ``formatted_context`` is non-empty, prepends a single ``system``-role
:class:`ChatMessage` to the context.

Error policy: read-side failures degrade silently. If the SDK raises,
we log at ERROR and leave the ``ChatContext`` untouched — a Synap blip
must never prevent a live call from starting.
"""

from __future__ import annotations

import logging
from typing import Optional

from maximem_synap import MaximemSynapSDK

from livekit.agents import ChatContext, ChatMessage

logger = logging.getLogger(__name__)


_PREAMBLE = (
    "Relevant long-term memory about this user (from Synap):\n{body}\n"
    "Use this context only when it helps answer the user's current turn."
)


async def preload_synap_context(
    chat_ctx: ChatContext,
    sdk: MaximemSynapSDK,
    *,
    user_id: str,
    customer_id: str = "",
    mode: str = "accurate",
    max_results: int = 20,
    include_conversation_context: bool = False,
    search_query: Optional[str] = None,
) -> Optional[ChatMessage]:
    """Prepend Synap long-term memory to ``chat_ctx`` as a system message.

    Args:
        chat_ctx: The :class:`ChatContext` about to be handed to an
            :class:`Agent`. Mutated in-place — a single system message is
            inserted at the head of the item list.
        sdk: Configured :class:`MaximemSynapSDK`.
        user_id: Required — Synap memory is user-scoped.
        customer_id: Optional customer/org scope. Empty string means
            customer-less.
        mode: Synap fetch mode (``"accurate"`` or ``"fast"``).
        max_results: Cap on ``sdk.fetch`` results.
        include_conversation_context: Whether to request Synap's recent
            conversation block alongside semantic memory.
        search_query: Optional pre-call query text. When absent, Synap
            returns the user's ambient context (no query bias).

    Returns:
        The inserted :class:`ChatMessage`, or ``None`` if nothing was
        injected (empty context, SDK failure, or formatted_context empty).
    """
    if sdk is None:
        raise ValueError("preload_synap_context requires a non-None sdk")
    if not user_id:
        raise ValueError("preload_synap_context requires a non-empty user_id")

    try:
        response = await sdk.fetch(
            user_id=user_id,
            customer_id=customer_id or None,
            search_query=[search_query] if search_query else None,
            max_results=max_results,
            mode=mode,
            include_conversation_context=include_conversation_context,
        )
    except Exception as exc:  # noqa: BLE001 — read-side graceful degrade
        logger.error(
            "preload_synap_context: sdk.fetch failed user_id=%s error=%s",
            user_id, exc, exc_info=True,
        )
        return None

    formatted = getattr(response, "formatted_context", None) or ""
    if not formatted.strip():
        return None

    body = _PREAMBLE.format(body=formatted.strip())
    # ChatContext prepends by creation_time; we want this message at the
    # head regardless of when the user populated chat_ctx, so insert
    # directly into the items list instead of relying on add_message's
    # time-ordered insertion.
    msg = ChatMessage(role="system", content=[body])
    chat_ctx.items.insert(0, msg)
    return msg


__all__ = ["preload_synap_context"]
