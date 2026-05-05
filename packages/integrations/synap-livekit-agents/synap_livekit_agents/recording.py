"""Record LiveKit ``AgentSession`` turns back to Synap.

:func:`attach_synap_recording` wires a callback onto the session's
``conversation_item_added`` event. In LiveKit 1.x, that event fires once
per committed chat item (both user transcripts and assistant responses),
replacing the 0.x ``user_speech_committed`` / ``agent_speech_committed``
pair. We dispatch on the payload's ``item.role`` and call
``sdk.conversation.record_message`` for each.

Error policy: LiveKit's :class:`EventEmitter` invokes listeners
synchronously and swallows raised exceptions (``logger.exception``). Our
callback must be sync, but ``sdk.conversation.record_message`` is async —
we schedule it via :func:`asyncio.create_task` and install a done-callback
that catches ``SynapIntegrationError`` and friends, logs them at ERROR,
and swallows. Callbacks never raise — a Synap write blip must not tear
down the realtime session.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional

from maximem_synap import MaximemSynapSDK

from synap_integrations_common import wrap_sdk_errors_async

logger = logging.getLogger(__name__)


def attach_synap_recording(
    session: Any,
    sdk: MaximemSynapSDK,
    *,
    user_id: str,
    customer_id: str = "",
    conversation_id: Optional[str] = None,
) -> str:
    """Attach a ``conversation_item_added`` listener that persists turns.

    Args:
        session: The :class:`AgentSession` (or any object with an
            ``on(event_name, callback)`` method compatible with LiveKit's
            :class:`EventEmitter`).
        sdk: Configured :class:`MaximemSynapSDK`.
        user_id: Required — Synap conversations are user-scoped.
        customer_id: Optional customer/org scope. Empty string means
            customer-less.
        conversation_id: Explicit conversation id for this call. Auto-
            generated (``livekit-<hex>``) when absent.

    Returns:
        The ``conversation_id`` used by the listener — hold onto it if
        you want to stitch downstream reads against the same conversation.
    """
    if sdk is None:
        raise ValueError("attach_synap_recording requires a non-None sdk")
    if not user_id:
        raise ValueError("attach_synap_recording requires a non-empty user_id")
    if session is None or not hasattr(session, "on"):
        raise ValueError(
            "attach_synap_recording requires a session with an .on(event, cb) method"
        )

    conv_id = conversation_id or f"livekit-{uuid.uuid4().hex[:12]}"

    def _on_item(event: Any) -> None:
        item = getattr(event, "item", None)
        if item is None:
            return
        role = getattr(item, "role", None)
        if role not in ("user", "assistant"):
            # Ignore system/developer artifacts and non-message items
            # (e.g. AgentHandoff carries no role).
            return
        text = getattr(item, "text_content", None)
        if callable(text):
            try:
                text = text()
            except Exception:  # noqa: BLE001 — defensive
                text = None
        if not text:
            return

        # Schedule the async write. If there's no running loop (e.g. the
        # event fires from a synchronous context), fall back to
        # ``asyncio.run`` on a short-lived loop.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = _record(sdk, conv_id, role, text, user_id, customer_id)
        if loop is not None:
            task = loop.create_task(coro)
            task.add_done_callback(_log_write_failure)
        else:
            try:
                asyncio.run(coro)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "attach_synap_recording: record_message failed "
                    "user_id=%s conversation_id=%s role=%s error=%s",
                    user_id, conv_id, role, exc, exc_info=True,
                )

    session.on("conversation_item_added", _on_item)
    return conv_id


async def _record(
    sdk: MaximemSynapSDK,
    conversation_id: str,
    role: str,
    content: str,
    user_id: str,
    customer_id: str,
) -> None:
    async with wrap_sdk_errors_async(
        "livekit.record_turn",
        logger,
        conversation_id=conversation_id,
        user_id=user_id,
        role=role,
    ):
        await sdk.conversation.record_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            user_id=user_id,
            customer_id=customer_id,
        )


def _log_write_failure(task: "asyncio.Task[Any]") -> None:
    exc = task.exception()
    if exc is None:
        return
    # wrap_sdk_errors_async has already logged the underlying cause;
    # this is belt-and-braces so the task's exception is never "lost"
    # (uncaught task exceptions surface as RuntimeWarnings in newer
    # asyncio versions, which would confuse operators).
    logger.error("attach_synap_recording: write task failed error=%s", exc)


__all__ = ["attach_synap_recording"]
