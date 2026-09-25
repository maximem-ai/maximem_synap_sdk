"""Report what an agent is doing, when there is a stream to report it on.

Every integration hooks a different framework, and they all have the same job
once a hook fires: turn it into a Synap stream event. Four copies of that is
four places for the role to be wrong, the ids to be forgotten, or an exception
to escape into somebody's agent loop.

**Two rules, and they are the whole design.**

*Silent when there is no stream.* These calls only work while
``sdk.instance.listen()`` is running. Most callers never start one, and for
them every function here is a no-op. An integration must behave the same as it
did before for a user who has not opted into streaming.

*Never raises.* A hook runs inside the framework's own loop. A telemetry call
that throws there takes down the agent run, and no context is worth that. Every
function swallows everything and logs at debug.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


def stream_is_active(sdk: Any) -> bool:
    """Whether there is a stream to report on.

    Never raises: an SDK built differently, or half-initialised, answers no.
    """
    try:
        return bool(sdk.instance.is_listening)
    except Exception:  # noqa: BLE001
        return False


async def report_turn(
    sdk: Any,
    *,
    role: str,
    content: str,
    conversation_id: str = "",
    user_id: str = "",
    customer_id: str = "",
    metadata: Optional[Mapping[str, str]] = None,
) -> bool:
    """Report a user or assistant turn. Returns whether it was sent.

    `assistant_message` is the event the anticipation agent acts on: it is the
    moment a turn ends and the next one can be predicted. An integration that
    reports everything else and not this one gets no anticipation at all.
    """
    event_type = "user_message" if role == "user" else "assistant_message"
    return await _send(
        sdk, "send_message",
        content=content,
        role="user" if role == "user" else "assistant",
        event_type=event_type,
        conversation_id=conversation_id,
        user_id=user_id,
        customer_id=customer_id,
        metadata=dict(metadata or {}),
    )


async def report_tool_call(
    sdk: Any,
    *,
    tool_name: str,
    tool_args: Optional[Any] = None,
    tool_call_id: str = "",
    conversation_id: str = "",
    user_id: str = "",
    customer_id: str = "",
) -> bool:
    """Report a tool the agent is invoking."""
    return await _send(
        sdk, "record_tool_call",
        tool_name,
        tool_args if isinstance(tool_args, dict) else None,
        tool_call_id=tool_call_id or None,
        conversation_id=conversation_id,
        user_id=user_id,
        customer_id=customer_id,
    )


async def report_tool_result(
    sdk: Any,
    *,
    result: Any,
    tool_name: str = "",
    tool_call_id: str = "",
    conversation_id: str = "",
    user_id: str = "",
    customer_id: str = "",
) -> bool:
    """Report what a tool returned.

    ⚠ A tool result is usually the caller's customer data. It is an
    anticipation hint and never becomes a long-term memory, but it does leave
    their process. An integration passes through what the framework gives it
    and does not go looking for more.
    """
    return await _send(
        sdk, "record_tool_result",
        result,
        tool_name=tool_name or None,
        tool_call_id=tool_call_id or None,
        conversation_id=conversation_id,
        user_id=user_id,
        customer_id=customer_id,
    )


async def report_reasoning(
    sdk: Any,
    *,
    content: str,
    step_index: Optional[int] = None,
    thought_type: str = "",
    conversation_id: str = "",
    user_id: str = "",
    customer_id: str = "",
) -> bool:
    """Report one reasoning step, when the framework exposes one.

    Most providers return a summary rather than raw reasoning, and some return
    nothing. An integration reports what it is given; a turn with no reasoning
    is an ordinary turn, not a gap to fill.
    """
    if not content:
        return False
    return await _send(
        sdk, "record_thinking",
        content,
        step_index=step_index,
        thought_type=thought_type or None,
        conversation_id=conversation_id,
        user_id=user_id,
        customer_id=customer_id,
    )


async def end_session(sdk: Any, conversation_id: str) -> bool:
    """Close the session for a conversation the integration knows has ended."""
    if not conversation_id:
        return False
    return await _send(sdk, "end_session", conversation_id)


async def _send(sdk: Any, method: str, *args: Any, **kwargs: Any) -> bool:
    """One call site for every report, so the two rules hold in one place."""
    if not stream_is_active(sdk):
        return False
    # `customer_id` is refused on a B2C instance, so an empty one is dropped
    # rather than sent as "". The same applies to any id the caller left out.
    cleaned = {k: v for k, v in kwargs.items() if v not in ("", None)}
    try:
        await getattr(sdk.instance, method)(*args, **cleaned)
        return True
    except Exception:  # noqa: BLE001 — a hook must never break the agent run
        logger.debug("synap stream report failed (%s)", method, exc_info=True)
        return False
