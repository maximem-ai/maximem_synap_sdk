"""Report a Google ADK run to Synap, so the next turn starts warm.

This package gave an agent tools and reported nothing about the run itself.
Synap saw whatever the model chose to search for, never the reply, the tools or
the model's own output. Anticipation had nothing to work with, so every fetch
stayed a cold retrieval.

``create_synap_callbacks`` returns the callback keyword arguments an ``LlmAgent``
takes::

    agent = LlmAgent(
        model="gemini-2.0-flash",
        tools=create_synap_tools(sdk, user_id="u1"),
        **create_synap_callbacks(sdk, user_id="u1", conversation_id=conv_id),
    )

Two rules hold throughout.

*Turns are always written; the rest needs a stream.* The model's reply and
the user's turn go on the ``sdk.instance.listen()`` stream when one is open and
over REST (``sdk.conversation.record_message``) when not, so Synap keeps the
conversation either way (REST needs a ``conversation_id``). Tool calls and
tool results are stream-only: without a stream those callbacks are no-ops.

*Never raises.* A callback runs inside the agent's own loop, and ADK treats a
returned value as an override of the model or tool result — so these return
``None`` always, and swallow everything.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    report_tool_call,
    record_turn,
    report_tool_result,
)

logger = logging.getLogger(__name__)


def create_synap_callbacks(
    sdk: MaximemSynapSDK,
    user_id: str,
    customer_id: str = "",
    conversation_id: Optional[str] = None,
    *,
    report_tool_results: bool = True,
) -> Dict[str, Any]:
    """Return ``before_tool_callback`` / ``after_tool_callback`` / ``after_model_callback``.

    Args:
        sdk: A configured SDK. The reply is written with or without an active
            ``listen()`` stream (REST when there is none); the tool callbacks
            are no-ops without one.
        user_id: Synap user scope. Required.
        customer_id: B2B only. Not accepted on a B2C instance.
        conversation_id: The conversation these runs belong to. Required for
            the events to mean anything: without it the server cannot say which
            conversation they belong to, no session is opened, and with no
            stream the reply is not written at all.
        report_tool_results: Whether tool results are reported. ⚠ A tool result
            is usually your own customer's data. It is an anticipation hint and
            never becomes a long-term memory, but it does leave your process.

    Returns:
        A dict to splat into ``LlmAgent(...)``.
    """
    if sdk is None:
        raise ValueError("create_synap_callbacks requires a non-None sdk")
    if not user_id:
        raise ValueError("create_synap_callbacks requires a non-empty user_id")

    ids = {
        "conversation_id": conversation_id or "",
        "user_id": user_id,
        "customer_id": customer_id,
    }
    # Tool calls and results arrive as separate callbacks with nothing tying
    # them together, so the pairing is kept here: one tool in flight per name,
    # which is what a sequential agent loop produces.
    calls: Dict[str, str] = {}
    step = {"n": 0}

    async def before_tool(tool: Any, args: Any, context: Any) -> None:
        name = str(getattr(tool, "name", "") or "")
        call_id = f"{name}:{step['n']}"
        step["n"] += 1
        calls[name] = call_id
        await report_tool_call(
            sdk, tool_name=name,
            tool_args=args if isinstance(args, dict) else None,
            tool_call_id=call_id, **ids,
        )
        # ADK reads a returned value as an override of the tool call. Always
        # None: this is telemetry, and telemetry must not change what runs.
        return None

    async def after_tool(tool: Any, args: Any, context: Any, tool_response: Any) -> None:
        if report_tool_results:
            name = str(getattr(tool, "name", "") or "")
            await report_tool_result(
                sdk, result=tool_response, tool_name=name,
                tool_call_id=calls.pop(name, ""), **ids,
            )
        return None

    async def after_model(context: Any, llm_response: Any) -> None:
        """Report the model's reply.

        This is the event anticipation acts on: the turn has ended, so the next
        one can be predicted and its context pushed before the user speaks
        again. An integration that reports tools and not this gets no
        anticipation at all.
        """
        # A streamed (SSE) run calls this once per partial chunk and once more
        # with the whole reply. Only the whole reply is a turn: writing the
        # chunks would store the reply several times over.
        if getattr(llm_response, "partial", None) is True:
            return None
        text = _response_text(llm_response)
        if text:
            await record_turn(sdk, role="assistant", content=text, **ids)
        return None

    return {
        "before_tool_callback": before_tool,
        "after_tool_callback": after_tool,
        "after_model_callback": after_model,
    }


async def report_user_turn(
    sdk: MaximemSynapSDK,
    content: str,
    *,
    conversation_id: str,
    user_id: str,
    customer_id: str = "",
) -> bool:
    """Write the user's turn before you run the agent.

    Goes on the ``listen()`` stream when one is open and over REST when not,
    so the turn is kept either way. Returns whether it was written.

    ``before_agent_callback`` fires with a context, not with what the user
    said, and the run's input is yours rather than ADK's. One line before the
    run is the whole cost, and without it the conversation Synap sees has
    answers and no questions.
    """
    return await record_turn(
        sdk, role="user", content=content or "",
        conversation_id=conversation_id, user_id=user_id, customer_id=customer_id,
    )


def _response_text(llm_response: Any) -> str:
    """The model's text out of an ``LlmResponse``.

    Reads the documented shape (``content.parts[].text``) and gives up rather
    than guessing: sending the wrong field as the agent's reply is worse than
    sending nothing.
    """
    if llm_response is None:
        return ""
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) or []
    texts = [
        str(getattr(part, "text", "") or "") for part in parts
        # A thought part is the model's reasoning, not its reply.
        if getattr(part, "text", None) and getattr(part, "thought", None) is not True
    ]
    return "".join(texts).strip()
