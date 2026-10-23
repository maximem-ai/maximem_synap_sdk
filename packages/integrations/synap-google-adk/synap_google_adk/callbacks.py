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

Three rules hold throughout.

*The stream is the default path.* The first callback to report opens
``sdk.instance.listen()`` if the caller has not, and the user's turn, the
model's reply, tool calls and results all go on it. When the stream cannot carry
a turn (``SYNAP_AUTO_LISTEN=false``, or it will not open) the two turns go over
REST (``sdk.conversation.record_message``, which needs a ``conversation_id``);
tool calls and results exist only on the stream.

*Synap's compaction replaces ADK's earlier turns.* ``before_model_callback``
rewrites ``llm_request.contents``: once Synap has compacted the conversation,
everything before the current turn is replaced by the compaction (in the system
instruction) and the turns after it. Before that, and whenever Synap cannot
answer, ADK's contents are sent untouched. ADK's own compaction stays installed
as the fallback.

*Never raises.* A callback runs inside the agent's own loop, and ADK treats a
returned value as an override of the model or tool result — so these return
``None`` always, and swallow everything.
"""

from __future__ import annotations

import logging
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    COMPACTION_PREFIX,
    fetch_compacted_history,
    is_synap_history,
    report_tool_call,
    record_turn,
    report_tool_result,
    splice_history,
)

logger = logging.getLogger(__name__)

# What `report_user_turn` wrote last, per SDK and conversation, so the callback
# that reads the same turn off the invocation does not write it a second time.
_manual_turns: "weakref.WeakKeyDictionary[Any, Dict[str, str]]" = weakref.WeakKeyDictionary()
# How many invocations the callbacks remember having written the user's turn for.
_SEEN_INVOCATIONS = 256


def create_synap_callbacks(
    sdk: MaximemSynapSDK,
    user_id: str,
    customer_id: str = "",
    conversation_id: Optional[str] = None,
    *,
    report_tool_results: bool = True,
) -> Dict[str, Any]:
    """Return the ``before_model`` / ``after_model`` / ``before_tool`` / ``after_tool`` callbacks.

    ``before_model_callback`` writes the user's turn and sends the model
    Synap's compaction in place of ADK's earlier turns. It takes the slot of
    your own ``before_model_callback``: to keep one, pass a list
    (``[callbacks["before_model_callback"], yours]``) after the splat.

    Args:
        sdk: A configured SDK. The first callback opens the ``listen()`` stream
            if you have not (``SYNAP_AUTO_LISTEN=false`` stops that, and turns
            then go over REST); the tool callbacks need the stream.
        user_id: Synap user scope. Required.
        customer_id: B2B only. Not accepted on a B2C instance.
        conversation_id: The conversation these runs belong to. Pass it:
            Synap's compaction replaces ADK's earlier turns only for a known
            conversation, and the REST fallback cannot write a turn without
            one.
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
    written: "OrderedDict[str, None]" = OrderedDict()

    # The parameter names are ADK's: it calls callbacks by keyword.
    async def before_model(callback_context: Any, llm_request: Any) -> None:
        """Write the user's turn, then send the model Synap's history.

        Returns ``None`` always: a returned response would skip the model call.
        """
        try:
            await _record_user_turn(sdk, callback_context, written, ids)
            await _splice_request(sdk, llm_request, ids["conversation_id"])
        except Exception:  # noqa: BLE001 — a callback must never end the run
            logger.warning("synap before_model callback failed", exc_info=True)
        return None

    async def before_tool(tool: Any, args: Any, tool_context: Any) -> None:
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

    async def after_tool(tool: Any, args: Any, tool_context: Any, tool_response: Any) -> None:
        if report_tool_results:
            name = str(getattr(tool, "name", "") or "")
            await report_tool_result(
                sdk, result=tool_response, tool_name=name,
                tool_call_id=calls.pop(name, ""), **ids,
            )
        return None

    async def after_model(callback_context: Any, llm_response: Any) -> None:
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
        # A response that calls a tool is the model thinking out loud on the way
        # to its answer: the turn is the response that follows the tool results.
        if _calls_a_tool(llm_response) or getattr(llm_response, "error_code", None):
            return None
        text = _response_text(llm_response)
        if text:
            await record_turn(sdk, role="assistant", content=text, **ids)
        return None

    return {
        "before_model_callback": before_model,
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
    """Write the user's turn yourself. Returns whether it was written.

    The callbacks already write it, from the invocation's user content, so most
    callers never need this. It is for a run that does not use them; if you call
    it and also use them, the turn is written once.
    """
    written = await record_turn(
        sdk, role="user", content=content or "",
        conversation_id=conversation_id, user_id=user_id, customer_id=customer_id,
    )
    if written:
        try:
            _manual_turns.setdefault(sdk, {})[conversation_id] = content
        except TypeError:  # an SDK that cannot be weakly referenced
            pass
    return written


async def _record_user_turn(
    sdk: Any, callback_context: Any, written: "OrderedDict[str, None]", ids: Dict[str, str],
) -> None:
    """The invocation's user message, once per invocation (a tool lap calls the
    model again with the same one)."""
    invocation_id = str(getattr(callback_context, "invocation_id", "") or "")
    if invocation_id:
        if invocation_id in written:
            return
        written[invocation_id] = None
        while len(written) > _SEEN_INVOCATIONS:
            written.popitem(last=False)
    text = _content_text(getattr(callback_context, "user_content", None))
    if not text or is_synap_history(text):
        return
    if _manual_turns.get(sdk, {}).pop(ids["conversation_id"], None) == text:
        return  # report_user_turn already wrote it
    await record_turn(sdk, role="user", content=text, **ids)


def _role_of(content: Any) -> str:
    parts = getattr(content, "parts", None) or []
    if any(getattr(p, "function_call", None) or getattr(p, "function_response", None) for p in parts):
        return "other"  # a tool exchange, not a turn
    return {"user": "user", "model": "assistant"}.get(getattr(content, "role", None) or "user", "other")


def _content_text(content: Any) -> str:
    parts = getattr(content, "parts", None) or []
    return "\n".join(
        str(p.text) for p in parts
        if getattr(p, "text", None) and getattr(p, "thought", None) is not True
    )


async def _splice_request(sdk: Any, llm_request: Any, conversation_id: str) -> None:
    """Replace the request's earlier contents with Synap's compaction and turns."""
    history = await fetch_compacted_history(sdk, conversation_id)
    if history is None:
        return
    from google.genai import types

    def make(role: str, text: str) -> Any:
        return types.Content(
            role="model" if role == "assistant" else "user", parts=[types.Part(text=text)])

    spliced = splice_history(
        list(llm_request.contents or []), history,
        role_of=_role_of, text_of=_content_text, make_message=make,
    )
    # The compaction goes in the system instruction, where ADK keeps the agent's
    # own: a user-role message would break the user/model alternation.
    llm_request.contents = [c for c in spliced if _content_text(c) != history.preamble]
    config = getattr(llm_request, "config", None)
    already = getattr(config, "system_instruction", None)
    if not (isinstance(already, str) and COMPACTION_PREFIX in already):
        llm_request.append_instructions([history.preamble])


def _calls_a_tool(llm_response: Any) -> bool:
    parts: List[Any] = getattr(getattr(llm_response, "content", None), "parts", None) or []
    return any(getattr(p, "function_call", None) for p in parts)


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
