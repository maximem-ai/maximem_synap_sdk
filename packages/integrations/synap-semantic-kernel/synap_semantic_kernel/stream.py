"""Report a Semantic Kernel agent to Synap: turns, tool calls, tool results.

Semantic Kernel has no single event bus, so the whole turn is picked up at
two places, and a caller wires both:

- :class:`SynapAgentThread` — a :class:`ChatHistoryAgentThread` that reports
  the **turns**. ``AgentThread.on_new_message`` is Semantic Kernel's one
  per-message hook, it is already async, and every message contributed to
  the chat by any participant goes through it: the user's, the agent's, and
  the tool traffic in between.
- :func:`attach_synap_filters` — an ``auto_function_invocation`` filter on
  the :class:`Kernel` that reports the **tool call** before the tool runs
  and the **tool result** after. It sits on the kernel rather than the
  thread, so it also covers a plain chat-completion loop that never builds
  an agent.

**Stream first, REST only as a fallback, never both.** The server persists
``user_message`` and ``assistant_message`` from the stream itself
(``grpc/servicer.py``, ``_persist_message``), so reporting a turn AND calling
``sdk.conversation.record_message`` writes it twice and extracts it twice.
:func:`record_turn` makes that choice, and the first turn opens the stream
when none is running (``SYNAP_AUTO_LISTEN=false`` turns that off).

**Synap's compaction replaces Semantic Kernel's.** The thread is also where the
agent reads its history (``get_messages``), so once Synap has compacted the
conversation the model is sent the compaction block and the turns after it in
place of the earlier turns (see :mod:`synap_semantic_kernel.history`).

Tool calls, tool results and reasoning are anticipation hints. The server does
not persist them, so they have no REST fallback and no double-write to avoid.

Nothing here raises into the agent run. ``on_new_message`` is awaited inside
``agent.get_response()`` and a filter sits in the middle of the
function-calling loop; an exception from either is the caller's agent falling
over because of a telemetry side-channel.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterable, Awaitable, Callable, Optional

from semantic_kernel.agents import ChatHistoryAgentThread
from semantic_kernel.contents import (
    AuthorRole,
    ChatHistory,
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
    ReasoningContent,
    TextContent,
)
from semantic_kernel.contents.kernel_content import KernelContent
from semantic_kernel.filters import FilterTypes

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import is_synap_history
from synap_integrations_common.stream_events import (
    end_session,
    record_turn,
    report_reasoning,
    report_tool_call,
    report_tool_result,
)

from synap_semantic_kernel.history import SynapChatHistoryReducer, splice_messages

logger = logging.getLogger(__name__)


# ─── shaping what we send ───────────────────────────────────────────────


def _jsonable(value: Any) -> Any:
    """What we can hand the SDK without it raising on the way out.

    ⚠ The SDK does ``json.dumps`` on a tool result and
    ``stream_events._send`` swallows every exception by design, so a result
    the dumps chokes on is not an error anybody sees: it is a result that
    silently never reaches the anticipation agent. Semantic Kernel hands the
    filter a :class:`FunctionResult`, a pydantic model carrying
    ``KernelFunctionMetadata``, which ``json.dumps`` cannot touch.
    """
    if isinstance(value, str):
        return value
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def _plain(value: Any) -> Any:
    """A ``KernelContent`` (``TextContent``, ``FunctionResultContent`` …) as
    the text it stands for. Every one of them defines ``__str__`` for exactly
    this."""
    return str(value) if isinstance(value, KernelContent) else value


def _tool_output(function_result: Any) -> Any:
    """What the tool returned, in a shape the SDK can serialise.

    ``FunctionResult.value`` is the tool's own return, but it is often a
    list of ``KernelContent`` rather than the plain value, so unwrap that
    before the serialisability check.
    """
    value = getattr(function_result, "value", function_result)
    if isinstance(value, list):
        value = [_plain(item) for item in value]
    else:
        value = _plain(value)
    return _jsonable(value)


def _tool_args(function_call_content: Any) -> dict:
    """The parsed arguments, not the JSON text the model emitted.

    ``FunctionCallContent.arguments`` is ``str | Mapping | None``: a string
    whenever the model produced it. ``report_tool_call`` forwards only a
    ``dict``, so sending the string would have sent no arguments at all, and
    a ``Mapping`` that is not a ``dict`` would go the same way —
    ``parse_arguments`` is typed to return the former.
    """
    if function_call_content is None:
        return {}
    raw = getattr(function_call_content, "arguments", None)
    parse = getattr(function_call_content, "parse_arguments", None)
    if callable(parse):
        try:
            parsed = parse()
        except Exception:  # noqa: BLE001 — SK raises on arguments it cannot parse
            parsed = None
        if parsed is not None:
            try:
                return dict(parsed)
            except (TypeError, ValueError):
                pass
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    return {"input": str(raw)}


def _tool_name(function_call_content: Any, context: Any = None) -> str:
    for candidate in (
        getattr(function_call_content, "name", None),
        getattr(getattr(context, "function", None), "fully_qualified_name", None),
        getattr(getattr(context, "function", None), "name", None),
        getattr(function_call_content, "function_name", None),
    ):
        if candidate:
            return str(candidate)
    return "tool"


def _turn_text(message: ChatMessageContent) -> str:
    """Every text block, not just the first.

    ``ChatMessageContent.content`` returns "the first ``TextContent``'s
    text" and drops the rest, so a reply the model split into several blocks
    would have arrived truncated.
    """
    parts = [item.text for item in message.items if isinstance(item, TextContent)]
    joined = "".join(part for part in parts if part)
    if joined.strip():
        return joined.strip()
    return (message.content or "").strip()


def _carries_tool_traffic(message: ChatMessageContent) -> bool:
    return any(
        isinstance(item, (FunctionCallContent, FunctionResultContent))
        for item in message.items
    )


# ─── the turns ──────────────────────────────────────────────────────────


class SynapAgentThread(ChatHistoryAgentThread):
    """A ``ChatHistoryAgentThread`` that reports each turn to Synap and sends
    the model Synap's compaction once there is one.

    Drop it in where you would pass a plain thread::

        thread = SynapAgentThread(
            sdk, conversation_id="conv-1", user_id="alice", customer_id="acme",
        )
        reply = await agent.get_response(messages="where is my order", thread=thread)
        ...
        await thread.delete()   # ends the Synap session too

    Args:
        sdk: Configured :class:`MaximemSynapSDK`.
        conversation_id: Synap conversation id for this thread. **Required** —
            a turn without one cannot be stitched to the rest of the chat.
        user_id: Required — Synap conversations are user-scoped.
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on
            a B2C instance (user_context_isolation=equals_customer): the server
            rejects a call carrying one with HTTP 400. Leave it unset there.
            Empty string means customer-less.
        chat_history: the thread's history. Left unset it is a
            :class:`SynapChatHistoryReducer`, which Synap's compaction replaces
            turns in. Pass your own (a Semantic Kernel reducer, say) and it
            stays as it is: the model is still sent Synap's compaction in place
            of the earlier turns, but your history is not rewritten.
        thread_id: passed straight through to :class:`ChatHistoryAgentThread`.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        *,
        conversation_id: str,
        user_id: str,
        customer_id: str = "",
        chat_history: Optional[ChatHistory] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapAgentThread requires a non-None sdk")
        if not conversation_id or not str(conversation_id).strip():
            raise ValueError("SynapAgentThread requires a non-empty conversation_id")
        if not user_id:
            raise ValueError("SynapAgentThread requires a non-empty user_id")
        if chat_history is None:
            chat_history = SynapChatHistoryReducer(sdk=sdk, conversation_id=conversation_id)
        super().__init__(chat_history=chat_history, thread_id=thread_id)
        self._sdk = sdk
        self._synap_conversation_id = conversation_id
        self._synap_user_id = user_id
        self._synap_customer_id = customer_id

    @property
    def synap_conversation_id(self) -> str:
        return self._synap_conversation_id

    async def _on_new_message(self, new_message: Any) -> None:
        if isinstance(new_message, str):
            new_message = ChatMessageContent(
                role=AuthorRole.USER, content=new_message
            )
        # The history is the agent's own state; it is updated first so a
        # Synap problem can never cost the caller a message.
        await super()._on_new_message(new_message)
        try:
            await self._report(new_message)
        except Exception as exc:  # noqa: BLE001 — never break the agent run
            logger.error(
                "SynapAgentThread: reporting a message failed "
                "conversation_id=%s error=%s",
                self._synap_conversation_id, exc, exc_info=True,
            )

    async def get_messages(self) -> AsyncIterable[ChatMessageContent]:
        """The history the agent sends the model: Synap's compaction in place of
        the turns before the current one, once Synap has compacted the
        conversation; the framework's own messages otherwise."""
        ours = isinstance(self._chat_history, SynapChatHistoryReducer)
        if ours:
            await self._chat_history.reduce()  # never raises: see splice_messages
        # Raises on a deleted thread, as the base class does.
        messages = [message async for message in super().get_messages()]
        if not ours:
            spliced = await splice_messages(
                self._sdk, self._synap_conversation_id, messages)
            if spliced is not None:
                messages = spliced
        for message in messages:
            yield message

    async def _delete(self) -> None:
        """Deleting the thread ends the Synap session for its conversation."""
        await end_session(self._sdk, self._synap_conversation_id)
        await super()._delete()

    async def _report(self, message: ChatMessageContent) -> None:
        for item in getattr(message, "items", []):
            if isinstance(item, ReasoningContent) and (item.text or "").strip():
                await report_reasoning(
                    self._sdk, content=item.text.strip(),
                    conversation_id=self._synap_conversation_id,
                    user_id=self._synap_user_id,
                    customer_id=self._synap_customer_id,
                )

        role = getattr(message, "role", None)
        if role not in (AuthorRole.USER, AuthorRole.ASSISTANT):
            # TOOL, SYSTEM and DEVELOPER are not turns. The tool traffic is
            # reported by the filter, which has the call id to tie it
            # together; this hook does not.
            return
        if _carries_tool_traffic(message):
            # An assistant message that carries a tool call is a lap of the
            # function-calling loop, not the end of the turn. Semantic Kernel
            # streams its text into the joined message that closes the turn,
            # so reporting it here would extract the same words twice.
            return
        text = _turn_text(message)
        if not text or is_synap_history(text):
            return
        await self._record_turn(
            "user" if role == AuthorRole.USER else "assistant", text
        )

    async def _record_turn(self, role: str, content: str) -> None:
        # Stream first (opened if need be), REST only if the stream cannot
        # carry it, never both: the server persists the turn from the stream
        # itself, so a `record_message` on top of a delivered event stores it
        # twice and extracts it twice.
        await record_turn(
            self._sdk, role=role, content=content,
            conversation_id=self._synap_conversation_id,
            user_id=self._synap_user_id, customer_id=self._synap_customer_id,
        )


# ─── the tool call and its result ───────────────────────────────────────


def synap_auto_function_filter(
    sdk: MaximemSynapSDK,
    *,
    conversation_id: str,
    user_id: str,
    customer_id: str = "",
) -> Callable[[Any, Callable[[Any], Awaitable[None]]], Awaitable[None]]:
    """Build the ``auto_function_invocation`` filter that reports tool traffic.

    The filter wraps the tool: the call goes out before ``next(context)`` so
    the anticipation agent hears about it while the tool is still running,
    and the result goes out after. Both carry
    ``AutoFunctionInvocationContext.function_call_content.id`` as the
    ``tool_call_id``, which is what lets them be read as one exchange rather
    than two unrelated events.

    Use :func:`attach_synap_filters` unless you need the filter object
    itself, for example to remove it later.
    """
    if sdk is None:
        raise ValueError("synap_auto_function_filter requires a non-None sdk")
    if not conversation_id or not str(conversation_id).strip():
        raise ValueError(
            "synap_auto_function_filter requires a non-empty conversation_id"
        )
    if not user_id:
        raise ValueError("synap_auto_function_filter requires a non-empty user_id")

    async def _filter(context: Any, next: Callable[[Any], Awaitable[None]]) -> None:
        function_call_content = getattr(context, "function_call_content", None)
        call_id = str(getattr(function_call_content, "id", "") or "")
        name = _tool_name(function_call_content, context)

        await report_tool_call(
            sdk,
            tool_name=name,
            tool_args=_tool_args(function_call_content),
            tool_call_id=call_id,
            conversation_id=conversation_id,
            user_id=user_id, customer_id=customer_id,
        )
        try:
            await next(context)
        finally:
            # `finally` because a filter further in may terminate the loop
            # and set the result without returning normally. Nothing is
            # reported when there is no result to report: an integration
            # passes through what the framework gives it.
            function_result = getattr(context, "function_result", None)
            if function_result is not None:
                await report_tool_result(
                    sdk,
                    result=_tool_output(function_result),
                    tool_name=name,
                    tool_call_id=call_id,
                    conversation_id=conversation_id,
                    user_id=user_id, customer_id=customer_id,
                )

    _filter.__name__ = "synap_auto_function_filter"
    return _filter


def attach_synap_filters(
    kernel: Any,
    sdk: MaximemSynapSDK,
    *,
    conversation_id: str,
    user_id: str,
    customer_id: str = "",
) -> Callable[[Any, Callable[[Any], Awaitable[None]]], Awaitable[None]]:
    """Register the tool-reporting filter on a :class:`Kernel`.

    Returns the filter, so it can be handed back to ``kernel.remove_filter``.
    """
    if kernel is None or not hasattr(kernel, "add_filter"):
        raise ValueError(
            "attach_synap_filters requires a kernel with an .add_filter method"
        )
    synap_filter = synap_auto_function_filter(
        sdk,
        conversation_id=conversation_id,
        user_id=user_id,
        customer_id=customer_id,
    )
    kernel.add_filter(FilterTypes.AUTO_FUNCTION_INVOCATION, synap_filter)
    return synap_filter


__all__ = [
    "SynapAgentThread",
    "attach_synap_filters",
    "synap_auto_function_filter",
]
