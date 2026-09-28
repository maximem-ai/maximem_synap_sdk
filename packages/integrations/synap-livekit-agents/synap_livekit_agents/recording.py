"""Report a LiveKit ``AgentSession`` to Synap: turns, tool calls, tool results.

:func:`attach_synap_recording` wires four listeners onto one session:

- ``conversation_item_added`` — one committed chat item, user or assistant.
  This is the turn. In LiveKit 1.x it replaced the 0.x
  ``user_speech_committed`` / ``agent_speech_committed`` pair.
- ``tool_execution_updated`` — the flat tool lifecycle. Its
  ``ToolCallStarted`` update fires the moment a tool is dispatched, which on
  a voice call is the moment the caller starts hearing nothing.
- ``function_tools_executed`` — the batch that finished, carrying each call
  next to its output. This is where the result comes from, and it is also
  the fallback for a LiveKit old enough not to emit
  ``tool_execution_updated``.
- ``close`` — the call ended, so the conversation's session can be closed
  rather than left to expire.

**Stream first, REST only as a fallback, never both.** The server persists
``user_message`` and ``assistant_message`` from the stream itself
(``grpc/servicer.py``, ``_persist_message``), so reporting a turn AND calling
``sdk.conversation.record_message`` writes it twice and extracts it twice.
:func:`report_turn` returns whether it actually went out, which is what makes
the choice decidable rather than guessed.

Tool calls, tool results and reasoning are anticipation hints. The server does
not persist them, so they have no REST fallback and no double-write to avoid.

Error policy: LiveKit's :class:`EventEmitter` **refuses an async callback**
outright (``.on()`` raises ``ValueError`` for one), so every listener here is
synchronous and hands its work to the loop. Failures are logged at ERROR and
swallowed — a Synap write blip must not tear down a realtime session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Coroutine, Optional

from maximem_synap import MaximemSynapSDK

from synap_integrations_common import wrap_sdk_errors_async
from synap_integrations_common.stream_events import (
    end_session,
    report_tool_call,
    report_tool_result,
    report_turn,
)

logger = logging.getLogger(__name__)

# How many ids to remember for de-duplication. A voice call runs for minutes
# and makes tens of tool calls, so this never fills; it exists so an hours-long
# session cannot grow the set without bound.
_SEEN_LIMIT = 512


def _jsonable(value: Any) -> Any:
    """What we can hand the SDK without it raising on the way out.

    ⚠ The SDK does ``json.dumps`` on a tool result, and
    ``stream_events._send`` swallows every exception by design. A result the
    dumps chokes on is therefore not an error anybody sees: it is a result
    that silently never reaches the anticipation agent. LiveKit types
    ``FunctionCallOutput.output`` as ``str``, but the value is whatever the
    tool returned once a provider adapter has been through it, so check
    rather than trust the annotation.
    """
    if isinstance(value, str):
        return value
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def _tool_args(function_call: Any) -> dict:
    """The parsed arguments, not the JSON text LiveKit carries them in.

    ``FunctionCall.arguments`` is a **string**: the raw JSON the model
    emitted. Passing it through as-is would hand the anticipation agent
    ``'{"pnr": "QX41RT"}'`` as one opaque blob to reason over instead of the
    arguments themselves — and ``report_tool_call`` drops a non-dict on the
    floor, so it would have arrived as nothing at all.
    """
    raw = getattr(function_call, "arguments", None)
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {"input": str(raw)}
    return parsed if isinstance(parsed, dict) else {"input": str(raw)}


def _text_of(item: Any) -> Optional[str]:
    """The text of a chat item. ``text_content`` is a property on LiveKit's
    own ``ChatMessage``, but stay tolerant of a double that makes it a
    method."""
    text = getattr(item, "text_content", None)
    if callable(text):
        try:
            text = text()
        except Exception:  # noqa: BLE001 — defensive
            return None
    return text or None


class _SessionReporter:
    """Per-attachment state: what has been reported, and in what order.

    Per instance and not per class: two concurrent calls must not silence
    each other's events.
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        conversation_id: str,
        user_id: str,
        customer_id: str,
    ) -> None:
        self.sdk = sdk
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.customer_id = customer_id
        # Turns already reported, keyed on the item id LiveKit stamps.
        self._seen_items: dict = {}
        # Tool calls / results already reported, keyed on `call_id`. A call
        # reported from `tool_execution_updated` must not be reported again
        # when `function_tools_executed` carries the same call.
        self._seen_calls: dict = {}
        self._seen_results: dict = {}
        # Tail of the report chain, so reports land in the order the session
        # produced them. Without it a tool result scheduled a second after
        # its call could still be awaited first.
        self._tail: Optional["asyncio.Task[Any]"] = None

    # ── de-duplication ──────────────────────────────────────────────────

    @staticmethod
    def _claim(seen: dict, key: str) -> bool:
        """Record `key` and say whether this is the first time we have seen it."""
        if key in seen:
            return False
        seen[key] = None
        while len(seen) > _SEEN_LIMIT:
            seen.pop(next(iter(seen)))
        return True

    # ── dispatch ────────────────────────────────────────────────────────

    def dispatch(self, make_coro: Callable[[], Coroutine[Any, Any, Any]]) -> None:
        """Run one report off the synchronous event callback.

        LiveKit's ``EventEmitter.on`` refuses an async callback, so the work
        has to be handed to the loop. With no loop running (a synchronous
        driver, or a test) the coroutine is driven to completion instead.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            try:
                asyncio.run(make_coro())
            except Exception as exc:  # noqa: BLE001 — callbacks never raise
                logger.error(
                    "attach_synap_recording: report failed "
                    "user_id=%s conversation_id=%s error=%s",
                    self.user_id, self.conversation_id, exc, exc_info=True,
                )
            return

        previous = self._tail

        async def _ordered() -> None:
            if previous is not None:
                # `wait` neither re-raises the previous report's exception nor
                # turns its cancellation into ours; it only waits.
                try:
                    await asyncio.wait({previous})
                except Exception:  # noqa: BLE001
                    pass
            await make_coro()

        task = loop.create_task(_ordered())
        self._tail = task
        task.add_done_callback(_log_write_failure)

    # ── the events ──────────────────────────────────────────────────────

    def on_conversation_item_added(self, event: Any) -> None:
        item = getattr(event, "item", None)
        if item is None:
            return
        role = getattr(item, "role", None)
        if role not in ("user", "assistant"):
            # Ignore system/developer artifacts and non-message items
            # (e.g. AgentHandoff carries no role).
            return
        text = _text_of(item)
        if not text:
            # An assistant message whose only content is a tool call has no
            # text. That is not a turn; the tool events carry it.
            return
        item_id = getattr(item, "id", None)
        if isinstance(item_id, str) and item_id:
            if not self._claim(self._seen_items, item_id):
                return
        self.dispatch(lambda: self._record_turn(role, text))

    def on_tool_execution_updated(self, event: Any) -> None:
        """The dispatch moment, which on a voice call is the dead air.

        A tool call is the clearest signal of what the agent is about to
        need, and the gap between a caller's question and the answer is the
        window anticipation has to work in. ``function_tools_executed`` only
        fires once the whole batch has finished, so reporting the call from
        there would throw that window away.
        """
        update = getattr(event, "update", None)
        if getattr(update, "type", None) != "tool_call_started":
            # `tool_call_updated` is a spoken progress note, `tool_call_ended`
            # carries the text voiced to the caller rather than what the tool
            # returned, and `tool_reply_updated` is speech lifecycle. The
            # authoritative result comes from `function_tools_executed`.
            return
        function_call = getattr(update, "function_call", None)
        if function_call is None:
            return
        call_id = str(getattr(function_call, "call_id", "") or "")
        if call_id and not self._claim(self._seen_calls, call_id):
            return
        name = str(getattr(function_call, "name", "") or "tool")
        args = _tool_args(function_call)
        self.dispatch(lambda: self._report_call(name, args, call_id))

    def on_function_tools_executed(self, event: Any) -> None:
        """Calls paired with their outputs, by list position and by call_id.

        The call half is reported only when ``tool_execution_updated`` did
        not already do it: a LiveKit old enough to lack that event, or a path
        that skips it, still gets a complete picture.
        """
        calls = list(getattr(event, "function_calls", None) or [])
        outputs = list(getattr(event, "function_call_outputs", None) or [])

        for function_call in calls:
            call_id = str(getattr(function_call, "call_id", "") or "")
            if call_id and not self._claim(self._seen_calls, call_id):
                continue
            name = str(getattr(function_call, "name", "") or "tool")
            args = _tool_args(function_call)
            self.dispatch(
                lambda n=name, a=args, c=call_id: self._report_call(n, a, c)
            )

        for output in outputs:
            call_id = str(getattr(output, "call_id", "") or "")
            if call_id and not self._claim(self._seen_results, call_id):
                continue
            name = str(getattr(output, "name", "") or "")
            result = _jsonable(getattr(output, "output", None))
            self.dispatch(
                lambda r=result, n=name, c=call_id: self._report_result(r, n, c)
            )

    def on_close(self, *_args: Any) -> None:
        """The call ended. Close the conversation's session rather than
        leaving the server to expire it.

        Takes ``*args`` because ``EventEmitter.emit`` re-raises a ``TypeError``
        from a listener, so a listener must not care how many arguments the
        event carries.
        """
        self.dispatch(lambda: end_session(self.sdk, self.conversation_id))

    # ── the reports ─────────────────────────────────────────────────────

    async def _record_turn(self, role: str, content: str) -> None:
        # ⚠ Stream first, REST only for what the stream did not take, never
        # both: the server persists the turn from the stream itself, so a
        # `record_message` on top of a delivered event stores it twice and
        # extracts it twice.
        if await report_turn(
            self.sdk, role=role, content=content,
            conversation_id=self.conversation_id,
            user_id=self.user_id, customer_id=self.customer_id,
        ):
            return
        async with wrap_sdk_errors_async(
            "livekit.record_turn",
            logger,
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            role=role,
        ):
            await self.sdk.conversation.record_message(
                conversation_id=self.conversation_id,
                role=role,
                content=content,
                user_id=self.user_id,
                customer_id=self.customer_id,
            )

    async def _report_call(self, name: str, args: dict, call_id: str) -> None:
        await report_tool_call(
            self.sdk,
            tool_name=name,
            tool_args=args,
            tool_call_id=call_id,
            conversation_id=self.conversation_id,
            user_id=self.user_id, customer_id=self.customer_id,
        )

    async def _report_result(self, result: Any, name: str, call_id: str) -> None:
        await report_tool_result(
            self.sdk,
            result=result,
            tool_name=name,
            tool_call_id=call_id,
            conversation_id=self.conversation_id,
            user_id=self.user_id, customer_id=self.customer_id,
        )


def attach_synap_recording(
    session: Any,
    sdk: MaximemSynapSDK,
    *,
    user_id: str,
    customer_id: str = "",
    conversation_id: Optional[str] = None,
) -> str:
    """Attach the Synap listeners to a LiveKit ``AgentSession``.

    Args:
        session: The :class:`AgentSession` (or any object with an
            ``on(event_name, callback)`` method compatible with LiveKit's
            :class:`EventEmitter`).
        sdk: Configured :class:`MaximemSynapSDK`.
        user_id: Required — Synap conversations are user-scoped.
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on a
            B2C instance (user_context_isolation=equals_customer): the server
            rejects a call carrying one with HTTP 400. Leave it unset there. Empty string means
            customer-less.
        conversation_id: Explicit conversation id for this call. Auto-
            generated (``livekit-<hex>``) when absent.

    Returns:
        The ``conversation_id`` used by the listeners — hold onto it if
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
    reporter = _SessionReporter(sdk, conv_id, user_id, customer_id)

    # `EventEmitter.on` stores by event name and never validates it, so
    # registering an event a older LiveKit does not emit is inert rather than
    # an error.
    session.on("conversation_item_added", reporter.on_conversation_item_added)
    session.on("tool_execution_updated", reporter.on_tool_execution_updated)
    session.on("function_tools_executed", reporter.on_function_tools_executed)
    session.on("close", reporter.on_close)
    return conv_id


def _log_write_failure(task: "asyncio.Task[Any]") -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is None:
        return
    # wrap_sdk_errors_async has already logged the underlying cause;
    # this is belt-and-braces so the task's exception is never "lost"
    # (uncaught task exceptions surface as RuntimeWarnings in newer
    # asyncio versions, which would confuse operators).
    logger.error("attach_synap_recording: write task failed error=%s", exc)


__all__ = ["attach_synap_recording"]
