"""Report a MAF turn on Synap's live gRPC stream.

Everything in this module is a no-op when ``sdk.instance.listen()`` is not
running, and none of it raises: it sits inside somebody's agent turn, and a
telemetry call that throws there ends their run. Both properties come from
``synap_integrations_common.stream_events``; the job here is to translate MAF's
shapes into what those helpers want, and to translate them *correctly*.

**Stream first, REST only as a fallback, never both.** The server writes the
conversation row from the stream event itself
(``grpc/servicer.py::_persist_message`` calls the same
``record_conversation_event`` the REST route does), so reporting a turn *and*
calling ``sdk.conversation.record_message`` stores it twice and extracts it
twice. Every writer in this package asks :func:`report_message_turn` first and
only falls back when it answers ``False``.

Where MAF's five events live, all confirmed by running a real agent turn
against ``agent-framework-core`` 1.19 rather than read off the type stubs:

``user`` / ``assistant`` turn
    ``ContextProvider.before_run`` sees the input messages and
    ``after_run`` sees ``context.response.messages``. Both fire **once per
    agent run**, not once per model call, so a tool loop does not re-report
    the question the way a per-model-call hook would.

tool call / tool result
    A ``ContextProvider`` may add *function middleware* to the invocation
    (``context.extend_middleware``), and that middleware wraps each tool
    invocation: the call on the way in, the result on the way out.
    :class:`SynapToolReporter` is that middleware.

reasoning
    A ``text_reasoning`` content on an assistant message. It is deliberately
    absent from ``Message.text``, so reporting the turn text and the reasoning
    separately does not send the same words twice.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Mapping, Optional

from agent_framework import FunctionMiddleware
from synap_integrations_common.stream_events import (
    report_reasoning,
    report_tool_call,
    report_tool_result,
    report_turn,
)

logger = logging.getLogger(__name__)

#: The only roles that have a stream event. ``report_turn`` maps anything that
#: is not ``"user"`` onto ``assistant_message``, so a ``system`` message put
#: through it would be filed as something the assistant said. System messages
#: therefore keep the REST path they have always had.
STREAMABLE_ROLES = frozenset({"user", "assistant"})

#: Marks a ``SessionContext`` that already carries a tool reporter, so wiring
#: both :class:`~synap_microsoft_agent.context_provider.SynapContextProvider`
#: and the harness memory provider onto one agent reports each tool once
#: rather than twice. Per invocation: the context is rebuilt for every run.
_REPORTER_INSTALLED = "_synap_tool_reporter_installed"

#: Where ``before_run`` notes the input turns it already put on the stream, so
#: ``after_run`` does not send them again. Keyed by ``source_id`` because two
#: providers on one agent have two independent answers.
_REPORTED_INPUTS = "_synap_reported_inputs"


async def report_message_turn(
    sdk: Any,
    *,
    role: str,
    content: str,
    conversation_id: str,
    user_id: str,
    customer_id: str = "",
) -> bool:
    """Put one turn on the stream. Returns whether it went out.

    ``False`` means the caller still owns the write and should fall back to
    ``record_message``: no stream is open, the role has no stream event, or
    the send failed.
    """
    if role not in STREAMABLE_ROLES or not content or not conversation_id:
        return False
    return await report_turn(
        sdk,
        role=role,
        content=content,
        conversation_id=conversation_id,
        user_id=user_id,
        customer_id=customer_id,
    )


async def report_message_reasoning(
    sdk: Any,
    message: Any,
    *,
    conversation_id: str,
    user_id: str,
    customer_id: str = "",
    first_step: int = 0,
) -> int:
    """Report every reasoning step carried by one message, in order.

    Returns how many went out, so a caller walking several messages can keep
    ``step_index`` running across them.

    A turn with no reasoning is an ordinary turn. Most providers return a
    summary or nothing at all, and this reports what it is handed.
    """
    sent = 0
    for content in getattr(message, "contents", None) or []:
        if getattr(content, "type", None) != "text_reasoning":
            continue
        text = (getattr(content, "text", None) or "").strip()
        if not text:
            continue
        if await report_reasoning(
            sdk,
            content=text,
            step_index=first_step + sent,
            conversation_id=conversation_id,
            user_id=user_id,
            customer_id=customer_id,
        ):
            sent += 1
    return sent


def tool_arguments(arguments: Any) -> Optional[dict]:
    """The tool's arguments as a plain dict, or ``None``.

    ⚠ ``report_tool_call`` drops anything that is not a ``dict``
    (``tool_args if isinstance(tool_args, dict) else None``), and MAF's
    ``FunctionInvocationContext.arguments`` is documented as "``BaseModel`` or
    ``Mapping``". A tool whose arguments arrive as a model would therefore have
    reported its name with no arguments at all, silently. Unwrap the model.
    """
    try:
        if isinstance(arguments, Mapping):
            return _json_safe(dict(arguments))
        dump = getattr(arguments, "model_dump", None)
        if callable(dump):
            try:
                dumped = dump(mode="json")
            except TypeError:  # a model_dump without the mode keyword
                dumped = dump()
            if isinstance(dumped, Mapping):
                return _json_safe(dict(dumped))
    except Exception:  # noqa: BLE001 — a hook must never break the agent run
        logger.debug("synap: could not read tool arguments", exc_info=True)
    return None


def tool_result_payload(result: Any) -> Any:
    """What the tool returned, in a shape the SDK can serialise.

    ⚠ This is the one that bites. The SDK does ``json.dumps`` on the result,
    MAF hands function middleware a ``list[Content]``, and ``Content`` is not
    JSON serialisable — measured, not assumed:
    ``TypeError: Object of type Content is not JSON serializable``.
    ``stream_events._send`` swallows every exception by design, so passing the
    list straight through would report **no tool result at all**, with nothing
    logged above debug and nothing failing. Unwrap the contents, and then make
    sure whatever is left actually serialises.
    """
    try:
        if result is None:
            return None
        if isinstance(result, (list, tuple)):
            unwrapped = [_content_payload(item) for item in result]
            unwrapped = [item for item in unwrapped if item is not None]
            if not unwrapped:
                return None
            return _json_safe(unwrapped[0] if len(unwrapped) == 1 else unwrapped)
        return _json_safe(_content_payload(result))
    except Exception:  # noqa: BLE001 — a hook must never break the agent run
        logger.debug("synap: could not read the tool result", exc_info=True)
        return None


def _content_payload(item: Any) -> Any:
    """One ``Content`` reduced to what it is actually carrying."""
    content_type = getattr(item, "type", None)
    if content_type is None:
        return item  # not a Content at all: a raw return value, pass it on
    text = getattr(item, "text", None)
    if text:
        return text
    inner = getattr(item, "result", None)
    if inner is not None:
        return inner
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return str(item)


def _json_safe(value: Any) -> Any:
    """``value`` if ``json.dumps`` will take it, else its string form.

    The last line of defence for the silent drop above: an integration cannot
    know every shape a tool can return, but it can guarantee that what it
    hands the SDK survives serialisation.
    """
    if isinstance(value, str):
        return value
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


class SynapToolReporter(FunctionMiddleware):
    """Function middleware that reports one tool invocation, call and result.

    Registered by a context provider through ``context.extend_middleware``, so
    it is scoped to the invocation rather than installed on the agent.

    ⚠ It must really subclass ``FunctionMiddleware``. MAF's
    ``categorize_middleware`` sorts an object it does not recognise into
    *agent* middleware, and ``SessionContext.extend_middleware`` refuses agent
    middleware from a context provider. The first version of this class was a
    plain object: every tool call and every tool result was dropped, the
    exception was caught here, and the whole unit suite stayed green. It was
    only visible in a real agent run.

    The call and its result share a ``tool_call_id``, which is what lets the
    anticipation agent follow a turn with several tools in flight. MAF puts the
    provider's own id on ``context.metadata["call_id"]``; that landed after the
    1.0 floor this package supports, so an id is minted per invocation when it
    is missing. Either way both events in one ``process`` get the same one.
    """

    def __init__(
        self,
        sdk: Any,
        *,
        conversation_id: str,
        user_id: str,
        customer_id: str = "",
    ) -> None:
        self.sdk = sdk
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.customer_id = customer_id

    async def process(self, context: Any, call_next: Any) -> None:
        call_id = ""
        tool_name = "tool"
        try:
            call_id = str(context.metadata.get("call_id") or "")
            tool_name = str(getattr(context.function, "name", "") or "tool")
        except Exception:  # noqa: BLE001
            logger.debug("synap: could not read the tool invocation", exc_info=True)
        if not call_id:
            call_id = f"maf-{uuid.uuid4().hex}"

        await report_tool_call(
            self.sdk,
            tool_name=tool_name,
            tool_args=tool_arguments(getattr(context, "arguments", None)),
            tool_call_id=call_id,
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            customer_id=self.customer_id,
        )

        # Not wrapped: a tool that raises has no result to report, and the
        # exception is MAF's to handle. Our reporting must not alter it.
        await call_next()

        payload = tool_result_payload(getattr(context, "result", None))
        if payload is None:
            return
        await report_tool_result(
            self.sdk,
            result=payload,
            tool_name=tool_name,
            tool_call_id=call_id,
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            customer_id=self.customer_id,
        )


def install_tool_reporter(
    context: Any,
    sdk: Any,
    *,
    source_id: str,
    conversation_id: str,
    user_id: str,
    customer_id: str = "",
) -> bool:
    """Add a :class:`SynapToolReporter` to this invocation. Returns whether it did.

    Called from ``before_run`` and unconditional on whether a stream is open:
    ``listen()`` can start after the agent is built, and the reporter is a
    no-op until it does.

    Installed at most once per invocation. Two Synap providers on one agent
    (the context provider and the harness memory provider) share a
    ``SessionContext``, and two reporters would report every tool twice.
    """
    if not conversation_id:
        # Nothing to attribute the tool to. Reporting it into an unnamed
        # conversation puts it somewhere nobody reads back.
        return False
    extend = getattr(context, "extend_middleware", None)
    if not callable(extend):
        # A MAF older than provider-added middleware. The turn still reports.
        logger.debug("synap: this agent-framework has no extend_middleware")
        return False
    try:
        metadata = context.metadata
        if metadata.get(_REPORTER_INSTALLED):
            return False
        extend(
            source_id,
            SynapToolReporter(
                sdk,
                conversation_id=conversation_id,
                user_id=user_id,
                customer_id=customer_id,
            ),
        )
        metadata[_REPORTER_INSTALLED] = True
        return True
    except Exception:  # noqa: BLE001 — a hook must never break the agent run
        # WARNING, not debug. A refused install costs every tool call and
        # every tool result for the whole run, and DEBUG is off in virtually
        # every production logger, so at debug this outage is invisible.
        logger.warning(
            "synap: could not install the tool reporter, so no tool call or "
            "tool result will reach anticipation on this run",
            exc_info=True,
        )
        return False


def mark_reported(context: Any, source_id: str, message: Any) -> None:
    """Note that ``message`` already went out on the stream this invocation."""
    try:
        store = context.metadata.setdefault(_REPORTED_INPUTS, {})
        store.setdefault(source_id, set()).add(id(message))
    except Exception:  # noqa: BLE001
        logger.debug("synap: could not mark a reported turn", exc_info=True)


def already_reported(context: Any, source_id: str, message: Any) -> bool:
    """Whether ``before_run`` already put ``message`` on the stream.

    Identity, not content: a user who says "yes" twice in one turn said it
    twice, and keying on the words would swallow the second.
    """
    try:
        store = context.metadata.get(_REPORTED_INPUTS) or {}
        return id(message) in store.get(source_id, ())
    except Exception:  # noqa: BLE001
        return False


__all__ = [
    "STREAMABLE_ROLES",
    "SynapToolReporter",
    "already_reported",
    "install_tool_reporter",
    "mark_reported",
    "report_message_reasoning",
    "report_message_turn",
    "tool_arguments",
    "tool_result_payload",
]
