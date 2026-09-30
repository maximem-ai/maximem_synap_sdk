"""Report a deepagents run on Synap's live stream.

The rest of this package is a read path: a backend the agent's file tools reach
through, middleware that puts recalled memory in the prompt, two tools the model
can call. None of it tells Synap what the agent is *doing*, so the anticipation
agent saw a run it could not anticipate: no question, no tools, no answer.

:class:`SynapStreamMiddleware` is the write half. Add it to ``middleware=[...]``
and the five events go out on the gRPC stream ``sdk.instance.listen()`` opened:

===================  ========================================================
event                where it comes from
===================  ========================================================
user turn            ``before_agent`` — the entry node, once per invocation
reasoning            ``after_model`` — the reasoning blocks on the AIMessage
tool call            ``wrap_tool_call`` — before the tool runs
tool result          ``wrap_tool_call`` — what the tool returned
assistant turn       ``after_agent`` — the exit node, once per invocation
===================  ========================================================

Three choices in that table are deliberate, and each of them is a bug that was
shipped once already in the LangChain handler this is modelled on.

**The user turn hangs off ``before_agent``, not ``before_model``.** A tool loop
calls the model again after every tool result with the same human message still
in the list. A per-model-call hook therefore reports the question once per lap,
and Synap extracts it that many times. ``before_agent`` is the entry node: it
runs once per ``invoke``/``ainvoke``. The message id is still keyed and skipped
on a repeat, because a graph resumed from a checkpoint after an interrupt
re-enters that node with the same question.

**The assistant turn hangs off ``after_agent``, not ``after_model``.** The model
speaks on every lap, and the text it produces alongside a tool call is not the
answer to anything. ``assistant_message`` is the event anticipation acts on — it
means *a turn just ended* — so it is sent once, from the exit node, carrying the
final AIMessage.

**The call and the result both come from ``wrap_tool_call``.** They have to
share a ``tool_call_id`` or the anticipation agent cannot pair them when two
tools are in flight, and reading the call from the AIMessage and the result from
somewhere else is how that pairing drifts. Here both sides read
``request.tool_call["id"]``, so they cannot disagree. ``request.tool_call["args"]``
is the parsed dict, never a repr of one.

What a deepagents tool hands back is a ``ToolMessage`` or a ``Command``, and the
SDK JSON-encodes a tool result before sending it. ``json.dumps`` can encode
neither, and the stream helper swallows the failure by design, so passing either
one straight through means every tool result in the run is silently never
reported: no error, no log above debug, nothing failing.
:func:`_tool_output` unwraps them first.

Two rules hold throughout, the same two ``synap_integrations_common`` holds:
silent when no stream is running, and never raises into the graph.

Wiring::

    from deepagents import create_deep_agent
    from maximem_synap import MaximemSynapSDK
    from synap_deepagents import SynapMemoryMiddleware, SynapStreamMiddleware

    sdk = MaximemSynapSDK(api_key="sk-...")
    await sdk.instance.listen()

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-5",
        middleware=[
            SynapMemoryMiddleware(sdk=sdk, user_id="alice"),
            SynapStreamMiddleware(
                sdk=sdk, conversation_id="conv-123", user_id="alice",
            ),
        ],
    )

One middleware covers one conversation: the de-duplication is per instance.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, List, Optional

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ResponseT,
    ToolCallRequest,
)
from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    report_reasoning,
    report_tool_call,
    report_tool_result,
    report_turn,
    run_async,
)

logger = logging.getLogger(__name__)


# ── reading what the framework hands us ──────────────────────────────────────


def _message_text(message: Any) -> str:
    """The text of a LangChain message, whatever shape its content is in.

    ``.text`` is a property in langchain-core 1.x. Older versions exposed it as
    a method and the 1.x value stays callable for compatibility, so it is tested
    for ``str`` before being used — calling the property raises a deprecation
    warning on every turn. A message whose content is a block list is rebuilt
    from its text blocks, because reading only the first one loses the rest.
    """
    if message is None:
        return ""
    text = getattr(message, "text", None)
    if text is not None and not isinstance(text, str) and callable(text):
        try:
            text = text()
        except Exception:  # noqa: BLE001 — a wrapper, not a failure
            text = None
    if isinstance(text, str) and text.strip():
        return text.strip()

    content = (
        message.get("content")
        if isinstance(message, dict)
        else getattr(message, "content", None)
    )
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()
    return ""


def _message_kind(message: Any) -> str:
    """``"human"`` / ``"ai"`` / ... for a message object or a plain dict.

    Middleware sees either, depending on how the graph was invoked.
    """
    kind = getattr(message, "type", None)
    if kind is None and isinstance(message, dict):
        kind = message.get("type") or message.get("role")
    return str(kind or "")


def _last_of_kind(messages: Optional[List[Any]], kinds: tuple) -> Any:
    """The most recent message of one of ``kinds``, or ``None``."""
    for message in reversed(list(messages or [])):
        if _message_kind(message) in kinds:
            return message
    return None


def _message_key(message: Any, text: str) -> str:
    """A stable key for one message, for the per-instance de-duplication.

    LangGraph's ``add_messages`` reducer stamps an id on everything it stores,
    which is the same across every lap of a turn and across a resume. A graph
    that does not use that reducer leaves it unset, and then the text stands in:
    two identical messages back to back are indistinguishable there, and
    reporting one of them twice is worse than reporting it once.
    """
    message_id = (
        message.get("id")
        if isinstance(message, dict)
        else getattr(message, "id", None)
    )
    return str(message_id) if message_id else f"content:{text}"


def _reasoning_steps(message: Any) -> List[str]:
    """Every reasoning step carried on one AIMessage.

    ``content_blocks`` is langchain-core 1.x and normalises the provider
    spellings — a native reasoning block, or the
    ``additional_kwargs["reasoning_content"]`` that Ollama, DeepSeek, xAI and
    Groq use — into one shape. On an older core, or a custom message wrapper
    that does not implement it, the raw content list is read instead.
    """
    if message is None:
        return []
    try:
        blocks: Any = message.content_blocks
    except Exception:  # noqa: BLE001 — a wrapper, not a failure
        logger.debug(
            "synap_deepagents: no content_blocks on %s, reading raw content",
            type(message).__name__,
            exc_info=True,
        )
        blocks = getattr(message, "content", None)
    if not isinstance(blocks, list):
        return []

    steps: List[str] = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "reasoning":
            continue
        text = str(block.get("reasoning") or "").strip()
        if text:
            steps.append(text)
    return steps


def _json_safe(value: Any) -> Any:
    """``value`` if the SDK can JSON-encode it, else its string form.

    The SDK calls ``json.dumps`` on a tool result and ``stream_events`` swallows
    whatever that raises, so an un-encodable result is not an error anywhere —
    it is a result that never arrives. Falling back to ``str`` is what deepagents'
    own tool node does when it puts a result in front of the model, so the
    anticipation agent reads the same thing the model read.
    """
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def _tool_output(result: Any, tool_call_id: str) -> Any:
    """What a deepagents tool returned, in a shape the SDK can put on the wire.

    Two shapes arrive here. A ``ToolMessage`` carries the answer on ``.content``.
    A ``Command`` — which is what every deepagents tool that also updates the
    agent's state returns, the whole filesystem set among them — carries it
    inside ``update["messages"]``, and the entry to read is the one stamped with
    this call's id rather than any state the command also happens to write.
    Neither object is JSON-encodable, and the SDK JSON-encodes a tool result.
    """
    content = getattr(result, "content", None)
    if content is not None and hasattr(result, "tool_call_id"):
        return _json_safe(content)

    update = getattr(result, "update", None)
    if isinstance(update, dict):
        messages = update.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if getattr(message, "tool_call_id", None) != tool_call_id:
                    continue
                return _json_safe(getattr(message, "content", None))

    return _json_safe(result)


# ── the middleware ───────────────────────────────────────────────────────────


class SynapStreamMiddleware(AgentMiddleware[AgentState, ContextT, ResponseT]):
    """Report a deepagents run's five events on Synap's live stream.

    Args:
        sdk: An initialised :class:`~maximem_synap.MaximemSynapSDK`.
        conversation_id: Synap conversation id. **Required** — an event with no
            conversation has nothing to be part of.
        user_id: External user id. **Required.**
        customer_id: B2B instances only, where it is REQUIRED. NOT accepted on a
            B2C instance (``user_context_isolation=equals_customer``): the server
            rejects a call carrying one. Leave it unset there.

    Turn reporting is stream first, REST only as a fallback, and never both. The
    server persists ``user_message`` and ``assistant_message`` from the stream
    itself (``grpc/servicer.py``), so a ``record_message`` on top of a delivered
    stream event writes the turn twice and extracts it twice.
    :func:`~synap_integrations_common.stream_events.report_turn` returns whether
    the event actually went out, which is what makes the choice decidable rather
    than guessed. Tool calls, tool results and reasoning have no REST equivalent:
    with no stream open they are simply not reported.

    Example::

        agent = create_deep_agent(
            model="anthropic:claude-sonnet-5",
            middleware=[SynapStreamMiddleware(
                sdk=sdk, conversation_id="conv-1", user_id="alice",
            )],
        )
    """

    state_schema = AgentState

    def __init__(
        self,
        *,
        sdk: MaximemSynapSDK,
        conversation_id: str,
        user_id: str,
        customer_id: str = "",
    ) -> None:
        if sdk is None:
            raise ValueError("SynapStreamMiddleware requires a non-None sdk")
        if not conversation_id or not str(conversation_id).strip():
            raise ValueError(
                "SynapStreamMiddleware requires a non-empty conversation_id"
            )
        if not user_id or not str(user_id).strip():
            raise ValueError("SynapStreamMiddleware requires a non-empty user_id")

        super().__init__()
        self.sdk = sdk
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.customer_id = customer_id
        # Which messages have already been reported. Per instance, not per
        # class: two conversations must not silence each other.
        self._reported: set = set()
        # Ordinal of the next reasoning step within the current turn.
        self._step_index = 0

    # ── turn recording ───────────────────────────────────────────────────

    async def _record(self, role: str, content: str) -> None:
        """Stream first; REST only when there was no stream. Never both."""
        if await report_turn(
            self.sdk,
            role=role,
            content=content,
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            customer_id=self.customer_id,
        ):
            return
        try:
            await self.sdk.conversation.record_message(
                conversation_id=self.conversation_id,
                role=role,
                content=content,
                user_id=self.user_id,
                customer_id=self.customer_id,
            )
        except Exception as exc:  # noqa: BLE001 — middleware must not raise
            logger.error(
                "SynapStreamMiddleware: record_message failed "
                "role=%s conversation_id=%s error=%s",
                role,
                self.conversation_id,
                exc,
                exc_info=True,
            )

    # ── the user turn: once per invocation ───────────────────────────────

    async def _areport_user(self, state: Any) -> None:
        message = _last_of_kind(
            (state or {}).get("messages"), ("human", "user")
        )
        if message is None:
            return
        text = _message_text(message)
        if not text:
            return
        key = f"user:{_message_key(message, text)}"
        if key in self._reported:
            return
        self._reported.add(key)
        await self._record("user", text)

    async def abefore_agent(
        self, state: AgentState, runtime: Any
    ) -> Optional[dict]:
        """Report the question this invocation is answering."""
        self._step_index = 0
        await self._areport_user(state)
        return None

    def before_agent(self, state: AgentState, runtime: Any) -> Optional[dict]:
        """Synchronous entry point. See :meth:`abefore_agent`."""
        self._step_index = 0
        _bridge(self._areport_user(state), "before_agent")
        return None

    # ── reasoning: once per model call ───────────────────────────────────

    async def _areport_reasoning(self, state: Any) -> None:
        message = _last_of_kind((state or {}).get("messages"), ("ai",))
        if message is None:
            return
        key = f"reasoning:{_message_key(message, _message_text(message))}"
        if key in self._reported:
            return
        self._reported.add(key)
        for thought in _reasoning_steps(message):
            await report_reasoning(
                self.sdk,
                content=thought,
                step_index=self._step_index,
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                customer_id=self.customer_id,
            )
            self._step_index += 1

    async def aafter_model(
        self, state: AgentState, runtime: Any
    ) -> Optional[dict]:
        """Report what the model thought before it spoke, if it says."""
        await self._areport_reasoning(state)
        return None

    def after_model(self, state: AgentState, runtime: Any) -> Optional[dict]:
        """Synchronous entry point. See :meth:`aafter_model`."""
        _bridge(self._areport_reasoning(state), "after_model")
        return None

    # ── the assistant turn: once per invocation ──────────────────────────

    async def _areport_assistant(self, state: Any) -> None:
        message = _last_of_kind((state or {}).get("messages"), ("ai",))
        if message is None:
            return
        text = _message_text(message)
        if not text:
            # The run ended on a tool call rather than an answer. There is no
            # turn to close, and reaching further back for the text of an
            # earlier lap would report deliberation as the final answer.
            return
        key = f"assistant:{_message_key(message, text)}"
        if key in self._reported:
            return
        self._reported.add(key)
        await self._record("assistant", text)

    async def aafter_agent(
        self, state: AgentState, runtime: Any
    ) -> Optional[dict]:
        """Report the answer. This is the event anticipation acts on."""
        await self._areport_assistant(state)
        return None

    def after_agent(self, state: AgentState, runtime: Any) -> Optional[dict]:
        """Synchronous entry point. See :meth:`aafter_agent`."""
        _bridge(self._areport_assistant(state), "after_agent")
        return None

    # ── tools: the call and its result, sharing one id ───────────────────

    def _tool_ids(self, request: ToolCallRequest) -> tuple:
        call = getattr(request, "tool_call", None) or {}
        name = str(call.get("name") or "tool")
        args = call.get("args")
        return name, (args if isinstance(args, dict) else None), str(call.get("id") or "")

    async def _areport_tool_call(self, request: ToolCallRequest) -> None:
        name, args, call_id = self._tool_ids(request)
        await report_tool_call(
            self.sdk,
            tool_name=name,
            tool_args=args,
            tool_call_id=call_id,
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            customer_id=self.customer_id,
        )

    async def _areport_tool_result(
        self, request: ToolCallRequest, result: Any
    ) -> None:
        name, _args, call_id = self._tool_ids(request)
        await report_tool_result(
            self.sdk,
            result=_tool_output(result, call_id),
            tool_name=name,
            tool_call_id=call_id,
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            customer_id=self.customer_id,
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        """Report the call, run the tool, report the result.

        Both reports read ``request.tool_call["id"]``, so the pair cannot drift
        apart. The handler runs whatever the reporting did: a telemetry
        side-channel does not get to decide whether a tool executes.
        """
        await _aswallow(self._areport_tool_call(request), "tool_call")
        result = await handler(request)
        await _aswallow(self._areport_tool_result(request, result), "tool_result")
        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """Synchronous entry point. See :meth:`awrap_tool_call`."""
        _bridge(self._areport_tool_call(request), "tool_call")
        result = handler(request)
        _bridge(self._areport_tool_result(request, result), "tool_result")
        return result


# ── bridges ──────────────────────────────────────────────────────────────────


def _bridge(coro: Any, site: str) -> None:
    """Drive one report coroutine from a synchronous hook, swallowing all of it.

    ``stream_events`` already swallows its own failures; this guards the bridge
    itself, which can fail on a loop ``nest_asyncio`` cannot patch. A middleware
    that raises takes down the graph run with it.
    """
    try:
        run_async(coro)
    except Exception as exc:  # noqa: BLE001 — telemetry never breaks a run
        close = getattr(coro, "close", None)
        if callable(close):
            # A coroutine the bridge never started would warn on GC otherwise.
            close()
        logger.debug(
            "synap_deepagents %s stream report failed: %s", site, exc, exc_info=True
        )


async def _aswallow(coro: Any, site: str) -> None:
    """Await one report, swallowing anything it raises."""
    try:
        await coro
    except Exception as exc:  # noqa: BLE001 — telemetry never breaks a run
        logger.debug(
            "synap_deepagents %s stream report failed: %s", site, exc, exc_info=True
        )


__all__ = ["SynapStreamMiddleware"]
