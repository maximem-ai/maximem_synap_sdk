"""Report a CrewAI run to Synap, so the anticipation agent sees the whole turn.

This package gave CrewAI a storage backend and a one-shot backstory builder.
Synap saw memories going in and coming out, and never saw the run itself: not
the request, not the answer, not the tools, not the reasoning. Anticipation had
nothing to work with, so every fetch was a cold retrieval.

``SynapCrewAIListener`` fills that in. CrewAI publishes its whole lifecycle on
a global event bus, and this listener subscribes to the five events that matter:

=================================  ====================================
CrewAI event                        Synap event
=================================  ====================================
``CrewKickoffStartedEvent``         user turn (the crew's inputs)
``CrewKickoffCompletedEvent``       assistant turn (the crew's answer)
``LiteAgentExecutionCompletedEvent`` assistant turn (a standalone agent)
``ToolUsageStartedEvent``           tool call
``ToolUsageFinishedEvent``          tool result
``ToolUsageErrorEvent``             tool result (the failure)
``AgentReasoningCompletedEvent``    reasoning
=================================  ====================================

A tool call and its result share an id. CrewAI's bus pairs its own scoped
events: it stamps every ending event with the ``event_id`` of the starting
event it closes, so ``ToolUsageFinishedEvent.started_event_id`` **is** the
``ToolUsageStartedEvent.event_id``. That is the id both legs travel with.

⚠ **The event bus runs handlers on its own event loop.** CrewAI dispatches
sync handlers on a worker thread and async handlers on a private
``CrewAIEventsLoop`` thread, and neither is the loop your
``sdk.instance.listen()`` stream lives on. A gRPC stream cannot be written
from a foreign loop, so an ``async def`` handler here would look correct,
report nothing, and say nothing about it. The handlers are therefore
deliberately synchronous, and each one hands its coroutine back to the
listening loop with :func:`asyncio.run_coroutine_threadsafe`. That loop is
captured when you build the listener; see ``loop`` and :meth:`bind_loop`.

Two rules hold throughout, and they come from ``stream_events``.

*The stream is the default.* The first event opens ``sdk.instance.listen()``
if you have not, and every turn, tool call, tool result and reasoning step
goes on it. ``SYNAP_AUTO_LISTEN=false`` (or
``synap_integrations_common.set_auto_listen(sdk, False)``) turns that off; the
user and assistant turns are then written over REST
(``sdk.conversation.record_message``) and the rest is dropped. With no loop
captured the report runs on the common background loop, which is also where
the stream then opens.

*Synap's compaction replaces CrewAI's history.* The listener also registers a
``before_llm_call`` hook. On the first model call of an executor, once Synap has
compacted the conversation, the messages CrewAI built (anything before the
current request, such as a history passed to ``agent.kickoff(messages=...)``)
are replaced by Synap's compaction and the turns after it. Before the first
compaction, or when Synap cannot answer, CrewAI's messages are untouched. The
hook is process-wide, like every CrewAI LLM hook, so it applies to each crew
that runs while the listener is live.

*Never raises.* A handler runs inside CrewAI's own dispatch. Nothing here
raises into it, and nothing here blocks it: the report is handed to the
listening loop and the handler returns.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from crewai.events import (
    AgentReasoningCompletedEvent,
    BaseEventListener,
    CrewKickoffCompletedEvent,
    CrewKickoffStartedEvent,
    LiteAgentExecutionCompletedEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
    crewai_event_bus,
)

from maximem_synap import MaximemSynapSDK
from synap_integrations_common import (
    fetch_compacted_history,
    is_synap_history,
    record_turn,
    report_reasoning,
    report_tool_call,
    report_tool_result,
    run_async,
    run_blocking,
    splice_history,
    stream_is_active,
)

logger = logging.getLogger(__name__)

_NO_LOOP_WARNING = (
    "SynapCrewAIListener: a Synap stream you opened is live but no event loop "
    "was captured, so nothing can be reported. CrewAI runs event handlers on "
    "its own threads, and a gRPC stream can only be written from the loop that "
    "sdk.instance.listen() runs on. Build the listener from inside that loop, "
    "pass loop=..., call listener.bind_loop() from a coroutine running on it, "
    "or leave listen() to the listener."
)

# The static block build_synap_st_backstory puts in a backstory. Once Synap's
# own history is spliced in it says the same thing, older.
_ST_BLOCK = re.compile(
    r"<synap_short_term_context>.*?</synap_short_term_context>\s*", re.DOTALL
)


class SynapCrewAIListener(BaseEventListener):
    """Reports a CrewAI run on the Synap stream.

    Building it registers the handlers on CrewAI's global event bus, which is
    how CrewAI intends listeners to work. Keep a reference to it: the
    registration lasts as long as the process, and :meth:`stop` is the way to
    undo it.

    Args:
        sdk: A configured :class:`MaximemSynapSDK`. The first event opens
            its ``listen()`` stream unless you already did; see the module
            docstring for the opt-out.
        user_id: Synap user scope. **Required.**
        conversation_id: The conversation these runs belong to. **Required.**
            Without it the server cannot say which conversation an event is
            part of, and no session is opened.
        customer_id: B2B only. Not accepted on a B2C instance.
        loop: The event loop ``sdk.instance.listen()`` runs on. Defaults to
            the running loop at construction, which is the right one whenever
            you build the listener from the same async code that started the
            stream. Only needed when you build it from synchronous code; see
            :meth:`bind_loop`.
        report_tool_results: Whether tool results are reported. On by default,
            because a result is what tells anticipation how the turn is
            actually going. ⚠ A tool result is usually your own customer's
            data. It is an anticipation hint and never becomes a long-term
            memory, but it does leave your process. Turn this off if that is
            not something you want to send.
        report_kickoff_inputs: Whether the crew's ``inputs`` are reported as
            the user's turn. On by default, because a conversation of answers
            with no questions is not much to anticipate from. ⚠ CrewAI has no
            user turn of its own: ``inputs`` are template variables, which is
            the closest thing to a request the framework has. Turn this off
            and report the real thing yourself with
            :func:`report_user_turn` if your inputs are not what the user said.

    Example::

        listener = SynapCrewAIListener(
            sdk, user_id="user-1", conversation_id="conv-1",
        )
        result = await crew.kickoff_async(inputs={"question": "where is my order"})
    """

    def __init__(
        self,
        sdk: MaximemSynapSDK,
        user_id: str,
        conversation_id: str,
        customer_id: str = "",
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        report_tool_results: bool = True,
        report_kickoff_inputs: bool = True,
    ) -> None:
        if sdk is None:
            raise ValueError("SynapCrewAIListener requires a non-None sdk")
        if not user_id:
            raise ValueError("SynapCrewAIListener requires a non-empty user_id")
        if not conversation_id:
            raise ValueError(
                "SynapCrewAIListener requires a non-empty conversation_id"
            )

        # Everything the handlers read has to exist before super().__init__(),
        # which is what registers them.
        self._sdk = sdk
        self._user_id = user_id
        self._conversation_id = conversation_id
        self._customer_id = customer_id
        self._report_tool_results = report_tool_results
        self._report_kickoff_inputs = report_kickoff_inputs
        self._loop = loop or _running_loop()
        self._no_loop_warned = False
        self._step = 0
        self._last_inputs = ""
        # Only used when the bus could not pair the two legs of a tool call
        # itself. One tool in flight per name, which is what a sequential
        # agent loop produces.
        self._calls: Dict[str, str] = {}
        self._registered: List[Tuple[type, Any]] = []
        self._hook_registered = False

        super().__init__()
        self._register_history_hook()

    # ── which loop the reports go back to ───────────────────────────────

    def bind_loop(
        self, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        """Point the listener at the loop your Synap stream runs on.

        Only needed when the listener was built from synchronous code, so
        there was no running loop to capture. Call it with no arguments from
        inside a coroutine running on the right loop::

            async def main():
                async with sdk.instance.listen():
                    listener.bind_loop()
                    crew.kickoff()
        """
        self._loop = loop or _running_loop()
        self._no_loop_warned = False

    def stop(self) -> None:
        """Unregister every handler from CrewAI's global event bus.

        The bus is a process-wide singleton and keeps a reference to each
        handler, so a listener that is never stopped outlives the run that
        needed it, along with the SDK it closes over.
        """
        for event_type, handler in self._registered:
            crewai_event_bus.off(event_type, handler)
        self._registered.clear()
        if self._hook_registered:
            try:
                from crewai.hooks.llm_hooks import unregister_before_llm_call_hook

                unregister_before_llm_call_hook(self._before_llm_call)
            except Exception:  # noqa: BLE001
                logger.debug("synap_crewai: could not unregister the history hook",
                             exc_info=True)
            self._hook_registered = False

    # ── the ids every event carries ─────────────────────────────────────

    def _ids(self) -> dict:
        return {
            "conversation_id": self._conversation_id,
            "user_id": self._user_id,
            "customer_id": self._customer_id,
        }

    def _dispatch(self, coro: Any) -> None:
        """Hand a report to the listening loop and return.

        Fire and forget on purpose. CrewAI does not wait for a handler, and
        neither should the agent's run wait for our telemetry.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            self._dispatch_without_loop(coro)
            return
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception:  # noqa: BLE001 — a handler must never raise
            coro.close()
            logger.debug("synap_crewai: could not schedule a report", exc_info=True)
            return
        future.add_done_callback(_swallow)

    def _dispatch_without_loop(self, coro: Any) -> None:
        """Run a report when no loop was captured: on the common background loop.

        That is the loop the stream opens on when the listener has to open it,
        and the loop the SDK's REST client is used from by every other
        integration helper. The handler's own thread waits for it, which is
        CrewAI's worker and not the crew. A stream you opened yourself on a loop
        we were not told about cannot be written from here, so that is the one
        case that still warns and drops.
        """
        if _foreign_stream(self._sdk):
            self._warn_no_loop()
            coro.close()
            return
        try:
            run_blocking(coro)
        except Exception:  # noqa: BLE001 — a handler must never raise
            coro.close()
            logger.debug("synap_crewai: a report failed", exc_info=True)

    def _warn_no_loop(self) -> None:
        # Loud, and once. A stream that is open and reporting nothing is the
        # failure that looks like success, so it does not get a debug line.
        if self._no_loop_warned:
            return
        self._no_loop_warned = True
        logger.warning(_NO_LOOP_WARNING)

    # ── the handlers ────────────────────────────────────────────────────

    def setup_listeners(self, crewai_event_bus: Any) -> None:
        """Subscribe to the bus. Called for us by ``BaseEventListener``."""

        def register(event_type: type, handler: Any) -> None:
            crewai_event_bus.on(event_type)(handler)
            self._registered.append((event_type, handler))

        def on_kickoff_started(source: Any, event: Any) -> None:
            self._step = 0
            text = _inputs_text(getattr(event, "inputs", None))
            self._last_inputs = text
            if not self._report_kickoff_inputs or not text:
                return
            self._write_turn("user", text)

        def on_kickoff_completed(source: Any, event: Any) -> None:
            self._report_answer(getattr(event, "output", None))

        def on_lite_agent_completed(source: Any, event: Any) -> None:
            # `agent.kickoff()` runs a LiteAgent, which never goes through a
            # crew, so this is the only place its answer appears.
            self._report_answer(getattr(event, "output", None))

        def on_tool_started(source: Any, event: Any) -> None:
            name = str(getattr(event, "tool_name", "") or "tool")
            call_id = str(getattr(event, "event_id", "") or "")
            if call_id:
                self._calls[name] = call_id
            self._dispatch(
                report_tool_call(
                    self._sdk,
                    tool_name=name,
                    tool_args=_tool_args(getattr(event, "tool_args", None)),
                    tool_call_id=call_id,
                    **self._ids(),
                )
            )

        def on_tool_finished(source: Any, event: Any) -> None:
            self._report_tool_outcome(event, getattr(event, "output", None))

        def on_tool_error(source: Any, event: Any) -> None:
            # A tool that raised still produced an outcome, and a call with no
            # result is a loose end the anticipation agent cannot close.
            error = getattr(event, "error", None)
            self._report_tool_outcome(event, f"error: {error}")

        def on_reasoning_completed(source: Any, event: Any) -> None:
            plan = getattr(event, "plan", None)
            if not isinstance(plan, str) or not plan.strip():
                return
            self._step += 1
            self._dispatch(
                report_reasoning(
                    self._sdk,
                    content=plan.strip(),
                    step_index=self._step,
                    thought_type="plan",
                    **self._ids(),
                )
            )

        register(CrewKickoffStartedEvent, on_kickoff_started)
        register(CrewKickoffCompletedEvent, on_kickoff_completed)
        register(LiteAgentExecutionCompletedEvent, on_lite_agent_completed)
        register(ToolUsageStartedEvent, on_tool_started)
        register(ToolUsageFinishedEvent, on_tool_finished)
        register(ToolUsageErrorEvent, on_tool_error)
        register(AgentReasoningCompletedEvent, on_reasoning_completed)

    # ── Synap's compaction in place of CrewAI's history ─────────────────

    def _register_history_hook(self) -> None:
        """Put :meth:`_before_llm_call` on CrewAI's global LLM hook list.

        An executor snapshots that list when it is built, so the hook has to
        be registered before the crew kicks off, which building the listener
        guarantees. A CrewAI without LLM hooks simply keeps its own history.
        """
        try:
            from crewai.hooks.llm_hooks import register_before_llm_call_hook
        except ImportError:
            logger.debug("synap_crewai: this CrewAI has no before_llm_call hooks")
            return
        register_before_llm_call_hook(self._before_llm_call)
        self._hook_registered = True

    def _before_llm_call(self, context: Any) -> None:
        """Replace the turns CrewAI holds with Synap's compaction, once per run.

        Only the first model call of an executor: the history is fixed by then,
        and later calls are the tool loop, which CrewAI's own context-window
        handling still covers. Always returns ``None``: ``False`` would block
        the call.
        """
        try:
            if getattr(context, "iterations", 0) != 0:
                return None
            messages = context.messages
            if not isinstance(messages, list) or not messages:
                return None
            if any(is_synap_history(_message_text(m)) for m in messages):
                return None  # already applied, by this listener or another
            history = run_async(
                fetch_compacted_history(self._sdk, self._conversation_id)
            )
            if history is None:
                return None
            history = self._without_current_inputs(history)
            messages[:] = splice_history(
                messages,
                history,
                role_of=_message_role,
                text_of=_message_text,
                make_message=lambda role, text: {"role": role, "content": text},
            )
            _drop_static_short_term(messages)
        except Exception:  # noqa: BLE001 — CrewAI's own history still stands
            logger.debug("synap_crewai: history splice failed", exc_info=True)
        return None

    def _without_current_inputs(self, history: Any) -> Any:
        """Synap already holds this kickoff's inputs as the newest user turn.

        CrewAI is about to send the same request, rendered into the task
        prompt, as the current turn; the recorded copy would be a second one.
        """
        now = " ".join(self._last_inputs.split())
        turns = list(history.turns)
        if now and turns and turns[-1][0] == "user" \
                and " ".join(turns[-1][1].split()) == now:
            turns.pop()
            return dataclasses.replace(history, turns=tuple(turns))
        return history

    # ── shared by the handlers above ────────────────────────────────────

    def _report_answer(self, output: Any) -> None:
        """Report the run's answer.

        This is the event anticipation acts on: the turn has ended, so the
        next one can be predicted and its context pushed before the user asks
        again. An integration that reports tools and not this gets no
        anticipation at all.
        """
        text = _output_text(output)
        if not text:
            return
        self._write_turn("assistant", text)

    def _write_turn(self, role: str, text: str) -> None:
        """Write a turn: on the stream (opened if need be), over REST if it cannot.

        Same path as every other report, so a turn is never written twice and
        never depends on whether the caller thought to open a stream. Never
        raises.
        """
        self._dispatch(
            record_turn(self._sdk, role=role, content=text, **self._ids())
        )

    def _report_tool_outcome(self, event: Any, result: Any) -> None:
        if not self._report_tool_results:
            return
        name = str(getattr(event, "tool_name", "") or "")
        # The bus stamps an ending event with the id of the starting event it
        # closes, so this is the same id the call went out with. It is set by
        # the bus rather than by us, which is why it holds no matter which
        # order the two handlers happen to run in.
        call_id = str(getattr(event, "started_event_id", "") or "")
        if not call_id:
            # Only reached when the bus could not pair them, which happens when
            # the ending event already carries a `parent_event_id`. ⚠ This half
            # is order-dependent: CrewAI runs each handler on a worker thread
            # of its own, so an ending event that arrived before its start was
            # handled finds nothing here and the result goes out unpaired.
            # A real tool runs between the two events, so the window is small,
            # and the cost when it loses is one result the anticipation agent
            # cannot tie to its call rather than a wrong pairing.
            call_id = self._calls.pop(name, "")
        else:
            self._calls.pop(name, None)
        self._dispatch(
            report_tool_result(
                self._sdk,
                result=result,
                tool_name=name,
                tool_call_id=call_id,
                **self._ids(),
            )
        )


async def report_user_turn(
    sdk: MaximemSynapSDK,
    content: str,
    *,
    conversation_id: str,
    user_id: str,
    customer_id: str = "",
) -> bool:
    """Write what the user actually said, before you kick the crew off.

    Goes on the ``listen()`` stream when one is open and over REST when not,
    so the turn is kept either way. Returns whether it was written.

    CrewAI has no user turn. A crew is kicked off with ``inputs``, which are
    template variables rather than a message, and
    :class:`SynapCrewAIListener` reports those because a conversation of
    answers with no questions is not much to anticipate from. When you have
    the real request, send it with this instead and build the listener with
    ``report_kickoff_inputs=False``.
    """
    return await record_turn(
        sdk,
        role="user",
        content=content or "",
        conversation_id=conversation_id,
        user_id=user_id,
        customer_id=customer_id,
    )


def _foreign_stream(sdk: Any) -> bool:
    """A stream the caller opened, on a loop this package was not told about.

    One the common helper opened records its loop; one opened by the caller
    does not, and cannot be written from the background loop.
    """
    if not stream_is_active(sdk):
        return False
    try:
        state = vars(sdk).get("_synap_auto_stream")
    except TypeError:
        return False
    return not (isinstance(state, dict) and state.get("loop") is not None)


def _message_role(message: Any) -> str:
    role = message.get("role") if isinstance(message, dict) else getattr(message, "role", "")
    return str(role or "")


def _message_text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content blocks
        return "\n".join(
            str(b.get("text", "")) for b in content if isinstance(b, dict)
        )
    return ""


def _drop_static_short_term(messages: List[Any]) -> None:
    """Remove the backstory snapshot from the system prompt.

    ``build_synap_st_backstory`` can only run when the agent is built; the
    history just spliced in is the same conversation, current.
    """
    for message in messages:
        if _message_role(message) != "system" or not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and "<synap_short_term_context>" in content:
            message["content"] = _ST_BLOCK.sub("", content)


def _running_loop() -> Optional[asyncio.AbstractEventLoop]:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _swallow(future: Any) -> None:
    """Read the result so a failed report is never an unretrieved exception.

    ``stream_events`` swallows everything already, so reaching the except here
    means the loop itself went away mid-report.
    """
    try:
        future.result()
    except Exception:  # noqa: BLE001
        logger.debug("synap_crewai: a report did not complete", exc_info=True)


def _inputs_text(inputs: Any) -> str:
    """The crew's kickoff inputs as one string.

    A single input is the request itself. Several are labelled, because
    ``{"city": "Paris", "days": 3}`` means nothing without its keys.
    """
    if inputs is None:
        return ""
    if isinstance(inputs, str):
        return inputs.strip()
    if not isinstance(inputs, dict) or not inputs:
        return ""
    if len(inputs) == 1:
        return str(next(iter(inputs.values()))).strip()
    return "\n".join(f"{key}: {value}" for key, value in inputs.items()).strip()


def _output_text(output: Any) -> str:
    """The answer text out of whatever CrewAI produced.

    ``CrewOutput`` carries the answer on ``.raw``; a ``LiteAgent`` hands back
    a plain string.
    """
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    raw = getattr(output, "raw", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return str(output).strip()


def _tool_args(tool_args: Any) -> Optional[dict]:
    """``tool_args`` as a dict.

    CrewAI types the field as ``dict | str``, and a string there is the
    argument the model wrote, so it is passed through under one key rather
    than dropped.
    """
    if isinstance(tool_args, dict):
        return tool_args
    if isinstance(tool_args, str) and tool_args.strip():
        return {"input": tool_args}
    return None


__all__ = ["SynapCrewAIListener", "report_user_turn"]
