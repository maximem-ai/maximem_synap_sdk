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

*Silent without a stream.* Everything here needs an active
``sdk.instance.listen()``. Most callers do not have one, and for them every
handler returns immediately. Adding this listener must not change behaviour
for someone who has not opted into streaming.

*Never raises.* A handler runs inside CrewAI's own dispatch. Nothing here
raises into it, and nothing here blocks it: the report is handed to the
listening loop and the handler returns.
"""

from __future__ import annotations

import asyncio
import logging
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
    report_reasoning,
    report_tool_call,
    report_tool_result,
    report_turn,
    stream_is_active,
)

logger = logging.getLogger(__name__)

_NO_LOOP_WARNING = (
    "SynapCrewAIListener: a Synap stream is open but no event loop was "
    "captured, so nothing can be reported. CrewAI runs event handlers on its "
    "own threads, and a gRPC stream can only be written from the loop that "
    "sdk.instance.listen() runs on. Build the listener from inside that loop, "
    "pass loop=..., or call listener.bind_loop() from a coroutine running on "
    "it."
)


class SynapCrewAIListener(BaseEventListener):
    """Reports a CrewAI run on the Synap stream.

    Building it registers the handlers on CrewAI's global event bus, which is
    how CrewAI intends listeners to work. Keep a reference to it: the
    registration lasts as long as the process, and :meth:`stop` is the way to
    undo it.

    Args:
        sdk: A configured :class:`MaximemSynapSDK`. Without an active
            ``listen()`` stream every handler is a no-op.
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
        # Only used when the bus could not pair the two legs of a tool call
        # itself. One tool in flight per name, which is what a sequential
        # agent loop produces.
        self._calls: Dict[str, str] = {}
        self._registered: List[Tuple[type, Any]] = []

        super().__init__()

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
            self._warn_no_loop()
            coro.close()
            return
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception:  # noqa: BLE001 — a handler must never raise
            coro.close()
            logger.debug("synap_crewai: could not schedule a report", exc_info=True)
            return
        future.add_done_callback(_swallow)

    def _warn_no_loop(self) -> None:
        # Loud, and once. A stream that is open and reporting nothing is the
        # failure that looks like success, so it does not get a debug line.
        if self._no_loop_warned:
            return
        self._no_loop_warned = True
        logger.warning(_NO_LOOP_WARNING)

    def _live(self) -> bool:
        """Whether there is anything to report to.

        Checked in the handler rather than only inside the helpers so that a
        caller with no stream pays nothing at all, and so that the missing-loop
        warning fires only for someone who actually opted into streaming.
        """
        return stream_is_active(self._sdk)

    # ── the handlers ────────────────────────────────────────────────────

    def setup_listeners(self, crewai_event_bus: Any) -> None:
        """Subscribe to the bus. Called for us by ``BaseEventListener``."""

        def register(event_type: type, handler: Any) -> None:
            crewai_event_bus.on(event_type)(handler)
            self._registered.append((event_type, handler))

        def on_kickoff_started(source: Any, event: Any) -> None:
            self._step = 0
            if not self._report_kickoff_inputs or not self._live():
                return
            text = _inputs_text(getattr(event, "inputs", None))
            if not text:
                return
            self._dispatch(
                report_turn(self._sdk, role="user", content=text, **self._ids())
            )

        def on_kickoff_completed(source: Any, event: Any) -> None:
            self._report_answer(getattr(event, "output", None))

        def on_lite_agent_completed(source: Any, event: Any) -> None:
            # `agent.kickoff()` runs a LiteAgent, which never goes through a
            # crew, so this is the only place its answer appears.
            self._report_answer(getattr(event, "output", None))

        def on_tool_started(source: Any, event: Any) -> None:
            if not self._live():
                return
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
            if not self._live():
                return
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

    # ── shared by the handlers above ────────────────────────────────────

    def _report_answer(self, output: Any) -> None:
        """Report the run's answer.

        This is the event anticipation acts on: the turn has ended, so the
        next one can be predicted and its context pushed before the user asks
        again. An integration that reports tools and not this gets no
        anticipation at all.
        """
        if not self._live():
            return
        text = _output_text(output)
        if not text:
            return
        self._dispatch(
            report_turn(self._sdk, role="assistant", content=text, **self._ids())
        )

    def _report_tool_outcome(self, event: Any, result: Any) -> None:
        if not self._report_tool_results or not self._live():
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
    """Report what the user actually said, before you kick the crew off.

    CrewAI has no user turn. A crew is kicked off with ``inputs``, which are
    template variables rather than a message, and
    :class:`SynapCrewAIListener` reports those because a conversation of
    answers with no questions is not much to anticipate from. When you have
    the real request, send it with this instead and build the listener with
    ``report_kickoff_inputs=False``.
    """
    return await report_turn(
        sdk,
        role="user",
        content=content or "",
        conversation_id=conversation_id,
        user_id=user_id,
        customer_id=customer_id,
    )


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
