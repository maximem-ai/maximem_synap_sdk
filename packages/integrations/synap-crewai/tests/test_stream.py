"""Tests for SynapCrewAIListener — the live-stream tap for CrewAI.

These drive CrewAI's **real** event bus rather than calling the handlers
directly, because the two things most likely to be wrong are not in our
mapping:

- whether a tool call and its result really do arrive carrying a shared id
  (CrewAI stamps an ending event with the ``event_id`` of the starting event
  it closes, and that pairing is the bus's, not ours), and
- whether a report reaches the loop the Synap stream lives on. The bus runs
  handlers on its own threads, and an ``async def`` handler would look correct
  and silently report nothing.

The rest of the contract, from ``stream_events``:

- **Stream first, and never a second write.** The server persists
  ``user_message`` and ``assistant_message`` from the stream itself, so an
  integration that reports a turn on the stream and also calls
  ``sdk.conversation.record_message`` writes the turn twice and extracts it
  twice. This integration is stream-only, so ``record_message`` must never be
  called at all.
- **Silent when no stream is running.**
- **Never raises, and never blocks**, because a handler runs inside CrewAI's
  own dispatch.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from crewai.events import (
    AgentReasoningCompletedEvent,
    CrewKickoffCompletedEvent,
    CrewKickoffStartedEvent,
    LiteAgentExecutionCompletedEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
    crewai_event_bus,
)

from synap_crewai.stream import (
    SynapCrewAIListener,
    _inputs_text,
    _output_text,
    _tool_args,
    report_user_turn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _streaming_sdk(is_listening: bool = True):
    sdk = MagicMock()
    sdk.instance.is_listening = is_listening
    sdk.instance.send_message = AsyncMock()
    sdk.instance.record_thinking = AsyncMock()
    sdk.instance.record_tool_call = AsyncMock()
    sdk.instance.record_tool_result = AsyncMock()
    sdk.conversation.record_message = AsyncMock()
    return sdk


def _listener(sdk, **kwargs):
    return SynapCrewAIListener(
        sdk, user_id="user-1", conversation_id="conv-1", customer_id="cust-1", **kwargs
    )


async def _settle(predicate=None, timeout: float = 3.0) -> None:
    """Let the bus's worker thread run and our reports land on this loop.

    A handler hands its report to this loop and returns, so the test has to
    yield before the report has happened.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(0.01)
        if predicate is not None and predicate():
            return
    if predicate is not None:
        await asyncio.sleep(0.05)


def _emit(event) -> None:
    crewai_event_bus.emit(object(), event)


async def _tool_call_landed(sdk) -> None:
    """Wait until the tool call has been reported before ending the tool.

    CrewAI hands every event to a worker thread of its own, so emitting
    ``tool_usage_started`` and ``tool_usage_finished`` back to back can run the
    two handlers in either order. Real runs do not look like that: a tool has
    to execute between the two events. Waiting here keeps the test about the
    pairing logic instead of about the bus's thread scheduling, which it
    cannot control and is not trying to pin.
    """
    await _settle(lambda: sdk.instance.record_tool_call.await_count)


def _both_legs(sdk):
    """A call AND its result have both been reported."""
    return (
        sdk.instance.record_tool_call.await_count
        and sdk.instance.record_tool_result.await_count
    )


def _started(tool_name="lookup_order", tool_args=None):
    return ToolUsageStartedEvent(
        tool_name=tool_name, tool_args=tool_args if tool_args is not None else {}
    )


def _finished(tool_name="lookup_order", output="shipped", tool_args=None):
    return ToolUsageFinishedEvent(
        tool_name=tool_name,
        tool_args=tool_args if tool_args is not None else {},
        started_at=datetime.now(),
        finished_at=datetime.now(),
        output=output,
    )


def _errored(tool_name="lookup_order", error="nope"):
    return ToolUsageErrorEvent(
        tool_name=tool_name, tool_args={}, error=RuntimeError(error)
    )


@pytest.fixture
async def live():
    """A listener wired to a listening SDK, unregistered afterwards.

    Async on purpose: the listener captures the loop it is built on, and that
    has to be the loop the test runs on.

    CrewAI's bus is a process-wide singleton, so a listener left registered
    would double-report in every test that follows.
    """
    sdk = _streaming_sdk(True)
    listener = _listener(sdk)
    try:
        yield sdk, listener
    finally:
        listener.stop()


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_init_raises_on_none_sdk():
    with pytest.raises(ValueError, match="non-None sdk"):
        SynapCrewAIListener(None, user_id="u", conversation_id="c")


def test_init_raises_on_empty_user_id():
    with pytest.raises(ValueError, match="non-empty user_id"):
        SynapCrewAIListener(_streaming_sdk(), user_id="", conversation_id="c")


def test_init_raises_on_empty_conversation_id():
    with pytest.raises(ValueError, match="non-empty conversation_id"):
        SynapCrewAIListener(_streaming_sdk(), user_id="u", conversation_id="")


def test_a_failed_construction_registers_nothing():
    """Validation runs before the handlers go on the global bus."""
    before = len(crewai_event_bus._sync_handlers.get(ToolUsageStartedEvent, ()))
    with pytest.raises(ValueError):
        SynapCrewAIListener(_streaming_sdk(), user_id="", conversation_id="c")
    assert len(crewai_event_bus._sync_handlers.get(ToolUsageStartedEvent, ())) == before


# ---------------------------------------------------------------------------
# Stream first, and never a second write
# ---------------------------------------------------------------------------


class TestStreamFirstAndNeverBoth:
    @pytest.mark.asyncio
    async def test_the_crews_inputs_go_out_as_the_user_turn(self, live):
        sdk, _ = live
        _emit(CrewKickoffStartedEvent(crew_name="c", inputs={"q": "where is my order"}))
        await _settle(lambda: sdk.instance.send_message.await_count)
        kw = sdk.instance.send_message.await_args.kwargs
        assert kw["event_type"] == "user_message"
        assert kw["role"] == "user"
        assert kw["content"] == "where is my order"

    @pytest.mark.asyncio
    async def test_the_crews_answer_is_the_anticipation_moment(self, live):
        sdk, _ = live
        _emit(
            CrewKickoffCompletedEvent(
                crew_name="c", output=MagicMock(raw="it ships tomorrow")
            )
        )
        await _settle(lambda: sdk.instance.send_message.await_count)
        kw = sdk.instance.send_message.await_args.kwargs
        assert kw["event_type"] == "assistant_message"
        assert kw["content"] == "it ships tomorrow"

    @pytest.mark.asyncio
    async def test_nothing_is_ALSO_written_over_rest(self, live):
        """The double-write the stream-first rule exists to prevent."""
        sdk, _ = live
        _emit(CrewKickoffStartedEvent(crew_name="c", inputs={"q": "hi"}))
        _emit(CrewKickoffCompletedEvent(crew_name="c", output="done"))
        await _settle(lambda: sdk.instance.send_message.await_count >= 2)
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_standalone_agent_run_reports_its_answer(self, live):
        """`agent.kickoff()` never goes through a crew, so this is the only
        place its answer appears."""
        sdk, _ = live
        _emit(
            LiteAgentExecutionCompletedEvent(agent_info={}, output="the lite answer")
        )
        await _settle(lambda: sdk.instance.send_message.await_count)
        assert (
            sdk.instance.send_message.await_args.kwargs["content"] == "the lite answer"
        )

    @pytest.mark.asyncio
    async def test_with_no_stream_nothing_goes_out_anywhere(self):
        """Someone who never called listen() must see no change at all."""
        sdk = _streaming_sdk(False)
        listener = _listener(sdk)
        try:
            _emit(CrewKickoffStartedEvent(crew_name="c", inputs={"q": "hi"}))
            _emit(_started())
            _emit(_finished())
            _emit(CrewKickoffCompletedEvent(crew_name="c", output="done"))
            await _settle(timeout=0.4)
            sdk.instance.send_message.assert_not_awaited()
            sdk.instance.record_tool_call.assert_not_awaited()
            sdk.conversation.record_message.assert_not_awaited()
        finally:
            listener.stop()

    @pytest.mark.asyncio
    async def test_the_ids_ride_on_every_event(self, live):
        sdk, _ = live
        _emit(CrewKickoffCompletedEvent(crew_name="c", output="done"))
        await _settle(lambda: sdk.instance.send_message.await_count)
        kw = sdk.instance.send_message.await_args.kwargs
        assert kw["conversation_id"] == "conv-1"
        assert kw["user_id"] == "user-1"
        assert kw["customer_id"] == "cust-1"


# ---------------------------------------------------------------------------
# The whole turn, not just its two ends
# ---------------------------------------------------------------------------


class TestTheWholeTurn:
    @pytest.mark.asyncio
    async def test_a_tool_call_is_reported(self, live):
        sdk, _ = live
        _emit(_started(tool_args={"pnr": "QX41RT"}))
        await _settle(lambda: sdk.instance.record_tool_call.await_count)
        args, kwargs = sdk.instance.record_tool_call.await_args
        assert args[0] == "lookup_order"
        assert args[1] == {"pnr": "QX41RT"}
        assert kwargs["tool_call_id"]

    @pytest.mark.asyncio
    async def test_a_tool_result_is_reported(self, live):
        sdk, _ = live
        _emit(_started())
        await _tool_call_landed(sdk)
        _emit(_finished(output="shipped"))
        await _settle(lambda: sdk.instance.record_tool_result.await_count)
        args, kwargs = sdk.instance.record_tool_result.await_args
        assert args[0] == "shipped"
        assert kwargs["tool_name"] == "lookup_order"

    @pytest.mark.asyncio
    async def test_the_call_and_its_result_share_an_id(self, live):
        """Without this the anticipation agent sees a call and a result and
        cannot tell they are the same tool invocation.

        The id comes from CrewAI's own scope pairing: the bus stamps
        ``tool_usage_finished`` with the ``event_id`` of the
        ``tool_usage_started`` it closes.
        """
        sdk, _ = live
        _emit(_started())
        await _tool_call_landed(sdk)
        _emit(_finished())
        await _settle(lambda: _both_legs(sdk))
        call_id = sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
        result_id = sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"]
        assert call_id and call_id == result_id

    @pytest.mark.asyncio
    async def test_the_result_still_finds_its_call_when_the_bus_does_not_pair_them(
        self, live
    ):
        """CrewAI only pairs an ending event that arrives inside the scope its
        start opened. An event carrying its own ``parent_event_id`` skips that
        path entirely and arrives with no ``started_event_id``, and the pairing
        then has to come from us or the result is orphaned.
        """
        sdk, _ = live
        _emit(_started())
        await _tool_call_landed(sdk)
        finished = _finished()
        finished.parent_event_id = "somewhere-else"
        _emit(finished)
        await _settle(lambda: _both_legs(sdk))
        assert finished.started_event_id is None
        call_id = sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
        assert (
            sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"]
            == call_id
        )

    @pytest.mark.asyncio
    async def test_a_tool_that_raised_still_reports_a_result(self, live):
        """A call with no result is a loose end the anticipation agent
        cannot close."""
        sdk, _ = live
        _emit(_started())
        await _tool_call_landed(sdk)
        _emit(_errored(error="upstream 503"))
        await _settle(lambda: _both_legs(sdk))
        args, kwargs = sdk.instance.record_tool_result.await_args
        assert "upstream 503" in args[0]
        assert (
            kwargs["tool_call_id"]
            == sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
        )

    @pytest.mark.asyncio
    async def test_reasoning_is_reported_from_the_plan(self, live):
        sdk, _ = live
        _emit(
            AgentReasoningCompletedEvent(
                agent_role="support",
                task_id="t1",
                plan="look up the booking first",
                ready=True,
            )
        )
        await _settle(lambda: sdk.instance.record_thinking.await_count)
        sdk.instance.record_thinking.assert_awaited_once()
        args, kwargs = sdk.instance.record_thinking.await_args
        assert args[0] == "look up the booking first"
        assert kwargs["thought_type"] == "plan"
        # Ordinal within the run, so the dashboard can replay the timeline in
        # the order the agent actually thought.
        assert kwargs["step_index"] == 1

    @pytest.mark.asyncio
    async def test_reasoning_steps_are_numbered_in_order_within_a_run(self, live):
        sdk, _ = live
        for plan in ("first look it up", "then check the refund window"):
            _emit(
                AgentReasoningCompletedEvent(
                    agent_role="support", task_id="t1", plan=plan, ready=True
                )
            )
        await _settle(lambda: sdk.instance.record_thinking.await_count >= 2)
        steps = [
            c.kwargs["step_index"]
            for c in sdk.instance.record_thinking.await_args_list
        ]
        assert steps == [1, 2]

    @pytest.mark.asyncio
    async def test_a_new_run_starts_the_numbering_again(self, live):
        sdk, _ = live
        _emit(
            AgentReasoningCompletedEvent(
                agent_role="support", task_id="t1", plan="think", ready=True
            )
        )
        await _settle(lambda: sdk.instance.record_thinking.await_count)
        _emit(CrewKickoffStartedEvent(crew_name="c", inputs={"q": "a new question"}))
        await _settle(lambda: sdk.instance.send_message.await_count)
        _emit(
            AgentReasoningCompletedEvent(
                agent_role="support", task_id="t2", plan="think again", ready=True
            )
        )
        await _settle(lambda: sdk.instance.record_thinking.await_count >= 2)
        steps = [
            c.kwargs["step_index"]
            for c in sdk.instance.record_thinking.await_args_list
        ]
        assert steps == [1, 1]

    @pytest.mark.asyncio
    async def test_an_empty_plan_reports_nothing(self, live):
        sdk, _ = live
        _emit(
            AgentReasoningCompletedEvent(
                agent_role="support", task_id="t1", plan="   ", ready=True
            )
        )
        await _settle(timeout=0.3)
        sdk.instance.record_thinking.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tool_results_can_be_turned_off(self):
        """A tool result is the caller's own customer data leaving their
        process, so it has to be refusable."""
        sdk = _streaming_sdk(True)
        listener = _listener(sdk, report_tool_results=False)
        try:
            _emit(_started())
            await _tool_call_landed(sdk)
            _emit(_finished())
            await _settle(lambda: sdk.instance.record_tool_call.await_count)
            await _settle(timeout=0.3)
            sdk.instance.record_tool_call.assert_awaited_once()
            sdk.instance.record_tool_result.assert_not_awaited()
        finally:
            listener.stop()

    @pytest.mark.asyncio
    async def test_kickoff_inputs_can_be_turned_off(self):
        """CrewAI's inputs are template variables, not a message. A caller
        with the real request reports it themselves."""
        sdk = _streaming_sdk(True)
        listener = _listener(sdk, report_kickoff_inputs=False)
        try:
            _emit(CrewKickoffStartedEvent(crew_name="c", inputs={"q": "hi"}))
            await _settle(timeout=0.3)
            sdk.instance.send_message.assert_not_awaited()
        finally:
            listener.stop()

    @pytest.mark.asyncio
    async def test_report_user_turn_sends_what_the_user_actually_said(self):
        sdk = _streaming_sdk(True)
        assert await report_user_turn(
            sdk, "where is my order", conversation_id="c", user_id="u"
        )
        kw = sdk.instance.send_message.await_args.kwargs
        assert kw["event_type"] == "user_message"
        assert kw["content"] == "where is my order"

    @pytest.mark.asyncio
    async def test_report_user_turn_with_no_stream_sends_nothing(self):
        sdk = _streaming_sdk(False)
        assert await report_user_turn(sdk, "hi", conversation_id="c", user_id="u") is False


# ---------------------------------------------------------------------------
# The loop the reports have to land on
# ---------------------------------------------------------------------------


class TestItReportsOnTheListeningLoop:
    """CrewAI dispatches handlers on its own threads and its own event loop.
    A gRPC stream cannot be written from a foreign loop, so a report that ran
    where the handler runs would fail and say nothing about it.
    """

    @pytest.mark.asyncio
    async def test_the_report_runs_on_the_loop_the_listener_captured(self, live):
        sdk, _ = live
        seen = {}

        async def record(*args, **kwargs):
            seen["loop"] = asyncio.get_running_loop()

        sdk.instance.send_message = AsyncMock(side_effect=record)
        _emit(CrewKickoffCompletedEvent(crew_name="c", output="done"))
        await _settle(lambda: "loop" in seen)
        assert seen["loop"] is asyncio.get_running_loop()

    @pytest.mark.asyncio
    async def test_a_listener_built_off_the_loop_warns_and_reports_nothing(self, caplog):
        """Built from synchronous code there is no loop to capture. That is a
        stream that is open and delivering nothing, which is the failure that
        looks like success, so it gets a warning rather than a debug line.
        """
        sdk = _streaming_sdk(True)
        listener = await asyncio.to_thread(
            SynapCrewAIListener, sdk, "user-1", "conv-1"
        )
        try:
            assert listener._loop is None
            with caplog.at_level(logging.WARNING, logger="synap_crewai.stream"):
                _emit(CrewKickoffCompletedEvent(crew_name="c", output="done"))
                await _settle(lambda: bool(caplog.records))
            sdk.instance.send_message.assert_not_awaited()
            assert any("no event loop was captured" in r.message for r in caplog.records)
        finally:
            listener.stop()

    @pytest.mark.asyncio
    async def test_bind_loop_attaches_one_afterwards(self):
        sdk = _streaming_sdk(True)
        listener = await asyncio.to_thread(
            SynapCrewAIListener, sdk, "user-1", "conv-1"
        )
        try:
            listener.bind_loop()
            _emit(CrewKickoffCompletedEvent(crew_name="c", output="done"))
            await _settle(lambda: sdk.instance.send_message.await_count)
            sdk.instance.send_message.assert_awaited_once()
        finally:
            listener.stop()

    @pytest.mark.asyncio
    async def test_a_listener_with_no_loop_and_no_stream_says_nothing(self, caplog):
        """No warning for someone who never opted into streaming: there is
        nothing to report, so nothing is wrong."""
        sdk = _streaming_sdk(False)
        listener = await asyncio.to_thread(
            SynapCrewAIListener, sdk, "user-1", "conv-1"
        )
        try:
            with caplog.at_level(logging.WARNING, logger="synap_crewai.stream"):
                _emit(CrewKickoffCompletedEvent(crew_name="c", output="done"))
                await _settle(timeout=0.4)
            assert caplog.records == []
        finally:
            listener.stop()


# ---------------------------------------------------------------------------
# Nothing here breaks or delays the crew
# ---------------------------------------------------------------------------


class TestNoneOfItBreaksTheCrew:
    @pytest.mark.asyncio
    async def test_a_dead_sdk_does_not_stop_the_reports_that_follow(self, live):
        sdk, _ = live
        sdk.instance.record_tool_call = AsyncMock(side_effect=RuntimeError("boom"))
        _emit(_started())
        _emit(CrewKickoffCompletedEvent(crew_name="c", output="done"))
        await _settle(lambda: sdk.instance.send_message.await_count)
        sdk.instance.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_an_sdk_with_no_instance_namespace_does_not_raise(self):
        """An older SDK, or a test double. Every handler must no-op."""
        sdk = MagicMock(spec=[])
        listener = SynapCrewAIListener(sdk, user_id="u", conversation_id="c")
        try:
            _emit(CrewKickoffStartedEvent(crew_name="c", inputs={"q": "hi"}))
            _emit(_started())
            _emit(_finished())
            _emit(CrewKickoffCompletedEvent(crew_name="c", output="done"))
            await _settle(timeout=0.4)
        finally:
            listener.stop()

    @pytest.mark.asyncio
    async def test_the_handler_does_not_wait_for_the_report(self, live):
        """Telemetry is handed to the listening loop and the handler returns.
        A crew must never wait on our network call.
        """
        sdk, _ = live
        slow = asyncio.Event()

        async def never_finishes(*args, **kwargs):
            await slow.wait()

        sdk.instance.send_message = AsyncMock(side_effect=never_finishes)

        started = time.monotonic()
        _emit(CrewKickoffCompletedEvent(crew_name="c", output="done"))
        # Blocks this loop, so the scheduled report cannot possibly run while
        # we wait: what returns is the handler, not the report.
        crewai_event_bus.flush(timeout=5.0)
        assert time.monotonic() - started < 1.0
        slow.set()
        await _settle(lambda: sdk.instance.send_message.await_count)

    @pytest.mark.asyncio
    async def test_a_report_that_cannot_be_handed_over_is_closed_not_leaked(
        self, live
    ):
        """``is_closed()`` is checked first, but a loop can close in the gap
        between that check and the hand-off, and then
        ``run_coroutine_threadsafe`` raises.

        Two things have to hold. The handler must not raise, because it is
        running inside CrewAI's own dispatch. And the report must be closed
        rather than dropped: a coroutine that is created and never awaited
        leaks and warns, at a moment when the process is already unhappy.
        """
        _, listener = live
        shut = MagicMock()
        shut.is_closed.return_value = False  # claims to still be open
        shut.call_soon_threadsafe.side_effect = RuntimeError("Event loop is closed")
        listener.bind_loop(shut)

        async def report():
            return None

        coro = report()
        listener._dispatch(coro)  # must not raise

        with pytest.raises(RuntimeError, match="cannot reuse"):
            await coro


# ---------------------------------------------------------------------------
# Leaving the global bus as we found it
# ---------------------------------------------------------------------------


class TestStop:
    @pytest.mark.asyncio
    async def test_a_stopped_listener_reports_nothing(self):
        sdk = _streaming_sdk(True)
        listener = _listener(sdk)
        listener.stop()
        _emit(CrewKickoffCompletedEvent(crew_name="c", output="done"))
        await _settle(timeout=0.4)
        sdk.instance.send_message.assert_not_awaited()

    def test_stop_is_safe_to_call_twice(self):
        listener = _listener(_streaming_sdk(True))
        listener.stop()
        listener.stop()

    @pytest.mark.asyncio
    async def test_two_listeners_do_not_leak_into_each_other(self):
        """The bus is a process-wide singleton, so a listener that outlives
        its run would double-report for the next one."""
        first_sdk = _streaming_sdk(True)
        first = _listener(first_sdk)
        first.stop()

        second_sdk = _streaming_sdk(True)
        second = _listener(second_sdk)
        try:
            _emit(CrewKickoffCompletedEvent(crew_name="c", output="done"))
            await _settle(lambda: second_sdk.instance.send_message.await_count)
            second_sdk.instance.send_message.assert_awaited_once()
            first_sdk.instance.send_message.assert_not_awaited()
        finally:
            second.stop()


# ---------------------------------------------------------------------------
# The shape readers
# ---------------------------------------------------------------------------


class TestReadingTheShapes:
    def test_one_input_is_the_request_itself(self):
        assert _inputs_text({"question": "where is my order"}) == "where is my order"

    def test_several_inputs_keep_their_labels(self):
        """`{"city": "Paris", "days": 3}` means nothing without its keys."""
        assert _inputs_text({"city": "Paris", "days": 3}) == "city: Paris\ndays: 3"

    def test_no_inputs_is_nothing_to_report(self):
        assert _inputs_text(None) == ""
        assert _inputs_text({}) == ""

    def test_output_text_prefers_the_raw_answer(self):
        assert _output_text(MagicMock(raw="  the answer  ")) == "the answer"

    def test_output_text_reads_a_plain_string(self):
        assert _output_text("lite answer") == "lite answer"

    def test_output_text_of_nothing_is_empty(self):
        assert _output_text(None) == ""

    def test_tool_args_passes_a_dict_through(self):
        assert _tool_args({"pnr": "QX41"}) == {"pnr": "QX41"}

    def test_tool_args_wraps_a_string_rather_than_dropping_it(self):
        """CrewAI types the field as dict | str, and a string there is the
        argument the model wrote."""
        assert _tool_args("QX41RT") == {"input": "QX41RT"}

    def test_tool_args_of_nothing_is_none(self):
        assert _tool_args(None) is None
        assert _tool_args("") is None
