"""Tests for create_synap_recorder — Smolagents step-callback turn recorder."""

from types import SimpleNamespace

import pytest

from synap_smolagents import create_synap_recorder
from synap_smolagents.callbacks import _as_text


def step(model_output, step_number=1, error=None):
    return SimpleNamespace(
        model_output=model_output, step_number=step_number, error=error
    )


# ── validation ───────────────────────────────────────────────────────────────


def test_requires_sdk():
    with pytest.raises(ValueError):
        create_synap_recorder(None, "alice", "conv")


def test_requires_user_id(mock_sdk):
    with pytest.raises(ValueError):
        create_synap_recorder(mock_sdk, "", "conv")


def test_requires_conversation_id(mock_sdk):
    with pytest.raises(ValueError):
        create_synap_recorder(mock_sdk, "alice", "")


# ── recording ────────────────────────────────────────────────────────────────


def test_records_action_step_with_per_step_doc_id(mock_sdk):
    recorder = create_synap_recorder(mock_sdk, "alice", "conv1", customer_id="acme")
    recorder(step("agent did a thing", step_number=2), agent=None)
    mock_sdk.memories.create.assert_called_once()
    kwargs = mock_sdk.memories.create.call_args.kwargs
    assert kwargs["document"] == "agent did a thing"
    assert kwargs["document_id"] == "smolagents-conv1-2"
    assert kwargs["document_type"] == "ai-chat-conversation"
    assert kwargs["user_id"] == "alice"
    assert kwargs["customer_id"] == "acme"
    assert kwargs["metadata"]["step"] == 2


def test_per_step_ids_differ_across_steps(mock_sdk):
    recorder = create_synap_recorder(mock_sdk, "alice", "conv1")
    recorder(step("s1", step_number=1))
    recorder(step("s2", step_number=2))
    ids = [c.kwargs["document_id"] for c in mock_sdk.memories.create.call_args_list]
    assert ids == ["smolagents-conv1-1", "smolagents-conv1-2"]


def test_skips_error_steps(mock_sdk):
    recorder = create_synap_recorder(mock_sdk, "alice", "conv1")
    recorder(step("output", error=ValueError("boom")))
    mock_sdk.memories.create.assert_not_called()


def test_skips_empty_output(mock_sdk):
    recorder = create_synap_recorder(mock_sdk, "alice", "conv1")
    recorder(step(""))
    recorder(step(None))
    recorder(step("   "))
    mock_sdk.memories.create.assert_not_called()


def test_coerces_list_model_output(mock_sdk):
    recorder = create_synap_recorder(mock_sdk, "alice", "conv1")
    recorder(step([{"type": "text", "text": "hello"}, {"text": "world"}]))
    assert mock_sdk.memories.create.call_args.kwargs["document"] == "hello\nworld"


def test_failure_is_logged_not_raised(failing_sdk):
    recorder = create_synap_recorder(failing_sdk, "alice", "conv1")
    # must NOT raise — a raising callback would abort the agent run
    recorder(step("something"))


# ── field-name guard against upstream renames ────────────────────────────────


def test_actionstep_has_expected_fields():
    from smolagents.memory import ActionStep

    fields = set(ActionStep.__dataclass_fields__)
    assert {"model_output", "error", "step_number"} <= fields


def test_as_text_helper():
    assert _as_text(None) == ""
    assert _as_text("hi") == "hi"
    assert _as_text([{"text": "a"}, "b"]) == "a\nb"


# ───────────────────────────────────────────────────────────────────────────
# Stream reporting — the five events the recorder never put on the wire
#
# Contract (from callbacks.py and stream_events.py):
# - silent when no stream is running, so nothing changes for a caller who
#   has not opted in;
# - never raises, because a raising step callback aborts the agent run;
# - a tool call and its result share a tool_call_id.
# ───────────────────────────────────────────────────────────────────────────

from unittest.mock import AsyncMock, MagicMock  # noqa: E402


def _streaming_sdk(is_listening=True):
    sdk = MagicMock()
    sdk.instance.is_listening = is_listening
    sdk.instance.send_message = AsyncMock()
    sdk.instance.record_thinking = AsyncMock()
    sdk.instance.record_tool_call = AsyncMock()
    sdk.instance.record_tool_result = AsyncMock()
    sdk.memories.create = AsyncMock()
    return sdk


def _recorder_for(sdk, **kwargs):
    return create_synap_recorder(
        sdk, "alice", "conv1", customer_id="acme", **kwargs
    )


def _action_step(**overrides):
    fields = {
        "step_number": 1,
        "model_output": "Thought: look it up",
        "tool_calls": None,
        "observations": None,
        "error": None,
        "is_final_answer": False,
        "action_output": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _tool_call(name="lookup_order", arguments=None, id="call_7"):
    return SimpleNamespace(name=name, arguments=arguments, id=id)


def _final_answer_step(output):
    return SimpleNamespace(output=output)


class TestStreamOnlyNoRestDoubleWrite:
    def test_no_stream_means_no_events_at_all(self):
        sdk = _streaming_sdk(is_listening=False)
        _recorder_for(sdk)(_action_step(), agent=None)
        sdk.instance.send_message.assert_not_awaited()
        sdk.instance.record_thinking.assert_not_awaited()

    def test_the_step_ingest_still_happens_without_a_stream(self):
        """The recorder's original behaviour is untouched."""
        sdk = _streaming_sdk(is_listening=False)
        _recorder_for(sdk)(_action_step(), agent=None)
        sdk.memories.create.assert_called_once()

    def test_ingest_steps_can_be_turned_off_without_touching_the_stream(self):
        """A caller who does not want the same text extracted twice while a
        stream is running turns the document ingest off, not the reporting."""
        sdk = _streaming_sdk()
        _recorder_for(sdk, ingest_steps=False)(_action_step(), agent=None)
        sdk.memories.create.assert_not_called()
        sdk.instance.record_thinking.assert_awaited_once()

    def test_the_recorder_never_calls_record_message(self):
        """Stream first, and never a REST conversation write on top of it:
        the server persists user_message/assistant_message from the stream,
        so a record_message here would store and extract the turn twice."""
        sdk = _streaming_sdk()
        sdk.conversation.record_message = AsyncMock()
        rec = _recorder_for(sdk)
        rec(_action_step(), agent=SimpleNamespace(task="where is my order"))
        rec(_final_answer_step("it ships tomorrow"), agent=None)
        sdk.conversation.record_message.assert_not_awaited()


class TestTheFiveEvents:
    def test_the_user_turn_comes_from_agent_task(self):
        sdk = _streaming_sdk()
        _recorder_for(sdk)(
            _action_step(), agent=SimpleNamespace(task="where is my order")
        )
        kw = sdk.instance.send_message.await_args.kwargs
        assert kw["event_type"] == "user_message"
        assert kw["content"] == "where is my order"

    def test_the_user_turn_is_reported_once_per_task(self):
        sdk = _streaming_sdk()
        agent = SimpleNamespace(task="where is my order")
        rec = _recorder_for(sdk)
        rec(_action_step(step_number=1), agent=agent)
        rec(_action_step(step_number=2), agent=agent)
        user_turns = [
            c for c in sdk.instance.send_message.await_args_list
            if c.kwargs["event_type"] == "user_message"
        ]
        assert len(user_turns) == 1

    def test_a_new_task_reports_a_new_user_turn(self):
        sdk = _streaming_sdk()
        rec = _recorder_for(sdk)
        rec(_action_step(), agent=SimpleNamespace(task="first"))
        rec(_action_step(), agent=SimpleNamespace(task="second"))
        contents = [
            c.kwargs["content"] for c in sdk.instance.send_message.await_args_list
            if c.kwargs["event_type"] == "user_message"
        ]
        assert contents == ["first", "second"]

    def test_report_user_task_can_be_turned_off(self):
        sdk = _streaming_sdk()
        _recorder_for(sdk, report_user_task=False)(
            _action_step(), agent=SimpleNamespace(task="where is my order")
        )
        assert not [
            c for c in sdk.instance.send_message.await_args_list
            if c.kwargs["event_type"] == "user_message"
        ]

    def test_the_assistant_turn_comes_from_the_final_answer_step(self):
        """This is the anticipation moment: the turn has ended."""
        sdk = _streaming_sdk()
        _recorder_for(sdk)(_final_answer_step("it ships tomorrow"), agent=None)
        kw = sdk.instance.send_message.await_args.kwargs
        assert kw["event_type"] == "assistant_message"
        assert kw["role"] == "assistant"
        assert kw["content"] == "it ships tomorrow"

    def test_a_final_action_step_also_reports_the_assistant_turn(self):
        """Registered against ActionStep alone, a caller still gets the turn."""
        sdk = _streaming_sdk()
        _recorder_for(sdk)(
            _action_step(is_final_answer=True, action_output="it ships tomorrow"),
            agent=None,
        )
        assert [
            c.kwargs["content"] for c in sdk.instance.send_message.await_args_list
            if c.kwargs["event_type"] == "assistant_message"
        ] == ["it ships tomorrow"]

    def test_the_answer_is_not_reported_twice_by_both_steps(self):
        """_run_stream finalises the final ActionStep and then a
        FinalAnswerStep carrying the same answer."""
        sdk = _streaming_sdk()
        rec = _recorder_for(sdk)
        rec(_action_step(is_final_answer=True, action_output="42"), agent=None)
        rec(_final_answer_step("42"), agent=None)
        assert len([
            c for c in sdk.instance.send_message.await_args_list
            if c.kwargs["event_type"] == "assistant_message"
        ]) == 1

    def test_a_tool_call_is_reported_with_the_frameworks_own_id(self):
        sdk = _streaming_sdk()
        _recorder_for(sdk)(
            _action_step(tool_calls=[_tool_call(arguments={"pnr": "QX41RT"})]),
            agent=None,
        )
        kw = sdk.instance.record_tool_call.await_args.kwargs
        assert kw["tool_call_id"] == "call_7"
        assert sdk.instance.record_tool_call.await_args.args[0] == "lookup_order"
        assert sdk.instance.record_tool_call.await_args.args[1] == {"pnr": "QX41RT"}

    def test_a_code_agent_string_argument_is_carried_not_dropped(self):
        sdk = _streaming_sdk()
        _recorder_for(sdk)(
            _action_step(tool_calls=[
                _tool_call(name="python_interpreter", arguments="print(1)")
            ]),
            agent=None,
        )
        assert sdk.instance.record_tool_call.await_args.args[1] == {"input": "print(1)"}

    def test_the_call_and_its_result_share_an_id(self):
        """Without this the anticipation agent sees a call and a result and
        cannot tell they are the same tool invocation."""
        sdk = _streaming_sdk()
        _recorder_for(sdk)(
            _action_step(
                tool_calls=[_tool_call()], observations="status: shipped"
            ),
            agent=None,
        )
        assert (
            sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
            == sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"]
        )

    def test_parallel_calls_report_a_result_with_no_id_rather_than_a_guess(self):
        """Smolagents merges every call's output into one observations string,
        so attributing it to one of them would be invented."""
        sdk = _streaming_sdk()
        _recorder_for(sdk)(
            _action_step(
                tool_calls=[_tool_call(id="a"), _tool_call(id="b")],
                observations="one\ntwo",
            ),
            agent=None,
        )
        assert sdk.instance.record_tool_call.await_count == 2
        assert "tool_call_id" not in sdk.instance.record_tool_result.await_args.kwargs

    def test_report_tool_results_can_be_turned_off(self):
        sdk = _streaming_sdk()
        _recorder_for(sdk, report_tool_results=False)(
            _action_step(tool_calls=[_tool_call()], observations="status: shipped"),
            agent=None,
        )
        sdk.instance.record_tool_result.assert_not_awaited()

    def test_a_plan_is_reported_as_reasoning(self):
        sdk = _streaming_sdk()
        _recorder_for(sdk)(
            SimpleNamespace(plan="First look up the booking"), agent=None
        )
        sdk.instance.record_thinking.assert_awaited_once()
        assert (
            sdk.instance.record_thinking.await_args.args[0]
            == "First look up the booking"
        )
        assert sdk.instance.record_thinking.await_args.kwargs["thought_type"] == "plan"

    def test_the_model_output_is_reported_as_reasoning(self):
        sdk = _streaming_sdk()
        _recorder_for(sdk)(_action_step(model_output="Thought: look it up"), agent=None)
        assert (
            sdk.instance.record_thinking.await_args.kwargs["thought_type"] == "action"
        )

    def test_an_empty_plan_reports_nothing(self):
        sdk = _streaming_sdk()
        _recorder_for(sdk)(SimpleNamespace(plan="   "), agent=None)
        sdk.instance.record_thinking.assert_not_awaited()


class TestNoneOfItAbortsTheRun:
    """The recorder runs in a `finally` with no surrounding try/except, so a
    raising callback takes down the whole agent run."""

    def test_a_dead_stream_does_not_raise(self):
        sdk = _streaming_sdk()
        sdk.instance.send_message = AsyncMock(side_effect=RuntimeError("boom"))
        sdk.instance.record_tool_call = AsyncMock(side_effect=RuntimeError("boom"))
        _recorder_for(sdk)(
            _action_step(tool_calls=[_tool_call()], observations="x"),
            agent=SimpleNamespace(task="t"),
        )

    def test_an_sdk_with_no_instance_namespace_does_not_raise(self):
        sdk = MagicMock(spec=["memories"])
        sdk.memories.create = AsyncMock()
        _recorder_for(sdk)(_action_step(), agent=SimpleNamespace(task="t"))

    def test_an_agent_without_a_task_does_not_raise(self):
        sdk = _streaming_sdk()
        _recorder_for(sdk)(_action_step(), agent=SimpleNamespace())

    def test_a_step_whose_field_explodes_does_not_raise(self):
        """stream_events swallows its own failures; this covers the bridge
        around them, which reads the step's fields."""
        class Exploding:
            step_number = 1
            model_output = None
            tool_calls = None
            error = None

            @property
            def observations(self):
                raise RuntimeError("boom")

        sdk = _streaming_sdk()
        _recorder_for(sdk)(Exploding(), agent=None)


# ── field-name guards against upstream renames ───────────────────────────────


def test_the_step_fields_the_stream_reads_still_exist():
    from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep, ToolCall

    assert {"tool_calls", "observations", "is_final_answer", "action_output"} <= set(
        ActionStep.__dataclass_fields__
    )
    assert "plan" in PlanningStep.__dataclass_fields__
    assert "output" in FinalAnswerStep.__dataclass_fields__
    assert {"name", "arguments", "id"} <= set(ToolCall.__dataclass_fields__)


def test_registering_against_memorystep_is_what_delivers_every_step():
    """The registry walks __mro__, so MemoryStep catches all three types.
    Registered against ActionStep the final answer never arrives."""
    from smolagents.memory import ActionStep, FinalAnswerStep, MemoryStep, PlanningStep

    assert issubclass(ActionStep, MemoryStep)
    assert issubclass(PlanningStep, MemoryStep)
    assert issubclass(FinalAnswerStep, MemoryStep)

    from synap_smolagents import synap_step_callbacks

    mapping = synap_step_callbacks(MagicMock(), "alice", "conv1")
    assert list(mapping) == [MemoryStep]


def test_a_whole_run_of_real_smolagents_steps_reports_the_whole_turn():
    """Doubles cannot catch an upstream shape change; real steps can.

    This is the sequence _run_stream actually produces: a plan, an action
    step, the final action step, and the FinalAnswerStep that repeats its
    answer.
    """
    from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep, ToolCall
    from smolagents.monitoring import Timing

    sdk = _streaming_sdk()
    agent = MagicMock()
    agent.task = "where is my order"
    rec = _recorder_for(sdk)

    rec(PlanningStep(model_input_messages=[], model_output_message=None,
                     plan="First look up the booking",
                     timing=Timing(start_time=0.0)), agent=agent)
    rec(ActionStep(step_number=1, timing=Timing(start_time=0.0),
                   model_output="Thought: look it up",
                   tool_calls=[ToolCall(name="lookup_order",
                                        arguments={"pnr": "QX"}, id="call_1")],
                   observations="status: shipped"), agent=agent)
    rec(ActionStep(step_number=2, timing=Timing(start_time=0.0),
                   is_final_answer=True,
                   action_output="it ships tomorrow"), agent=agent)
    rec(FinalAnswerStep(output="it ships tomorrow"), agent=agent)

    assert [(c.kwargs["event_type"], c.kwargs["content"])
            for c in sdk.instance.send_message.await_args_list] == [
        ("user_message", "where is my order"),
        ("assistant_message", "it ships tomorrow"),
    ]
    assert [c.kwargs["thought_type"]
            for c in sdk.instance.record_thinking.await_args_list] == ["plan", "action"]
    assert sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"] == "call_1"
    assert sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"] == "call_1"
