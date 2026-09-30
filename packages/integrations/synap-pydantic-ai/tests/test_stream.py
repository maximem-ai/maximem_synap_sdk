"""Tests for the Pydantic AI live-stream reporting.

The rule these pin is "stream first, REST only as a fallback, never both".
The server persists ``user_message`` and ``assistant_message`` from the
stream itself, so a ``record_message`` on top of a delivered stream event
writes the same turn twice and extracts it twice.

**Why the events here are stand-ins rather than the real classes.**
``pydantic_ai`` is not installed in this repo's venv, and ``stream.py``
deliberately imports nothing from it: it matches on the ``event_kind``
discriminator and reads attributes, so a version that renames a class
degrades to reporting less rather than to an ImportError. The shapes below
were read off ``pydantic-ai-slim`` 2.51.0 (run) and 1.0.10 (source), and the
whole path was driven end to end against 2.51.0 with a real ``Agent``,
``FunctionModel`` and real ``ThinkingPart`` / ``PartDeltaEvent`` objects
before these were written. The one difference between those two versions
that matters is covered below: the tool-result event renamed its payload
from ``result`` to ``part``, and ``PartEndEvent`` does not exist before 2.x.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_pydantic_ai.deps import SynapDeps
from synap_pydantic_ai.stream import (
    SynapEventStreamHandler,
    _sendable,
    _tool_args,
    synap_run,
    synap_run_sync,
)


# ---------------------------------------------------------------------------
# Stand-ins for the Pydantic AI shapes, matching the real ones field for field
# ---------------------------------------------------------------------------


@dataclass
class ToolCallPart:
    tool_name: str
    args: Any = None
    tool_call_id: str = ""
    part_kind: str = "tool-call"

    def args_as_dict(self) -> Dict[str, Any]:
        if isinstance(self.args, dict):
            return self.args
        return json.loads(self.args)


@dataclass
class ToolReturnPart:
    tool_name: str
    content: Any = None
    tool_call_id: str = ""
    part_kind: str = "tool-return"


@dataclass
class ThinkingPart:
    content: str = ""
    part_kind: str = "thinking"


@dataclass
class TextPart:
    content: str = ""
    part_kind: str = "text"


@dataclass
class ThinkingPartDelta:
    content_delta: Optional[str] = None
    part_delta_kind: str = "thinking"


@dataclass
class TextPartDelta:
    content_delta: str = ""
    part_delta_kind: str = "text"


@dataclass
class FunctionToolCallEvent:
    part: ToolCallPart
    event_kind: str = "function_tool_call"

    @property
    def tool_call_id(self) -> str:
        return self.part.tool_call_id


@dataclass
class FunctionToolResultEvent:
    """Pydantic AI 2.x: the payload is ``part``."""

    part: ToolReturnPart
    event_kind: str = "function_tool_result"

    @property
    def tool_call_id(self) -> str:
        return self.part.tool_call_id


@dataclass
class LegacyFunctionToolResultEvent:
    """Pydantic AI 1.x: the same payload, named ``result``."""

    result: ToolReturnPart
    event_kind: str = "function_tool_result"

    @property
    def tool_call_id(self) -> str:
        return self.result.tool_call_id


@dataclass
class PartStartEvent:
    index: int
    part: Any
    event_kind: str = "part_start"


@dataclass
class PartEndEvent:
    """2.x only. 1.x has to build a thought out of deltas alone."""

    index: int
    part: Any
    event_kind: str = "part_end"


@dataclass
class PartDeltaEvent:
    index: int
    delta: Any
    event_kind: str = "part_delta"


@dataclass
class FinalResultEvent:
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    event_kind: str = "final_result"


async def _stream(*events):
    for event in events:
        yield event


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _streaming_sdk(is_listening: bool):
    sdk = MagicMock()
    sdk.instance.is_listening = is_listening
    sdk.instance.send_message = AsyncMock()
    sdk.instance.record_thinking = AsyncMock()
    sdk.instance.record_tool_call = AsyncMock()
    sdk.instance.record_tool_result = AsyncMock()
    sdk.conversation.record_message = AsyncMock()
    return sdk


def _deps(sdk, conversation_id="conv-1"):
    return SynapDeps(
        sdk=sdk, user_id="u1", customer_id="c1", conversation_id=conversation_id,
    )


def _handler(sdk):
    return SynapEventStreamHandler(
        sdk, conversation_id="conv-1", user_id="u1", customer_id="c1",
    )


class FakeAgent:
    """An agent whose ``run`` takes ``event_stream_handler``, as 1.x and 2.x do."""

    def __init__(self, output="Your order shipped.", raises=None):
        self._output = output
        self._raises = raises
        self.calls: List[Dict[str, Any]] = []

    async def run(self, user_prompt=None, *, deps=None, event_stream_handler=None,
                  **kwargs):
        self.calls.append({
            "user_prompt": user_prompt,
            "deps": deps,
            "event_stream_handler": event_stream_handler,
            **kwargs,
        })
        if self._raises is not None:
            raise self._raises
        return MagicMock(output=self._output)


class OldFakeAgent:
    """A Pydantic AI old enough not to have the hook at all."""

    def __init__(self, output="ok"):
        self._output = output
        self.calls: List[Dict[str, Any]] = []

    async def run(self, user_prompt=None, *, deps=None, **kwargs):
        self.calls.append({"user_prompt": user_prompt, "deps": deps, **kwargs})
        return MagicMock(output=self._output)


# ---------------------------------------------------------------------------
# _sendable / _tool_args
# ---------------------------------------------------------------------------


class TestSendable:
    def test_a_string_passes_through(self):
        assert _sendable("shipped") == "shipped"

    def test_a_dict_passes_through(self):
        assert _sendable({"status": "shipped"}) == {"status": "shipped"}

    def test_something_json_cannot_encode_becomes_its_string(self):
        """The SDK JSON-encodes a tool result and the stream helper swallows
        every failure, so passing an unencodable object through is not an
        error anybody sees: it is a result that never gets reported."""

        class Order:
            def __str__(self):
                return "Order(QX41RT)"

        assert _sendable(Order()) == "Order(QX41RT)"

    def test_none_passes_through(self):
        assert _sendable(None) is None


class TestToolArgs:
    def test_a_dict_is_used_as_is(self):
        part = ToolCallPart("lookup_order", args={"pnr": "QX41RT"})
        assert _tool_args(part) == {"pnr": "QX41RT"}

    def test_a_raw_json_string_is_parsed(self):
        """Providers that stream arguments hand over the JSON text."""
        part = ToolCallPart("lookup_order", args='{"pnr": "QX41RT"}')
        assert _tool_args(part) == {"pnr": "QX41RT"}

    def test_malformed_json_yields_nothing_rather_than_raising(self):
        part = ToolCallPart("lookup_order", args="{not json")
        assert _tool_args(part) is None

    def test_json_that_is_not_an_object_yields_nothing(self):
        part = ToolCallPart("lookup_order", args="[1, 2]")
        assert _tool_args(part) is None


# ---------------------------------------------------------------------------
# The three events in between
# ---------------------------------------------------------------------------


class TestTheThreeEventsInBetween:
    @pytest.mark.asyncio
    async def test_a_tool_call_is_reported(self):
        sdk = _streaming_sdk(True)
        await _handler(sdk)(None, _stream(FunctionToolCallEvent(
            ToolCallPart("lookup_order", {"pnr": "QX41RT"}, "call_1"))))
        sdk.instance.record_tool_call.assert_awaited_once()
        assert sdk.instance.record_tool_call.await_args.args[0] == "lookup_order"
        assert sdk.instance.record_tool_call.await_args.args[1] == {"pnr": "QX41RT"}

    @pytest.mark.asyncio
    async def test_a_tool_result_is_reported(self):
        sdk = _streaming_sdk(True)
        await _handler(sdk)(None, _stream(FunctionToolResultEvent(
            ToolReturnPart("lookup_order", "shipped", "call_1"))))
        sdk.instance.record_tool_result.assert_awaited_once()
        assert sdk.instance.record_tool_result.await_args.args[0] == "shipped"
        assert sdk.instance.record_tool_result.await_args.kwargs[
            "tool_name"] == "lookup_order"

    @pytest.mark.asyncio
    async def test_the_call_and_its_result_share_an_id(self):
        """Without it the anticipation agent cannot tell which result belongs
        to which of several tools in flight."""
        sdk = _streaming_sdk(True)
        await _handler(sdk)(None, _stream(
            FunctionToolCallEvent(ToolCallPart("t", {"a": 1}, "call_1")),
            FunctionToolResultEvent(ToolReturnPart("t", "out", "call_1")),
        ))
        assert (sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
                == sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"]
                == "call_1")

    @pytest.mark.asyncio
    async def test_the_1x_result_event_still_reports(self):
        """Pydantic AI 1.x names the payload ``result``, 2.x names it
        ``part``. Both keep ``tool_call_id`` on the event."""
        sdk = _streaming_sdk(True)
        await _handler(sdk)(None, _stream(LegacyFunctionToolResultEvent(
            ToolReturnPart("lookup_order", "shipped", "call_1"))))
        sdk.instance.record_tool_result.assert_awaited_once()
        assert sdk.instance.record_tool_result.await_args.args[0] == "shipped"
        assert sdk.instance.record_tool_result.await_args.kwargs[
            "tool_call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_a_result_json_cannot_encode_is_still_reported(self):
        class Order:
            def __str__(self):
                return "Order(QX41RT)"

        sdk = _streaming_sdk(True)
        await _handler(sdk)(None, _stream(FunctionToolResultEvent(
            ToolReturnPart("lookup_order", Order(), "call_1"))))
        sent = sdk.instance.record_tool_result.await_args.args[0]
        assert sent == "Order(QX41RT)"
        json.dumps(sent)

    @pytest.mark.asyncio
    async def test_reasoning_is_built_from_the_deltas(self):
        """The 1.x shape: no ``PartEndEvent`` exists, so the deltas alone
        have to add up to the thought."""
        sdk = _streaming_sdk(True)
        await _handler(sdk)(None, _stream(
            PartStartEvent(0, ThinkingPart("I should ")),
            PartDeltaEvent(0, ThinkingPartDelta("look up ")),
            PartDeltaEvent(0, ThinkingPartDelta("the booking")),
        ))
        sdk.instance.record_thinking.assert_awaited_once()
        assert sdk.instance.record_thinking.await_args.args[0] == (
            "I should look up the booking")

    @pytest.mark.asyncio
    async def test_a_part_end_corrects_the_accumulated_text(self):
        """2.x sends the finished part as well as the deltas. Adding both
        would report the thought twice over."""
        sdk = _streaming_sdk(True)
        await _handler(sdk)(None, _stream(
            PartStartEvent(0, ThinkingPart("I should ")),
            PartDeltaEvent(0, ThinkingPartDelta("look up the booking")),
            PartEndEvent(0, ThinkingPart("I should look up the booking")),
        ))
        assert sdk.instance.record_thinking.await_args.args[0] == (
            "I should look up the booking")

    @pytest.mark.asyncio
    async def test_reasoning_is_reported_only_once_the_stream_has_ended(self):
        """A thought arrives in pieces, so there is nothing to report until
        the node that produced it is done."""
        sdk = _streaming_sdk(True)
        seen: List[int] = []

        async def events():
            yield PartStartEvent(0, ThinkingPart("half "))
            seen.append(sdk.instance.record_thinking.await_count)
            yield PartDeltaEvent(0, ThinkingPartDelta("a thought"))
            seen.append(sdk.instance.record_thinking.await_count)

        await _handler(sdk)(None, events())
        assert seen == [0, 0]
        assert sdk.instance.record_thinking.await_count == 1

    @pytest.mark.asyncio
    async def test_each_thought_carries_its_own_step_index(self):
        sdk = _streaming_sdk(True)
        await _handler(sdk)(None, _stream(
            PartStartEvent(0, ThinkingPart("first")),
            PartStartEvent(1, ThinkingPart("second")),
        ))
        calls = sdk.instance.record_thinking.await_args_list
        assert [c.args[0] for c in calls] == ["first", "second"]
        assert [c.kwargs["step_index"] for c in calls] == [0, 1]

    @pytest.mark.asyncio
    async def test_the_step_index_keeps_counting_across_nodes(self):
        """Pydantic AI calls the handler once per graph node, so a run's
        reasoning arrives in several streams."""
        sdk = _streaming_sdk(True)
        handler = _handler(sdk)
        await handler(None, _stream(PartStartEvent(0, ThinkingPart("first"))))
        await handler(None, _stream(PartStartEvent(0, ThinkingPart("second"))))
        assert [c.kwargs["step_index"]
                for c in sdk.instance.record_thinking.await_args_list] == [0, 1]

    @pytest.mark.asyncio
    async def test_a_blank_thought_reports_nothing(self):
        sdk = _streaming_sdk(True)
        await _handler(sdk)(None, _stream(PartStartEvent(0, ThinkingPart("   "))))
        sdk.instance.record_thinking.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_text_is_not_mistaken_for_reasoning(self):
        """The answer is the assistant turn, which ``synap_run`` reports from
        the result. Reporting it here too would write it twice."""
        sdk = _streaming_sdk(True)
        await _handler(sdk)(None, _stream(
            PartStartEvent(0, TextPart("Your order ")),
            PartDeltaEvent(0, TextPartDelta("shipped.")),
            PartEndEvent(0, TextPart("Your order shipped.")),
            FinalResultEvent(),
        ))
        sdk.instance.record_thinking.assert_not_awaited()
        sdk.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_handler_never_reports_a_turn(self):
        """It covers the three events in between and nothing else; the turns
        are ``synap_run``'s, and reporting from both is the double-write."""
        sdk = _streaming_sdk(True)
        await _handler(sdk)(None, _stream(
            FunctionToolCallEvent(ToolCallPart("t", {"a": 1}, "call_1")),
            FunctionToolResultEvent(ToolReturnPart("t", "out", "call_1")),
            PartStartEvent(0, ThinkingPart("thinking")),
            FinalResultEvent(),
        ))
        sdk.instance.send_message.assert_not_awaited()
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_nothing_is_reported_when_there_is_no_stream(self):
        sdk = _streaming_sdk(False)
        await _handler(sdk)(None, _stream(
            FunctionToolCallEvent(ToolCallPart("t", {"a": 1}, "call_1")),
            PartStartEvent(0, ThinkingPart("thinking")),
        ))
        sdk.instance.record_tool_call.assert_not_awaited()
        sdk.instance.record_thinking.assert_not_awaited()


# ---------------------------------------------------------------------------
# None of it breaks the run
# ---------------------------------------------------------------------------


class TestNoneOfItBreaksTheRun:
    """A handler that throws inside the agent's own loop takes down the run,
    so every one of these must be silent on failure."""

    @pytest.mark.asyncio
    async def test_a_dead_sdk_does_not_raise(self):
        sdk = _streaming_sdk(True)
        sdk.instance.record_tool_call = AsyncMock(side_effect=RuntimeError("boom"))
        sdk.instance.record_thinking = AsyncMock(side_effect=RuntimeError("boom"))
        await _handler(sdk)(None, _stream(
            FunctionToolCallEvent(ToolCallPart("t", {"a": 1}, "call_1")),
            PartStartEvent(0, ThinkingPart("thinking")),
        ))

    @pytest.mark.asyncio
    async def test_an_sdk_with_no_instance_namespace_does_not_raise(self):
        sdk = MagicMock(spec=[])
        handler = SynapEventStreamHandler(sdk, conversation_id="c", user_id="u")
        await handler(None, _stream(
            FunctionToolCallEvent(ToolCallPart("t", {"a": 1}, "call_1")),
            FunctionToolResultEvent(ToolReturnPart("t", "out", "call_1")),
            PartStartEvent(0, ThinkingPart("thinking")),
        ))

    @pytest.mark.asyncio
    async def test_an_unreadable_event_does_not_stop_the_later_ones(self):
        """One malformed event must not cost the rest of the stream."""

        class Exploding:
            event_kind = "function_tool_call"

            @property
            def part(self):
                raise RuntimeError("boom")

        sdk = _streaming_sdk(True)
        await _handler(sdk)(None, _stream(
            Exploding(),
            FunctionToolResultEvent(ToolReturnPart("t", "out", "call_1")),
        ))
        sdk.instance.record_tool_result.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_stream_that_raises_does_not_raise_out(self):
        sdk = _streaming_sdk(True)

        async def events():
            yield FunctionToolCallEvent(ToolCallPart("t", {"a": 1}, "call_1"))
            raise RuntimeError("the stream broke")

        await _handler(sdk)(None, events())
        sdk.instance.record_tool_call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_an_unknown_event_kind_is_ignored(self):
        """A newer Pydantic AI emits events this handler has never heard of."""
        sdk = _streaming_sdk(True)

        @dataclass
        class SomethingNew:
            event_kind: str = "something_the_future_added"

        await _handler(sdk)(None, _stream(SomethingNew()))
        sdk.instance.record_tool_call.assert_not_awaited()
        sdk.instance.record_thinking.assert_not_awaited()

    def test_construction_rejects_a_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapEventStreamHandler(None)


# ---------------------------------------------------------------------------
# synap_run — stream first, REST only as a fallback, never both
# ---------------------------------------------------------------------------


class TestStreamFirstRestFallback:
    @pytest.mark.asyncio
    async def test_the_user_turn_goes_out_on_the_stream(self):
        sdk = _streaming_sdk(True)
        await synap_run(FakeAgent(), "where is my order", deps=_deps(sdk))
        first = sdk.instance.send_message.await_args_list[0].kwargs
        assert first["event_type"] == "user_message"
        assert first["content"] == "where is my order"

    @pytest.mark.asyncio
    async def test_and_is_NOT_also_written_over_rest(self):
        """The double-write this rule exists to prevent."""
        sdk = _streaming_sdk(True)
        await synap_run(FakeAgent(), "where is my order", deps=_deps(sdk))
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_assistant_turn_is_the_anticipation_moment(self):
        sdk = _streaming_sdk(True)
        await synap_run(FakeAgent("it ships tomorrow"), "when", deps=_deps(sdk))
        last = sdk.instance.send_message.await_args_list[-1].kwargs
        assert last["event_type"] == "assistant_message"
        assert last["role"] == "assistant"
        assert last["content"] == "it ships tomorrow"

    @pytest.mark.asyncio
    async def test_with_no_stream_both_turns_fall_back_to_rest(self):
        sdk = _streaming_sdk(False)
        await synap_run(FakeAgent("it ships tomorrow"), "when", deps=_deps(sdk))
        roles = [c.kwargs["role"]
                 for c in sdk.conversation.record_message.await_args_list]
        assert roles == ["user", "assistant"]
        sdk.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_user_turn_goes_out_before_the_run(self):
        sdk = _streaming_sdk(True)
        order: List[str] = []
        sdk.instance.send_message = AsyncMock(
            side_effect=lambda *a, **k: order.append(k["event_type"]))

        class Recording(FakeAgent):
            async def run(self, *a, **kw):
                order.append("agent.run")
                return await super().run(*a, **kw)

        await synap_run(Recording(), "where is my order", deps=_deps(sdk))
        assert order == ["user_message", "agent.run", "assistant_message"]

    @pytest.mark.asyncio
    async def test_a_run_with_no_conversation_id_still_streams(self):
        sdk = _streaming_sdk(True)
        await synap_run(FakeAgent(), "hi", deps=_deps(sdk, conversation_id=None))
        assert sdk.instance.send_message.await_count == 2

    @pytest.mark.asyncio
    async def test_a_run_with_no_conversation_id_has_no_rest_to_fall_back_to(self):
        """``record_message`` records against a conversation; the stream
        opens its own. With neither there is nowhere to put the turn."""
        sdk = _streaming_sdk(False)
        await synap_run(FakeAgent(), "hi", deps=_deps(sdk, conversation_id=None))
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_rest_failure_does_not_break_the_run(self):
        sdk = _streaming_sdk(False)
        sdk.conversation.record_message = AsyncMock(side_effect=RuntimeError("boom"))
        result = await synap_run(FakeAgent("ok"), "hi", deps=_deps(sdk))
        assert result.output == "ok"

    @pytest.mark.asyncio
    async def test_an_empty_prompt_reports_no_user_turn(self):
        sdk = _streaming_sdk(True)
        await synap_run(FakeAgent(), None, deps=_deps(sdk))
        assert sdk.instance.send_message.await_count == 1

    @pytest.mark.asyncio
    async def test_a_structured_output_is_still_the_assistant_turn(self):
        sdk = _streaming_sdk(True)

        @dataclass
        class Answer:
            status: str = "shipped"

        await synap_run(FakeAgent(Answer()), "when", deps=_deps(sdk))
        assert "shipped" in sdk.instance.send_message.await_args_list[-1].kwargs[
            "content"]

    @pytest.mark.asyncio
    async def test_a_failing_run_reports_the_user_turn_and_no_answer(self):
        sdk = _streaming_sdk(True)
        with pytest.raises(RuntimeError, match="model down"):
            await synap_run(
                FakeAgent(raises=RuntimeError("model down")), "hi", deps=_deps(sdk))
        assert sdk.instance.send_message.await_count == 1
        assert sdk.instance.send_message.await_args.kwargs[
            "event_type"] == "user_message"


# ---------------------------------------------------------------------------
# synap_run — wiring
# ---------------------------------------------------------------------------


class TestSynapRunWiring:
    @pytest.mark.asyncio
    async def test_it_installs_the_handler_for_the_run(self):
        sdk = _streaming_sdk(True)
        agent = FakeAgent()
        await synap_run(agent, "hi", deps=_deps(sdk))
        handler = agent.calls[0]["event_stream_handler"]
        assert isinstance(handler, SynapEventStreamHandler)
        assert handler.conversation_id == "conv-1"
        assert handler.user_id == "u1"
        assert handler.customer_id == "c1"

    @pytest.mark.asyncio
    async def test_a_caller_supplied_handler_is_left_alone(self):
        sdk = _streaming_sdk(True)
        agent = FakeAgent()
        mine = object()
        await synap_run(agent, "hi", deps=_deps(sdk), event_stream_handler=mine)
        assert agent.calls[0]["event_stream_handler"] is mine

    @pytest.mark.asyncio
    async def test_a_version_without_the_hook_is_not_handed_one(self):
        """Passing an argument the agent does not know is a TypeError inside
        the caller's run, which is exactly what must never happen."""
        sdk = _streaming_sdk(True)
        agent = OldFakeAgent()
        await synap_run(agent, "hi", deps=_deps(sdk))
        assert "event_stream_handler" not in agent.calls[0]

    @pytest.mark.asyncio
    async def test_a_version_without_the_hook_still_reports_both_turns(self):
        sdk = _streaming_sdk(True)
        await synap_run(OldFakeAgent("answered"), "hi", deps=_deps(sdk))
        assert [c.kwargs["event_type"]
                for c in sdk.instance.send_message.await_args_list] == [
            "user_message", "assistant_message"]

    @pytest.mark.asyncio
    async def test_a_version_without_the_hook_says_so_once(self, caplog):
        import logging

        import synap_pydantic_ai.stream as stream_module

        stream_module._warned_no_hook = False
        sdk = _streaming_sdk(True)
        with caplog.at_level(logging.WARNING, logger="synap_pydantic_ai.stream"):
            await synap_run(OldFakeAgent(), "hi", deps=_deps(sdk))
            await synap_run(OldFakeAgent(), "hi", deps=_deps(sdk))
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "event_stream_handler" in warnings[0].getMessage()

    @pytest.mark.asyncio
    async def test_deps_and_extra_kwargs_reach_the_agent(self):
        sdk = _streaming_sdk(True)
        agent = FakeAgent()
        deps = _deps(sdk)
        history = [{"role": "user", "content": "earlier"}]
        await synap_run(agent, "hi", deps=deps, message_history=history)
        assert agent.calls[0]["deps"] is deps
        assert agent.calls[0]["message_history"] is history
        assert agent.calls[0]["user_prompt"] == "hi"

    @pytest.mark.asyncio
    async def test_the_agents_result_is_returned_untouched(self):
        sdk = _streaming_sdk(True)
        agent = FakeAgent("the answer")
        result = await synap_run(agent, "hi", deps=_deps(sdk))
        assert result.output == "the answer"

    @pytest.mark.asyncio
    async def test_a_fresh_handler_per_run_so_step_indexes_restart(self):
        sdk = _streaming_sdk(True)
        agent = FakeAgent()
        await synap_run(agent, "one", deps=_deps(sdk))
        await synap_run(agent, "two", deps=_deps(sdk))
        assert (agent.calls[0]["event_stream_handler"]
                is not agent.calls[1]["event_stream_handler"])

    @pytest.mark.asyncio
    async def test_it_rejects_a_none_agent(self):
        with pytest.raises(ValueError, match="non-None agent"):
            await synap_run(None, "hi", deps=_deps(_streaming_sdk(True)))

    @pytest.mark.asyncio
    async def test_it_rejects_missing_deps(self):
        with pytest.raises(ValueError, match="SynapDeps"):
            await synap_run(FakeAgent(), "hi", deps=None)


# ---------------------------------------------------------------------------
# synap_run_sync
# ---------------------------------------------------------------------------


class TestSynapRunSync:
    def test_it_runs_the_whole_turn_from_sync_code(self):
        sdk = _streaming_sdk(True)
        agent = FakeAgent("it ships tomorrow")
        result = synap_run_sync(agent, "when", deps=_deps(sdk))
        assert result.output == "it ships tomorrow"
        assert [c.kwargs["event_type"]
                for c in sdk.instance.send_message.await_args_list] == [
            "user_message", "assistant_message"]
