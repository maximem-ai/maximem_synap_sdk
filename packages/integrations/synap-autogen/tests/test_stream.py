"""Tests for SynapStreamingChatContext — the live-stream tap for AutoGen.

The contract these pin, from ``stream_events`` and from the module docstring:

- **Stream first, and never a second write.** The server persists
  ``user_message`` and ``assistant_message`` from the stream itself, so an
  integration that reports a turn on the stream and also calls
  ``sdk.conversation.record_message`` writes the turn twice and extracts it
  twice. This integration is stream-only, so ``record_message`` must never be
  called at all.
- **Silent when no stream is running.** Adding this context must not change
  behaviour for someone who never called ``sdk.instance.listen()``.
- **Never raises.** ``add_message`` runs inside the agent's own loop, and an
  exception there ends the run.
- **A tool call and its result share an id**, which for AutoGen is
  ``FunctionCall.id`` and ``FunctionExecutionResult.call_id``.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from autogen_core import FunctionCall
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    SystemMessage,
    UserMessage,
)

# `AssistantAgent` is in autogen-agentchat, which is a test-only dependency: the
# runtime needs autogen-core alone. Declared in the `[dev]` extra so CI runs
# these, and gated here so a contributor who installed only the runtime deps gets
# a stated skip rather than a ModuleNotFoundError they have to diagnose.
try:
    from autogen_agentchat.agents import AssistantAgent

    AGENTCHAT_AVAILABLE = True
except ImportError:  # pragma: no cover — depends on what is installed
    AssistantAgent = None  # type: ignore[assignment]
    AGENTCHAT_AVAILABLE = False

requires_agentchat = pytest.mark.skipif(
    not AGENTCHAT_AVAILABLE,
    reason="needs a real AssistantAgent (autogen-agentchat, in the [dev] extra)",
)

from synap_autogen.stream import (
    SynapStreamingChatContext,
    _reply_text,
    _tool_args,
    _user_text,
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


def _ctx(sdk, **kwargs):
    return SynapStreamingChatContext(
        sdk,
        conversation_id="conv-1",
        user_id="user-1",
        customer_id="cust-1",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_init_raises_on_none_sdk():
    with pytest.raises(ValueError, match="non-None sdk"):
        SynapStreamingChatContext(None, conversation_id="c", user_id="u")


def test_init_raises_on_empty_conversation_id():
    with pytest.raises(ValueError, match="non-empty conversation_id"):
        SynapStreamingChatContext(_streaming_sdk(), conversation_id="", user_id="u")


def test_init_raises_on_empty_user_id():
    with pytest.raises(ValueError, match="non-empty user_id"):
        SynapStreamingChatContext(_streaming_sdk(), conversation_id="c", user_id="")


def test_customer_id_defaults_to_empty_string():
    ctx = SynapStreamingChatContext(
        _streaming_sdk(), conversation_id="c", user_id="u"
    )
    assert ctx._customer_id == ""


# ---------------------------------------------------------------------------
# Stream first, and never a second write
# ---------------------------------------------------------------------------


class TestStreamFirstAndNeverBoth:
    @pytest.mark.asyncio
    async def test_a_user_turn_goes_out_on_the_stream(self):
        sdk = _streaming_sdk(True)
        await _ctx(sdk).add_message(UserMessage(content="where is my order", source="u"))
        sdk.instance.send_message.assert_awaited_once()
        kw = sdk.instance.send_message.await_args.kwargs
        assert kw["event_type"] == "user_message"
        assert kw["role"] == "user"
        assert kw["content"] == "where is my order"

    @pytest.mark.asyncio
    async def test_and_is_NOT_also_written_over_rest(self):
        """The double-write the stream-first rule exists to prevent.

        The server persists the turn from the stream event itself, so a
        ``record_message`` on top of it stores and extracts the same turn
        twice.
        """
        sdk = _streaming_sdk(True)
        ctx = _ctx(sdk)
        await ctx.add_message(UserMessage(content="hello", source="u"))
        await ctx.add_message(AssistantMessage(content="hi there", source="a"))
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_assistant_turn_is_the_anticipation_moment(self):
        sdk = _streaming_sdk(True)
        await _ctx(sdk).add_message(
            AssistantMessage(content="it ships tomorrow", source="a")
        )
        kw = sdk.instance.send_message.await_args.kwargs
        assert kw["event_type"] == "assistant_message"
        assert kw["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_with_no_stream_nothing_goes_out_anywhere(self):
        """Someone who never called listen() must see no change at all."""
        sdk = _streaming_sdk(False)
        ctx = _ctx(sdk)
        await ctx.add_message(UserMessage(content="hello", source="u"))
        await ctx.add_message(AssistantMessage(content="hi", source="a"))
        sdk.instance.send_message.assert_not_awaited()
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_ids_ride_on_every_event(self):
        sdk = _streaming_sdk(True)
        await _ctx(sdk).add_message(UserMessage(content="hello", source="u"))
        kw = sdk.instance.send_message.await_args.kwargs
        assert kw["conversation_id"] == "conv-1"
        assert kw["user_id"] == "user-1"
        assert kw["customer_id"] == "cust-1"


# ---------------------------------------------------------------------------
# The whole turn, not just its two ends
# ---------------------------------------------------------------------------


class TestTheWholeTurn:
    @pytest.mark.asyncio
    async def test_a_tool_call_is_reported(self):
        sdk = _streaming_sdk(True)
        await _ctx(sdk).add_message(
            AssistantMessage(
                content=[
                    FunctionCall(
                        id="call_1",
                        name="lookup_order",
                        arguments=json.dumps({"pnr": "QX41RT"}),
                    )
                ],
                source="a",
            )
        )
        sdk.instance.record_tool_call.assert_awaited_once()
        args, kwargs = sdk.instance.record_tool_call.await_args
        assert args[0] == "lookup_order"
        assert args[1] == {"pnr": "QX41RT"}
        assert kwargs["tool_call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_every_tool_call_in_one_message_is_reported(self):
        sdk = _streaming_sdk(True)
        await _ctx(sdk).add_message(
            AssistantMessage(
                content=[
                    FunctionCall(id="c1", name="a", arguments="{}"),
                    FunctionCall(id="c2", name="b", arguments="{}"),
                ],
                source="a",
            )
        )
        assert sdk.instance.record_tool_call.await_count == 2

    @pytest.mark.asyncio
    async def test_a_tool_result_is_reported(self):
        sdk = _streaming_sdk(True)
        await _ctx(sdk).add_message(
            FunctionExecutionResultMessage(
                content=[
                    FunctionExecutionResult(
                        content="shipped", name="lookup_order", call_id="call_1"
                    )
                ]
            )
        )
        sdk.instance.record_tool_result.assert_awaited_once()
        args, kwargs = sdk.instance.record_tool_result.await_args
        assert args[0] == "shipped"
        assert kwargs["tool_name"] == "lookup_order"

    @pytest.mark.asyncio
    async def test_the_call_and_its_result_share_an_id(self):
        """Without this the anticipation agent sees a call and a result and
        cannot tell they are the same tool invocation."""
        sdk = _streaming_sdk(True)
        ctx = _ctx(sdk)
        await ctx.add_message(
            AssistantMessage(
                content=[FunctionCall(id="call_1", name="t", arguments="{}")],
                source="a",
            )
        )
        await ctx.add_message(
            FunctionExecutionResultMessage(
                content=[
                    FunctionExecutionResult(content="out", name="t", call_id="call_1")
                ]
            )
        )
        assert (
            sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
            == sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"]
        )

    @pytest.mark.asyncio
    async def test_tool_results_can_be_turned_off(self):
        """A tool result is the caller's own customer data leaving their
        process, so it has to be refusable."""
        sdk = _streaming_sdk(True)
        await _ctx(sdk, report_tool_results=False).add_message(
            FunctionExecutionResultMessage(
                content=[
                    FunctionExecutionResult(content="pii", name="t", call_id="c1")
                ]
            )
        )
        sdk.instance.record_tool_result.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reasoning_is_reported_from_the_thought(self):
        sdk = _streaming_sdk(True)
        await _ctx(sdk).add_message(
            AssistantMessage(
                content=[FunctionCall(id="c1", name="t", arguments="{}")],
                thought="I should look up the booking first",
                source="a",
            )
        )
        sdk.instance.record_thinking.assert_awaited_once()
        args, kwargs = sdk.instance.record_thinking.await_args
        assert args[0] == "I should look up the booking first"
        assert kwargs["thought_type"] == "model_thought"
        # Ordinal within the turn, so the dashboard can replay the reasoning
        # timeline in the order the agent actually thought.
        assert kwargs["step_index"] == 1

    @pytest.mark.asyncio
    async def test_reasoning_steps_are_numbered_in_order(self):
        sdk = _streaming_sdk(True)
        ctx = _ctx(sdk)
        for thought in ("first look it up", "now check the refund window"):
            await ctx.add_message(
                AssistantMessage(
                    content=[FunctionCall(id="c", name="t", arguments="{}")],
                    thought=thought,
                    source="a",
                )
            )
        steps = [
            c.kwargs["step_index"]
            for c in sdk.instance.record_thinking.await_args_list
        ]
        assert steps == [1, 2]

    @pytest.mark.asyncio
    async def test_a_message_with_no_thought_reports_no_reasoning(self):
        sdk = _streaming_sdk(True)
        await _ctx(sdk).add_message(AssistantMessage(content="plain answer", source="a"))
        sdk.instance.record_thinking.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reasoning_can_be_turned_off(self):
        sdk = _streaming_sdk(True)
        await _ctx(sdk, report_thoughts=False).add_message(
            AssistantMessage(content="answer", thought="thinking", source="a")
        )
        sdk.instance.record_thinking.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_system_prompt_is_not_a_turn(self):
        sdk = _streaming_sdk(True)
        await _ctx(sdk).add_message(SystemMessage(content="You are helpful"))
        sdk.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_empty_assistant_message_reports_nothing(self):
        sdk = _streaming_sdk(True)
        await _ctx(sdk).add_message(AssistantMessage(content="   ", source="a"))
        sdk.instance.send_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# The gap AutoGen leaves on its own default path
# ---------------------------------------------------------------------------


class TestTheGapAutogenLeaves:
    """With ``reflect_on_tool_use=False`` — AutoGen's default when no
    ``output_content_type`` is set — a turn that ends in a tool call answers
    with a ToolCallSummaryMessage that AutoGen never writes into the model
    context. Without ``report_reply`` such a turn reports its tools and then
    stops, and anticipation never sees the assistant_message it acts on.
    """

    @pytest.mark.asyncio
    async def test_report_reply_sends_a_reply_the_context_never_saw(self):
        sdk = _streaming_sdk(True)
        result = MagicMock()
        result.messages = [MagicMock(to_text=MagicMock(return_value="shipped"))]
        assert await _ctx(sdk).report_reply(result) is True
        kw = sdk.instance.send_message.await_args.kwargs
        assert kw["event_type"] == "assistant_message"
        assert kw["content"] == "shipped"

    @pytest.mark.asyncio
    async def test_report_reply_does_not_write_the_reply_twice(self):
        """It is safe to call on every turn: a reply already reported through
        add_message is not reported again."""
        sdk = _streaming_sdk(True)
        ctx = _ctx(sdk)
        await ctx.add_message(
            AssistantMessage(content="it ships tomorrow", source="a")
        )
        result = MagicMock()
        result.messages = [
            MagicMock(to_text=MagicMock(return_value="it ships tomorrow"))
        ]
        assert await ctx.report_reply(result) is False
        assert sdk.instance.send_message.await_count == 1

    @pytest.mark.asyncio
    async def test_report_reply_with_no_stream_sends_nothing(self):
        sdk = _streaming_sdk(False)
        result = MagicMock()
        result.messages = [MagicMock(to_text=MagicMock(return_value="shipped"))]
        assert await _ctx(sdk).report_reply(result) is False

    @pytest.mark.asyncio
    async def test_report_reply_on_nothing_reports_nothing(self):
        sdk = _streaming_sdk(True)
        assert await _ctx(sdk).report_reply(None) is False
        sdk.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_report_reply_on_nothing_after_a_real_turn_stays_quiet(self):
        """The interesting case for the empty check. Once a turn has been
        reported, an unreadable reply no longer matches it, so nothing but an
        explicit empty check stops a blank assistant turn going out.
        """
        sdk = _streaming_sdk(True)
        ctx = _ctx(sdk)
        await ctx.add_message(AssistantMessage(content="it ships tomorrow", source="a"))
        assert await ctx.report_reply(None) is False
        assert sdk.instance.send_message.await_count == 1


# ---------------------------------------------------------------------------
# Nothing here breaks the agent's run
# ---------------------------------------------------------------------------


class TestNoneOfItBreaksTheRun:
    """add_message is awaited inside the agent's own loop. An exception there
    ends the run, so every one of these must be silent on failure rather than
    merely unlikely to fail."""

    @pytest.mark.asyncio
    async def test_a_dead_sdk_does_not_raise(self):
        sdk = _streaming_sdk(True)
        sdk.instance.send_message = AsyncMock(side_effect=RuntimeError("boom"))
        await _ctx(sdk).add_message(UserMessage(content="hi", source="u"))

    @pytest.mark.asyncio
    async def test_an_sdk_with_no_instance_namespace_does_not_raise(self):
        """An older SDK, or a test double. Every call must no-op."""
        sdk = MagicMock(spec=[])
        ctx = SynapStreamingChatContext(sdk, conversation_id="c", user_id="u")
        await ctx.add_message(UserMessage(content="hi", source="u"))
        await ctx.add_message(
            AssistantMessage(
                content=[FunctionCall(id="c1", name="t", arguments="{}")], source="a"
            )
        )
        await ctx.report_reply("done")

    @pytest.mark.asyncio
    async def test_a_message_of_an_unreadable_shape_does_not_raise(self):
        sdk = _streaming_sdk(True)
        broken = MagicMock(spec=AssistantMessage)
        type(broken).content = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("no content"))
        )
        await _ctx(sdk).add_message(broken)

    @pytest.mark.asyncio
    async def test_the_model_still_sees_the_message_when_reporting_fails(self):
        """The inner context is the agent's memory of the conversation. A
        failed report must not cost it a message."""
        sdk = _streaming_sdk(True)
        sdk.instance.send_message = AsyncMock(side_effect=RuntimeError("boom"))
        inner = UnboundedChatCompletionContext()
        ctx = _ctx(sdk, inner=inner)
        await ctx.add_message(UserMessage(content="hi", source="u"))
        assert len(await inner.get_messages()) == 1


# ---------------------------------------------------------------------------
# Reads and state are pass-through
# ---------------------------------------------------------------------------


class TestItChangesNothingTheModelSees:
    @pytest.mark.asyncio
    async def test_get_messages_returns_what_the_inner_context_holds(self):
        inner = UnboundedChatCompletionContext()
        ctx = _ctx(_streaming_sdk(False), inner=inner)
        await ctx.add_message(UserMessage(content="hi", source="u"))
        assert [m.content for m in await ctx.get_messages()] == ["hi"]

    @pytest.mark.asyncio
    async def test_clear_clears_the_inner_context(self):
        inner = UnboundedChatCompletionContext()
        ctx = _ctx(_streaming_sdk(False), inner=inner)
        await ctx.add_message(UserMessage(content="hi", source="u"))
        await ctx.clear()
        assert await inner.get_messages() == []

    @pytest.mark.asyncio
    async def test_state_round_trips_through_the_inner_context(self):
        sdk = _streaming_sdk(False)
        source = _ctx(sdk, inner=UnboundedChatCompletionContext())
        await source.add_message(UserMessage(content="hi", source="u"))
        target = _ctx(sdk, inner=UnboundedChatCompletionContext())
        await target.load_state(await source.save_state())
        assert [m.content for m in await target.get_messages()] == ["hi"]

    @pytest.mark.asyncio
    async def test_restoring_a_conversation_does_not_replay_it_on_the_stream(self):
        """Loading saved history is not a turn happening now. Replaying it
        would have anticipation react to history as if it were live."""
        sdk = _streaming_sdk(False)
        source = _ctx(sdk, inner=UnboundedChatCompletionContext())
        await source.add_message(UserMessage(content="hi", source="u"))
        state = await source.save_state()

        live = _streaming_sdk(True)
        await _ctx(live, inner=UnboundedChatCompletionContext()).load_state(state)
        live.instance.send_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# The shape readers
# ---------------------------------------------------------------------------


class TestReadingTheShapes:
    def test_user_text_reads_a_plain_string(self):
        assert _user_text("  hello  ") == "hello"

    def test_user_text_keeps_only_the_words_from_multimodal_content(self):
        """The repr of an image is not something a person said."""
        assert _user_text(["look at this", object()]) == "look at this"

    def test_user_text_of_nothing_is_empty(self):
        assert _user_text(None) == ""

    def test_tool_args_parses_the_json_autogen_sends(self):
        assert _tool_args('{"pnr": "QX41"}') == {"pnr": "QX41"}

    def test_tool_args_passes_unparseable_arguments_through(self):
        """A model can emit something that is not JSON. Dropping it loses the
        only record of what the tool was asked for."""
        assert _tool_args("not json at all") == {"arguments": "not json at all"}

    def test_tool_args_passes_a_json_non_object_through(self):
        assert _tool_args("[1, 2]") == {"arguments": "[1, 2]"}

    def test_tool_args_of_nothing_is_none(self):
        assert _tool_args(None) is None
        assert _tool_args("") is None

    def test_reply_text_reads_the_LAST_message_of_a_task_result(self):
        """A TaskResult carries the whole exchange. The reply is the end of
        it, not the task that started it."""
        result = MagicMock()
        result.messages = [
            MagicMock(to_text=MagicMock(return_value="where is my order")),
            MagicMock(to_text=MagicMock(return_value="it ships tomorrow")),
        ]
        assert _reply_text(result) == "it ships tomorrow"

    def test_reply_text_reads_a_response(self):
        response = MagicMock(spec=["chat_message"])
        response.chat_message = MagicMock(to_text=MagicMock(return_value="the reply"))
        assert _reply_text(response) == "the reply"

    def test_reply_text_reads_a_plain_string(self):
        assert _reply_text(" done ") == "done"

    def test_reply_text_of_nothing_is_empty(self):
        assert _reply_text(None) == ""


# ---------------------------------------------------------------------------
# Against a real AssistantAgent
#
# Everything above drives the context directly. This drives AutoGen, with a
# stub model client standing in for the provider, and asserts on what came out
# of the stream — because the thing that can be wrong is not the mapping but
# whether AutoGen routes these messages through the model context at all.
# ---------------------------------------------------------------------------


def _fake_model_client(results):
    from autogen_core.models import ChatCompletionClient, ModelInfo, RequestUsage

    class _FakeClient(ChatCompletionClient):
        def __init__(self):
            self._results = list(results)
            self._i = 0

        async def create(self, messages, **kwargs):
            result = self._results[self._i]
            self._i += 1
            return result

        def create_stream(self, *a, **k):
            raise NotImplementedError

        async def close(self):
            pass

        def actual_usage(self):
            return RequestUsage(0, 0)

        def total_usage(self):
            return RequestUsage(0, 0)

        def count_tokens(self, messages, **k):
            return 0

        def remaining_tokens(self, messages, **k):
            return 1000

        @property
        def capabilities(self):
            return self.model_info

        @property
        def model_info(self):
            return ModelInfo(
                vision=False,
                function_calling=True,
                json_output=False,
                family="unknown",
                structured_output=False,
            )

    return _FakeClient()


def _tool_then_answer():
    from autogen_core.models import CreateResult, RequestUsage

    return [
        CreateResult(
            finish_reason="function_calls",
            content=[
                FunctionCall(
                    id="call_1",
                    name="lookup_order",
                    arguments=json.dumps({"pnr": "QX41RT"}),
                )
            ],
            usage=RequestUsage(0, 0),
            cached=False,
            thought="I should look up the booking first",
        ),
        CreateResult(
            finish_reason="stop",
            content="Your order ships tomorrow.",
            usage=RequestUsage(0, 0),
            cached=False,
        ),
    ]


def _lookup_order(pnr: str) -> str:
    return "shipped"


@requires_agentchat
class TestAgainstARealAssistantAgent:
    @pytest.mark.asyncio
    async def test_a_reflecting_run_reports_all_five_events(self):
        sdk = _streaming_sdk(True)
        ctx = _ctx(sdk)
        agent = AssistantAgent(
            name="support",
            model_client=_fake_model_client(_tool_then_answer()),
            tools=[_lookup_order],
            model_context=ctx,
            reflect_on_tool_use=True,
        )
        await agent.run(task="where is my order")

        # The turn, in order: the question and the answer on the message
        # channel, and the three events in between on their own.
        events = [
            c.kwargs["event_type"] for c in sdk.instance.send_message.await_args_list
        ]
        assert events == ["user_message", "assistant_message"]
        assert (
            sdk.instance.send_message.await_args_list[0].kwargs["content"]
            == "where is my order"
        )
        assert (
            sdk.instance.send_message.await_args_list[1].kwargs["content"]
            == "Your order ships tomorrow."
        )
        sdk.instance.record_thinking.assert_awaited_once()
        assert (
            sdk.instance.record_thinking.await_args.args[0]
            == "I should look up the booking first"
        )
        sdk.instance.record_tool_call.assert_awaited_once()
        sdk.instance.record_tool_result.assert_awaited_once()
        assert (
            sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
            == sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"]
            == "call_1"
        )

    @pytest.mark.asyncio
    async def test_the_default_run_loses_the_assistant_turn_without_report_reply(self):
        """AutoGen's own default. The tool-call summary never reaches the
        model context, so the turn ends with no assistant_message — which is
        the one event anticipation acts on."""
        sdk = _streaming_sdk(True)
        ctx = _ctx(sdk)
        agent = AssistantAgent(
            name="support",
            model_client=_fake_model_client(_tool_then_answer()),
            tools=[_lookup_order],
            model_context=ctx,
            reflect_on_tool_use=False,
        )
        result = await agent.run(task="where is my order")

        events = [
            c.kwargs["event_type"] for c in sdk.instance.send_message.await_args_list
        ]
        assert "assistant_message" not in events

        assert await ctx.report_reply(result) is True
        assert (
            sdk.instance.send_message.await_args.kwargs["event_type"]
            == "assistant_message"
        )

    @pytest.mark.asyncio
    async def test_a_run_with_no_stream_reports_nothing(self):
        sdk = _streaming_sdk(False)
        ctx = _ctx(sdk)
        agent = AssistantAgent(
            name="support",
            model_client=_fake_model_client(_tool_then_answer()),
            tools=[_lookup_order],
            model_context=ctx,
            reflect_on_tool_use=True,
        )
        result = await agent.run(task="where is my order")
        assert result.messages[-1].to_text() == "Your order ships tomorrow."
        sdk.instance.send_message.assert_not_awaited()
        sdk.conversation.record_message.assert_not_awaited()
