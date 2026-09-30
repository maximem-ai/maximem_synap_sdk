"""Tests for SynapStreamMiddleware — the five events on Synap's live stream.

These pin the contract from the module docstring:

- stream first, REST only as a fallback, never both;
- the user turn goes out once per invocation, not once per lap of a tool loop;
- the assistant turn goes out once, from the exit node, and is the final answer;
- a tool call and its result share one ``tool_call_id``;
- what is reported as a tool result survives ``json.dumps``, because the SDK
  encodes it and ``stream_events`` swallows whatever that raises;
- nothing in here can raise into the graph.
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from synap_deepagents.stream import (
    SynapStreamMiddleware,
    _json_safe,
    _message_text,
    _reasoning_steps,
    _tool_output,
)


# ── fixtures ─────────────────────────────────────────────────────────────────


def _streaming_sdk(is_listening: bool):
    sdk = MagicMock()
    sdk.instance.is_listening = is_listening
    sdk.instance.send_message = AsyncMock()
    sdk.instance.record_thinking = AsyncMock()
    sdk.instance.record_tool_call = AsyncMock()
    sdk.instance.record_tool_result = AsyncMock()
    sdk.conversation.record_message = AsyncMock()
    return sdk


def _mw(sdk):
    return SynapStreamMiddleware(
        sdk=sdk, conversation_id="conv-1", user_id="user-1", customer_id="cust-1",
    )


def _request(name="lookup_order", args=None, call_id="call-1"):
    request = MagicMock()
    request.tool_call = {
        "name": name,
        "args": {"pnr": "QX41RT"} if args is None else args,
        "id": call_id,
    }
    return request


# ── construction ─────────────────────────────────────────────────────────────


def test_requires_sdk():
    with pytest.raises(ValueError, match="non-None sdk"):
        SynapStreamMiddleware(sdk=None, conversation_id="c", user_id="u")


def test_requires_conversation_id(mock_sdk):
    with pytest.raises(ValueError, match="non-empty conversation_id"):
        SynapStreamMiddleware(sdk=mock_sdk, conversation_id="  ", user_id="u")


def test_requires_user_id(mock_sdk):
    with pytest.raises(ValueError, match="non-empty user_id"):
        SynapStreamMiddleware(sdk=mock_sdk, conversation_id="c", user_id="")


def test_construction_does_no_sdk_io(failing_sdk):
    SynapStreamMiddleware(sdk=failing_sdk, conversation_id="c", user_id="u")


# ── stream first, REST only as a fallback ────────────────────────────────────


class TestStreamFirstRestFallback:
    async def test_a_turn_goes_out_on_the_stream_when_one_is_open(self):
        sdk = _streaming_sdk(True)
        await _mw(sdk)._record("user", "where is my order")
        sdk.instance.send_message.assert_awaited_once()
        assert (
            sdk.instance.send_message.await_args.kwargs["event_type"]
            == "user_message"
        )

    async def test_and_is_NOT_also_written_over_rest(self):
        """The double-write this rule exists to prevent."""
        sdk = _streaming_sdk(True)
        await _mw(sdk)._record("user", "where is my order")
        sdk.conversation.record_message.assert_not_awaited()

    async def test_with_no_stream_it_falls_back_to_rest(self):
        sdk = _streaming_sdk(False)
        await _mw(sdk)._record("assistant", "it ships tomorrow")
        sdk.conversation.record_message.assert_awaited_once()
        sdk.instance.send_message.assert_not_awaited()

    async def test_the_assistant_turn_is_the_anticipation_moment(self):
        sdk = _streaming_sdk(True)
        await _mw(sdk)._record("assistant", "it ships tomorrow")
        kwargs = sdk.instance.send_message.await_args.kwargs
        assert kwargs["event_type"] == "assistant_message"
        assert kwargs["role"] == "assistant"


# ── the user turn ────────────────────────────────────────────────────────────


class TestTheUserTurn:
    async def test_before_agent_reports_the_question(self):
        sdk = _streaming_sdk(True)
        await _mw(sdk).abefore_agent(
            {"messages": [HumanMessage("where is my order", id="h1")]}, None
        )
        assert (
            sdk.instance.send_message.await_args.kwargs["content"]
            == "where is my order"
        )

    async def test_it_reads_the_most_recent_human_message(self):
        sdk = _streaming_sdk(True)
        await _mw(sdk).abefore_agent(
            {
                "messages": [
                    HumanMessage("first", id="h1"),
                    AIMessage("answer", id="a1"),
                    HumanMessage("second", id="h2"),
                ]
            },
            None,
        )
        assert sdk.instance.send_message.await_args.kwargs["content"] == "second"

    async def test_a_resumed_run_does_not_report_the_same_turn_twice(self):
        """``before_agent`` re-enters on a resume with the same question."""
        sdk = _streaming_sdk(True)
        middleware = _mw(sdk)
        state = {"messages": [HumanMessage("where is my order", id="h1")]}
        await middleware.abefore_agent(state, None)
        await middleware.abefore_agent(state, None)
        assert sdk.instance.send_message.await_count == 1

    async def test_the_same_words_in_two_turns_are_two_turns(self):
        sdk = _streaming_sdk(True)
        middleware = _mw(sdk)
        await middleware.abefore_agent(
            {"messages": [HumanMessage("hello", id="h1")]}, None
        )
        await middleware.abefore_agent(
            {"messages": [HumanMessage("hello", id="h2")]}, None
        )
        assert sdk.instance.send_message.await_count == 2

    async def test_two_middlewares_do_not_silence_each_other(self):
        """The de-duplication is per instance, not per class."""
        sdk = _streaming_sdk(True)
        state = {"messages": [HumanMessage("hello", id="h1")]}
        await _mw(sdk).abefore_agent(state, None)
        await _mw(sdk).abefore_agent(state, None)
        assert sdk.instance.send_message.await_count == 2

    async def test_no_human_message_reports_nothing(self):
        sdk = _streaming_sdk(True)
        await _mw(sdk).abefore_agent({"messages": [AIMessage("hi", id="a1")]}, None)
        sdk.instance.send_message.assert_not_awaited()
        sdk.conversation.record_message.assert_not_awaited()

    async def test_a_plain_dict_message_is_read_too(self):
        sdk = _streaming_sdk(True)
        await _mw(sdk).abefore_agent(
            {"messages": [{"role": "user", "content": "from a dict"}]}, None
        )
        assert (
            sdk.instance.send_message.await_args.kwargs["content"] == "from a dict"
        )


# ── the assistant turn ───────────────────────────────────────────────────────


class TestTheAssistantTurn:
    async def test_after_agent_reports_the_answer(self):
        sdk = _streaming_sdk(True)
        await _mw(sdk).aafter_agent(
            {
                "messages": [
                    HumanMessage("where is my order", id="h1"),
                    AIMessage("it ships tomorrow", id="a1"),
                ]
            },
            None,
        )
        kwargs = sdk.instance.send_message.await_args.kwargs
        assert kwargs["event_type"] == "assistant_message"
        assert kwargs["content"] == "it ships tomorrow"

    async def test_an_intermediate_lap_is_not_a_turn(self):
        """``after_model`` fires per lap and must not close the turn.

        A three-tool turn reported from the model hook is four assistant
        messages, and anticipation acts on every one of them.
        """
        sdk = _streaming_sdk(True)
        middleware = _mw(sdk)
        state = {
            "messages": [
                HumanMessage("where is my order", id="h1"),
                AIMessage("let me look", id="a1"),
            ]
        }
        await middleware.aafter_model(state, None)
        assert sdk.instance.send_message.await_count == 0

    async def test_a_run_that_ends_on_a_tool_call_reports_no_answer(self):
        sdk = _streaming_sdk(True)
        await _mw(sdk).aafter_agent(
            {
                "messages": [
                    HumanMessage("q", id="h1"),
                    AIMessage(
                        "",
                        id="a1",
                        tool_calls=[
                            {"name": "t", "args": {}, "id": "call-1"},
                        ],
                    ),
                ]
            },
            None,
        )
        sdk.instance.send_message.assert_not_awaited()
        sdk.conversation.record_message.assert_not_awaited()

    async def test_it_is_reported_once(self):
        sdk = _streaming_sdk(True)
        middleware = _mw(sdk)
        state = {"messages": [AIMessage("it ships tomorrow", id="a1")]}
        await middleware.aafter_agent(state, None)
        await middleware.aafter_agent(state, None)
        assert sdk.instance.send_message.await_count == 1


# ── reasoning ────────────────────────────────────────────────────────────────


class TestReasoning:
    async def test_a_reasoning_block_is_reported(self):
        sdk = _streaming_sdk(True)
        message = AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "the PNR looks like a booking"},
                {"type": "text", "text": "let me look"},
            ],
            id="a1",
        )
        await _mw(sdk).aafter_model({"messages": [message]}, None)
        sdk.instance.record_thinking.assert_awaited_once()
        assert (
            sdk.instance.record_thinking.await_args.args[0]
            == "the PNR looks like a booking"
        )

    async def test_reasoning_content_from_the_provider_kwarg_is_reported(self):
        """DeepSeek, Groq, xAI and Ollama put it in additional_kwargs."""
        sdk = _streaming_sdk(True)
        message = AIMessage(
            content="let me look",
            id="a1",
            additional_kwargs={"reasoning_content": "I need the booking first"},
        )
        await _mw(sdk).aafter_model({"messages": [message]}, None)
        sdk.instance.record_thinking.assert_awaited_once()

    async def test_steps_are_numbered_within_the_turn(self):
        sdk = _streaming_sdk(True)
        middleware = _mw(sdk)
        await middleware.abefore_agent({"messages": []}, None)
        for index, text in enumerate(["first thought", "second thought"]):
            await middleware.aafter_model(
                {
                    "messages": [
                        AIMessage(
                            content=[{"type": "reasoning", "reasoning": text}],
                            id=f"a{index}",
                        )
                    ]
                },
                None,
            )
        steps = [
            call.kwargs["step_index"]
            for call in sdk.instance.record_thinking.await_args_list
        ]
        assert steps == [0, 1]

    async def test_a_turn_with_no_reasoning_reports_none(self):
        sdk = _streaming_sdk(True)
        await _mw(sdk).aafter_model(
            {"messages": [AIMessage("plain answer", id="a1")]}, None
        )
        sdk.instance.record_thinking.assert_not_awaited()

    async def test_the_same_message_is_not_mined_twice(self):
        sdk = _streaming_sdk(True)
        middleware = _mw(sdk)
        state = {
            "messages": [
                AIMessage(content=[{"type": "reasoning", "reasoning": "hm"}], id="a1")
            ]
        }
        await middleware.aafter_model(state, None)
        await middleware.aafter_model(state, None)
        assert sdk.instance.record_thinking.await_count == 1


# ── tools ────────────────────────────────────────────────────────────────────


class TestTools:
    async def test_the_call_is_reported_before_the_tool_runs(self):
        sdk = _streaming_sdk(True)
        order = []

        async def handler(request):
            order.append("tool")
            return ToolMessage(content="shipped", tool_call_id="call-1")

        sdk.instance.record_tool_call.side_effect = lambda *a, **k: order.append(
            "call"
        )
        await _mw(sdk).awrap_tool_call(_request(), handler)
        assert order == ["call", "tool"]

    async def test_the_parsed_args_are_reported_not_a_repr(self):
        sdk = _streaming_sdk(True)

        async def handler(request):
            return ToolMessage(content="shipped", tool_call_id="call-1")

        await _mw(sdk).awrap_tool_call(_request(), handler)
        assert sdk.instance.record_tool_call.await_args.args[1] == {"pnr": "QX41RT"}

    async def test_the_call_and_its_result_share_an_id(self):
        sdk = _streaming_sdk(True)

        async def handler(request):
            return ToolMessage(content="shipped", tool_call_id="call-1")

        await _mw(sdk).awrap_tool_call(_request(call_id="call-9"), handler)
        assert (
            sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
            == sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"]
            == "call-9"
        )

    async def test_the_tool_still_runs_when_the_stream_fails(self):
        sdk = _streaming_sdk(True)
        sdk.instance.record_tool_call.side_effect = RuntimeError("stream died")
        sdk.instance.record_tool_result.side_effect = RuntimeError("stream died")
        ran = []

        async def handler(request):
            ran.append(True)
            return ToolMessage(content="shipped", tool_call_id="call-1")

        result = await _mw(sdk).awrap_tool_call(_request(), handler)
        assert ran == [True]
        assert result.content == "shipped"

    async def test_the_tool_still_runs_when_the_REPORT_ITSELF_raises(self):
        """A request shaped differently from the one this was written against.

        ``stream_events`` swallows its own failures, so the test above never
        reaches this middleware's guard. Everything before the helper — reading
        the id, the name, the args off whatever the framework passed — is this
        module's own code, and it runs inside the tool path.
        """

        class ExplodingRequest:
            @property
            def tool_call(self):
                raise RuntimeError("the framework changed shape")

        sdk = _streaming_sdk(True)
        ran = []

        async def handler(request):
            ran.append(True)
            return ToolMessage(content="shipped", tool_call_id="call-1")

        result = await _mw(sdk).awrap_tool_call(ExplodingRequest(), handler)
        assert ran == [True]
        assert result.content == "shipped"

    async def test_the_handler_result_is_returned_unchanged(self):
        sdk = _streaming_sdk(True)
        sentinel = ToolMessage(content="shipped", tool_call_id="call-1")

        async def handler(request):
            return sentinel

        assert await _mw(sdk).awrap_tool_call(_request(), handler) is sentinel

    def test_the_sync_half_reports_both_sides(self):
        """``invoke()`` never reaches ``awrap_tool_call``."""
        sdk = _streaming_sdk(True)

        def handler(request):
            return ToolMessage(content="shipped", tool_call_id="call-1")

        _mw(sdk).wrap_tool_call(_request(), handler)
        sdk.instance.record_tool_call.assert_awaited_once()
        sdk.instance.record_tool_result.assert_awaited_once()


# ── the tool result survives serialisation ───────────────────────────────────


class TestTheToolResultSurvivesSerialisation:
    """The SDK does ``json.dumps`` on a tool result and ``stream_events``
    swallows what that raises, so an un-encodable result is not an error: it is
    a result the anticipation agent never sees."""

    def test_a_toolmessage_is_unwrapped_to_its_content(self):
        message = ToolMessage(content="shipped", tool_call_id="call-1")
        assert _tool_output(message, "call-1") == "shipped"

    def test_a_raw_toolmessage_would_not_have_survived(self):
        with pytest.raises(TypeError):
            json.dumps(ToolMessage(content="shipped", tool_call_id="call-1"))

    def test_a_command_is_unwrapped_to_the_message_for_this_call(self):
        """Every deepagents tool that writes state returns one of these."""
        command = Command(
            update={
                "files": {"/memories/a.md": "x"},
                "messages": [ToolMessage(content="wrote it", tool_call_id="call-1")],
            }
        )
        assert _tool_output(command, "call-1") == "wrote it"

    def test_a_command_carrying_another_calls_message_is_not_mistaken_for_this_one(
        self,
    ):
        command = Command(
            update={
                "messages": [ToolMessage(content="other", tool_call_id="call-2")]
            }
        )
        assert _tool_output(command, "call-1") != "other"

    def test_everything_reported_is_json_encodable(self):
        for result in (
            ToolMessage(content="shipped", tool_call_id="call-1"),
            Command(
                update={
                    "messages": [
                        ToolMessage(content="wrote it", tool_call_id="call-1")
                    ]
                }
            ),
            {"status": "shipped"},
            "plain string",
            object(),
        ):
            json.dumps(_tool_output(result, "call-1"))

    def test_a_plain_result_is_left_alone(self):
        assert _tool_output({"status": "shipped"}, "call-1") == {"status": "shipped"}

    def test_json_safe_falls_back_to_the_string_form(self):
        class Unencodable:
            def __str__(self):
                return "the readable form"

        assert _json_safe(Unencodable()) == "the readable form"


# ── nothing here breaks the graph ────────────────────────────────────────────


class TestNoneOfItBreaksTheGraph:
    """A middleware that raises takes down the whole graph run."""

    async def test_a_dead_sdk_does_not_raise_from_before_agent(self, failing_sdk):
        middleware = SynapStreamMiddleware(
            sdk=failing_sdk, conversation_id="c", user_id="u"
        )
        await middleware.abefore_agent(
            {"messages": [HumanMessage("hi", id="h1")]}, None
        )

    async def test_a_dead_sdk_does_not_raise_from_after_agent(self, failing_sdk):
        middleware = SynapStreamMiddleware(
            sdk=failing_sdk, conversation_id="c", user_id="u"
        )
        await middleware.aafter_agent({"messages": [AIMessage("hi", id="a1")]}, None)

    async def test_an_sdk_with_no_instance_namespace_does_not_raise(self):
        sdk = MagicMock()
        del sdk.instance
        sdk.conversation.record_message = AsyncMock()
        middleware = SynapStreamMiddleware(
            sdk=sdk, conversation_id="c", user_id="u"
        )
        await middleware.abefore_agent(
            {"messages": [HumanMessage("hi", id="h1")]}, None
        )

    async def test_a_state_with_no_messages_does_not_raise(self):
        sdk = _streaming_sdk(True)
        middleware = _mw(sdk)
        await middleware.abefore_agent({}, None)
        await middleware.aafter_model({}, None)
        await middleware.aafter_agent({}, None)

    def test_the_sync_halves_return_none(self):
        sdk = _streaming_sdk(True)
        middleware = _mw(sdk)
        state = {"messages": [HumanMessage("hi", id="h1")]}
        assert middleware.before_agent(state, None) is None
        assert middleware.after_model(state, None) is None
        assert middleware.after_agent(state, None) is None

    def test_a_bridge_that_cannot_run_the_coroutine_is_still_silent(
        self, monkeypatch
    ):
        """``run_async`` itself can fail, on a loop ``nest_asyncio`` cannot patch.

        That failure is outside everything ``stream_events`` guards, and it
        happens inside a graph node, so without the bridge's own guard it ends
        the run.
        """
        from synap_deepagents import stream as stream_module

        def boom(coro):
            raise RuntimeError("no loop to run this on")

        monkeypatch.setattr(stream_module, "run_async", boom)
        sdk = _streaming_sdk(True)
        middleware = _mw(sdk)
        state = {"messages": [HumanMessage("hi", id="h1")]}

        assert middleware.before_agent(state, None) is None
        assert middleware.after_model(state, None) is None
        assert middleware.after_agent(state, None) is None

        def handler(request):
            return ToolMessage(content="shipped", tool_call_id="call-1")

        assert (
            middleware.wrap_tool_call(_request(), handler).content == "shipped"
        )


# ── helpers ──────────────────────────────────────────────────────────────────


def test_message_text_joins_block_lists():
    message = AIMessage(
        content=[
            {"type": "text", "text": "Block one. "},
            {"type": "text", "text": "Block two."},
            {"type": "tool_use", "id": "t1"},
        ],
        id="a1",
    )
    assert _message_text(message) == "Block one. Block two."


def test_message_text_joins_block_lists_without_a_text_property():
    """The branch a real ``AIMessage`` never reaches.

    ``AIMessage.text`` already concatenates for us, so the block-list code
    below it only runs for a plain dict or a message wrapper that has no
    ``.text`` — and that is exactly where reading the first block only would
    lose the rest of the answer.
    """

    class Blocky:
        content = [
            {"type": "text", "text": "Block one. "},
            {"type": "text", "text": "Block two."},
            {"type": "tool_use", "id": "t1"},
        ]

    assert _message_text(Blocky()) == "Block one. Block two."
    assert (
        _message_text({"role": "assistant", "content": Blocky.content})
        == "Block one. Block two."
    )


def test_message_text_of_nothing_is_empty():
    assert _message_text(None) == ""


def test_reasoning_steps_of_a_plain_message_is_empty():
    assert _reasoning_steps(AIMessage("plain", id="a1")) == []


def test_reasoning_steps_skips_blank_blocks():
    message = AIMessage(
        content=[{"type": "reasoning", "reasoning": "   "}], id="a1"
    )
    assert _reasoning_steps(message) == []
