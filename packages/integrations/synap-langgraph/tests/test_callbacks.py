"""Tests for SynapLangGraphCallbackHandler.

The rule these pin is "stream first, REST only as a fallback, never both".
The server persists ``user_message`` and ``assistant_message`` from the
stream itself, so a ``record_message`` on top of a delivered stream event
writes the same turn twice and extracts it twice.

Beyond that, three things are specific to a graph and are what this handler
exists for:

- ``on_agent_action`` never fires in LangGraph, so reasoning is read off the
  AIMessage's reasoning content blocks in ``on_llm_end`` instead.
- a ReAct graph calls the model again after every tool result with the human
  message still in the list, so the user turn is reported once per turn
  rather than once per lap.
- the tool call and its result share ``run_id``, inherited from the
  LangChain handler and re-pinned here because a graph is where several
  tools are in flight at once.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from langchain_core.messages import ToolMessage

from synap_langgraph.callbacks import (
    SynapLangGraphCallbackHandler,
    _reasoning_steps,
    _tool_output,
)


# ---------------------------------------------------------------------------
# Fixtures
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


def _handler_for(sdk):
    return SynapLangGraphCallbackHandler(
        sdk=sdk, conversation_id="conv-1", user_id="user-1", customer_id="cust-1",
    )


def _human(content: str, msg_id=None):
    return MagicMock(type="human", content=content, id=msg_id)


def _llm_result(content, *, message_id="ai-1"):
    """An LLMResult whose first generation carries ``content`` blocks.

    ``content_blocks`` is the langchain-core 1.x normaliser; a MagicMock
    would answer it with another MagicMock, so it is set explicitly to the
    list the real property would return.
    """
    message = MagicMock()
    message.content = content
    message.content_blocks = content if isinstance(content, list) else [
        {"type": "text", "text": content}
    ]
    message.id = message_id
    generation = MagicMock()
    generation.message = message
    generation.text = "" if isinstance(content, list) else content
    return MagicMock(generations=[[generation]])


# ---------------------------------------------------------------------------
# _reasoning_steps
# ---------------------------------------------------------------------------


class TestReasoningSteps:
    def test_reads_a_standard_reasoning_block(self):
        result = _llm_result([{"type": "reasoning", "reasoning": "look up the booking"}])
        assert _reasoning_steps(result) == ["look up the booking"]

    def test_reads_every_reasoning_block_in_order(self):
        result = _llm_result([
            {"type": "reasoning", "reasoning": "first"},
            {"type": "text", "text": "answer"},
            {"type": "reasoning", "reasoning": "second"},
        ])
        assert _reasoning_steps(result) == ["first", "second"]

    def test_a_plain_text_answer_has_no_reasoning(self):
        assert _reasoning_steps(_llm_result("just an answer")) == []

    def test_a_blank_reasoning_block_is_not_a_step(self):
        result = _llm_result([{"type": "reasoning", "reasoning": "   "}])
        assert _reasoning_steps(result) == []

    def test_empty_generations_yield_nothing(self):
        assert _reasoning_steps(MagicMock(generations=[])) == []
        assert _reasoning_steps(MagicMock(generations=[[]])) == []

    def test_falls_back_to_raw_content_when_content_blocks_is_unavailable(self):
        """An older langchain-core, or a custom message wrapper."""

        class NoContentBlocks:
            content = [{"type": "reasoning", "reasoning": "from raw content"}]

            @property
            def content_blocks(self):
                raise AttributeError("no such property")

        generation = MagicMock()
        generation.message = NoContentBlocks()
        assert _reasoning_steps(MagicMock(generations=[[generation]])) == [
            "from raw content"
        ]

    def test_only_the_first_generation_is_read(self):
        """Candidates whose text is thrown away did not reach the user."""
        kept = MagicMock()
        kept.message = MagicMock(
            content_blocks=[{"type": "reasoning", "reasoning": "kept"}]
        )
        dropped = MagicMock()
        dropped.message = MagicMock(
            content_blocks=[{"type": "reasoning", "reasoning": "dropped"}]
        )
        result = MagicMock(generations=[[kept, dropped]])
        assert _reasoning_steps(result) == ["kept"]


# ---------------------------------------------------------------------------
# _tool_output
# ---------------------------------------------------------------------------


class TestToolOutput:
    def test_a_toolmessage_is_unwrapped_to_its_content(self):
        message = ToolMessage(
            content="shipped", name="lookup_order", tool_call_id="call_1")
        assert _tool_output(message) == "shipped"

    def test_a_plain_string_passes_through(self):
        assert _tool_output("shipped") == "shipped"

    def test_a_dict_passes_through(self):
        assert _tool_output({"status": "shipped"}) == {"status": "shipped"}

    def test_a_message_like_object_from_another_library_is_unwrapped(self):
        class NotALangChainMessage:
            content = "shipped"
            tool_call_id = "call_1"

        assert _tool_output(NotALangChainMessage()) == "shipped"

    def test_an_object_with_content_but_no_tool_call_id_passes_through(self):
        class Envelope:
            content = "inner"

        envelope = Envelope()
        assert _tool_output(envelope) is envelope


# ---------------------------------------------------------------------------
# Stream first, REST only as a fallback, never both
# ---------------------------------------------------------------------------


class TestStreamFirstRestFallback:
    @pytest.mark.asyncio
    async def test_the_user_turn_goes_out_on_the_stream(self):
        sdk = _streaming_sdk(True)
        await _handler_for(sdk).on_chat_model_start(
            serialized={}, messages=[[_human("where is my order", "h1")]],
            run_id=uuid4(),
        )
        sdk.instance.send_message.assert_awaited_once()
        assert sdk.instance.send_message.await_args.kwargs[
            "event_type"] == "user_message"

    @pytest.mark.asyncio
    async def test_and_is_NOT_also_written_over_rest(self):
        """The double-write this rule exists to prevent."""
        sdk = _streaming_sdk(True)
        await _handler_for(sdk).on_chat_model_start(
            serialized={}, messages=[[_human("where is my order", "h1")]],
            run_id=uuid4(),
        )
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_with_no_stream_the_user_turn_falls_back_to_rest(self):
        sdk = _streaming_sdk(False)
        await _handler_for(sdk).on_chat_model_start(
            serialized={}, messages=[[_human("where is my order", "h1")]],
            run_id=uuid4(),
        )
        sdk.conversation.record_message.assert_awaited_once()
        sdk.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_assistant_turn_is_the_anticipation_moment(self):
        sdk = _streaming_sdk(True)
        await _handler_for(sdk).on_llm_end(
            _llm_result("it ships tomorrow"), run_id=uuid4(),
        )
        kw = sdk.instance.send_message.await_args.kwargs
        assert kw["event_type"] == "assistant_message"
        assert kw["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_the_assistant_turn_is_NOT_also_written_over_rest(self):
        sdk = _streaming_sdk(True)
        await _handler_for(sdk).on_llm_end(
            _llm_result("it ships tomorrow"), run_id=uuid4(),
        )
        sdk.conversation.record_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# The user turn fires once per turn, not once per lap
# ---------------------------------------------------------------------------


class TestTheUserTurnIsReportedOncePerTurn:
    @pytest.mark.asyncio
    async def test_a_react_loop_does_not_re_report_the_same_human_message(self):
        """LangGraph calls the model again after every tool result and the
        human message is still in the list, so without this the turn is
        ingested and extracted once per lap."""
        sdk = _streaming_sdk(True)
        handler = _handler_for(sdk)
        human = _human("where is my order", "h1")
        ai = MagicMock(type="ai", content="", id="ai-1")
        tool = MagicMock(type="tool", content="shipped", id="t-1")

        await handler.on_chat_model_start({}, [[human]], run_id=uuid4())
        await handler.on_chat_model_start({}, [[human, ai, tool]], run_id=uuid4())
        await handler.on_chat_model_start({}, [[human, ai, tool]], run_id=uuid4())

        assert sdk.instance.send_message.await_count == 1

    @pytest.mark.asyncio
    async def test_a_genuinely_new_user_message_is_reported(self):
        sdk = _streaming_sdk(True)
        handler = _handler_for(sdk)
        await handler.on_chat_model_start({}, [[_human("first", "h1")]], run_id=uuid4())
        await handler.on_chat_model_start(
            {}, [[_human("first", "h1"), _human("second", "h2")]], run_id=uuid4(),
        )
        assert sdk.instance.send_message.await_count == 2
        assert sdk.instance.send_message.await_args.kwargs["content"] == "second"

    @pytest.mark.asyncio
    async def test_the_same_words_in_two_turns_are_two_turns(self):
        """A user answering "yes" twice said two things. The id is what
        separates them, which is why the key is not the content."""
        sdk = _streaming_sdk(True)
        handler = _handler_for(sdk)
        await handler.on_chat_model_start({}, [[_human("yes", "h1")]], run_id=uuid4())
        await handler.on_chat_model_start(
            {}, [[_human("yes", "h1"), _human("yes", "h2")]], run_id=uuid4(),
        )
        assert sdk.instance.send_message.await_count == 2

    @pytest.mark.asyncio
    async def test_a_graph_without_message_ids_falls_back_to_content(self):
        """No ``add_messages`` reducer means no stamped id."""
        sdk = _streaming_sdk(True)
        handler = _handler_for(sdk)
        await handler.on_chat_model_start({}, [[_human("same text")]], run_id=uuid4())
        await handler.on_chat_model_start({}, [[_human("same text")]], run_id=uuid4())
        await handler.on_chat_model_start({}, [[_human("other text")]], run_id=uuid4())
        assert sdk.instance.send_message.await_count == 2

    @pytest.mark.asyncio
    async def test_two_handlers_do_not_share_the_de_duplication_state(self):
        sdk = _streaming_sdk(True)
        await _handler_for(sdk).on_chat_model_start(
            {}, [[_human("hello", "h1")]], run_id=uuid4())
        await _handler_for(sdk).on_chat_model_start(
            {}, [[_human("hello", "h1")]], run_id=uuid4())
        assert sdk.instance.send_message.await_count == 2

    @pytest.mark.asyncio
    async def test_no_human_message_reports_nothing(self):
        sdk = _streaming_sdk(True)
        system = MagicMock(type="system", content="you are helpful", id="s1")
        await _handler_for(sdk).on_chat_model_start({}, [[system]], run_id=uuid4())
        sdk.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_messages_reports_nothing(self):
        sdk = _streaming_sdk(True)
        await _handler_for(sdk).on_chat_model_start({}, [], run_id=uuid4())
        sdk.instance.send_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# The three events a graph would otherwise never report
# ---------------------------------------------------------------------------


class TestTheEventsInBetween:
    @pytest.mark.asyncio
    async def test_a_tool_call_is_reported(self):
        sdk = _streaming_sdk(True)
        rid = uuid4()
        await _handler_for(sdk).on_tool_start(
            {"name": "lookup_order"}, "PNR QX41RT", run_id=rid)
        sdk.instance.record_tool_call.assert_awaited_once()
        assert sdk.instance.record_tool_call.await_args.kwargs[
            "tool_call_id"] == str(rid)

    @pytest.mark.asyncio
    async def test_a_tool_call_carries_the_parsed_arguments(self):
        """LangGraph passes the parsed dict as ``inputs``. The repr string in
        ``input_str`` is not something the anticipation agent can read."""
        sdk = _streaming_sdk(True)
        await _handler_for(sdk).on_tool_start(
            {"name": "lookup_order"}, "{'pnr': 'QX41RT'}",
            run_id=uuid4(), inputs={"pnr": "QX41RT"},
        )
        assert sdk.instance.record_tool_call.await_args.args[1] == {"pnr": "QX41RT"}

    @pytest.mark.asyncio
    async def test_a_tool_call_without_inputs_falls_back_to_the_input_string(self):
        sdk = _streaming_sdk(True)
        await _handler_for(sdk).on_tool_start(
            {"name": "lookup_order"}, "PNR QX41RT", run_id=uuid4())
        assert sdk.instance.record_tool_call.await_args.args[1] == {
            "input": "PNR QX41RT"}

    @pytest.mark.asyncio
    async def test_a_tool_result_is_reported(self):
        sdk = _streaming_sdk(True)
        await _handler_for(sdk).on_tool_end({"status": "shipped"}, run_id=uuid4())
        sdk.instance.record_tool_result.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_toolnode_result_is_unwrapped_to_something_sendable(self):
        """``ToolNode`` hands over the whole ``ToolMessage`` and the SDK
        JSON-encodes a tool result. ``json.dumps`` cannot encode a
        ``ToolMessage``, and the stream helper swallows the failure, so
        passing it through means no tool result is ever reported."""
        import json

        sdk = _streaming_sdk(True)
        message = ToolMessage(
            content="shipped", name="lookup_order", tool_call_id="call_1")
        await _handler_for(sdk).on_tool_end(message, run_id=uuid4())
        sent = sdk.instance.record_tool_result.await_args.args[0]
        assert sent == "shipped"
        json.dumps(sent)  # would raise on the ToolMessage itself

    @pytest.mark.asyncio
    async def test_a_toolnode_result_carries_the_tool_name(self):
        sdk = _streaming_sdk(True)
        message = ToolMessage(
            content="shipped", name="lookup_order", tool_call_id="call_1")
        await _handler_for(sdk).on_tool_end(message, run_id=uuid4())
        assert sdk.instance.record_tool_result.await_args.kwargs[
            "tool_name"] == "lookup_order"

    @pytest.mark.asyncio
    async def test_the_call_and_its_result_share_an_id(self):
        """LangGraph hands ``on_tool_start`` and ``on_tool_end`` the same
        ``run_id``. Without it the anticipation agent cannot tell which
        result belongs to which of several tools in flight."""
        sdk = _streaming_sdk(True)
        rid = uuid4()
        handler = _handler_for(sdk)
        await handler.on_tool_start({"name": "t"}, "in", run_id=rid)
        await handler.on_tool_end("out", run_id=rid)
        assert (sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
                == sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"])

    @pytest.mark.asyncio
    async def test_reasoning_is_reported_from_the_ai_message(self):
        """``on_agent_action`` never fires in a graph, so this is the only
        place a LangGraph run's reasoning can come from."""
        sdk = _streaming_sdk(True)
        await _handler_for(sdk).on_llm_end(
            _llm_result([{"type": "reasoning", "reasoning": "look up the booking"}]),
            run_id=uuid4(),
        )
        sdk.instance.record_thinking.assert_awaited_once()
        assert sdk.instance.record_thinking.await_args.args[0] == "look up the booking"

    @pytest.mark.asyncio
    async def test_each_reasoning_step_carries_its_index(self):
        sdk = _streaming_sdk(True)
        await _handler_for(sdk).on_llm_end(
            _llm_result([
                {"type": "reasoning", "reasoning": "first"},
                {"type": "reasoning", "reasoning": "second"},
            ]),
            run_id=uuid4(),
        )
        indexes = [c.kwargs["step_index"]
                   for c in sdk.instance.record_thinking.await_args_list]
        assert indexes == [0, 1]

    @pytest.mark.asyncio
    async def test_reasoning_goes_out_before_the_assistant_turn(self):
        """The model reasoned and then answered; anticipation acts on the
        answer, so it has to be the later of the two."""
        sdk = _streaming_sdk(True)
        order = []
        sdk.instance.record_thinking = AsyncMock(
            side_effect=lambda *a, **k: order.append("reasoning"))
        sdk.instance.send_message = AsyncMock(
            side_effect=lambda *a, **k: order.append("turn"))
        await _handler_for(sdk).on_llm_end(
            _llm_result([
                {"type": "reasoning", "reasoning": "thinking"},
                {"type": "text", "text": "it ships tomorrow"},
            ]),
            run_id=uuid4(),
        )
        assert order == ["reasoning", "turn"]

    @pytest.mark.asyncio
    async def test_a_tool_calling_response_reports_reasoning_and_no_turn(self):
        """The AIMessage that asks for a tool has reasoning but no text, so
        it must not be reported as the assistant's answer."""
        sdk = _streaming_sdk(True)
        await _handler_for(sdk).on_llm_end(
            _llm_result([{"type": "reasoning", "reasoning": "I need the booking"}]),
            run_id=uuid4(),
        )
        sdk.instance.record_thinking.assert_awaited_once()
        sdk.instance.send_message.assert_not_awaited()
        sdk.conversation.record_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# None of it breaks the graph
# ---------------------------------------------------------------------------


class TestNoneOfItBreaksTheGraph:
    """A raising callback aborts the whole LangGraph run, so every one of
    these must be silent on failure, not merely unlikely to fail."""

    @pytest.mark.asyncio
    async def test_reasoning_on_a_dead_sdk_does_not_raise(self):
        sdk = _streaming_sdk(True)
        sdk.instance.record_thinking = AsyncMock(side_effect=RuntimeError("boom"))
        await _handler_for(sdk).on_llm_end(
            _llm_result([{"type": "reasoning", "reasoning": "thinking"}]),
            run_id=uuid4(),
        )

    @pytest.mark.asyncio
    async def test_a_tool_call_on_a_dead_sdk_does_not_raise(self):
        sdk = _streaming_sdk(True)
        sdk.instance.record_tool_call = AsyncMock(side_effect=RuntimeError("boom"))
        await _handler_for(sdk).on_tool_start({"name": "t"}, "in", run_id=uuid4())

    @pytest.mark.asyncio
    async def test_a_user_turn_on_a_dead_sdk_does_not_raise(self):
        sdk = _streaming_sdk(False)
        sdk.conversation.record_message = AsyncMock(side_effect=RuntimeError("boom"))
        await _handler_for(sdk).on_chat_model_start(
            {}, [[_human("hi", "h1")]], run_id=uuid4())

    @pytest.mark.asyncio
    async def test_an_sdk_with_no_instance_namespace_does_not_raise(self):
        """An older SDK, or a test double. Every stream call must no-op."""
        sdk = MagicMock(spec=["conversation"])
        sdk.conversation.record_message = AsyncMock()
        handler = SynapLangGraphCallbackHandler(
            sdk=sdk, conversation_id="c", user_id="u")
        await handler.on_tool_start({"name": "t"}, "in", run_id=uuid4())
        await handler.on_tool_end("out", run_id=uuid4())
        await handler.on_llm_end(
            _llm_result([{"type": "reasoning", "reasoning": "thinking"}]),
            run_id=uuid4(),
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_rejects_a_none_sdk(self):
        with pytest.raises(ValueError, match="non-None sdk"):
            SynapLangGraphCallbackHandler(sdk=None, conversation_id="c", user_id="u")

    def test_rejects_an_empty_conversation_id(self):
        with pytest.raises(ValueError, match="non-empty conversation_id"):
            SynapLangGraphCallbackHandler(
                sdk=MagicMock(), conversation_id="", user_id="u")

    def test_rejects_an_empty_user_id(self):
        with pytest.raises(ValueError, match="non-empty user_id"):
            SynapLangGraphCallbackHandler(
                sdk=MagicMock(), conversation_id="c", user_id="")

    def test_customer_id_defaults_to_empty(self):
        handler = SynapLangGraphCallbackHandler(
            sdk=MagicMock(), conversation_id="c", user_id="u")
        assert handler.customer_id == ""
