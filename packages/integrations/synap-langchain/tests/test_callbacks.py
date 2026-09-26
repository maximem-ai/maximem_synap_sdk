"""Tests for SynapCallbackHandler and the _extract_text helper.

Documented error-handling contract (from callbacks.py docstring):
- 'LangChain callbacks must not raise — a raising callback aborts the whole chain.'
- SDK failures are logged at ERROR (not DEBUG) and swallowed.
- _extract_text prefers multi-block list content over .text to avoid truncation.
"""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage, SystemMessage

from synap_langchain.callbacks import SynapCallbackHandler, _extract_text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sdk():
    sdk = MagicMock()
    sdk.conversation.record_message = AsyncMock()
    return sdk


@pytest.fixture
def handler(mock_sdk):
    return SynapCallbackHandler(
        sdk=mock_sdk,
        conversation_id="conv-1",
        user_id="user-1",
        customer_id="cust-1",
    )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_init_raises_on_none_sdk():
    with pytest.raises(ValueError, match="non-None sdk"):
        SynapCallbackHandler(sdk=None, conversation_id="c", user_id="u")


def test_init_raises_on_empty_conversation_id(mock_sdk):
    with pytest.raises(ValueError, match="non-empty conversation_id"):
        SynapCallbackHandler(sdk=mock_sdk, conversation_id="", user_id="u")


def test_init_raises_on_empty_user_id(mock_sdk):
    with pytest.raises(ValueError, match="non-empty user_id"):
        SynapCallbackHandler(sdk=mock_sdk, conversation_id="c", user_id="")


def test_customer_id_defaults_to_empty_string(mock_sdk):
    h = SynapCallbackHandler(sdk=mock_sdk, conversation_id="c", user_id="u")
    assert h.customer_id == ""


# ---------------------------------------------------------------------------
# _extract_text — unit tests for the dispatch logic
# ---------------------------------------------------------------------------


def test_extract_text_returns_plain_text_from_generation():
    """.text is returned when there is no message attribute."""
    gen = MagicMock(spec=[])  # no message attr
    gen.text = "plain text response"
    assert _extract_text(gen) == "plain text response"


def test_extract_text_prefers_message_when_content_is_string():
    """Falls through to message.content (string) when .text is empty."""
    message = MagicMock()
    message.content = "content from message"
    gen = MagicMock()
    gen.message = message
    gen.text = ""
    assert _extract_text(gen) == "content from message"


def test_extract_text_concatenates_list_content_parts():
    """Multi-block list content is fully concatenated (not just first block)."""
    message = MagicMock()
    message.content = [
        {"type": "text", "text": "Block one. "},
        {"type": "text", "text": "Block two."},
        {"type": "tool_use", "id": "tool-1"},  # non-text; should be skipped
    ]
    gen = MagicMock()
    gen.message = message
    assert _extract_text(gen) == "Block one. Block two."


def test_extract_text_concatenates_mixed_list_string_and_dict():
    """String items in the content list are also included."""
    message = MagicMock()
    message.content = [
        "raw string part",
        {"type": "text", "text": " dict part"},
    ]
    gen = MagicMock()
    gen.message = message
    assert _extract_text(gen) == "raw string part dict part"


def test_extract_text_returns_empty_string_for_empty_content():
    gen = MagicMock(spec=[])
    gen.text = ""
    assert _extract_text(gen) == ""


def test_extract_text_skips_non_text_dict_entries():
    """Dict entries without type='text' are silently skipped."""
    message = MagicMock()
    message.content = [
        {"type": "image_url", "url": "http://example.com/img.png"},
        {"type": "text", "text": "only this"},
    ]
    gen = MagicMock()
    gen.message = message
    assert _extract_text(gen) == "only this"


# ---------------------------------------------------------------------------
# on_chat_model_start — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_chat_model_start_records_last_human_message(handler, mock_sdk):
    """The last human message in the last batch is recorded as 'user'."""
    human_msg = MagicMock(type="human", content="hello world")
    system_msg = MagicMock(type="system", content="you are helpful")

    await handler.on_chat_model_start(
        serialized={},
        messages=[[system_msg, human_msg]],
        run_id=uuid4(),
    )

    mock_sdk.conversation.record_message.assert_awaited_once_with(
        conversation_id="conv-1",
        role="user",
        content="hello world",
        user_id="user-1",
        customer_id="cust-1",
    )


@pytest.mark.asyncio
async def test_on_chat_model_start_empty_messages_does_not_call_sdk(handler, mock_sdk):
    """Empty messages list → no SDK call (guard branch)."""
    await handler.on_chat_model_start(
        serialized={}, messages=[], run_id=uuid4(),
    )
    mock_sdk.conversation.record_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_chat_model_start_no_human_message_does_not_call_sdk(handler, mock_sdk):
    """Batch with no human message → no SDK call."""
    system_msg = MagicMock(type="system", content="sys")
    await handler.on_chat_model_start(
        serialized={},
        messages=[[system_msg]],
        run_id=uuid4(),
    )
    mock_sdk.conversation.record_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_chat_model_start_uses_last_batch(handler, mock_sdk):
    """Only the last message batch is examined (messages[-1])."""
    early_human = MagicMock(type="human", content="earlier turn")
    late_human = MagicMock(type="human", content="latest turn")
    await handler.on_chat_model_start(
        serialized={},
        messages=[[early_human], [late_human]],
        run_id=uuid4(),
    )
    kw = mock_sdk.conversation.record_message.call_args.kwargs
    assert kw["content"] == "latest turn"


# ---------------------------------------------------------------------------
# on_chat_model_start — failure path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_chat_model_start_swallows_sdk_error(handler, mock_sdk):
    """SDK failure during user-message recording must NOT raise (contract: callbacks never raise)."""
    mock_sdk.conversation.record_message.side_effect = RuntimeError("sdk boom")
    human_msg = MagicMock(type="human", content="hi")

    # Must not raise
    await handler.on_chat_model_start(
        serialized={},
        messages=[[human_msg]],
        run_id=uuid4(),
    )


# ---------------------------------------------------------------------------
# on_llm_end — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_llm_end_records_assistant_response(handler, mock_sdk):
    gen = MagicMock(text="I can help with that")
    response = MagicMock(generations=[[gen]])

    await handler.on_llm_end(response=response, run_id=uuid4())

    mock_sdk.conversation.record_message.assert_awaited_once_with(
        conversation_id="conv-1",
        role="assistant",
        content="I can help with that",
        user_id="user-1",
        customer_id="cust-1",
    )


@pytest.mark.asyncio
async def test_on_llm_end_empty_generations_does_not_call_sdk(handler, mock_sdk):
    """Empty generations list → no SDK call."""
    response = MagicMock(generations=[])
    await handler.on_llm_end(response=response, run_id=uuid4())
    mock_sdk.conversation.record_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_llm_end_empty_first_generation_does_not_call_sdk(handler, mock_sdk):
    """Empty first generation batch → no SDK call."""
    response = MagicMock(generations=[[]])
    await handler.on_llm_end(response=response, run_id=uuid4())
    mock_sdk.conversation.record_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_llm_end_empty_text_does_not_call_sdk(handler, mock_sdk):
    """Generation that extracts empty text → no SDK call (guard branch)."""
    gen = MagicMock()
    gen.text = ""
    gen.message = None
    response = MagicMock(generations=[[gen]])
    await handler.on_llm_end(response=response, run_id=uuid4())
    mock_sdk.conversation.record_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# on_llm_end — failure path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_llm_end_swallows_sdk_error(handler, mock_sdk):
    """SDK failure during assistant-message recording must NOT raise."""
    mock_sdk.conversation.record_message.side_effect = Exception("fail")
    gen = MagicMock(text="response")
    response = MagicMock(generations=[[gen]])

    # Must not raise
    await handler.on_llm_end(response=response, run_id=uuid4())


@pytest.mark.asyncio
async def test_on_llm_end_logs_error_on_failure(handler, mock_sdk, caplog):
    """SDK failure during on_llm_end is logged at ERROR level."""
    import logging
    mock_sdk.conversation.record_message.side_effect = RuntimeError("boom")
    gen = MagicMock(text="text")
    response = MagicMock(generations=[[gen]])

    with caplog.at_level(logging.ERROR, logger="synap_langchain.callbacks"):
        await handler.on_llm_end(response=response, run_id=uuid4())

    assert len(caplog.records) >= 1


# ---------------------------------------------------------------------------
# The live stream
#
# This handler recorded the user turn and the assistant turn and stopped
# there, so anticipation saw a conversation with questions and answers and
# nothing in between: no tool calls, no results, no reasoning. Those are the
# events that say what the agent is about to need.
#
# The rule these pin is "stream first, REST only as a fallback, never both".
# The server persists `user_message` and `assistant_message` from the stream
# itself, so a `record_message` on top of a delivered stream event writes the
# same turn twice and extracts it twice.
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
    return SynapCallbackHandler(
        sdk=sdk, conversation_id="conv-1", user_id="user-1", customer_id="cust-1",
    )


class TestStreamFirstRestFallback:
    @pytest.mark.asyncio
    async def test_a_turn_goes_out_on_the_stream_when_one_is_open(self):
        sdk = _streaming_sdk(True)
        await _handler_for(sdk)._record("user", "where is my order")
        sdk.instance.send_message.assert_awaited_once()
        assert sdk.instance.send_message.await_args.kwargs["event_type"] == "user_message"

    @pytest.mark.asyncio
    async def test_and_is_NOT_also_written_over_rest(self):
        """The double-write this rule exists to prevent."""
        sdk = _streaming_sdk(True)
        await _handler_for(sdk)._record("user", "where is my order")
        sdk.conversation.record_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_with_no_stream_it_falls_back_to_rest(self):
        sdk = _streaming_sdk(False)
        await _handler_for(sdk)._record("assistant", "it ships tomorrow")
        sdk.conversation.record_message.assert_awaited_once()
        sdk.instance.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_assistant_turn_is_the_anticipation_moment(self):
        sdk = _streaming_sdk(True)
        await _handler_for(sdk)._record("assistant", "it ships tomorrow")
        kw = sdk.instance.send_message.await_args.kwargs
        assert kw["event_type"] == "assistant_message"
        assert kw["role"] == "assistant"


class TestTheThreeEventsItNeverReported:
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
    async def test_a_tool_result_is_reported(self):
        sdk = _streaming_sdk(True)
        rid = uuid4()
        await _handler_for(sdk).on_tool_end({"status": "shipped"}, run_id=rid)
        sdk.instance.record_tool_result.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_the_call_and_its_result_share_an_id(self):
        """Without this the anticipation agent sees a call and a result and
        cannot tell they are the same tool invocation."""
        sdk = _streaming_sdk(True)
        rid = uuid4()
        h = _handler_for(sdk)
        await h.on_tool_start({"name": "t"}, "in", run_id=rid)
        await h.on_tool_end("out", run_id=rid)
        assert (sdk.instance.record_tool_call.await_args.kwargs["tool_call_id"]
                == sdk.instance.record_tool_result.await_args.kwargs["tool_call_id"])

    @pytest.mark.asyncio
    async def test_reasoning_is_reported_from_the_action_log(self):
        sdk = _streaming_sdk(True)
        action = MagicMock()
        action.log = "I should look up the booking first"
        await _handler_for(sdk).on_agent_action(action, run_id=uuid4())
        sdk.instance.record_thinking.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_an_empty_reasoning_log_reports_nothing(self):
        sdk = _streaming_sdk(True)
        action = MagicMock()
        action.log = "   "
        await _handler_for(sdk).on_agent_action(action, run_id=uuid4())
        sdk.instance.record_thinking.assert_not_awaited()


class TestNoneOfItBreaksTheChain:
    """A raising callback aborts the whole LangChain run, so every one of
    these must be silent on failure, not merely unlikely to fail."""

    @pytest.mark.asyncio
    async def test_a_tool_call_on_a_dead_sdk_does_not_raise(self):
        sdk = _streaming_sdk(True)
        sdk.instance.record_tool_call = AsyncMock(side_effect=RuntimeError("boom"))
        await _handler_for(sdk).on_tool_start({"name": "t"}, "in", run_id=uuid4())

    @pytest.mark.asyncio
    async def test_reasoning_on_a_dead_sdk_does_not_raise(self):
        sdk = _streaming_sdk(True)
        sdk.instance.record_thinking = AsyncMock(side_effect=RuntimeError("boom"))
        action = MagicMock(); action.log = "thinking"
        await _handler_for(sdk).on_agent_action(action, run_id=uuid4())

    @pytest.mark.asyncio
    async def test_an_sdk_with_no_instance_namespace_does_not_raise(self):
        """An older SDK, or a test double. Every call must no-op."""
        sdk = MagicMock(spec=[])
        h = SynapCallbackHandler(sdk=sdk, conversation_id="c", user_id="u")
        await h.on_tool_start({"name": "t"}, "in", run_id=uuid4())
        await h.on_tool_end("out", run_id=uuid4())
