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
