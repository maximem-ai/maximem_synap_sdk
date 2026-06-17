"""Tests for SynapChatMemory.

Documented error-handling contract (from memory.py docstring):
- aget (read path): best-effort — swallows SDK errors, returns partial messages + ERROR log.
- aput (write path): strict — wraps SDK errors as SynapIntegrationError and re-raises.
  The caller explicitly wrote data and has a right to know if it didn't land.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from synap_llamaindex.memory import SynapChatMemory
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Fixtures — local, tests own their context
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sdk():
    sdk = MagicMock()
    sdk.fetch = AsyncMock()
    sdk.conversation = MagicMock()
    sdk.conversation.context = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock()
    sdk.conversation.record_message = AsyncMock(return_value={
        "message_id": "msg-001", "conversation_id": "conv-1",
    })
    sdk.cache = MagicMock()
    return sdk


@pytest.fixture
def memory(mock_sdk):
    return SynapChatMemory(
        sdk=mock_sdk,
        conversation_id="conv-1",
        user_id="u1",
        customer_id="c1",
    )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_surface_exports():
    import synap_llamaindex
    assert hasattr(synap_llamaindex, "SynapChatMemory")
    assert "SynapChatMemory" in synap_llamaindex.__all__


def test_import_all_public():
    from synap_llamaindex import SynapChatMemory, SynapRetriever, synap_st_chat_message
    assert SynapChatMemory is not None
    assert SynapRetriever is not None
    assert synap_st_chat_message is not None


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


def test_init_raises_on_none_sdk():
    with pytest.raises(ValueError, match="non-None sdk"):
        SynapChatMemory(sdk=None, conversation_id="c", user_id="u")


def test_init_raises_on_empty_conversation_id(mock_sdk):
    with pytest.raises(ValueError, match="non-empty conversation_id"):
        SynapChatMemory(sdk=mock_sdk, conversation_id="", user_id="u")


def test_init_raises_on_empty_user_id(mock_sdk):
    with pytest.raises(ValueError, match="non-empty user_id"):
        SynapChatMemory(sdk=mock_sdk, conversation_id="c", user_id="")


def test_customer_id_defaults_to_empty_string(mock_sdk):
    """customer_id is optional; default is '' (not None)."""
    mem = SynapChatMemory(sdk=mock_sdk, conversation_id="c", user_id="u")
    assert mem._customer_id == ""


def test_from_defaults_happy_path(mock_sdk):
    mem = SynapChatMemory.from_defaults(
        sdk=mock_sdk, conversation_id="conv-1", user_id="u1", customer_id="c1"
    )
    assert isinstance(mem, SynapChatMemory)
    assert mem._conversation_id == "conv-1"
    assert mem._user_id == "u1"
    assert mem._customer_id == "c1"


def test_from_defaults_raises_when_no_sdk():
    with pytest.raises(ValueError, match="from_defaults requires sdk"):
        SynapChatMemory.from_defaults(conversation_id="c", user_id="u")


# ---------------------------------------------------------------------------
# aget — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aget_returns_system_plus_recent_messages(memory, mock_sdk):
    """aget returns system context message + recent messages from SDK."""
    from datetime import datetime, timezone
    msg_user = MagicMock(role="user", content="hello")
    msg_asst = MagicMock(role="assistant", content="hi there")
    mock_ctx = MagicMock(
        recent_messages=[msg_user, msg_asst],
        formatted_context="Compacted summary",
        available=True,
    )
    mock_sdk.conversation.context.get_context_for_prompt.return_value = mock_ctx
    mock_sdk.fetch.return_value = MagicMock(formatted_context=None)

    msgs = await memory.aget()

    assert len(msgs) >= 2
    # The system prompt from formatted_context + user and assistant messages
    roles = [m.role for m in msgs]
    assert MessageRole.USER in roles
    assert MessageRole.ASSISTANT in roles
    mock_sdk.conversation.context.get_context_for_prompt.assert_awaited_once()


@pytest.mark.asyncio
async def test_aget_with_input_calls_fetch(memory, mock_sdk):
    """When input is provided, aget calls sdk.fetch with search_query."""
    mock_sdk.fetch.return_value = MagicMock(
        formatted_context="Relevant memory: User likes tea."
    )
    mock_sdk.conversation.context.get_context_for_prompt.return_value = MagicMock(
        recent_messages=[], formatted_context=None
    )

    msgs = await memory.aget(input="tea preferences")

    mock_sdk.fetch.assert_awaited_once()
    call_kwargs = mock_sdk.fetch.call_args.kwargs
    assert call_kwargs["conversation_id"] == "conv-1"
    assert call_kwargs["user_id"] == "u1"
    assert call_kwargs["search_query"] == ["tea preferences"]
    assert call_kwargs["include_conversation_context"] is False
    # System message from fetch result should be present
    system_msgs = [m for m in msgs if m.role == MessageRole.SYSTEM]
    assert any("Relevant memory" in str(m.content) for m in system_msgs)


@pytest.mark.asyncio
async def test_aget_no_input_skips_fetch(memory, mock_sdk):
    """Without input, aget should NOT call sdk.fetch."""
    mock_sdk.conversation.context.get_context_for_prompt.return_value = MagicMock(
        recent_messages=[], formatted_context=None
    )

    await memory.aget()

    mock_sdk.fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_aget_formatted_context_becomes_system_message(memory, mock_sdk):
    """A non-empty formatted_context is wrapped in a SYSTEM ChatMessage."""
    mock_sdk.conversation.context.get_context_for_prompt.return_value = MagicMock(
        recent_messages=[],
        formatted_context="Conversation history text",
    )
    mock_sdk.fetch.return_value = MagicMock(formatted_context=None)

    msgs = await memory.aget()

    system_msgs = [m for m in msgs if m.role == MessageRole.SYSTEM]
    assert len(system_msgs) >= 1
    assert any("Conversation history" in str(m.content) for m in system_msgs)


@pytest.mark.asyncio
async def test_aget_user_role_maps_to_user(memory, mock_sdk):
    """Messages with role='user' map to MessageRole.USER."""
    msg = MagicMock(role="user", content="user message")
    mock_sdk.conversation.context.get_context_for_prompt.return_value = MagicMock(
        recent_messages=[msg], formatted_context=None
    )
    mock_sdk.fetch.return_value = MagicMock(formatted_context=None)

    msgs = await memory.aget()
    user_msgs = [m for m in msgs if m.role == MessageRole.USER]
    assert len(user_msgs) == 1
    assert user_msgs[0].content == "user message"


@pytest.mark.asyncio
async def test_aget_assistant_role_maps_to_assistant(memory, mock_sdk):
    """Messages with role='assistant' map to MessageRole.ASSISTANT."""
    msg = MagicMock(role="assistant", content="assistant reply")
    mock_sdk.conversation.context.get_context_for_prompt.return_value = MagicMock(
        recent_messages=[msg], formatted_context=None
    )
    mock_sdk.fetch.return_value = MagicMock(formatted_context=None)

    msgs = await memory.aget()
    asst_msgs = [m for m in msgs if m.role == MessageRole.ASSISTANT]
    assert len(asst_msgs) == 1
    assert asst_msgs[0].content == "assistant reply"


@pytest.mark.asyncio
async def test_aget_appends_local_messages(memory, mock_sdk):
    """Messages previously stored via aput are appended at the end."""
    mock_sdk.conversation.context.get_context_for_prompt.return_value = MagicMock(
        recent_messages=[], formatted_context=None
    )
    mock_sdk.fetch.return_value = MagicMock(formatted_context=None)
    # Pre-populate local messages
    memory._messages = [ChatMessage(role=MessageRole.USER, content="local-msg")]

    msgs = await memory.aget()

    local = [m for m in msgs if m.content == "local-msg"]
    assert len(local) == 1


# ---------------------------------------------------------------------------
# aget — failure paths (documented: swallow + log ERROR)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aget_swallows_fetch_failure_and_continues(memory, mock_sdk):
    """fetch failure must NOT abort aget — context.get_context_for_prompt is still called."""
    mock_sdk.fetch.side_effect = RuntimeError("fetch boom")
    mock_sdk.conversation.context.get_context_for_prompt.return_value = MagicMock(
        recent_messages=[MagicMock(role="user", content="fallback")],
        formatted_context=None,
    )

    msgs = await memory.aget(input="something")

    # Must not raise; conversation context should still be returned
    assert any(m.content == "fallback" for m in msgs)
    mock_sdk.conversation.context.get_context_for_prompt.assert_awaited_once()


@pytest.mark.asyncio
async def test_aget_swallows_context_failure_and_returns_partial(memory, mock_sdk):
    """get_context_for_prompt failure must NOT propagate — returns what was collected."""
    mock_sdk.fetch.return_value = MagicMock(
        formatted_context="Relevant: tea lover."
    )
    mock_sdk.conversation.context.get_context_for_prompt.side_effect = RuntimeError("ctx boom")

    # input provided so fetch is also called
    msgs = await memory.aget(input="tea")

    # Must not raise; fetch result IS available
    assert not any(isinstance(m, Exception) for m in msgs)


@pytest.mark.asyncio
async def test_aget_logs_error_on_fetch_failure(memory, mock_sdk, caplog):
    """SDK fetch failure must be logged at ERROR with conversation_id context."""
    mock_sdk.fetch.side_effect = RuntimeError("fetch exploded")
    mock_sdk.conversation.context.get_context_for_prompt.return_value = MagicMock(
        recent_messages=[], formatted_context=None
    )

    with caplog.at_level(logging.ERROR, logger="synap_llamaindex.memory"):
        await memory.aget(input="query")

    assert any("conv-1" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_aget_logs_error_on_context_failure(memory, mock_sdk, caplog):
    """get_context_for_prompt failure must be logged at ERROR."""
    mock_sdk.fetch.return_value = MagicMock(formatted_context=None)
    mock_sdk.conversation.context.get_context_for_prompt.side_effect = RuntimeError("boom")

    with caplog.at_level(logging.ERROR, logger="synap_llamaindex.memory"):
        await memory.aget()

    assert any("conv-1" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# aput — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aput_records_user_message(memory, mock_sdk):
    """aput with USER role calls record_message with role='user'."""
    msg = ChatMessage(role=MessageRole.USER, content="hi there")

    await memory.aput(msg)

    mock_sdk.conversation.record_message.assert_awaited_once()
    kw = mock_sdk.conversation.record_message.call_args.kwargs
    assert kw["role"] == "user"
    assert kw["content"] == "hi there"
    assert kw["conversation_id"] == "conv-1"
    assert kw["user_id"] == "u1"
    assert kw["customer_id"] == "c1"


@pytest.mark.asyncio
async def test_aput_records_assistant_message(memory, mock_sdk):
    """aput with ASSISTANT role calls record_message with role='assistant'."""
    msg = ChatMessage(role=MessageRole.ASSISTANT, content="hello there")

    await memory.aput(msg)

    kw = mock_sdk.conversation.record_message.call_args.kwargs
    assert kw["role"] == "assistant"
    assert kw["content"] == "hello there"


@pytest.mark.asyncio
async def test_aput_appends_to_local_messages(memory, mock_sdk):
    """aput always appends the message to _messages regardless of role."""
    msg = ChatMessage(role=MessageRole.USER, content="stored locally")

    await memory.aput(msg)

    assert len(memory._messages) == 1
    assert memory._messages[0].content == "stored locally"


@pytest.mark.asyncio
async def test_aput_system_role_skips_record_message(memory, mock_sdk):
    """System messages are stored locally but NOT forwarded to record_message."""
    msg = ChatMessage(role=MessageRole.SYSTEM, content="you are helpful")

    await memory.aput(msg)

    mock_sdk.conversation.record_message.assert_not_awaited()
    # Still stored locally
    assert len(memory._messages) == 1


# ---------------------------------------------------------------------------
# aput — failure path (documented: raise SynapIntegrationError)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aput_raises_synap_integration_error_on_sdk_failure(memory, mock_sdk):
    """aput MUST raise SynapIntegrationError when the SDK record_message fails.

    wrap_sdk_errors_async is used in memory.py — SDK errors are surfaced as
    SynapIntegrationError. Callers need to know when writes fail.
    """
    mock_sdk.conversation.record_message.side_effect = RuntimeError("record boom")

    with pytest.raises(SynapIntegrationError):
        await memory.aput(ChatMessage(role=MessageRole.USER, content="hello"))


@pytest.mark.asyncio
async def test_aput_error_preserves_original_cause(memory, mock_sdk):
    """SynapIntegrationError must chain the original SDK exception as __cause__."""
    original = RuntimeError("original error")
    mock_sdk.conversation.record_message.side_effect = original

    with pytest.raises(SynapIntegrationError) as exc_info:
        await memory.aput(ChatMessage(role=MessageRole.USER, content="hi"))

    assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# set / aset / reset / areset
# ---------------------------------------------------------------------------


def test_set_replaces_local_messages(memory):
    msgs = [
        ChatMessage(role=MessageRole.USER, content="msg1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="msg2"),
    ]
    memory.set(msgs)
    assert memory._messages == msgs


@pytest.mark.asyncio
async def test_aset_replaces_local_messages(memory):
    msgs = [ChatMessage(role=MessageRole.USER, content="async-set")]
    await memory.aset(msgs)
    assert memory._messages == msgs


def test_reset_clears_messages(memory):
    memory._messages = [ChatMessage(role=MessageRole.USER, content="old")]
    memory.reset()
    assert memory._messages == []


@pytest.mark.asyncio
async def test_areset_clears_messages(memory):
    memory._messages = [ChatMessage(role=MessageRole.USER, content="old")]
    await memory.areset()
    assert memory._messages == []


# ---------------------------------------------------------------------------
# Sync wrappers — get / get_all / put
# ---------------------------------------------------------------------------


def test_get_all_delegates_to_get(memory, mock_sdk):
    """get_all() is documented to return get() — they must return the same type."""
    mock_sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=MagicMock(recent_messages=[], formatted_context=None)
    )
    mock_sdk.fetch = AsyncMock(return_value=MagicMock(formatted_context=None))
    result = memory.get_all()
    assert isinstance(result, list)


def test_get_sync_returns_list(memory, mock_sdk):
    """get() is a sync wrapper around aget(); must return a list."""
    mock_sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=MagicMock(recent_messages=[], formatted_context=None)
    )
    mock_sdk.fetch = AsyncMock(return_value=MagicMock(formatted_context=None))
    result = memory.get()
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_aget_all_delegates_to_aget(memory, mock_sdk):
    """aget_all() calls aget() without input."""
    mock_sdk.conversation.context.get_context_for_prompt.return_value = MagicMock(
        recent_messages=[], formatted_context=None
    )
    mock_sdk.fetch.return_value = MagicMock(formatted_context=None)
    result = await memory.aget_all()
    assert isinstance(result, list)
