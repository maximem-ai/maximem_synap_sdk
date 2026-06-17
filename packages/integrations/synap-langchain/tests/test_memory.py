"""Tests for SynapChatMessageHistory (and SynapMemory alias).

Documented error-handling contract (from memory.py docstring):
- aget_messages: best-effort — swallows SDK errors, returns empty list + ERROR log.
- aadd_messages: strict — wraps SDK errors as SynapIntegrationError and re-raises.
  The caller explicitly invoked ``add``; silently dropping messages masks outages.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from synap_langchain.memory import SynapChatMessageHistory, SynapMemory
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Fixtures — local, no dependency on shared harness (tests own their context)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sdk():
    sdk = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock()
    sdk.conversation.record_message = AsyncMock()
    sdk.cache.clear = MagicMock()
    return sdk


@pytest.fixture
def history(mock_sdk):
    return SynapChatMessageHistory(
        sdk=mock_sdk,
        conversation_id="conv-1",
        user_id="user-1",
        customer_id="cust-1",
    )


# ---------------------------------------------------------------------------
# Construction & public-surface
# ---------------------------------------------------------------------------


def test_backward_compat_alias():
    """SynapMemory is the documented backward-compatible alias."""
    assert SynapMemory is SynapChatMessageHistory


def test_init_raises_on_none_sdk():
    with pytest.raises(ValueError, match="non-None sdk"):
        SynapChatMessageHistory(sdk=None, conversation_id="c", user_id="u")


def test_init_raises_on_empty_conversation_id(mock_sdk):
    with pytest.raises(ValueError, match="non-empty conversation_id"):
        SynapChatMessageHistory(sdk=mock_sdk, conversation_id="", user_id="u")


def test_init_raises_on_empty_user_id(mock_sdk):
    with pytest.raises(ValueError, match="non-empty user_id"):
        SynapChatMessageHistory(sdk=mock_sdk, conversation_id="c", user_id="")


def test_customer_id_defaults_to_empty_string(mock_sdk):
    """customer_id is optional; default is '' (not None)."""
    h = SynapChatMessageHistory(sdk=mock_sdk, conversation_id="c", user_id="u")
    assert h.customer_id == ""


# ---------------------------------------------------------------------------
# aget_messages — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aget_messages_returns_human_and_ai_messages(history, mock_sdk):
    mock_msg1 = MagicMock(role="user", content="hello")
    mock_msg2 = MagicMock(role="assistant", content="hi there")
    mock_ctx = MagicMock(recent_messages=[mock_msg1, mock_msg2])
    mock_sdk.conversation.context.get_context_for_prompt.return_value = mock_ctx

    msgs = await history.aget_messages()

    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "hello"
    assert isinstance(msgs[1], AIMessage)
    assert msgs[1].content == "hi there"
    mock_sdk.conversation.context.get_context_for_prompt.assert_awaited_once_with(
        conversation_id="conv-1",
    )


@pytest.mark.asyncio
async def test_aget_messages_empty_recent_messages(history, mock_sdk):
    """An empty recent_messages list → empty result (no crash)."""
    mock_sdk.conversation.context.get_context_for_prompt.return_value = MagicMock(
        recent_messages=[]
    )
    msgs = await history.aget_messages()
    assert msgs == []


@pytest.mark.asyncio
async def test_aget_messages_none_ctx(history, mock_sdk):
    """A falsy (None) context response → empty result (no crash)."""
    mock_sdk.conversation.context.get_context_for_prompt.return_value = None
    msgs = await history.aget_messages()
    assert msgs == []


@pytest.mark.asyncio
async def test_aget_messages_none_recent_messages(history, mock_sdk):
    """recent_messages=None → empty result (no crash)."""
    ctx = MagicMock()
    ctx.recent_messages = None
    mock_sdk.conversation.context.get_context_for_prompt.return_value = ctx
    msgs = await history.aget_messages()
    assert msgs == []


@pytest.mark.asyncio
async def test_aget_messages_unknown_role_maps_to_human(history, mock_sdk):
    """Messages with unrecognised role default to HumanMessage."""
    mock_msg = MagicMock(role="system", content="you are helpful")
    mock_ctx = MagicMock(recent_messages=[mock_msg])
    mock_sdk.conversation.context.get_context_for_prompt.return_value = mock_ctx

    msgs = await history.aget_messages()
    assert len(msgs) == 1
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "you are helpful"


# ---------------------------------------------------------------------------
# aget_messages — failure path (documented: swallow + empty list)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aget_messages_swallows_sdk_error_and_returns_empty(history, mock_sdk):
    """SDK failure during message retrieval must NOT propagate — empty list returned."""
    mock_sdk.conversation.context.get_context_for_prompt.side_effect = Exception("sdk outage")

    msgs = await history.aget_messages()

    assert msgs == []  # must not raise


@pytest.mark.asyncio
async def test_aget_messages_logs_error_on_failure(history, mock_sdk, caplog):
    """SDK failure must be logged at ERROR level with conversation_id context."""
    import logging
    mock_sdk.conversation.context.get_context_for_prompt.side_effect = RuntimeError("boom")

    with caplog.at_level(logging.ERROR, logger="synap_langchain.memory"):
        await history.aget_messages()

    assert len(caplog.records) >= 1
    assert any("conv-1" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# aadd_messages — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aadd_messages_records_each_message(history, mock_sdk):
    """Both HumanMessage and AIMessage are forwarded with the correct role."""
    msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
    await history.aadd_messages(msgs)

    assert mock_sdk.conversation.record_message.await_count == 2
    calls = mock_sdk.conversation.record_message.call_args_list
    assert calls[0].kwargs["role"] == "user"
    assert calls[0].kwargs["content"] == "hi"
    assert calls[1].kwargs["role"] == "assistant"
    assert calls[1].kwargs["content"] == "hello"


@pytest.mark.asyncio
async def test_aadd_messages_passes_conversation_and_user_ids(history, mock_sdk):
    """conversation_id, user_id, and customer_id are forwarded on every call."""
    await history.aadd_messages([HumanMessage(content="hello")])

    kw = mock_sdk.conversation.record_message.call_args.kwargs
    assert kw["conversation_id"] == "conv-1"
    assert kw["user_id"] == "user-1"
    assert kw["customer_id"] == "cust-1"


@pytest.mark.asyncio
async def test_aadd_messages_empty_sequence_is_noop(history, mock_sdk):
    await history.aadd_messages([])
    mock_sdk.conversation.record_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# aadd_messages — failure path (documented: raise SynapIntegrationError)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aadd_messages_raises_synap_integration_error_on_sdk_failure(history, mock_sdk):
    """aadd_messages MUST raise SynapIntegrationError when the SDK fails.

    The docstring is explicit: 'aadd_messages does raise. The caller explicitly
    invoked add; silently dropping messages masks ingestion outages.'

    wrap_sdk_errors_async converts any non-SynapIntegrationError exception into
    SynapIntegrationError and re-raises it — this is the correct, documented
    behavior. Tests MUST assert it.
    """
    mock_sdk.conversation.record_message.side_effect = RuntimeError("sdk boom")

    with pytest.raises(SynapIntegrationError):
        await history.aadd_messages([HumanMessage(content="hi")])


@pytest.mark.asyncio
async def test_aadd_messages_error_preserves_original_cause(history, mock_sdk):
    """The SynapIntegrationError must chain the original SDK exception as __cause__."""
    original = RuntimeError("original error")
    mock_sdk.conversation.record_message.side_effect = original

    with pytest.raises(SynapIntegrationError) as exc_info:
        await history.aadd_messages([HumanMessage(content="hi")])

    assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# clear / aclear
# ---------------------------------------------------------------------------


def test_clear_delegates_to_cache(history, mock_sdk):
    history.clear()
    mock_sdk.cache.clear.assert_called_once()


@pytest.mark.asyncio
async def test_aclear_delegates_to_cache(history, mock_sdk):
    await history.aclear()
    mock_sdk.cache.clear.assert_called_once()
