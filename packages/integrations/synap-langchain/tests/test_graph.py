"""Tests for create_synap_node (LangGraph integration).

Documented error-handling contract (from graph.py / wrap_sdk_errors_async):
- SDK errors are wrapped as SynapIntegrationError and re-raised.
- Callers (LangGraph) handle the exception; the node never silently swallows failures.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from synap_langchain.graph import create_synap_node
from synap_integrations_common import SynapIntegrationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sdk():
    sdk = MagicMock()
    sdk.fetch = AsyncMock()
    return sdk


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_create_synap_node_raises_on_none_sdk():
    with pytest.raises(ValueError, match="non-None sdk"):
        create_synap_node(None, user_id="u1")  # type: ignore[arg-type]


def test_create_synap_node_raises_on_empty_user_id(mock_sdk):
    with pytest.raises(ValueError, match="non-empty user_id"):
        create_synap_node(mock_sdk, user_id="")


def test_create_synap_node_returns_callable(mock_sdk):
    node = create_synap_node(mock_sdk, user_id="u1")
    assert callable(node)


def test_create_synap_node_function_is_named(mock_sdk):
    """Node function has a predictable __name__ for graph introspection."""
    node = create_synap_node(mock_sdk, user_id="u1")
    assert node.__name__ == "synap_memory"


# ---------------------------------------------------------------------------
# Happy path — query extraction and state injection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_extracts_query_from_human_messages(mock_sdk):
    """Human message (LangChain message object) is used as the search query."""
    mock_sdk.fetch.return_value = MagicMock(formatted_context="context here")

    node = create_synap_node(mock_sdk, user_id="u1", conversation_id="conv-1")
    human_msg = MagicMock(type="human", content="what is my budget?")
    state = {"messages": [human_msg]}

    result = await node(state)

    assert result == {"synap_context": "context here"}
    call_kwargs = mock_sdk.fetch.call_args.kwargs
    assert call_kwargs["search_query"] == ["what is my budget?"]
    assert call_kwargs["conversation_id"] == "conv-1"
    assert call_kwargs["user_id"] == "u1"


@pytest.mark.asyncio
async def test_node_extracts_query_from_dict_messages(mock_sdk):
    """Dict-style messages (e.g., {'role': 'user', 'content': '...'}) also work."""
    mock_sdk.fetch.return_value = MagicMock(formatted_context="ctx")

    node = create_synap_node(mock_sdk, user_id="u1")
    state = {"messages": [{"role": "user", "content": "hello from dict"}]}

    await node(state)
    assert mock_sdk.fetch.call_args.kwargs["search_query"] == ["hello from dict"]


@pytest.mark.asyncio
async def test_node_no_messages_passes_none_query(mock_sdk):
    """Empty message list → search_query=None (broad fetch)."""
    mock_sdk.fetch.return_value = MagicMock(formatted_context="")

    node = create_synap_node(mock_sdk, user_id="u1")
    result = await node({"messages": []})

    assert result == {"synap_context": ""}
    assert mock_sdk.fetch.call_args.kwargs["search_query"] is None


@pytest.mark.asyncio
async def test_node_uses_last_human_message_when_multiple_exist(mock_sdk):
    """Multiple messages: last human message (reversed scan) is used as query."""
    mock_sdk.fetch.return_value = MagicMock(formatted_context="ctx")

    node = create_synap_node(mock_sdk, user_id="u1")
    msg1 = MagicMock(type="human", content="first")
    msg2 = MagicMock(type="assistant", content="mid")
    msg3 = MagicMock(type="human", content="latest")
    state = {"messages": [msg1, msg2, msg3]}

    await node(state)
    assert mock_sdk.fetch.call_args.kwargs["search_query"] == ["latest"]


@pytest.mark.asyncio
async def test_node_custom_state_key(mock_sdk):
    """state_key parameter controls which key receives the context."""
    mock_sdk.fetch.return_value = MagicMock(formatted_context="ctx")

    node = create_synap_node(mock_sdk, user_id="u1", state_key="memory")
    result = await node({"messages": []})

    assert "memory" in result
    assert result["memory"] == "ctx"


@pytest.mark.asyncio
async def test_node_custom_messages_key(mock_sdk):
    """messages_key parameter controls which state key is read for messages."""
    mock_sdk.fetch.return_value = MagicMock(formatted_context="ctx")

    node = create_synap_node(mock_sdk, user_id="u1", messages_key="chat_history")
    human_msg = MagicMock(type="human", content="custom key query")
    result = await node({"chat_history": [human_msg]})

    assert mock_sdk.fetch.call_args.kwargs["search_query"] == ["custom key query"]


@pytest.mark.asyncio
async def test_node_reads_conversation_id_from_state_when_not_set(mock_sdk):
    """conversation_id falls back to state['conversation_id'] when not fixed at creation."""
    mock_sdk.fetch.return_value = MagicMock(formatted_context="")

    node = create_synap_node(mock_sdk, user_id="u1")  # no conversation_id
    await node({"messages": [], "conversation_id": "from-state-123"})

    assert mock_sdk.fetch.call_args.kwargs["conversation_id"] == "from-state-123"


@pytest.mark.asyncio
async def test_node_passes_include_scope_labels(mock_sdk):
    """include_scope_labels is forwarded to sdk.fetch."""
    mock_sdk.fetch.return_value = MagicMock(formatted_context="ctx")

    node = create_synap_node(mock_sdk, user_id="u1", include_scope_labels=True)
    await node({"messages": []})

    assert mock_sdk.fetch.call_args.kwargs["include_scope_labels"] is True


@pytest.mark.asyncio
async def test_node_none_formatted_context_becomes_empty_string(mock_sdk):
    """response.formatted_context=None → state key value is ''."""
    mock_sdk.fetch.return_value = MagicMock(formatted_context=None)

    node = create_synap_node(mock_sdk, user_id="u1")
    result = await node({"messages": []})

    assert result["synap_context"] == ""


# ---------------------------------------------------------------------------
# Failure path — SDK raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_raises_synap_integration_error_on_sdk_failure(mock_sdk):
    """SDK failure is wrapped as SynapIntegrationError (wrap_sdk_errors_async contract)."""
    mock_sdk.fetch.side_effect = RuntimeError("sdk boom")

    node = create_synap_node(mock_sdk, user_id="u1")

    with pytest.raises(SynapIntegrationError):
        await node({"messages": []})
