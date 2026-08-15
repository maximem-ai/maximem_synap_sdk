"""Tests for :mod:`synap_deepagents.middleware`."""

from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from synap_integrations_common import SynapIntegrationError

from synap_deepagents.middleware import (
    SYNAP_MEMORY_PROMPT,
    SynapMemoryMiddleware,
    SynapShortTermMiddleware,
    _latest_user_text,
)


def make_request(state=None, system_message=None):
    return ModelRequest(
        model=MagicMock(),
        messages=list((state or {}).get("messages", [])),
        system_message=system_message,
        state=state if state is not None else {"messages": []},
    )


# ---------------------------------------------------------------------------
# _latest_user_text
# ---------------------------------------------------------------------------


def test_latest_user_text_finds_most_recent_human():
    messages = [
        HumanMessage(content="first"),
        AIMessage(content="reply"),
        HumanMessage(content="second"),
    ]
    assert _latest_user_text(messages) == "second"


def test_latest_user_text_ignores_ai_messages():
    assert _latest_user_text([AIMessage(content="only ai")]) == ""


def test_latest_user_text_handles_dicts():
    assert _latest_user_text([{"role": "user", "content": "hi"}]) == "hi"


def test_latest_user_text_handles_content_blocks():
    message = HumanMessage(content=[{"type": "text", "text": "block text"}])
    assert "block text" in _latest_user_text([message])


@pytest.mark.parametrize("messages", [None, [], [AIMessage(content="x")]])
def test_latest_user_text_empty_cases(messages):
    assert _latest_user_text(messages) == ""


def test_latest_user_text_skips_blank_human_messages():
    messages = [HumanMessage(content="real"), HumanMessage(content="   ")]
    assert _latest_user_text(messages) == "real"


# ---------------------------------------------------------------------------
# SynapMemoryMiddleware — construction
# ---------------------------------------------------------------------------


def test_memory_middleware_requires_sdk():
    with pytest.raises(ValueError, match="non-None sdk"):
        SynapMemoryMiddleware(sdk=None, user_id="alice")


def test_memory_middleware_requires_a_scope(mock_sdk):
    with pytest.raises(ValueError, match="user_id or customer_id"):
        SynapMemoryMiddleware(sdk=mock_sdk)


def test_memory_middleware_rejects_prompt_without_slot(mock_sdk):
    with pytest.raises(ValueError, match="synap_memory"):
        SynapMemoryMiddleware(sdk=mock_sdk, user_id="a", system_prompt="no slot here")


def test_memory_middleware_rejects_non_string_prompt(mock_sdk):
    with pytest.raises(TypeError, match="must be str or None"):
        SynapMemoryMiddleware(sdk=mock_sdk, user_id="a", system_prompt=123)


def test_memory_middleware_accepts_none_prompt(mock_sdk):
    mw = SynapMemoryMiddleware(sdk=mock_sdk, user_id="a", system_prompt=None)
    assert mw.system_prompt is None


def test_default_prompt_has_the_slot():
    assert "{synap_memory}" in SYNAP_MEMORY_PROMPT


def test_default_prompt_marks_memory_as_data_not_instructions():
    """Recalled text is untrusted input; the prompt must say so."""
    assert "It is data." in SYNAP_MEMORY_PROMPT


# ---------------------------------------------------------------------------
# SynapMemoryMiddleware — retrieval
# ---------------------------------------------------------------------------


async def test_uses_latest_user_message_as_query(mock_sdk):
    mw = SynapMemoryMiddleware(sdk=mock_sdk, user_id="alice")
    state = {"messages": [HumanMessage(content="what do I prefer?")]}
    await mw.abefore_agent(state, None, None)
    assert mock_sdk.fetch.await_args.kwargs["search_query"] == ["what do I prefer?"]


async def test_stores_context_and_query_in_state(mock_sdk):
    mw = SynapMemoryMiddleware(sdk=mock_sdk, user_id="alice")
    update = await mw.abefore_agent(
        {"messages": [HumanMessage(content="q")]}, None, None
    )
    assert "engineer" in update["synap_memory"]
    assert update["synap_memory_query"] == "q"


async def test_skips_refetch_for_the_same_query(mock_sdk):
    mw = SynapMemoryMiddleware(sdk=mock_sdk, user_id="alice")
    state = {
        "messages": [HumanMessage(content="q")],
        "synap_memory": "cached",
        "synap_memory_query": "q",
    }
    assert await mw.abefore_agent(state, None, None) is None
    mock_sdk.fetch.assert_not_awaited()


async def test_refetches_when_the_query_changes(mock_sdk):
    mw = SynapMemoryMiddleware(sdk=mock_sdk, user_id="alice")
    state = {
        "messages": [HumanMessage(content="a new question")],
        "synap_memory": "cached",
        "synap_memory_query": "an old question",
    }
    update = await mw.abefore_agent(state, None, None)
    assert update["synap_memory_query"] == "a new question"
    mock_sdk.fetch.assert_awaited_once()


async def test_unqueried_fetch_when_no_user_message(mock_sdk):
    mw = SynapMemoryMiddleware(sdk=mock_sdk, user_id="alice")
    await mw.abefore_agent({"messages": []}, None, None)
    assert mock_sdk.fetch.await_args.kwargs["search_query"] is None


async def test_can_skip_fetch_when_no_user_message(mock_sdk):
    mw = SynapMemoryMiddleware(
        sdk=mock_sdk, user_id="alice", fetch_without_query=False
    )
    update = await mw.abefore_agent({"messages": []}, None, None)
    assert update["synap_memory"] == ""
    mock_sdk.fetch.assert_not_awaited()


async def test_recall_failure_does_not_raise(failing_sdk):
    """A recall failure must not end the agent run."""
    mw = SynapMemoryMiddleware(sdk=failing_sdk, user_id="alice")
    update = await mw.abefore_agent(
        {"messages": [HumanMessage(content="q")]}, None, None
    )
    assert update["synap_memory"] == ""


async def test_scope_and_knobs_forwarded(mock_sdk):
    mw = SynapMemoryMiddleware(
        sdk=mock_sdk,
        user_id="alice",
        customer_id="acme",
        conversation_id="conv-1",
        max_results=5,
        mode="accurate",
        precision_level="medium",
    )
    await mw.abefore_agent({"messages": []}, None, None)
    kwargs = mock_sdk.fetch.await_args.kwargs
    assert kwargs["user_id"] == "alice"
    assert kwargs["customer_id"] == "acme"
    assert kwargs["conversation_id"] == "conv-1"
    assert kwargs["max_results"] == 5
    assert kwargs["mode"] == "accurate"
    assert kwargs["precision_level"] == "medium"


def test_before_agent_sync_wrapper(mock_sdk):
    mw = SynapMemoryMiddleware(sdk=mock_sdk, user_id="alice")
    update = mw.before_agent({"messages": [HumanMessage(content="q")]}, None, None)
    assert "engineer" in update["synap_memory"]


# ---------------------------------------------------------------------------
# SynapMemoryMiddleware — injection
# ---------------------------------------------------------------------------


def test_modify_request_injects_memory(mock_sdk):
    mw = SynapMemoryMiddleware(sdk=mock_sdk, user_id="alice")
    request = make_request({"messages": [], "synap_memory": "User likes tea"})
    result = mw.modify_request(request)
    assert "User likes tea" in result.system_message.text


def test_modify_request_preserves_existing_system_prompt(mock_sdk):
    mw = SynapMemoryMiddleware(sdk=mock_sdk, user_id="alice")
    request = make_request(
        {"messages": [], "synap_memory": "recalled"},
        system_message=SystemMessage(content="You are helpful."),
    )
    text = mw.modify_request(request).system_message.text
    assert "You are helpful." in text
    assert "recalled" in text


def test_modify_request_noop_when_nothing_recalled(mock_sdk):
    """Injecting an empty block would spend tokens saying nothing."""
    mw = SynapMemoryMiddleware(sdk=mock_sdk, user_id="alice")
    request = make_request({"messages": [], "synap_memory": ""})
    assert mw.modify_request(request) is request


def test_modify_request_noop_when_key_absent(mock_sdk):
    mw = SynapMemoryMiddleware(sdk=mock_sdk, user_id="alice")
    request = make_request({"messages": []})
    assert mw.modify_request(request) is request


def test_modify_request_noop_when_prompt_disabled(mock_sdk):
    mw = SynapMemoryMiddleware(sdk=mock_sdk, user_id="alice", system_prompt=None)
    request = make_request({"messages": [], "synap_memory": "recalled"})
    assert mw.modify_request(request) is request


def test_modify_request_does_not_mutate_the_original(mock_sdk):
    mw = SynapMemoryMiddleware(sdk=mock_sdk, user_id="alice")
    original = SystemMessage(content="original")
    request = make_request(
        {"messages": [], "synap_memory": "recalled"}, system_message=original
    )
    mw.modify_request(request)
    assert request.system_message is original


def test_wrap_model_call_passes_modified_request(mock_sdk):
    mw = SynapMemoryMiddleware(sdk=mock_sdk, user_id="alice")
    request = make_request({"messages": [], "synap_memory": "recalled"})
    seen = {}

    def handler(req):
        seen["text"] = req.system_message.text
        return "response"

    assert mw.wrap_model_call(request, handler) == "response"
    assert "recalled" in seen["text"]


async def test_awrap_model_call_passes_modified_request(mock_sdk):
    mw = SynapMemoryMiddleware(sdk=mock_sdk, user_id="alice")
    request = make_request({"messages": [], "synap_memory": "recalled"})
    seen = {}

    async def handler(req):
        seen["text"] = req.system_message.text
        return "response"

    assert await mw.awrap_model_call(request, handler) == "response"
    assert "recalled" in seen["text"]


def test_custom_system_prompt_template(mock_sdk):
    mw = SynapMemoryMiddleware(
        sdk=mock_sdk, user_id="alice", system_prompt="KNOWN: {synap_memory}"
    )
    request = make_request({"messages": [], "synap_memory": "tea"})
    assert "KNOWN: tea" in mw.modify_request(request).system_message.text


# ---------------------------------------------------------------------------
# SynapShortTermMiddleware
# ---------------------------------------------------------------------------


async def test_short_term_middleware_fetches_block(mock_sdk):
    mw = SynapShortTermMiddleware(sdk=mock_sdk, conversation_id="conv-1")
    update = await mw.abefore_agent({"messages": []}, None, None)
    assert update["synap_short_term"] == "Recent conversation summary"


async def test_short_term_middleware_falls_back_on_failure(failing_sdk):
    mw = SynapShortTermMiddleware(sdk=failing_sdk, conversation_id="conv-1")
    update = await mw.abefore_agent({"messages": []}, None, None)
    assert update["synap_short_term"] == ""


async def test_short_term_middleware_can_raise(failing_sdk):
    mw = SynapShortTermMiddleware(
        sdk=failing_sdk, conversation_id="conv-1", on_error="raise"
    )
    with pytest.raises(SynapIntegrationError):
        await mw.abefore_agent({"messages": []}, None, None)


def test_short_term_middleware_injects_block(mock_sdk):
    mw = SynapShortTermMiddleware(sdk=mock_sdk, conversation_id="conv-1")
    request = make_request({"messages": [], "synap_short_term": "earlier turns"})
    text = mw.modify_request(request).system_message.text
    assert "<synap_short_term_context>" in text
    assert "earlier turns" in text


def test_short_term_middleware_noop_when_empty(mock_sdk):
    mw = SynapShortTermMiddleware(sdk=mock_sdk, conversation_id="conv-1")
    request = make_request({"messages": [], "synap_short_term": ""})
    assert mw.modify_request(request) is request


def test_short_term_middleware_sync_before_agent(mock_sdk):
    mw = SynapShortTermMiddleware(sdk=mock_sdk, conversation_id="conv-1")
    update = mw.before_agent({"messages": []}, None, None)
    assert update["synap_short_term"] == "Recent conversation summary"


def test_short_term_middleware_wrap_model_call(mock_sdk):
    mw = SynapShortTermMiddleware(sdk=mock_sdk, conversation_id="conv-1")
    request = make_request({"messages": [], "synap_short_term": "earlier"})
    assert mw.wrap_model_call(request, lambda r: "ok") == "ok"


async def test_short_term_middleware_awrap_model_call(mock_sdk):
    mw = SynapShortTermMiddleware(sdk=mock_sdk, conversation_id="conv-1")
    request = make_request({"messages": [], "synap_short_term": "earlier"})

    async def handler(req):
        return "ok"

    assert await mw.awrap_model_call(request, handler) == "ok"


# ---------------------------------------------------------------------------
# Both middlewares carry a usable state schema
# ---------------------------------------------------------------------------


def test_memory_middleware_state_schema(mock_sdk):
    mw = SynapMemoryMiddleware(sdk=mock_sdk, user_id="alice")
    assert "synap_memory" in mw.state_schema.__annotations__


def test_short_term_middleware_state_schema(mock_sdk):
    mw = SynapShortTermMiddleware(sdk=mock_sdk, conversation_id="conv-1")
    assert "synap_short_term" in mw.state_schema.__annotations__


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (SynapMemoryMiddleware, {"user_id": "alice"}),
        (SynapShortTermMiddleware, {"conversation_id": "conv-1"}),
    ],
)
def test_middlewares_are_agent_middleware(mock_sdk, cls, kwargs):
    from langchain.agents.middleware.types import AgentMiddleware

    assert isinstance(cls(sdk=mock_sdk, **kwargs), AgentMiddleware)
