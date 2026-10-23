"""Tests for :mod:`synap_deepagents.middleware`."""

from datetime import datetime, timezone
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
# SynapShortTermMiddleware: the compaction replaces the turns it covers
# ---------------------------------------------------------------------------

from langchain_core.messages import ToolMessage
from maximem_synap.models.context import ContextForPromptResponse, RecentMessage

COMPACTION = "## Conversation so far (compacted 1x)\nThey planned a Lisbon trip."


def compacted_sdk(mock_sdk, turns=()):
    mock_sdk.conversation.context.get_context_for_prompt.return_value = ContextForPromptResponse(
        formatted_context=f"{COMPACTION}\nRECENT-TURNS-DUPLICATE", context_block=COMPACTION,
        available=True, recent_messages=[RecentMessage(role=r, content=c, timestamp=datetime.now(timezone.utc), message_id=f"m{i}")
                         for i, (r, c) in enumerate(turns)],
    )
    return mock_sdk


OLD = [HumanMessage("old q1"), AIMessage("old a1"), HumanMessage("old q2"), AIMessage("old a2")]


async def spliced(mw, messages, system_message=None):
    seen = {}

    async def handler(req):
        seen["req"] = req
        return "ok"

    request = make_request({"messages": messages}, system_message=system_message)
    assert await mw.awrap_model_call(request, handler) == "ok"
    return seen["req"]


async def test_short_term_replaces_earlier_turns_with_the_compaction(mock_sdk):
    compacted_sdk(mock_sdk, turns=[("user", "kept q"), ("assistant", "kept a")])
    mw = SynapShortTermMiddleware(sdk=mock_sdk, conversation_id="conv-1")
    req = await spliced(mw, [*OLD, HumanMessage("now?")], SystemMessage("Be brief."))
    assert [m.content for m in req.messages] == ["kept q", "kept a", "now?"]
    text = req.system_message.text
    assert text.startswith("Be brief.")
    assert "<synap_short_term_context>" in text and "Lisbon trip" in text
    assert "RECENT-TURNS-DUPLICATE" not in text


async def test_short_term_is_asked_on_every_model_call(mock_sdk):
    """Not once per invocation: a long tool loop can cross the compaction."""
    compacted_sdk(mock_sdk)
    mw = SynapShortTermMiddleware(sdk=mock_sdk, conversation_id="conv-1")
    for _ in range(3):
        await spliced(mw, [HumanMessage("q")])
    assert mock_sdk.conversation.context.get_context_for_prompt.await_count == 3


async def test_short_term_keeps_the_current_tool_loop_paired(mock_sdk):
    compacted_sdk(mock_sdk)
    mw = SynapShortTermMiddleware(sdk=mock_sdk, conversation_id="conv-1")
    call = AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": "c1"}])
    req = await spliced(mw, [*OLD, HumanMessage("now?"), call, ToolMessage("r", tool_call_id="c1")])
    assert req.messages[-2:] == [call, ToolMessage("r", tool_call_id="c1")] or \
        [type(m) for m in req.messages[-2:]] == [AIMessage, ToolMessage]
    assert not any("old q" in str(m.content) for m in req.messages)


async def test_short_term_is_untouched_before_the_first_compaction(mock_sdk):
    mock_sdk.conversation.context.get_context_for_prompt.return_value = ContextForPromptResponse(
        formatted_context="recent turns", available=True)
    mw = SynapShortTermMiddleware(sdk=mock_sdk, conversation_id="conv-1")
    request = make_request({"messages": [*OLD, HumanMessage("now?")]})

    async def handler(req):
        return req

    assert await mw.awrap_model_call(request, handler) is request


async def test_short_term_falls_back_on_failure(failing_sdk):
    mw = SynapShortTermMiddleware(sdk=failing_sdk, conversation_id="conv-1")
    request = make_request({"messages": [HumanMessage("q")]})

    async def handler(req):
        return req

    assert await mw.awrap_model_call(request, handler) is request


async def test_short_term_can_raise(failing_sdk):
    mw = SynapShortTermMiddleware(sdk=failing_sdk, conversation_id="conv-1", on_error="raise")
    request = make_request({"messages": [HumanMessage("q")]})

    async def handler(req):
        return req

    with pytest.raises(SynapIntegrationError):
        await mw.awrap_model_call(request, handler)


def test_short_term_sync_path_splices_too(mock_sdk):
    compacted_sdk(mock_sdk)
    mw = SynapShortTermMiddleware(sdk=mock_sdk, conversation_id="conv-1")
    seen = {}
    mw.wrap_model_call(make_request({"messages": [*OLD, HumanMessage("now?")]}),
                       lambda r: seen.setdefault("req", r))
    assert [m.content for m in seen["req"].messages] == ["now?"]


async def test_a_layer_that_already_spliced_is_not_asked_again(mock_sdk):
    compacted_sdk(mock_sdk)
    mw = SynapShortTermMiddleware(sdk=mock_sdk, conversation_id="conv-1")
    first = await spliced(mw, [*OLD, HumanMessage("now?")])
    mock_sdk.conversation.context.get_context_for_prompt.reset_mock()

    async def handler(req):
        return req

    request = make_request({"messages": first.messages}, system_message=first.system_message)
    assert await mw.awrap_model_call(request, handler) is request
    mock_sdk.conversation.context.get_context_for_prompt.assert_not_awaited()


# ---------------------------------------------------------------------------
# Both middlewares carry a usable state schema
# ---------------------------------------------------------------------------


def test_memory_middleware_state_schema(mock_sdk):
    mw = SynapMemoryMiddleware(sdk=mock_sdk, user_id="alice")
    assert "synap_memory" in mw.state_schema.__annotations__


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


# ---------------------------------------------------------------------------
# before_agent is reachable the way the graph actually calls it
# ---------------------------------------------------------------------------
#
# ⚠ A middleware hook is not called with the arguments its signature declares.
# LangChain wraps it in a ``RunnableCallable``, which reads the signature and
# injects only the parameters it recognises by name AND annotation. ``config``
# annotated ``Any`` is not recognised: langgraph warns, skips it, and calls the
# hook without it — so a required ``config`` parameter meant the entry node of
# every deepagents run using these middlewares raised ``TypeError`` before the
# model was ever called. Every unit test passed, because a unit test calls the
# method directly and passes all three.


def _as_graph_node(middleware):
    """Wrap a middleware's ``before_agent`` exactly as LangChain's factory does."""
    from langgraph._internal._runnable import RunnableCallable

    return RunnableCallable(
        middleware.before_agent, middleware.abefore_agent, trace=False
    )


def _node_config():
    from langgraph._internal._constants import CONF, CONFIG_KEY_RUNTIME

    return {CONF: {CONFIG_KEY_RUNTIME: MagicMock()}}


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (SynapMemoryMiddleware, {"user_id": "alice"}),
    ],
)
def test_before_agent_runs_as_a_graph_node(mock_sdk, cls, kwargs):
    node = _as_graph_node(cls(sdk=mock_sdk, **kwargs))
    assert node.invoke({"messages": [HumanMessage("hello")]}, _node_config())


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (SynapMemoryMiddleware, {"user_id": "alice"}),
    ],
)
def test_before_agent_is_offered_its_config(mock_sdk, cls, kwargs):
    """The annotation is load-bearing: ``Any`` here means no config is passed."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        node = _as_graph_node(cls(sdk=mock_sdk, **kwargs))
    assert "config" in node.func_accepts
