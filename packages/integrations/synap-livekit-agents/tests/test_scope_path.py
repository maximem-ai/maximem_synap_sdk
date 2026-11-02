"""``scope_path`` (a deeper scope ladder) reaches every SDK call in this package.

Voice: the path is built once and rides the calls that were already made. It
adds no call per turn (pinned below).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from livekit.agents import ChatContext
from maximem_synap.models.context import ContextForPromptResponse

from synap_integrations_common.testing import RecordingSDK, make_ladder
from synap_livekit_agents import (
    attach_synap_recording,
    preload_synap_context,
    preload_synap_st,
    synap_search_tool,
    synap_store_tool,
)

PATH = {"customer": "acme", "team": "payments", "user": "dana"}
OTHER = {"customer": "evil", "team": "x", "user": "y"}


@pytest.fixture
def sdk():
    return RecordingSDK(ladder=make_ladder("deep"))


class Session:
    def __init__(self):
        self.h = {}

    def on(self, ev, cb):
        self.h[ev] = cb


def item(role, text, i):
    m = MagicMock()
    m.role, m.text_content, m.id = role, text, i
    return m


async def drain():
    await asyncio.gather(*[t for t in asyncio.all_tasks() if t is not asyncio.current_task()])


async def say(session, role, text, i):
    session.h["conversation_item_added"](MagicMock(item=item(role, text, i)))
    await drain()


# -- preload ----------------------------------------------------------------


async def test_preload_default_unchanged(sdk):
    await preload_synap_context(ChatContext(), sdk, user_id="u", customer_id="c")
    assert sdk.last_kwargs("fetch") == {
        "user_id": "u", "customer_id": "c", "search_query": None, "max_results": 20,
        "mode": "accurate", "include_conversation_context": False}


async def test_preload_carries_path(sdk):
    await preload_synap_context(ChatContext(), sdk, user_id="dana", customer_id="acme",
                                scope_path=PATH)
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


# -- recording --------------------------------------------------------------


async def test_turns_default_unchanged(sdk):
    s = Session()
    attach_synap_recording(s, sdk, user_id="u", customer_id="c", conversation_id="c1")
    await say(s, "user", "hi", "i1")
    assert "scope_path" not in sdk.last_kwargs("instance.send_message")


async def test_both_turns_carry_path(sdk):
    s = Session()
    attach_synap_recording(s, sdk, user_id="dana", customer_id="acme",
                           conversation_id="c1", scope_path=PATH)
    await say(s, "user", "hi", "i1")
    await say(s, "assistant", "hello", "i2")
    assert [k["scope_path"] for k in sdk.kwargs_of("instance.send_message")] == [PATH, PATH]


async def test_rest_fallback_turn_carries_path():
    sdk = RecordingSDK(ladder=make_ladder("deep"), listening=False)

    async def no_stream():
        raise RuntimeError("no stream")

    sdk.instance.listen = no_stream
    s = Session()
    attach_synap_recording(s, sdk, user_id="dana", conversation_id="c1", scope_path=PATH)
    await say(s, "user", "hi", "i1")
    assert sdk.last_kwargs("conversation.record_message")["scope_path"] == PATH


async def test_no_extra_network_call_per_turn(sdk):
    """One turn = exactly the calls it made without a path (one send_message)."""
    s = Session()
    attach_synap_recording(s, sdk, user_id="dana", conversation_id="c1", scope_path=PATH)
    await say(s, "user", "hi", "i1")
    assert [c.name for c in sdk.calls] == ["instance.send_message"]
    sdk.reset()
    await say(s, "assistant", "yo", "i2")
    assert [c.name for c in sdk.calls] == ["instance.send_message"]


async def test_tool_events_and_close_still_report_with_a_path(sdk):
    s = Session()
    attach_synap_recording(s, sdk, user_id="dana", conversation_id="c1", scope_path=PATH)
    fc = MagicMock(call_id="k1", arguments='{"a": 1}')
    fc.name = "lookup"
    out = MagicMock(call_id="k1", output="ok")
    out.name = "lookup"
    s.h["function_tools_executed"](MagicMock(function_calls=[fc], function_call_outputs=[out]))
    await drain()
    assert len(sdk.calls_to("instance.record_tool_call")) == 1
    assert len(sdk.calls_to("instance.record_tool_result")) == 1
    s.h["close"]()
    await drain()
    assert len(sdk.calls_to("instance.end_session")) == 1


async def test_older_sdk_not_sent_the_keyword_when_unset():
    sdk = RecordingSDK(accept_scope_path=False)
    s = Session()
    attach_synap_recording(s, sdk, user_id="u", conversation_id="c1")
    await say(s, "user", "hi", "i1")
    await preload_synap_context(ChatContext(), sdk, user_id="u")
    assert len(sdk.calls_to("instance.send_message")) == 1 and len(sdk.calls_to("fetch")) == 1


# -- tools: closed over -----------------------------------------------------


async def test_search_tool_default_unchanged(sdk):
    await synap_search_tool(sdk, user_id="u", customer_id="c")._func("q")
    assert sdk.last_kwargs("fetch") == {
        "user_id": "u", "customer_id": "c", "search_query": ["q"], "max_results": 10,
        "mode": "accurate"}


async def test_search_tool_carries_path(sdk):
    await synap_search_tool(sdk, user_id="dana", customer_id="acme", scope_path=PATH)._func("q")
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


async def test_store_tool_default_unchanged(sdk):
    await synap_store_tool(sdk, user_id="u")._func("c")
    kw = sdk.last_kwargs("memories.create")
    assert "scope" not in kw and kw["user_id"] == "u"


async def test_store_tool_carries_path_as_scope(sdk):
    await synap_store_tool(sdk, user_id="dana", scope_path=PATH)._func("c")
    kw = sdk.last_kwargs("memories.create")
    assert kw["scope"] == PATH and "scope_path" not in kw


def _tools(sdk):
    return (synap_search_tool(sdk, user_id="d", scope_path=PATH),
            synap_store_tool(sdk, user_id="d", scope_path=PATH))


def test_tool_schemas_have_no_scope_path(sdk):
    """The schema the model is shown, built by livekit's own functions."""
    import inspect
    import json

    from livekit.agents.llm.utils import build_legacy_openai_schema, build_strict_openai_schema

    for t in _tools(sdk):
        assert "scope_path" not in inspect.signature(t._func).parameters
        for build in (build_legacy_openai_schema, build_strict_openai_schema):
            dumped = json.dumps(build(t))
            assert "scope" not in dumped
            assert "query" in dumped or "content" in dumped  # it is the real schema


async def test_model_supplied_path_is_dropped_by_livekit_and_never_reaches_the_sdk(sdk):
    from livekit.agents.llm.utils import prepare_function_arguments

    search, store = _tools(sdk)
    for tool, args in ((search, {"query": "q"}), (store, {"content": "c"})):
        poisoned = {**args, "scope_path": OTHER, "scope": OTHER}
        try:
            a, kw = prepare_function_arguments(fnc=tool, json_arguments=poisoned)
        except Exception:  # rejecting it outright is also acceptable
            continue
        assert "scope" not in kw and "scope_path" not in kw
        await tool._func(*a, **kw)
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH
    assert sdk.last_kwargs("memories.create")["scope"] == PATH


async def test_model_supplied_path_cannot_reach_the_call(sdk):
    search, store = _tools(sdk)
    with pytest.raises(TypeError):
        await search._func("q", scope_path=OTHER)
    with pytest.raises(TypeError):
        await store._func("c", scope=OTHER)
    assert sdk.calls == []


# -- conversation-only read -------------------------------------------------


async def test_short_term_preload_works_with_a_path_configured(sdk):
    sdk.results["conversation.context.get_context_for_prompt"] = ContextForPromptResponse(
        formatted_context="x", context_block="## Conversation so far (compacted 1x)\nS",
        available=True)
    s = Session()
    attach_synap_recording(s, sdk, user_id="dana", conversation_id="c1", scope_path=PATH)
    await preload_synap_st(ChatContext(), sdk, conversation_id="c1")
    kw = sdk.last_kwargs("conversation.context.get_context_for_prompt")
    assert "scope_path" not in kw and "scope" not in kw
