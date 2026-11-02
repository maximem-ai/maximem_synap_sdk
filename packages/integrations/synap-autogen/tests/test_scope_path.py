"""The scope ladder path through synap-autogen, against ``RecordingSDK``.

Pinned here: a call without ``scope_path`` carries the keywords it always did;
with one, it reaches every SDK call this package makes; the two LLM-facing tools
close the path over at construction, so it is neither in their schema nor
readable from the model's arguments; the conversation-only short-term context
still works in a setup that uses a path.
"""

from __future__ import annotations

from synap_integrations_common.conversation_id import coerce_conversation_id
import json

import pytest
from autogen_core import CancellationToken
from autogen_core.models import AssistantMessage, UserMessage

from synap_autogen.tools import SearchMemoryArgs, StoreMemoryArgs
from synap_autogen import (
    SynapSearchTool,
    SynapShortTermChatContext,
    SynapStoreTool,
    SynapStreamingChatContext,
)
from synap_integrations_common import set_auto_listen
from synap_integrations_common.testing import RecordingSDK, make_ladder

PATH = {"customer": "acme", "team": "payments", "user": "dana"}
OTHER = {"customer": "evil", "team": "other", "user": "mallory"}


# ── tools ──────────────────────────────────────────────────────────────────


async def test_search_default_call_is_unchanged(ct):
    sdk = RecordingSDK()
    await SynapSearchTool(sdk, "dana", "acme", conversation_id="c1").run(
        SearchMemoryArgs(query="tea"), ct)
    assert sdk.last_kwargs("fetch") == {
        "conversation_id": coerce_conversation_id("c1"), "user_id": "dana", "customer_id": "acme",
        "search_query": ["tea"], "mode": "accurate", "include_conversation_context": False,
    }


async def test_store_default_call_is_unchanged(ct):
    sdk = RecordingSDK()
    await SynapStoreTool(sdk, "dana", "acme").run(StoreMemoryArgs(content="likes tea"), ct)
    assert sdk.last_kwargs("memories.create") == {
        "document": "likes tea", "user_id": "dana", "customer_id": "acme"}


async def test_search_path_reaches_fetch(ct):
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    await SynapSearchTool(sdk, "dana", "acme", scope_path=PATH).run(SearchMemoryArgs(query="tea"), ct)
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


async def test_store_path_reaches_memories_create_as_scope(ct):
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    await SynapStoreTool(sdk, "dana", "acme", scope_path=PATH).run(StoreMemoryArgs(content="x"), ct)
    kw = sdk.last_kwargs("memories.create")
    assert kw["scope"] == PATH and "scope_path" not in kw


@pytest.mark.parametrize("make", [
    lambda sdk: SynapSearchTool(sdk, "dana", "acme", scope_path=PATH),
    lambda sdk: SynapStoreTool(sdk, "dana", "acme", scope_path=PATH),
])
def test_tool_schema_does_not_expose_scope_path(make):
    tool = make(RecordingSDK(ladder=make_ladder("deep")))
    params = tool.schema["parameters"]
    assert "scope_path" not in json.dumps(params) and "scope" not in params.get("properties", {})
    assert "scope_path" not in json.dumps(tool.schema)


async def test_search_ignores_a_model_supplied_path(ct):
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    tool = SynapSearchTool(sdk, "dana", "acme", scope_path=PATH)
    await tool.run_json({"query": "tea", "scope_path": OTHER}, ct)
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


async def test_search_without_a_configured_path_ignores_a_model_supplied_one(ct):
    sdk = RecordingSDK()
    await SynapSearchTool(sdk, "dana", "acme").run_json({"query": "tea", "scope_path": OTHER}, ct)
    assert "scope_path" not in sdk.last_kwargs("fetch")


async def test_store_ignores_a_model_supplied_path(ct):
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    tool = SynapStoreTool(sdk, "dana", "acme", scope_path=PATH)
    await tool.run_json({"content": "x", "scope_path": OTHER, "scope": OTHER}, ct)
    kw = sdk.last_kwargs("memories.create")
    assert kw["scope"] == PATH and "scope_path" not in kw


async def test_store_without_a_configured_path_ignores_a_model_supplied_one(ct):
    sdk = RecordingSDK()
    await SynapStoreTool(sdk, "dana", "acme").run_json({"content": "x", "scope": OTHER}, ct)
    assert "scope" not in sdk.last_kwargs("memories.create")


# ── streaming context: user and assistant turns ────────────────────────────


def _ctx(sdk, **kw):
    return SynapStreamingChatContext(sdk, "conv-1", "dana", "acme", **kw)


async def test_streaming_default_turns_are_unchanged():
    sdk = RecordingSDK()
    ctx = _ctx(sdk)
    await ctx.add_message(UserMessage(content="where is my order", source="user"))
    await ctx.add_message(AssistantMessage(content="tomorrow", source="assistant"))
    sent = sdk.kwargs_of("instance.send_message")
    assert [k["event_type"] for k in sent] == ["user_message", "assistant_message"]
    assert all("scope_path" not in k for k in sent)


async def test_streaming_path_reaches_user_and_assistant_turns():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    ctx = _ctx(sdk, scope_path=PATH)
    await ctx.add_message(UserMessage(content="where is my order", source="user"))
    await ctx.add_message(AssistantMessage(content="tomorrow", source="assistant"))
    sent = sdk.kwargs_of("instance.send_message")
    assert [k["event_type"] for k in sent] == ["user_message", "assistant_message"]
    assert [k["scope_path"] for k in sent] == [PATH, PATH]


async def test_streaming_path_reaches_report_reply():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    ctx = _ctx(sdk, scope_path=PATH)
    assert await ctx.report_reply("the exact reply")
    assert sdk.last_kwargs("instance.send_message")["scope_path"] == PATH


async def test_streaming_path_reaches_the_tool_call_summary_stand_in():
    """The late stand-in reply (a tool result AutoGen never hands to the context)."""
    from autogen_core import FunctionCall
    from autogen_core.models import FunctionExecutionResult, FunctionExecutionResultMessage

    sdk = RecordingSDK(ladder=make_ladder("deep"))
    ctx = _ctx(sdk, scope_path=PATH)
    await ctx.add_message(AssistantMessage(
        content=[FunctionCall(id="c1", name="lookup", arguments="{}")], source="assistant"))
    await ctx.add_message(FunctionExecutionResultMessage(content=[
        FunctionExecutionResult(content="shipped", name="lookup", call_id="c1", is_error=False)]))
    await ctx.add_message(UserMessage(content="next question", source="user"))  # flushes the summary
    turns = [k for k in sdk.kwargs_of("instance.send_message") if k["event_type"] == "assistant_message"]
    assert turns and all(k["scope_path"] == PATH for k in turns)


async def test_streaming_path_reaches_the_rest_fallback():
    sdk = RecordingSDK(ladder=make_ladder("deep"), listening=False)
    set_auto_listen(sdk, False)
    ctx = _ctx(sdk, scope_path=PATH)
    await ctx.add_message(UserMessage(content="where is my order", source="user"))
    await ctx.add_message(AssistantMessage(content="tomorrow", source="assistant"))
    recorded = sdk.kwargs_of("conversation.record_message")
    assert [k["role"] for k in recorded] == ["user", "assistant"]
    assert [k["scope_path"] for k in recorded] == [PATH, PATH]


async def test_streaming_tool_events_carry_no_path():
    from autogen_core import FunctionCall
    from autogen_core.models import FunctionExecutionResult, FunctionExecutionResultMessage

    sdk = RecordingSDK(ladder=make_ladder("deep"))
    ctx = _ctx(sdk, scope_path=PATH)
    await ctx.add_message(AssistantMessage(
        content=[FunctionCall(id="c1", name="lookup", arguments="{}")], source="assistant"))
    await ctx.add_message(FunctionExecutionResultMessage(content=[
        FunctionExecutionResult(content="shipped", name="lookup", call_id="c1", is_error=False)]))
    for name in ("instance.record_tool_call", "instance.record_tool_result"):
        assert sdk.calls_to(name), name
        assert all("scope_path" not in k for k in sdk.kwargs_of(name))


# ── conversation-only reads ────────────────────────────────────────────────


async def test_short_term_context_takes_no_path_and_still_works():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    ctx = SynapShortTermChatContext(sdk, "conv-1")
    await ctx.add_message(UserMessage(content="hi", source="user"))
    assert [m.content for m in await ctx.get_messages()] == ["hi"]
    calls = sdk.kwargs_of("conversation.context.get_context_for_prompt")
    assert calls and all("scope_path" not in k and "user_id" not in k for k in calls)


async def test_streaming_context_with_a_path_still_reads_history_through_the_conversation():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    ctx = _ctx(sdk, scope_path=PATH)
    await ctx.add_message(UserMessage(content="hi", source="user"))
    await ctx.get_messages()
    calls = sdk.kwargs_of("conversation.context.get_context_for_prompt")
    assert calls and all("scope_path" not in k and "user_id" not in k for k in calls)


# ── ids that disagree with the path are refused at construction ─────────────

from synap_integrations_common import SynapIntegrationError  # noqa: E402


def test_construction_refuses_ids_that_disagree_with_the_path():
    sdk = RecordingSDK()
    for build in (
        lambda **k: SynapSearchTool(sdk, **k),
        lambda **k: SynapStoreTool(sdk, **k),
        lambda **k: SynapStreamingChatContext(sdk, "conv-1", **k),
    ):
        with pytest.raises(SynapIntegrationError):
            build(user_id="bob", customer_id="acme", scope_path=PATH)  # alice's path, bob's id
        with pytest.raises(SynapIntegrationError):
            build(user_id="dana", customer_id="other", scope_path=PATH)
        build(user_id="dana", customer_id="acme", scope_path=PATH)   # matching
        build(user_id="dana", customer_id="", scope_path=PATH)       # empty customer: path supplies it
        build(user_id="bob", customer_id="other")                     # no path: unchanged
    assert sdk.calls == []


# ── the default REST fallback is exactly what it was ────────────────────────


async def test_default_rest_fallback_kwargs_are_unchanged():
    sdk = RecordingSDK(listening=False)
    set_auto_listen(sdk, False)
    ctx = _ctx(sdk)
    await ctx.add_message(UserMessage(content="where is my order", source="user"))
    await ctx.add_message(AssistantMessage(content="tomorrow", source="assistant"))
    assert sdk.kwargs_of("conversation.record_message") == [
        {"conversation_id": coerce_conversation_id("conv-1"), "content": "where is my order", "role": "user",
         "user_id": "dana", "customer_id": "acme"},
        {"conversation_id": coerce_conversation_id("conv-1"), "content": "tomorrow", "role": "assistant",
         "user_id": "dana", "customer_id": "acme"},
    ]
