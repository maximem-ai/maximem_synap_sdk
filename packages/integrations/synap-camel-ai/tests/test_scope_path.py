"""The scope ladder path through synap-camel-ai, against ``RecordingSDK``.

A call without ``scope_path`` carries the keywords it always did; with one it
reaches every SDK call (recall ``fetch``, the user and assistant turns,
``memories.create`` in the store tool); the model-facing tools close it over
(not in their schema, not read from the model's arguments); the conversation
read for short-term context takes no path and still works.
"""

from __future__ import annotations

import json

import pytest

from camel.memories import MemoryRecord
from camel.messages import BaseMessage
from camel.types import OpenAIBackendRole

from synap_camel_ai import SynapAgentMemory, create_synap_tools, synap_st_system_message
from synap_integrations_common import set_auto_listen
from synap_integrations_common.testing import RecordingSDK, make_ladder

PATH = {"customer": "acme", "team": "payments", "user": "dana"}
OTHER = {"customer": "evil", "team": "other", "user": "mallory"}


def _user(text):
    return MemoryRecord(message=BaseMessage.make_user_message("user", text),
                        role_at_backend=OpenAIBackendRole.USER)


def _asst(text):
    return MemoryRecord(message=BaseMessage.make_assistant_message("assistant", text),
                        role_at_backend=OpenAIBackendRole.ASSISTANT)


def _by_name(tools):
    return {t.get_openai_tool_schema()["function"]["name"]: t for t in tools}


# ── SynapAgentMemory ───────────────────────────────────────────────────────


def test_memory_default_calls_are_unchanged():
    sdk = RecordingSDK()
    mem = SynapAgentMemory(sdk, "dana", customer_id="acme", conversation_id="conv-1")
    mem.write_records([_user("where is my order")])
    mem.retrieve()
    assert sdk.last_kwargs("fetch") == {
        "user_id": "dana", "customer_id": "acme", "search_query": ["where is my order"],
        "max_results": 5, "mode": "accurate", "include_conversation_context": False,
    }
    sent = sdk.kwargs_of("instance.send_message")
    assert sent and all("scope_path" not in k for k in sent)


def test_memory_path_reaches_the_recall_fetch():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    mem = SynapAgentMemory(sdk, "dana", customer_id="acme", conversation_id="conv-1",
                           scope_path=PATH)
    mem.write_records([_user("where is my order")])
    mem.retrieve()
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


def test_memory_path_reaches_user_and_assistant_turns_on_the_stream():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    mem = SynapAgentMemory(sdk, "dana", customer_id="acme", conversation_id="conv-1",
                           scope_path=PATH)
    mem.write_records([_user("where is my order")])
    mem.write_records([_asst("tomorrow")])
    sent = sdk.kwargs_of("instance.send_message")
    assert [k["event_type"] for k in sent] == ["user_message", "assistant_message"]
    assert [k["scope_path"] for k in sent] == [PATH, PATH]


def test_memory_path_reaches_the_rest_fallback():
    sdk = RecordingSDK(ladder=make_ladder("deep"), listening=False)
    set_auto_listen(sdk, False)
    mem = SynapAgentMemory(sdk, "dana", customer_id="acme", conversation_id="conv-1",
                           scope_path=PATH)
    mem.write_records([_user("where is my order")])
    mem.write_records([_asst("tomorrow")])
    recorded = sdk.kwargs_of("conversation.record_message")
    assert [k["role"] for k in recorded] == ["user", "assistant"]
    assert [k["scope_path"] for k in recorded] == [PATH, PATH]


def test_memory_default_rest_fallback_is_unchanged():
    sdk = RecordingSDK(listening=False)
    set_auto_listen(sdk, False)
    mem = SynapAgentMemory(sdk, "dana", customer_id="acme", conversation_id="conv-1")
    mem.write_records([_user("hi")])
    recorded = sdk.kwargs_of("conversation.record_message")
    assert recorded and all("scope_path" not in k for k in recorded)


def test_memory_keeps_its_own_copy_of_the_path():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    mine = dict(PATH)
    mem = SynapAgentMemory(sdk, "dana", conversation_id="conv-1", scope_path=mine)
    mine["user"] = "someone-else"
    mem.write_records([_user("q")])
    mem.retrieve()
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


def test_memory_with_a_path_still_reads_history_through_the_conversation():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    mem = SynapAgentMemory(sdk, "dana", conversation_id="conv-1", scope_path=PATH)
    mem.write_records([_user("q")])
    mem.retrieve()
    calls = sdk.kwargs_of("conversation.context.get_context_for_prompt")
    assert calls and all("scope_path" not in k and "user_id" not in k for k in calls)


# ── create_synap_tools ─────────────────────────────────────────────────────


def test_tools_default_calls_are_unchanged():
    sdk = RecordingSDK()
    tools = _by_name(create_synap_tools(sdk, "dana", customer_id="acme"))
    tools["search_memory"].func("tea")
    tools["store_memory"].func("likes tea")
    assert sdk.last_kwargs("fetch") == {
        "user_id": "dana", "customer_id": "acme", "search_query": ["tea"],
        "include_conversation_context": False,
    }
    assert sdk.last_kwargs("memories.create") == {
        "document": "likes tea", "user_id": "dana", "customer_id": "acme"}


def test_tools_path_reaches_fetch_and_create():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    tools = _by_name(create_synap_tools(sdk, "dana", customer_id="acme", scope_path=PATH))
    tools["search_memory"].func("tea")
    tools["store_memory"].func("likes tea")
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH
    create = sdk.last_kwargs("memories.create")
    assert create["scope"] == PATH and "scope_path" not in create


def test_tool_schemas_do_not_expose_scope_path():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    for tool in create_synap_tools(sdk, "dana", customer_id="acme", scope_path=PATH):
        schema = tool.get_openai_tool_schema()
        assert "scope" not in json.dumps(schema["function"]["parameters"])
        assert list(schema["function"]["parameters"]["properties"]) in (["query"], ["content"])


def test_tools_refuse_a_model_supplied_path():
    """camel calls a tool as ``tool(**arguments)``; an unknown ``scope_path`` / ``scope``
    argument must raise and must not reach the SDK."""
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    tools = _by_name(create_synap_tools(sdk, "dana", customer_id="acme", scope_path=PATH))
    for name, arg in (("search_memory", "query"), ("store_memory", "content")):
        for bad in ("scope_path", "scope"):
            # camel's FunctionTool wraps the function's TypeError (unknown keyword) in a ValueError
            with pytest.raises(ValueError, match="unexpected keyword argument"):
                tools[name](**{arg: "x", bad: OTHER})
    assert sdk.calls == []
    # and the real call still works, with the configured path only
    tools["search_memory"](query="tea")
    tools["store_memory"](content="likes tea")
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH
    assert sdk.last_kwargs("memories.create")["scope"] == PATH


def test_tools_without_a_configured_path_refuse_a_model_supplied_one_too():
    sdk = RecordingSDK()
    tools = _by_name(create_synap_tools(sdk, "dana", customer_id="acme"))
    for name, arg in (("search_memory", "query"), ("store_memory", "content")):
        for bad in ("scope_path", "scope"):
            # camel's FunctionTool wraps the function's TypeError (unknown keyword) in a ValueError
            with pytest.raises(ValueError, match="unexpected keyword argument"):
                tools[name](**{arg: "x", bad: OTHER})
    assert sdk.calls == []


# ── short-term: conversation only ──────────────────────────────────────────


def test_short_term_system_message_takes_no_path_and_still_works():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    out = synap_st_system_message(sdk, "conv-1", system_message="You are helpful.")
    assert "You are helpful." in str(out)
    calls = sdk.kwargs_of("conversation.context.get_context_for_prompt")
    assert calls and all("scope_path" not in k and "user_id" not in k for k in calls)


# ── ids that disagree with the path are refused at construction ─────────────

from synap_integrations_common import SynapIntegrationError  # noqa: E402


def test_construction_refuses_ids_that_disagree_with_the_path():
    sdk = RecordingSDK()
    builders = (
        lambda **k: SynapAgentMemory(sdk, **k),
        lambda user_id, customer_id="", scope_path=None: create_synap_tools(
            sdk, user_id, customer_id=customer_id, scope_path=scope_path),
    )
    for build in builders:
        with pytest.raises(SynapIntegrationError):
            build(user_id="bob", customer_id="acme", scope_path=PATH)  # alice's path, bob's id
        with pytest.raises(SynapIntegrationError):
            build(user_id="dana", customer_id="other", scope_path=PATH)
        build(user_id="dana", customer_id="acme", scope_path=PATH)
        build(user_id="dana", customer_id="", scope_path=PATH)
        build(user_id="bob", customer_id="other")  # no path: unchanged
    assert sdk.calls == []
