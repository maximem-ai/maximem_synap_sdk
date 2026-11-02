"""The scope ladder path through synap-claude-agent, against ``RecordingSDK``.

A call without ``scope_path`` carries the keywords it always did; with one it
reaches every SDK call (the prompt hook's ``fetch``, the user turn, the Stop
hook's assistant turn, ``report_assistant_turn``, and both MCP tools). The MCP
tools are exposed to the model, so the path is closed over: it is not in their
input schema and a model-supplied value is ignored. The session store and the
short-term hook read through the conversation and take no path.
"""

from __future__ import annotations

from synap_integrations_common.conversation_id import coerce_conversation_id
import json

import pytest

from synap_claude_agent import (
    create_synap_hooks,
    create_synap_session_store,
    create_synap_st_hook,
    report_assistant_turn,
)
from synap_claude_agent.mcp_server import _build_synap_tools, create_synap_mcp_server
from synap_integrations_common import set_auto_listen
from synap_integrations_common.testing import RecordingSDK, make_ladder

PATH = {"customer": "acme", "team": "payments", "user": "dana"}
OTHER = {"customer": "evil", "team": "other", "user": "mallory"}


def _hooks(sdk, **kw):
    return create_synap_hooks(sdk, "dana", "acme", conversation_id="conv-1", **kw)


async def _prompt(hooks, text="where is my order"):
    return await hooks["UserPromptSubmit"][0].hooks[0]({"prompt": text}, None, None)


async def _stop(hooks, reply="tomorrow"):
    return await hooks["Stop"][0].hooks[0]({"last_assistant_message": reply}, None, None)


# ── hooks ──────────────────────────────────────────────────────────────────


async def test_hooks_default_calls_are_unchanged():
    sdk = RecordingSDK()
    h = _hooks(sdk)
    await _prompt(h)
    await _stop(h)
    assert sdk.last_kwargs("fetch") == {
        "conversation_id": coerce_conversation_id("conv-1"), "user_id": "dana", "customer_id": "acme",
        "search_query": ["where is my order"], "max_results": 20, "mode": "accurate",
        "include_conversation_context": False,
    }
    sent = sdk.kwargs_of("instance.send_message")
    assert [k["event_type"] for k in sent] == ["user_message", "assistant_message"]
    assert all("scope_path" not in k for k in sent)


async def test_hooks_path_reaches_fetch_and_both_turns():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    h = _hooks(sdk, scope_path=PATH)
    await _prompt(h)
    await _stop(h)
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH
    sent = sdk.kwargs_of("instance.send_message")
    assert [k["event_type"] for k in sent] == ["user_message", "assistant_message"]
    assert [k["scope_path"] for k in sent] == [PATH, PATH]


async def test_hooks_path_reaches_the_rest_fallback():
    sdk = RecordingSDK(ladder=make_ladder("deep"), listening=False)
    set_auto_listen(sdk, False)
    h = _hooks(sdk, scope_path=PATH)
    await _prompt(h)
    await _stop(h)
    recorded = sdk.kwargs_of("conversation.record_message")
    assert [k["role"] for k in recorded] == ["user", "assistant"]
    assert [k["scope_path"] for k in recorded] == [PATH, PATH]


async def test_hooks_tool_events_carry_no_path():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    h = _hooks(sdk, scope_path=PATH)
    await h["PreToolUse"][0].hooks[0](
        {"tool_name": "Bash", "tool_input": {"c": "ls"}, "tool_use_id": "t1"}, "t1", None)
    await h["PostToolUse"][0].hooks[0](
        {"tool_name": "Bash", "tool_response": "ok", "tool_use_id": "t1"}, "t1", None)
    for name in ("instance.record_tool_call", "instance.record_tool_result"):
        assert sdk.calls_to(name), name
        assert all("scope_path" not in k for k in sdk.kwargs_of(name))


async def test_report_assistant_turn_default_and_path():
    plain = RecordingSDK()
    await report_assistant_turn(plain, "hi", conversation_id="c", user_id="dana", customer_id="acme")
    assert "scope_path" not in plain.last_kwargs("instance.send_message")

    sdk = RecordingSDK(ladder=make_ladder("deep"))
    await report_assistant_turn(sdk, "hi", conversation_id="c", user_id="dana",
                                customer_id="acme", scope_path=PATH)
    assert sdk.last_kwargs("instance.send_message")["scope_path"] == PATH


# ── MCP tools ──────────────────────────────────────────────────────────────


def _tools(sdk, **kw):
    search, remember = _build_synap_tools(sdk, "dana", "acme", "conv-1", "accurate", **kw)
    return search, remember


async def test_mcp_default_calls_are_unchanged():
    sdk = RecordingSDK()
    search, remember = _tools(sdk)
    await search.handler({"query": "tea"})
    await remember.handler({"content": "likes tea"})
    assert sdk.last_kwargs("fetch") == {
        "conversation_id": coerce_conversation_id("conv-1"), "user_id": "dana", "customer_id": "acme",
        "search_query": ["tea"], "max_results": 10, "mode": "accurate",
        "include_conversation_context": False,
    }
    assert sdk.last_kwargs("memories.create") == {
        "document": "likes tea", "user_id": "dana", "customer_id": "acme",
        "metadata": {"source": "claude_agent_sdk"},
    }


async def test_mcp_path_reaches_fetch_and_create():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    search, remember = _tools(sdk, scope_path=PATH)
    await search.handler({"query": "tea"})
    await remember.handler({"content": "likes tea"})
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH
    create = sdk.last_kwargs("memories.create")
    assert create["scope"] == PATH and "scope_path" not in create


async def test_mcp_path_survives_the_b2c_customer_id_retry():
    """The retry without a refused customer_id must still carry the path."""
    class Rejected(Exception):
        pass

    sdk = RecordingSDK(ladder=make_ladder("deep"))
    first = {"done": False}
    real = sdk.fetch

    async def flaky(*a, **kw):
        if not first["done"]:
            first["done"] = True
            raise Rejected("customer_id_not_accepted_on_b2c")
        return await real(*a, **kw)

    sdk.fetch = flaky
    search, _ = _tools(sdk, scope_path=PATH)
    await search.handler({"query": "tea"})
    kw = sdk.last_kwargs("fetch")
    assert kw["scope_path"] == PATH and kw["customer_id"] is None


def test_mcp_tool_schemas_do_not_expose_scope_path():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    for with_path in (None, PATH):
        search, remember = _tools(sdk, scope_path=with_path)
        # input_schema is {name: python type}; the model sees exactly these names.
        assert set(search.input_schema) == {"query", "max_results"}
        assert set(remember.input_schema) == {"content", "metadata"}


async def test_mcp_tools_ignore_a_model_supplied_path():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    search, remember = _tools(sdk, scope_path=PATH)
    await search.handler({"query": "tea", "scope_path": OTHER, "scope": OTHER})
    await remember.handler({"content": "x", "scope_path": OTHER, "scope": OTHER})
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH
    assert sdk.last_kwargs("memories.create")["scope"] == PATH
    assert not any(k.get("scope_path") == OTHER or k.get("scope") == OTHER
                   for c in sdk.calls for k in [c.kwargs])


async def test_mcp_without_a_configured_path_ignores_a_model_supplied_one():
    sdk = RecordingSDK()
    search, remember = _tools(sdk)
    await search.handler({"query": "tea", "scope_path": OTHER})
    await remember.handler({"content": "x", "scope": OTHER, "metadata": {"scope": OTHER}})
    assert "scope_path" not in sdk.last_kwargs("fetch")
    assert "scope" not in sdk.last_kwargs("memories.create")


def test_create_synap_mcp_server_hands_the_path_to_the_tools(monkeypatch):
    from synap_claude_agent import mcp_server

    seen = {}
    real = mcp_server._build_synap_tools

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(mcp_server, "_build_synap_tools", spy)
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    assert create_synap_mcp_server(sdk, "dana", "acme", scope_path=PATH)["type"] == "sdk"
    assert seen["scope_path"] == PATH
    seen.clear()
    create_synap_mcp_server(sdk, "dana", "acme")
    assert not seen.get("scope_path")


# ── conversation-only reads ────────────────────────────────────────────────


async def test_session_store_and_st_hook_take_no_path_and_still_work():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    store = create_synap_session_store(sdk, conversation_id="conv-1")
    await store.load({"project_key": "p", "session_id": "s1"})
    st = create_synap_st_hook(sdk, "conv-1")
    await st["UserPromptSubmit"][0].hooks[0]({"prompt": "hi"}, None, None)
    calls = sdk.kwargs_of("conversation.context.get_context_for_prompt")
    assert len(calls) >= 2
    assert all("scope_path" not in k and "user_id" not in k for k in calls)


async def test_prompt_hook_history_read_with_a_path_uses_the_conversation_only():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    await _prompt(_hooks(sdk, scope_path=PATH))
    calls = sdk.kwargs_of("conversation.context.get_context_for_prompt")
    assert calls and all("scope_path" not in k and "user_id" not in k for k in calls)


# ── ids that disagree with the path are refused ─────────────────────────────

from synap_integrations_common import SynapIntegrationError  # noqa: E402


def test_construction_refuses_ids_that_disagree_with_the_path():
    sdk = RecordingSDK()
    builders = (
        lambda u, c, p: create_synap_hooks(sdk, u, c, scope_path=p),
        lambda u, c, p: create_synap_mcp_server(sdk, u, c, scope_path=p),
        lambda u, c, p: _build_synap_tools(sdk, u, c, None, "accurate", scope_path=p),
    )
    for build in builders:
        with pytest.raises(SynapIntegrationError):
            build("bob", "acme", PATH)      # alice's path, bob's id
        with pytest.raises(SynapIntegrationError):
            build("dana", "other", PATH)
        build("dana", "acme", PATH)
        build("dana", "", PATH)
        build("bob", "other", None)         # no path: unchanged
    assert sdk.calls == []


async def test_report_assistant_turn_refuses_another_user_and_sends_nothing():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    with pytest.raises(SynapIntegrationError):
        await report_assistant_turn(sdk, "hi", conversation_id="c", user_id="bob",
                                    customer_id="acme", scope_path=PATH)
    assert sdk.calls == []
    assert await report_assistant_turn(sdk, "hi", conversation_id="c", user_id="dana",
                                       customer_id="acme", scope_path=PATH)


# ── the default REST fallback is exactly what it was ────────────────────────


async def test_default_rest_fallback_kwargs_are_unchanged():
    sdk = RecordingSDK(listening=False)
    set_auto_listen(sdk, False)
    h = _hooks(sdk)
    await _prompt(h)
    await _stop(h)
    assert sdk.kwargs_of("conversation.record_message") == [
        {"conversation_id": coerce_conversation_id("conv-1"), "content": "where is my order", "role": "user",
         "user_id": "dana", "customer_id": "acme"},
        {"conversation_id": coerce_conversation_id("conv-1"), "content": "tomorrow", "role": "assistant",
         "user_id": "dana", "customer_id": "acme"},
    ]
