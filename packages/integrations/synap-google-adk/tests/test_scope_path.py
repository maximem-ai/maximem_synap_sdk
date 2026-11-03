"""``scope_path`` (a deeper scope ladder) reaches every SDK call in this package."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from maximem_synap.models.context import ContextForPromptResponse

from synap_integrations_common.testing import RecordingSDK, make_ladder

from synap_google_adk import (
    create_synap_callbacks,
    create_synap_tools,
    report_user_turn,
    synap_st_instruction,
)

PATH = {"customer": "acme", "team": "payments", "user": "dana"}
OTHER = {"customer": "evil", "team": "x", "user": "y"}


@pytest.fixture
def sdk():
    return RecordingSDK(ladder=make_ladder("deep"))


def ctx(text="where is my order?", inv="inv-1"):
    return SimpleNamespace(
        invocation_id=inv,
        user_content=SimpleNamespace(role="user", parts=[SimpleNamespace(text=text, thought=None)]),
    )


def reply(text="It shipped."):
    return SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]))


# -- tools ------------------------------------------------------------------


async def test_tools_default_calls_unchanged(sdk):
    search, store = create_synap_tools(sdk, user_id="u1", customer_id="c1")
    await search.func("q")
    await store.func("note")
    assert sdk.last_kwargs("fetch") == {
        "conversation_id": None, "user_id": "u1", "customer_id": "c1",
        "search_query": ["q"], "mode": "accurate", "include_conversation_context": False,
    }
    assert sdk.last_kwargs("memories.create") == {
        "document": "note", "user_id": "u1", "customer_id": "c1"}


async def test_search_tool_carries_path(sdk):
    search, _ = create_synap_tools(sdk, user_id="dana", customer_id="acme", scope_path=PATH)
    await search.func("q")
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


async def test_store_tool_carries_path_as_scope(sdk):
    _, store = create_synap_tools(sdk, user_id="dana", customer_id="acme", scope_path=PATH)
    await store.func("note")
    kw = sdk.last_kwargs("memories.create")
    assert kw["scope"] == PATH and "scope_path" not in kw


def test_tools_do_not_expose_scope_path_to_the_model(sdk):
    search, store = create_synap_tools(sdk, user_id="dana", scope_path=PATH)
    for t, arg in ((search, "query"), (store, "content")):
        decl = str(t._get_declaration())
        assert arg in decl  # the declaration does describe the real parameter
        assert "scope" not in decl


async def test_model_supplied_path_is_rejected_or_ignored(sdk):
    search, store = create_synap_tools(sdk, user_id="dana", scope_path=PATH)
    for tool, args, rec in ((search, {"query": "q"}, "fetch"), (store, {"content": "c"}, "memories.create")):
        try:
            await tool.run_async(args={**args, "scope_path": OTHER, "scope": OTHER}, tool_context=None)
        except TypeError:
            continue  # rejected: also acceptable
        kw = sdk.last_kwargs(rec)
        assert kw.get("scope_path", kw.get("scope")) == PATH


async def test_model_supplied_path_not_honoured_when_none_configured(sdk):
    search, _ = create_synap_tools(sdk, user_id="dana")
    try:
        await search.run_async(args={"query": "q", "scope_path": OTHER}, tool_context=None)
    except TypeError:
        return
    assert "scope_path" not in sdk.last_kwargs("fetch")


# -- callbacks --------------------------------------------------------------


async def test_callbacks_default_turns_unchanged(sdk):
    cb = create_synap_callbacks(sdk, user_id="u1", customer_id="c1", conversation_id="c")
    await cb["before_model_callback"](ctx(), MagicMock(contents=[]))
    await cb["after_model_callback"](MagicMock(), reply())
    sends = sdk.kwargs_of("instance.send_message")
    assert len(sends) == 2 and all("scope_path" not in k for k in sends)
    assert sends[0]["user_id"] == "u1" and sends[0]["customer_id"] == "c1"


async def test_user_and_assistant_turns_carry_path(sdk):
    cb = create_synap_callbacks(sdk, user_id="dana", customer_id="acme",
                                conversation_id="c", scope_path=PATH)
    await cb["before_model_callback"](ctx(), MagicMock(contents=[]))
    await cb["after_model_callback"](MagicMock(), reply())
    sends = sdk.kwargs_of("instance.send_message")
    assert [k["event_type"] for k in sends] == ["user_message", "assistant_message"]
    assert [k["scope_path"] for k in sends] == [PATH, PATH]


async def test_rest_fallback_turns_carry_path():
    sdk = RecordingSDK(ladder=make_ladder("deep"), listening=False)

    async def no_stream():
        raise RuntimeError("no stream")

    sdk.instance.listen = no_stream
    cb = create_synap_callbacks(sdk, user_id="dana", customer_id="acme",
                                conversation_id="c", scope_path=PATH)
    await cb["before_model_callback"](ctx(), MagicMock(contents=[]))
    await cb["after_model_callback"](MagicMock(), reply())
    recs = sdk.kwargs_of("conversation.record_message")
    assert len(recs) == 2 and all(k["scope_path"] == PATH for k in recs)


async def test_report_user_turn_carries_path_and_default_does_not(sdk):
    await report_user_turn(sdk, "hi", conversation_id="c", user_id="dana", customer_id="acme",
                           scope_path=PATH)
    assert sdk.last_kwargs("instance.send_message")["scope_path"] == PATH
    await report_user_turn(sdk, "hi", conversation_id="c", user_id="dana")
    assert "scope_path" not in sdk.last_kwargs("instance.send_message")


async def test_tool_callbacks_still_report_with_a_path_configured(sdk):
    cb = create_synap_callbacks(sdk, user_id="dana", conversation_id="c", scope_path=PATH)
    t = SimpleNamespace(name="lookup")
    await cb["before_tool_callback"](t, {"a": 1}, MagicMock())
    await cb["after_tool_callback"](t, {}, MagicMock(), {"r": 1})
    assert len(sdk.calls_to("instance.record_tool_call")) == 1
    assert len(sdk.calls_to("instance.record_tool_result")) == 1


async def test_older_sdk_not_sent_the_keyword_when_unset():
    sdk = RecordingSDK(accept_scope_path=False)
    search, store = create_synap_tools(sdk, user_id="u1")
    await search.func("q")
    await store.func("n")
    cb = create_synap_callbacks(sdk, user_id="u1", conversation_id="c")
    await cb["before_model_callback"](ctx(), MagicMock(contents=[]))
    assert len(sdk.calls_to("fetch")) == 1 and len(sdk.calls_to("memories.create")) == 1
    assert len(sdk.calls_to("instance.send_message")) == 1


# -- conversation-only read: no path, still works ---------------------------


async def test_short_term_instruction_works_with_a_path_configured(sdk):
    sdk.results["conversation.context.get_context_for_prompt"] = ContextForPromptResponse(
        formatted_context="x", context_block="## Conversation so far (compacted 1x)\nS",
        available=True)
    create_synap_callbacks(sdk, user_id="dana", conversation_id="c", scope_path=PATH)
    out = await synap_st_instruction(sdk, "c", instruction="sys")(MagicMock())
    assert "compacted" in out
    kw = sdk.last_kwargs("conversation.context.get_context_for_prompt")
    assert "scope_path" not in kw and "scope" not in kw
