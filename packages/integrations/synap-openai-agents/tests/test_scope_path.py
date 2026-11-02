"""`scope_path` (a deeper scope ladder's level) through synap-openai-agents.

Every SDK call is recorded with its exact keyword arguments (`RecordingSDK`).
The keyword is sent ONLY when a path was given. The two tools the model can call
close the path over at construction: it is never in their schema and a value the
model supplies is not used.
"""

import json

import pytest
from agents import function_tool
from agents.tool_context import ToolContext

from synap_integrations_common import SynapIntegrationError
from synap_integrations_common.testing import RecordingSDK, recording_sdk  # noqa: F401
from synap_openai_agents import (
    SynapRunHooks,
    create_search_tool,
    create_store_tool,
    report_user_turn,
    synap_st_instructions,
)

PATH = {"customer": "acme", "team": "payments", "user": "dana"}
OTHER = {"customer": "rival", "user": "mallory"}


# --- create_search_tool: sdk.fetch ---------------------------------------------

async def test_search_default_fetch_kwargs_unchanged(recording_sdk):
    fn = create_search_tool(recording_sdk, user_id="dana", customer_id="acme",
                            conversation_id="c1")
    await fn("coffee")
    assert recording_sdk.last_kwargs("fetch") == {
        "conversation_id": "c1", "user_id": "dana", "customer_id": "acme",
        "search_query": ["coffee"], "mode": "accurate",
        "include_conversation_context": False,
    }


async def test_search_forwards_scope_path_to_fetch(recording_sdk):
    fn = create_search_tool(recording_sdk, user_id="dana", customer_id="acme",
                            scope_path=PATH)
    await fn("coffee")
    kw = recording_sdk.last_kwargs("fetch")
    assert kw["scope_path"] == PATH
    assert kw["user_id"] == "dana" and kw["customer_id"] == "acme"


async def test_search_empty_path_not_sent(recording_sdk):
    await create_search_tool(recording_sdk, user_id="dana", scope_path={})("q")
    assert "scope_path" not in recording_sdk.last_kwargs("fetch")


# --- create_store_tool: memories.create (the argument is `scope`) ----------------

async def test_store_default_create_kwargs_unchanged(recording_sdk):
    await create_store_tool(recording_sdk, user_id="dana", customer_id="acme")("likes tea")
    assert recording_sdk.last_kwargs("memories.create") == {
        "document": "likes tea", "user_id": "dana", "customer_id": "acme"}


async def test_store_forwards_path_as_scope(recording_sdk):
    await create_store_tool(recording_sdk, user_id="dana", customer_id="acme",
                            scope_path=PATH)("likes tea")
    kw = recording_sdk.last_kwargs("memories.create")
    assert kw["scope"] == PATH and "scope_path" not in kw


# --- LLM tools: the path is closed over, never model-controlled ------------------

def _schema_text(tool):
    return json.dumps(tool.params_json_schema)


@pytest.mark.parametrize("factory", [create_search_tool, create_store_tool])
def test_scope_path_is_not_in_the_tool_schema(recording_sdk, factory):
    fn = factory(recording_sdk, user_id="dana", scope_path=PATH)
    tool = function_tool(fn)
    props = tool.params_json_schema["properties"]
    assert set(props) == {"query"} or set(props) == {"content"}
    assert "scope" not in _schema_text(tool) and "customer" not in _schema_text(tool)


@pytest.mark.parametrize("factory,arg", [(create_search_tool, "query"),
                                         (create_store_tool, "content")])
def test_tool_function_signature_has_no_path_parameter(recording_sdk, factory, arg):
    import inspect
    fn = factory(recording_sdk, user_id="dana", scope_path=PATH)
    assert list(inspect.signature(fn).parameters) == [arg]


async def _call_as_model(tool, payload):
    ctx = ToolContext(context=None, tool_name=tool.name, tool_call_id="call-1",
                      tool_arguments=json.dumps(payload))
    return await tool.on_invoke_tool(ctx, json.dumps(payload))


async def test_model_supplied_scope_path_is_not_used_by_search(recording_sdk):
    tool = function_tool(create_search_tool(recording_sdk, user_id="dana", scope_path=PATH))
    try:
        await _call_as_model(tool, {"query": "x", "scope_path": OTHER})
    except Exception:  # noqa: BLE001 - rejecting the call is also acceptable
        pass
    # Rejected (no fetch) or run with the closed-over path; never the model's.
    assert all(kw.get("scope_path") == PATH for kw in recording_sdk.kwargs_of("fetch"))
    assert OTHER not in [kw.get("scope_path") for kw in recording_sdk.kwargs_of("fetch")]


async def test_model_supplied_scope_path_is_not_used_by_store(recording_sdk):
    tool = function_tool(create_store_tool(recording_sdk, user_id="dana", scope_path=PATH))
    try:
        await _call_as_model(tool, {"content": "x", "scope": OTHER, "scope_path": OTHER})
    except Exception:  # noqa: BLE001
        pass
    for kw in recording_sdk.kwargs_of("memories.create"):
        assert kw.get("scope") == PATH


async def test_model_call_without_closed_path_never_gains_one(recording_sdk):
    tool = function_tool(create_search_tool(recording_sdk, user_id="dana"))
    try:
        await _call_as_model(tool, {"query": "x", "scope_path": OTHER})
    except Exception:  # noqa: BLE001
        pass
    for kw in recording_sdk.kwargs_of("fetch"):
        assert "scope_path" not in kw


async def test_the_closed_over_path_is_a_copy(recording_sdk):
    path = dict(PATH)
    fn = create_search_tool(recording_sdk, user_id="dana", scope_path=path)
    path["user"] = "mallory"
    await fn("q")
    assert recording_sdk.last_kwargs("fetch")["scope_path"] == PATH


# --- SynapRunHooks: the two turns carry the path; events that cannot do not ------

class _Ctx:
    pass


async def _run_through(hooks):
    ctx = _Ctx()
    await hooks.on_llm_start(ctx, None, None, [{"role": "user", "content": "hello"}])
    await hooks.on_agent_end(ctx, None, "hi there")


async def test_hooks_default_turns_unchanged(recording_sdk):
    await _run_through(SynapRunHooks(recording_sdk, user_id="dana", customer_id="acme",
                                     conversation_id="c1"))
    sent = recording_sdk.kwargs_of("instance.send_message")
    assert [k["event_type"] for k in sent] == ["user_message", "assistant_message"]
    assert all("scope_path" not in k for k in sent)
    assert sent[0] == {"content": "hello", "role": "user", "event_type": "user_message",
                       "conversation_id": "c1", "user_id": "dana", "customer_id": "acme",
                       "metadata": {}}


async def test_hooks_user_and_assistant_turns_carry_scope_path(recording_sdk):
    await _run_through(SynapRunHooks(recording_sdk, user_id="dana", customer_id="acme",
                                     conversation_id="c1", scope_path=PATH))
    sent = recording_sdk.kwargs_of("instance.send_message")
    assert [k["event_type"] for k in sent] == ["user_message", "assistant_message"]
    assert all(k["scope_path"] == PATH for k in sent)


async def test_hooks_rest_fallback_turns_carry_scope_path():
    sdk = RecordingSDK(listening=False)

    async def refuse():
        raise RuntimeError("no stream")

    sdk.instance.listen = refuse
    await _run_through(SynapRunHooks(sdk, user_id="dana", conversation_id="c1",
                                     scope_path=PATH))
    rec = sdk.kwargs_of("conversation.record_message")
    assert len(rec) == 2 and all(k["scope_path"] == PATH for k in rec)


async def test_hooks_tool_events_take_no_path(recording_sdk):
    from types import SimpleNamespace
    h = SynapRunHooks(recording_sdk, user_id="dana", conversation_id="c1", scope_path=PATH)
    t = SimpleNamespace(name="lookup")
    await h.on_tool_start(_Ctx(), None, t)
    await h.on_tool_end(_Ctx(), None, t, "done")
    for name in ("instance.record_tool_call", "instance.record_tool_result"):
        assert recording_sdk.kwargs_of(name)
        assert all("scope_path" not in k and "scope" not in k
                   for k in recording_sdk.kwargs_of(name))


async def test_report_user_turn_default_and_path(recording_sdk):
    await report_user_turn(recording_sdk, "hello", conversation_id="c1", user_id="dana")
    assert "scope_path" not in recording_sdk.last_kwargs("instance.send_message")
    await report_user_turn(recording_sdk, "again", conversation_id="c1", user_id="dana",
                           scope_path=PATH)
    assert recording_sdk.last_kwargs("instance.send_message")["scope_path"] == PATH


# --- conversation-only reads still work with a configured path -------------------

async def test_short_term_and_history_work_with_a_configured_path(recording_sdk):
    from maximem_synap.models.context import ContextForPromptResponse
    recording_sdk.results["conversation.context.get_context_for_prompt"] = (
        ContextForPromptResponse(formatted_context="x", context_block="COMPACTED SUMMARY",
                                 available=True))
    hooks = SynapRunHooks(recording_sdk, user_id="dana", conversation_id="c1", scope_path=PATH)
    text = await synap_st_instructions(recording_sdk, "c1")(None, None)
    assert "COMPACTED SUMMARY" in text
    assert hooks.run_kwargs()["run_config"].call_model_input_filter is not None
    ctx_calls = recording_sdk.kwargs_of("conversation.context.get_context_for_prompt")
    assert ctx_calls and all("scope_path" not in k for k in ctx_calls)


async def test_old_sdk_is_fine_without_a_path():
    sdk = RecordingSDK(accept_scope_path=False)
    await create_search_tool(sdk, user_id="dana")("q")
    await create_store_tool(sdk, user_id="dana")("m")
    await _run_through(SynapRunHooks(sdk, user_id="dana", conversation_id="c1"))
    assert sdk.calls_to("fetch") and sdk.calls_to("memories.create")


# --- ids must match the path (alice's path, bob's call) ------------------------------------

from synap_integrations_common import SynapIntegrationError  # noqa: E402

ALICE = {"customer": "acme", "user": "alice"}


@pytest.mark.parametrize("ids", [
    {"user_id": "bob", "customer_id": "rival"},
    {"user_id": "bob", "customer_id": "acme"},
    {"user_id": "alice", "customer_id": "rival"},
])
async def test_report_user_turn_with_other_ids_is_refused_and_sends_nothing(recording_sdk, ids):
    with pytest.raises(SynapIntegrationError):
        await report_user_turn(recording_sdk, "hi", conversation_id="c", scope_path=ALICE, **ids)
    assert recording_sdk.calls == []


async def test_report_user_turn_with_matching_ids_still_works(recording_sdk):
    await report_user_turn(recording_sdk, "hi", conversation_id="c", user_id="alice",
                           customer_id="acme", scope_path=ALICE)
    assert recording_sdk.last_kwargs("instance.send_message")["scope_path"] == ALICE


async def test_report_user_turn_without_a_path_is_not_checked(recording_sdk):
    await report_user_turn(recording_sdk, "hi", conversation_id="c", user_id="bob",
                           customer_id="rival")
    k = recording_sdk.last_kwargs("instance.send_message")
    assert k["user_id"] == "bob" and "scope_path" not in k
