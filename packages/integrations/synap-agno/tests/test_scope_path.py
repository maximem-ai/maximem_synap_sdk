"""The scope ladder path through synap-agno.

Every test drives the package against ``RecordingSDK``, which keeps the exact
keyword arguments of each SDK call and refuses a keyword the installed SDK does
not take. Three things are pinned:

* a call made WITHOUT ``scope_path`` carries exactly the keywords it always did;
* with a ``scope_path`` it reaches every SDK call this package makes
  (``sdk.fetch``, ``memories.create``, and the user / assistant turns);
* the conversation-only reads (``synap_st_instructions``, history) take no path
  and still work when the package is configured with one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from agno.db.schemas.memory import UserMemory

from synap_agno import (
    AsyncSynapDb,
    SynapDb,
    create_synap_async_hooks,
    create_synap_hooks,
    synap_st_instructions,
)
from synap_integrations_common import set_auto_listen
from synap_integrations_common.testing import RecordingSDK, make_ladder

PATH = {"customer": "acme", "team": "payments", "user": "dana"}


def _run_context(session_id="sess-1", user_id="dana"):
    return SimpleNamespace(session_id=session_id, user_id=user_id, run_id="run-1")


def _run_input(text="where is my order"):
    return SimpleNamespace(input_content=text, input_content_string=lambda: text)


def _run_output(text="it ships tomorrow"):
    return SimpleNamespace(content=text, tools=None, reasoning_content=None, reasoning_steps=None)


# ── SynapDb: reads and writes ──────────────────────────────────────────────


def test_db_default_calls_are_unchanged():
    sdk = RecordingSDK()
    db = SynapDb(sdk, "acme")
    db.upsert_user_memory(UserMemory(memory="likes tea", user_id="dana"))
    db.get_user_memories(user_id="dana", search_content="tea")

    create = sdk.last_kwargs("memories.create")
    assert set(create) == {"document", "user_id", "customer_id", "metadata"}
    assert "scope" not in create
    fetch = sdk.last_kwargs("fetch")
    assert fetch == {
        "user_id": "dana", "customer_id": "acme", "search_query": ["tea"],
        "max_results": 50, "mode": "accurate", "include_conversation_context": False,
    }


def test_db_empty_scope_path_counts_as_unset():
    sdk = RecordingSDK()
    db = SynapDb(sdk, "acme", scope_path={})
    db.get_user_memories(user_id="dana")
    assert "scope_path" not in sdk.last_kwargs("fetch")


def test_db_write_carries_the_path_as_scope():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    db = SynapDb(sdk, "acme", scope_path=PATH)
    db.upsert_user_memory(UserMemory(memory="likes tea", user_id="dana"))
    assert sdk.last_kwargs("memories.create")["scope"] == PATH
    assert "scope_path" not in sdk.last_kwargs("memories.create")


def test_db_upsert_memories_carries_the_path_on_every_write():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    db = SynapDb(sdk, "acme", scope_path=PATH)
    db.upsert_memories([UserMemory(memory="a", user_id="dana"), UserMemory(memory="b", user_id="dana")])
    assert [k["scope"] for k in sdk.kwargs_of("memories.create")] == [PATH, PATH]


def test_db_read_carries_the_path():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    db = SynapDb(sdk, "acme", scope_path=PATH)
    db.get_user_memories(user_id="dana", search_content="tea")
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


def test_db_topics_read_carries_the_path():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    db = SynapDb(sdk, "acme", scope_path=PATH)
    db.get_all_memory_topics(user_id="dana")
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


def test_db_keeps_its_own_copy_of_the_path():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    mine = dict(PATH)
    db = SynapDb(sdk, "acme", scope_path=mine)
    mine["user"] = "someone-else"
    db.get_user_memories(user_id="dana")
    assert sdk.last_kwargs("fetch")["scope_path"] == PATH


async def test_async_db_default_unchanged_and_path_reaches_both_calls():
    plain = RecordingSDK()
    await AsyncSynapDb(plain, "acme").upsert_user_memory(UserMemory(memory="x", user_id="dana"))
    await AsyncSynapDb(plain, "acme").get_user_memories(user_id="dana")
    assert "scope" not in plain.last_kwargs("memories.create")
    assert "scope_path" not in plain.last_kwargs("fetch")

    sdk = RecordingSDK(ladder=make_ladder("deep"))
    db = AsyncSynapDb(sdk, "acme", scope_path=PATH)
    await db.upsert_user_memory(UserMemory(memory="x", user_id="dana"))
    await db.get_user_memories(user_id="dana")
    await db.get_all_memory_topics(user_id="dana")
    assert sdk.last_kwargs("memories.create")["scope"] == PATH
    assert [k["scope_path"] for k in sdk.kwargs_of("fetch")] == [PATH, PATH]


# ── hooks: the user and assistant turns ────────────────────────────────────


async def test_async_hooks_default_turns_are_unchanged():
    sdk = RecordingSDK()
    hooks = create_synap_async_hooks(sdk, customer_id="acme")
    await hooks["pre_hooks"][0](run_input=_run_input(), run_context=_run_context())
    await hooks["post_hooks"][0](run_output=_run_output(), run_context=_run_context())
    sent = sdk.kwargs_of("instance.send_message")
    assert [k["event_type"] for k in sent] == ["user_message", "assistant_message"]
    for k in sent:
        assert "scope_path" not in k
        assert k["customer_id"] == "acme" and k["user_id"] == "dana"


async def test_async_hooks_path_reaches_user_and_assistant_turns():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    hooks = create_synap_async_hooks(sdk, customer_id="acme", scope_path=PATH)
    await hooks["pre_hooks"][0](run_input=_run_input(), run_context=_run_context())
    await hooks["post_hooks"][0](run_output=_run_output(), run_context=_run_context())
    sent = sdk.kwargs_of("instance.send_message")
    assert [k["event_type"] for k in sent] == ["user_message", "assistant_message"]
    assert [k["scope_path"] for k in sent] == [PATH, PATH]


async def test_async_hooks_path_reaches_the_rest_fallback():
    sdk = RecordingSDK(ladder=make_ladder("deep"), listening=False)
    set_auto_listen(sdk, False)  # else the first turn opens the stream and goes on it
    hooks = create_synap_async_hooks(sdk, customer_id="acme", scope_path=PATH)
    await hooks["pre_hooks"][0](run_input=_run_input(), run_context=_run_context())
    await hooks["post_hooks"][0](run_output=_run_output(), run_context=_run_context())
    recorded = sdk.kwargs_of("conversation.record_message")
    assert [k["role"] for k in recorded] == ["user", "assistant"]
    assert [k["scope_path"] for k in recorded] == [PATH, PATH]


async def test_async_hooks_default_rest_fallback_is_unchanged():
    sdk = RecordingSDK(listening=False)
    set_auto_listen(sdk, False)
    hooks = create_synap_async_hooks(sdk, customer_id="acme")
    await hooks["pre_hooks"][0](run_input=_run_input(), run_context=_run_context())
    assert all("scope_path" not in k for k in sdk.kwargs_of("conversation.record_message"))
    assert sdk.calls_to("conversation.record_message")


def test_sync_hooks_path_reaches_user_and_assistant_turns():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    hooks = create_synap_hooks(sdk, customer_id="acme", scope_path=PATH)
    hooks["pre_hooks"][0](run_input=_run_input(), run_context=_run_context())
    hooks["post_hooks"][0](run_output=_run_output(), run_context=_run_context())
    sent = sdk.kwargs_of("instance.send_message")
    assert [k["event_type"] for k in sent] == ["user_message", "assistant_message"]
    assert [k["scope_path"] for k in sent] == [PATH, PATH]


def test_sync_hooks_default_turns_are_unchanged():
    sdk = RecordingSDK()
    hooks = create_synap_hooks(sdk, customer_id="acme")
    hooks["pre_hooks"][0](run_input=_run_input(), run_context=_run_context())
    hooks["post_hooks"][0](run_output=_run_output(), run_context=_run_context())
    assert all("scope_path" not in k for k in sdk.kwargs_of("instance.send_message"))


async def test_tool_events_carry_no_path():
    """Tool calls, results and reasoning belong to the conversation, which holds its path."""
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    out = _run_output()
    out.tools = [SimpleNamespace(tool_call_id="c1", tool_name="lookup", tool_args={"a": 1}, result="ok")]
    out.reasoning_content = "thinking"
    hooks = create_synap_async_hooks(sdk, customer_id="acme", scope_path=PATH)
    await hooks["post_hooks"][0](run_output=out, run_context=_run_context())
    for name in ("instance.record_tool_call", "instance.record_tool_result", "instance.record_thinking"):
        assert sdk.calls_to(name), name
        assert all("scope_path" not in k for k in sdk.kwargs_of(name))


# ── conversation-only reads still work in a package configured with a path ──


async def test_short_term_and_history_still_work_when_a_path_is_configured():
    sdk = RecordingSDK(ladder=make_ladder("deep"))
    # The short-term helper reads through the conversation: no identity, no path.
    await synap_st_instructions(sdk, "sess-1")()
    ctx_calls = sdk.kwargs_of("conversation.context.get_context_for_prompt")
    assert ctx_calls and all("scope_path" not in k and "user_id" not in k for k in ctx_calls)

    # The hooks' history read (prepare_history) is the same kind of call.
    sdk.reset()
    hooks = create_synap_async_hooks(sdk, customer_id="acme", scope_path=PATH)
    session = SimpleNamespace(get_messages=lambda *a, **k: [])  # prepare_history needs one
    await hooks["pre_hooks"][0](
        run_input=_run_input(), run_context=_run_context(), session=session)
    ctx_calls = sdk.kwargs_of("conversation.context.get_context_for_prompt")
    assert ctx_calls and all("scope_path" not in k and "user_id" not in k for k in ctx_calls)


# ── ids that disagree with the fixed path are refused (fail closed) ─────────
#
# Agno names the user on every call, and the server lets the path's `user` win
# over a flat id: without these checks a multi-user app would read and file as
# the path's user.

from synap_integrations_common import SynapIntegrationError  # noqa: E402

PATH2 = {"customer": "acme", "user": "dana"}


def _no_sdk_call(sdk):
    assert sdk.calls == [], [c.name for c in sdk.calls]


def test_construction_refuses_a_customer_id_that_disagrees_with_the_path():
    sdk = RecordingSDK(ladder=make_ladder("b2b"))
    with pytest.raises(SynapIntegrationError):
        SynapDb(sdk, "other-customer", scope_path=PATH2)
    with pytest.raises(SynapIntegrationError):
        AsyncSynapDb(sdk, "other-customer", scope_path=PATH2)
    SynapDb(sdk, "acme", scope_path=PATH2)  # matching: fine
    SynapDb(sdk, "", scope_path=PATH2)      # empty: the path supplies it
    SynapDb(sdk, "other-customer")          # no path: unchanged
    _no_sdk_call(sdk)


def test_construction_refuses_hooks_ids_that_disagree_with_the_path():
    sdk = RecordingSDK()
    for factory in (create_synap_hooks, create_synap_async_hooks):
        with pytest.raises(SynapIntegrationError):
            factory(sdk, customer_id="acme", user_id="bob", scope_path=PATH2)
        with pytest.raises(SynapIntegrationError):
            factory(sdk, customer_id="other", scope_path=PATH2)
        factory(sdk, customer_id="acme", user_id="dana", scope_path=PATH2)
        factory(sdk, customer_id="acme", user_id="bob")  # no path: unchanged
    _no_sdk_call(sdk)


def test_upsert_for_another_user_is_refused_and_nothing_is_sent():
    sdk = RecordingSDK()
    db = SynapDb(sdk, "acme", scope_path=PATH2)
    with pytest.raises(SynapIntegrationError):
        db.upsert_user_memory(UserMemory(memory="x", user_id="zed"))
    _no_sdk_call(sdk)


def test_upsert_memories_refuses_the_whole_batch_before_any_write():
    sdk = RecordingSDK()
    db = SynapDb(sdk, "acme", scope_path=PATH2)
    with pytest.raises(SynapIntegrationError):
        db.upsert_memories([UserMemory(memory="a", user_id="dana"),
                            UserMemory(memory="b", user_id="zed")])
    _no_sdk_call(sdk)


@pytest.mark.parametrize("call", [
    lambda db: db.get_user_memories(user_id="zed"),
    lambda db: db.get_all_memory_topics(user_id="zed"),
    lambda db: db.get_user_memory("00000000-0000-0000-0000-000000000001", user_id="zed"),
    lambda db: db.delete_user_memory("00000000-0000-0000-0000-000000000001", user_id="zed"),
    lambda db: db.delete_user_memories(["00000000-0000-0000-0000-000000000001"], user_id="zed"),
], ids=["get_user_memories", "topics", "get_user_memory", "delete_one", "delete_many"])
def test_db_calls_for_another_user_raise_and_send_nothing(call):
    sdk = RecordingSDK()
    db = SynapDb(sdk, "acme", scope_path=PATH2)
    with pytest.raises(SynapIntegrationError):
        call(db)
    _no_sdk_call(sdk)


def test_db_calls_for_the_path_user_still_work():
    sdk = RecordingSDK()
    db = SynapDb(sdk, "acme", scope_path=PATH2)
    db.upsert_user_memory(UserMemory(memory="x", user_id="dana"))
    db.upsert_user_memory(UserMemory(memory="y"))  # no id: the path names the user
    db.get_user_memories(user_id="dana")
    db.get_user_memories()
    assert len(sdk.calls_to("memories.create")) == 2
    assert [k["scope_path"] for k in sdk.kwargs_of("fetch")] == [PATH2, PATH2]


def test_db_without_a_path_serves_any_user_as_before():
    sdk = RecordingSDK()
    db = SynapDb(sdk, "acme")
    db.upsert_user_memory(UserMemory(memory="x", user_id="zed"))
    db.get_user_memories(user_id="zed")
    assert sdk.last_kwargs("memories.create")["user_id"] == "zed"
    assert sdk.last_kwargs("fetch")["user_id"] == "zed"


async def test_async_db_for_another_user_raises_and_sends_nothing():
    sdk = RecordingSDK()
    db = AsyncSynapDb(sdk, "acme", scope_path=PATH2)
    with pytest.raises(SynapIntegrationError):
        await db.upsert_user_memory(UserMemory(memory="x", user_id="zed"))
    with pytest.raises(SynapIntegrationError):
        await db.upsert_memories([UserMemory(memory="a", user_id="dana"),
                                  UserMemory(memory="b", user_id="zed")])
    with pytest.raises(SynapIntegrationError):
        await db.get_user_memories(user_id="zed")
    with pytest.raises(SynapIntegrationError):
        await db.get_all_memory_topics(user_id="zed")
    with pytest.raises(SynapIntegrationError):
        await db.get_user_memory("00000000-0000-0000-0000-000000000001", user_id="zed")
    with pytest.raises(SynapIntegrationError):
        await db.delete_user_memory("00000000-0000-0000-0000-000000000001", user_id="zed")
    with pytest.raises(SynapIntegrationError):
        await db.delete_user_memories(["00000000-0000-0000-0000-000000000001"], user_id="zed")
    _no_sdk_call(sdk)
    await db.upsert_user_memory(UserMemory(memory="x", user_id="dana"))
    assert sdk.last_kwargs("memories.create")["scope"] == PATH2


async def test_async_hooks_skip_a_run_for_another_user_and_send_nothing(caplog):
    sdk = RecordingSDK()
    hooks = create_synap_async_hooks(sdk, customer_id="acme", scope_path=PATH2)
    session = SimpleNamespace(get_messages=lambda *a, **k: [])
    ctx = _run_context(user_id="zed")
    with caplog.at_level("ERROR"):
        await hooks["pre_hooks"][0](run_input=_run_input(), run_context=ctx, session=session)
        await hooks["post_hooks"][0](run_output=_run_output(), run_context=ctx, session=session)
        await hooks["pre_hooks"][0](run_input=_run_input(), run_context=_run_context(),
                                    session=session, user_id="zed")  # per-run override
    _no_sdk_call(sdk)
    assert "NOT reported" in caplog.text and "zed" not in caplog.text


def test_sync_hooks_skip_a_run_for_another_user_and_send_nothing():
    sdk = RecordingSDK()
    hooks = create_synap_hooks(sdk, customer_id="acme", scope_path=PATH2)
    ctx = _run_context(user_id="zed")
    hooks["pre_hooks"][0](run_input=_run_input(), run_context=ctx)
    hooks["post_hooks"][0](run_output=_run_output(), run_context=ctx)
    _no_sdk_call(sdk)


async def test_hooks_still_report_the_path_user():
    sdk = RecordingSDK()
    hooks = create_synap_async_hooks(sdk, customer_id="acme", scope_path=PATH2)
    await hooks["pre_hooks"][0](run_input=_run_input(), run_context=_run_context(user_id="dana"))
    await hooks["post_hooks"][0](run_output=_run_output(), run_context=_run_context(user_id="dana"))
    assert [k["scope_path"] for k in sdk.kwargs_of("instance.send_message")] == [PATH2, PATH2]
