"""`scope_path` (a deeper scope ladder's level) through synap-nemo-agent-toolkit.

Every SDK call is recorded with its exact keyword arguments (`RecordingSDK`).
The keyword is sent ONLY when a path was given.
"""

from __future__ import annotations

import sys

import pytest

# test_short_term.py stubs `nat` in sys.modules (see test_stream.py): bind the real one.
for _name in [n for n in list(sys.modules) if n == "nat" or n.startswith("nat.")]:
    if getattr(sys.modules[_name], "__file__", None) is None:
        del sys.modules[_name]

from nat.data_models.intermediate_step import (  # noqa: E402
    IntermediateStep, IntermediateStepPayload, IntermediateStepType, StreamEventData,
)
from nat.data_models.invocation_node import InvocationNode  # noqa: E402
from nat.memory.models import MemoryItem  # noqa: E402

from synap_integrations_common.testing import RecordingSDK, recording_sdk  # noqa: E402,F401
from synap_nemo_agent_toolkit.editor import SynapMemoryEditor  # noqa: E402
from synap_nemo_agent_toolkit.plugin import SynapMemoryClientConfig  # noqa: E402
from synap_nemo_agent_toolkit.short_term import SynapShortTermFunction  # noqa: E402
from synap_nemo_agent_toolkit.stream import (  # noqa: E402
    SynapStreamExporter, SynapStreamExporterConfig,
)

PATH = {"customer": "acme", "team": "payments", "user": "dana"}
_ANCESTRY = InvocationNode(function_id="fn-1", function_name="agent")


def _step(event_type, *, name=None, data=None):
    return IntermediateStep(
        parent_id="root", function_ancestry=_ANCESTRY,
        payload=IntermediateStepPayload(UUID="s1", event_type=event_type, name=name, data=data),
    )


# --- SynapMemoryEditor: memories.create, fetch, recorded turns --------------------

async def test_editor_default_calls_unchanged(recording_sdk):
    ed = SynapMemoryEditor(recording_sdk, customer_id="acme")
    await ed.add_items([MemoryItem(user_id="dana", memory="likes tea", tags=[], metadata={})])
    await ed.search("tea", user_id="dana")
    assert recording_sdk.last_kwargs("memories.create") == {
        "document": "likes tea", "user_id": "dana", "customer_id": "acme",
        "document_type": "ai-chat-conversation",
        "metadata": {"nat_memory_item": True, "tags": []},
    }
    assert recording_sdk.last_kwargs("fetch") == {
        "user_id": "dana", "customer_id": "acme", "search_query": ["tea"],
        "max_results": 5, "mode": "accurate", "include_conversation_context": False,
    }


async def test_editor_write_carries_path_as_scope(recording_sdk):
    ed = SynapMemoryEditor(recording_sdk, customer_id="acme", scope_path=PATH)
    await ed.add_items([MemoryItem(user_id="dana", memory="likes tea", tags=[], metadata={})])
    kw = recording_sdk.last_kwargs("memories.create")
    assert kw["scope"] == PATH and "scope_path" not in kw


async def test_editor_search_carries_path(recording_sdk):
    ed = SynapMemoryEditor(recording_sdk, customer_id="acme", scope_path=PATH)
    await ed.search("tea", user_id="dana")
    assert recording_sdk.last_kwargs("fetch")["scope_path"] == PATH


async def test_editor_recorded_turn_carries_path(recording_sdk):
    ed = SynapMemoryEditor(recording_sdk, customer_id="acme", scope_path=PATH)
    await ed.add_items([MemoryItem(
        user_id="dana", memory="",
        conversation=[{"role": "user", "content": "hello there"}], tags=[], metadata={})])
    assert recording_sdk.last_kwargs("instance.send_message")["scope_path"] == PATH
    assert recording_sdk.calls_to("memories.create") == []


async def test_editor_recorded_turn_default_has_no_path(recording_sdk):
    ed = SynapMemoryEditor(recording_sdk, customer_id="acme")
    await ed.add_items([MemoryItem(
        user_id="dana", memory="",
        conversation=[{"role": "user", "content": "hello there"}], tags=[], metadata={})])
    assert "scope_path" not in recording_sdk.last_kwargs("instance.send_message")


async def test_editor_rest_fallback_turn_carries_path():
    # no stream and no conversation id minted by it: the turn cannot be recorded,
    # so it falls back to the document, which carries the scope.
    sdk = RecordingSDK(listening=False)

    async def refuse():
        raise RuntimeError("no stream")

    sdk.instance.listen = refuse
    ed = SynapMemoryEditor(sdk, scope_path=PATH)
    await ed.add_items([MemoryItem(
        user_id="dana", memory="",
        conversation=[{"role": "user", "content": "hello there"}], tags=[], metadata={})])
    assert sdk.last_kwargs("memories.create")["scope"] == PATH


async def test_editor_ignores_a_scope_path_passed_per_call(recording_sdk):
    ed = SynapMemoryEditor(recording_sdk, scope_path=PATH)
    await ed.search("tea", user_id="dana", scope_path={"user": "mallory"})
    assert recording_sdk.last_kwargs("fetch")["scope_path"] == PATH
    ed2 = SynapMemoryEditor(recording_sdk)
    await ed2.search("tea", user_id="dana", scope_path={"user": "mallory"})
    assert "scope_path" not in recording_sdk.last_kwargs("fetch")


async def test_editor_copies_the_path(recording_sdk):
    path = dict(PATH)
    ed = SynapMemoryEditor(recording_sdk, scope_path=path)
    path["user"] = "mallory"
    await ed.search("tea", user_id="dana")
    assert recording_sdk.last_kwargs("fetch")["scope_path"] == PATH


# --- YAML config -------------------------------------------------------------------

def test_memory_config_has_scope_path_default_empty():
    assert SynapMemoryClientConfig().scope_path == {}
    assert SynapMemoryClientConfig(scope_path=PATH).scope_path == PATH


def test_stream_config_has_scope_path_default_empty():
    assert SynapStreamExporterConfig().scope_path == {}
    assert SynapStreamExporterConfig(scope_path=PATH).scope_path == PATH


async def test_yaml_factory_passes_the_configured_path(monkeypatch):
    import contextlib
    from synap_nemo_agent_toolkit import plugin

    sdk = RecordingSDK()

    @contextlib.asynccontextmanager
    async def fake_shared(config):
        yield sdk

    monkeypatch.setattr(plugin, "shared_sdk", fake_shared)
    gen = plugin.synap_memory_client.__wrapped__ if hasattr(
        plugin.synap_memory_client, "__wrapped__") else plugin.synap_memory_client
    async with contextlib.asynccontextmanager(gen)(
            SynapMemoryClientConfig(scope_path=PATH), None) as editor:
        assert editor.scope_path == PATH
    async with contextlib.asynccontextmanager(gen)(SynapMemoryClientConfig(), None) as editor:
        assert editor.scope_path is None


# --- SynapStreamExporter: the user and assistant turns -----------------------------

def _exporter(sdk, **kw):
    return SynapStreamExporter(sdk=sdk, conversation_id="conv-1", user_id="dana",
                               customer_id="acme", **kw)


async def _run(exporter):
    await exporter._report(_step(IntermediateStepType.WORKFLOW_START,
                                 data=StreamEventData(input="where is my order")))
    await exporter._report(_step(IntermediateStepType.WORKFLOW_END,
                                 data=StreamEventData(output="tomorrow")))


async def test_exporter_default_turns_unchanged(recording_sdk):
    await _run(_exporter(recording_sdk))
    sent = recording_sdk.kwargs_of("instance.send_message")
    assert [k["event_type"] for k in sent] == ["user_message", "assistant_message"]
    assert sent[0] == {"content": "where is my order", "role": "user",
                       "event_type": "user_message", "conversation_id": "conv-1",
                       "user_id": "dana", "customer_id": "acme", "metadata": {}}
    assert all("scope_path" not in k for k in sent)


async def test_exporter_user_and_assistant_turns_carry_path(recording_sdk):
    await _run(_exporter(recording_sdk, scope_path=PATH))
    sent = recording_sdk.kwargs_of("instance.send_message")
    assert [k["event_type"] for k in sent] == ["user_message", "assistant_message"]
    assert all(k["scope_path"] == PATH for k in sent)


async def test_exporter_rest_fallback_turns_carry_path():
    sdk = RecordingSDK(listening=False)

    async def refuse():
        raise RuntimeError("no stream")

    sdk.instance.listen = refuse
    await _run(_exporter(sdk, scope_path=PATH))
    rec = sdk.kwargs_of("conversation.record_message")
    assert len(rec) == 2 and all(k["scope_path"] == PATH for k in rec)


async def test_exporter_tool_events_take_no_path(recording_sdk):
    ex = _exporter(recording_sdk, scope_path=PATH)
    await ex._report(_step(IntermediateStepType.TOOL_START, name="lookup",
                           data=StreamEventData(input={"a": 1})))
    await ex._report(_step(IntermediateStepType.TOOL_END, name="lookup",
                           data=StreamEventData(output="ok")))
    for name in ("instance.record_tool_call", "instance.record_tool_result"):
        assert recording_sdk.kwargs_of(name)
        assert all("scope_path" not in k and "scope" not in k
                   for k in recording_sdk.kwargs_of(name))


# --- conversation-only reads still work with a configured path ---------------------

async def test_short_term_function_works_alongside_a_configured_path(recording_sdk):
    SynapMemoryEditor(recording_sdk, scope_path=PATH)
    _exporter(recording_sdk, scope_path=PATH)
    out = await SynapShortTermFunction(recording_sdk)("conv-1")
    assert "Recent conversation summary" in out
    ctx_calls = recording_sdk.kwargs_of("conversation.context.get_context_for_prompt")
    assert ctx_calls and all("scope_path" not in k and "scope" not in k for k in ctx_calls)


async def test_old_sdk_is_fine_without_a_path():
    sdk = RecordingSDK(accept_scope_path=False)
    ed = SynapMemoryEditor(sdk)
    await ed.add_items([MemoryItem(user_id="dana", memory="x", tags=[], metadata={})])
    await ed.search("x", user_id="dana")
    await _run(_exporter(sdk))
    assert sdk.calls_to("memories.create") and sdk.calls_to("fetch")


async def test_stream_yaml_factory_passes_the_configured_path(monkeypatch):
    import contextlib
    from synap_nemo_agent_toolkit import stream

    sdk = RecordingSDK()

    @contextlib.asynccontextmanager
    async def fake_shared(config):
        yield sdk

    monkeypatch.setattr(stream, "shared_sdk", fake_shared)
    gen = getattr(stream.synap_stream_exporter, "__wrapped__", stream.synap_stream_exporter)
    async with contextlib.asynccontextmanager(gen)(
            SynapStreamExporterConfig(scope_path=PATH), None) as exporter:
        assert exporter.scope_path == PATH
    async with contextlib.asynccontextmanager(gen)(SynapStreamExporterConfig(), None) as exporter:
        assert exporter.scope_path is None


# --- ids must match the path (the verifier's scenario: alice's path, bob's call) ---------
# The server lets the path win, so bob's call would read and write ALICE's memory.

import logging  # noqa: E402

from synap_integrations_common import SynapIntegrationError  # noqa: E402

ALICE = {"customer": "acme", "user": "alice"}


def _alice_editor(sdk, **kw):
    return SynapMemoryEditor(sdk, customer_id="acme", scope_path=ALICE, **kw)


def _no_sdk_calls(sdk):
    assert sdk.calls == [], f"the SDK received {[c.name for c in sdk.calls]}"


async def test_search_as_bob_and_rival_is_refused_and_sends_nothing(recording_sdk):
    with pytest.raises(SynapIntegrationError):
        await _alice_editor(recording_sdk).search("x", user_id="bob", customer_id="rival")
    _no_sdk_calls(recording_sdk)


async def test_search_as_bob_alone_is_refused(recording_sdk):
    with pytest.raises(SynapIntegrationError):
        await _alice_editor(recording_sdk).search("x", user_id="bob")
    _no_sdk_calls(recording_sdk)


async def test_search_with_rival_customer_alone_is_refused(recording_sdk):
    with pytest.raises(SynapIntegrationError):
        await _alice_editor(recording_sdk).search("x", user_id="alice", customer_id="rival")
    _no_sdk_calls(recording_sdk)


async def test_search_with_matching_ids_still_works(recording_sdk):
    await _alice_editor(recording_sdk).search("x", user_id="alice", customer_id="acme")
    assert recording_sdk.last_kwargs("fetch")["scope_path"] == ALICE


async def test_add_items_memory_for_bob_is_refused(recording_sdk):
    with pytest.raises(SynapIntegrationError):
        await _alice_editor(recording_sdk).add_items(
            [MemoryItem(user_id="bob", memory="note", tags=[], metadata={})])
    _no_sdk_calls(recording_sdk)


async def test_add_items_turns_for_bob_are_refused(recording_sdk):
    with pytest.raises(SynapIntegrationError):
        await _alice_editor(recording_sdk).add_items([MemoryItem(
            user_id="bob", memory="",
            conversation=[{"role": "user", "content": "hello there"}], tags=[], metadata={})])
    _no_sdk_calls(recording_sdk)


async def test_add_items_with_a_rival_customer_configured_is_refused(recording_sdk):
    ed = SynapMemoryEditor(recording_sdk, customer_id="rival", scope_path=ALICE)
    with pytest.raises(SynapIntegrationError):
        await ed.add_items([MemoryItem(user_id="alice", memory="n", tags=[], metadata={})])
    _no_sdk_calls(recording_sdk)


async def test_record_turns_helper_refuses_bob_directly(recording_sdk):
    with pytest.raises(SynapIntegrationError):
        await _alice_editor(recording_sdk)._record_turns(
            "bob", [{"role": "user", "content": "hi there"}])
    _no_sdk_calls(recording_sdk)


async def test_add_items_matching_ids_still_works(recording_sdk):
    await _alice_editor(recording_sdk).add_items(
        [MemoryItem(user_id="alice", memory="note", tags=[], metadata={})])
    assert recording_sdk.last_kwargs("memories.create")["scope"] == ALICE


async def test_no_path_means_no_check(recording_sdk):
    ed = SynapMemoryEditor(recording_sdk, customer_id="acme")
    await ed.search("x", user_id="bob", customer_id="rival")
    await ed.add_items([MemoryItem(user_id="bob", memory="n", tags=[], metadata={})])
    assert recording_sdk.last_kwargs("fetch")["user_id"] == "bob"
    assert "scope" not in recording_sdk.last_kwargs("memories.create")


def _wf_start():
    return _step(IntermediateStepType.WORKFLOW_START, data=StreamEventData(input="hello"))


async def test_exporter_skips_every_event_for_bob_and_logs_an_error(recording_sdk, caplog):
    ex = SynapStreamExporter(sdk=recording_sdk, conversation_id="c", user_id="alice",
                             customer_id="acme", scope_path=ALICE)
    ex._context_state.user_id.set("bob")  # NAT's request context says bob
    events = [
        _wf_start(),
        _step(IntermediateStepType.TOOL_START, name="t", data=StreamEventData(input={"a": 1})),
        _step(IntermediateStepType.TOOL_END, name="t", data=StreamEventData(output="ok")),
        _step(IntermediateStepType.WORKFLOW_END, data=StreamEventData(output="bye")),
    ]
    with caplog.at_level(logging.ERROR):
        for e in events:
            await ex._report(e)
    _no_sdk_calls(recording_sdk)
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 4 and "do not match" in errors[0].getMessage()
    assert "bob" not in caplog.text and "alice" not in caplog.text  # fingerprints only


async def test_exporter_with_a_rival_customer_is_skipped(recording_sdk):
    ex = SynapStreamExporter(sdk=recording_sdk, conversation_id="c", user_id="alice",
                             customer_id="rival", scope_path=ALICE)
    await ex._report(_wf_start())
    _no_sdk_calls(recording_sdk)


async def test_exporter_for_alice_still_reports(recording_sdk):
    ex = SynapStreamExporter(sdk=recording_sdk, conversation_id="c", user_id="alice",
                             customer_id="acme", scope_path=ALICE)
    ex._context_state.user_id.set("alice")
    await ex._report(_wf_start())
    assert recording_sdk.last_kwargs("instance.send_message")["scope_path"] == ALICE


async def test_exporter_without_a_path_reports_whoever_nat_says(recording_sdk):
    ex = SynapStreamExporter(sdk=recording_sdk, conversation_id="c", user_id="alice",
                             customer_id="acme")
    ex._context_state.user_id.set("bob")
    await ex._report(_wf_start())
    sent = recording_sdk.last_kwargs("instance.send_message")
    assert sent["user_id"] == "bob" and "scope_path" not in sent
