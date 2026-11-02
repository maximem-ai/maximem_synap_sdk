"""`scope_path` (a deeper scope ladder's level) through synap-microsoft-agent.

Every SDK call is recorded with its exact keyword arguments (`RecordingSDK`).
The keyword is sent ONLY when a path was given. The store's model-facing tools
(search_transcripts, the file store's read/write) close the path over at
construction; a model cannot choose a level.
"""

from __future__ import annotations

import inspect

import pytest
from agent_framework import AgentResponse, Message, SessionContext

from synap_integrations_common.testing import RecordingSDK, recording_sdk  # noqa: F401
from synap_microsoft_agent import (
    SynapAgentFileStore,
    SynapContextProvider,
    SynapHistoryProvider,
    SynapMemoryStore,
    SynapShortTermContextProvider,
    create_synap_harness_memory,
)
from synap_microsoft_agent.stream import report_message_turn
from tests._harness import make_record, make_session, requires_harness

PATH = {"customer": "acme", "team": "payments", "user": "alice"}


def _no_stream():
    """An SDK whose stream will not open: turns go over REST."""
    sdk = RecordingSDK(listening=False)

    async def refuse():
        raise RuntimeError("no stream")

    sdk.instance.listen = refuse
    return sdk


def _ctx(session_id="conv-1", text="Hello", reply="Hi there"):
    ctx = SessionContext(session_id=session_id, input_messages=[Message(role="user", contents=[text])])
    ctx._response = AgentResponse(messages=[Message(role="assistant", contents=[reply])])
    return ctx


# --- stream.report_message_turn ---------------------------------------------------

async def test_report_message_turn_default_and_path(recording_sdk):
    await report_message_turn(recording_sdk, role="user", content="hi",
                              conversation_id="c1", user_id="alice", customer_id="acme")
    assert recording_sdk.last_kwargs("instance.send_message") == {
        "content": "hi", "role": "user", "event_type": "user_message",
        "conversation_id": "c1", "user_id": "alice", "customer_id": "acme", "metadata": {}}
    await report_message_turn(recording_sdk, role="user", content="hi", conversation_id="c1",
                              user_id="alice", customer_id="acme", scope_path=PATH)
    assert recording_sdk.last_kwargs("instance.send_message")["scope_path"] == PATH


# --- SynapContextProvider: fetch, both turn writes (stream and REST) ---------------

async def test_context_provider_default_calls_unchanged(recording_sdk):
    p = SynapContextProvider(recording_sdk, user_id="alice", customer_id="acme",
                             conversation_id="conv-1")
    ctx = _ctx()
    await p.before_run(agent=None, session=None, context=ctx, state={})
    assert recording_sdk.last_kwargs("fetch") == {
        "conversation_id": "conv-1", "user_id": "alice", "customer_id": "acme",
        "search_query": ["Hello"], "max_results": 20, "mode": "accurate",
        "include_conversation_context": False, "include_scope_labels": False}
    assert "scope_path" not in recording_sdk.last_kwargs("instance.send_message")


async def test_context_provider_fetch_carries_path(recording_sdk):
    p = SynapContextProvider(recording_sdk, user_id="alice", customer_id="acme",
                             conversation_id="conv-1", scope_path=PATH)
    await p.before_run(agent=None, session=None, context=_ctx(), state={})
    assert recording_sdk.last_kwargs("fetch")["scope_path"] == PATH


async def test_context_provider_input_turn_on_stream_carries_path(recording_sdk):
    p = SynapContextProvider(recording_sdk, user_id="alice", conversation_id="conv-1",
                             scope_path=PATH)
    await p.before_run(agent=None, session=None, context=_ctx(), state={})
    assert recording_sdk.last_kwargs("instance.send_message")["scope_path"] == PATH


async def test_context_provider_after_run_stream_turn_carries_path(recording_sdk):
    p = SynapContextProvider(recording_sdk, user_id="alice", conversation_id="conv-1",
                             scope_path=PATH)
    ctx = _ctx()
    ctx._response = AgentResponse(messages=[Message(role="assistant", contents=["Hi there"])])
    await p.after_run(agent=None, session=None, context=ctx, state={})
    sent = recording_sdk.kwargs_of("instance.send_message")
    assert sent and all(k["scope_path"] == PATH for k in sent)


async def test_context_provider_rest_turns_carry_path():
    sdk = _no_stream()
    p = SynapContextProvider(sdk, user_id="alice", customer_id="acme",
                             conversation_id="conv-1", scope_path=PATH)
    await p.after_run(agent=None, session=None, context=_ctx(), state={})
    rec = sdk.kwargs_of("conversation.record_message")
    assert len(rec) == 2 and all(k["scope_path"] == PATH for k in rec)


async def test_context_provider_rest_default_has_no_path():
    sdk = _no_stream()
    p = SynapContextProvider(sdk, user_id="alice", customer_id="acme", conversation_id="conv-1")
    await p.after_run(agent=None, session=None, context=_ctx(), state={})
    rec = sdk.kwargs_of("conversation.record_message")
    assert len(rec) == 2 and all("scope_path" not in k for k in rec)


# --- SynapHistoryProvider: turn writes; loading takes none --------------------------

async def test_history_provider_stream_and_rest_turns_carry_path(recording_sdk):
    msgs = [Message(role="user", contents=["hi"])]
    p = SynapHistoryProvider(recording_sdk, user_id="alice", customer_id="acme",
                             conversation_id="conv-1", scope_path=PATH)
    await p.save_messages("conv-1", msgs)
    assert recording_sdk.last_kwargs("instance.send_message")["scope_path"] == PATH
    sdk = _no_stream()
    await SynapHistoryProvider(sdk, user_id="alice", conversation_id="conv-1",
                               scope_path=PATH).save_messages("conv-1", msgs)
    assert sdk.last_kwargs("conversation.record_message")["scope_path"] == PATH


async def test_history_provider_default_unchanged_and_load_takes_no_path(recording_sdk):
    p = SynapHistoryProvider(recording_sdk, user_id="alice", customer_id="acme",
                             conversation_id="conv-1")
    await p.save_messages("conv-1", [Message(role="user", contents=["hi"])])
    assert "scope_path" not in recording_sdk.last_kwargs("instance.send_message")
    p2 = SynapHistoryProvider(recording_sdk, user_id="alice", conversation_id="conv-1",
                              scope_path=PATH)
    await p2.get_messages("conv-1")
    ctx_calls = recording_sdk.kwargs_of("conversation.context.get_context_for_prompt")
    assert ctx_calls and all("scope_path" not in k and "scope" not in k for k in ctx_calls)


async def test_short_term_provider_works_next_to_a_configured_path(recording_sdk):
    SynapContextProvider(recording_sdk, user_id="alice", conversation_id="c", scope_path=PATH)
    st = SynapShortTermContextProvider(recording_sdk, conversation_id="conv-1")
    ctx = _ctx()
    await st.before_run(agent=None, session=None, context=ctx, state={})
    ctx_calls = recording_sdk.kwargs_of("conversation.context.get_context_for_prompt")
    assert ctx_calls and all("scope_path" not in k for k in ctx_calls)


# --- SynapMemoryStore (harness) -----------------------------------------------------

@requires_harness
class TestMemoryStore:
    def test_default_calls_and_annotation_unchanged(self, recording_sdk):
        store = SynapMemoryStore(recording_sdk, user_id="alice", customer_id="acme")
        session = make_session()
        store.write_topic(session, make_record(), source_id="memory")
        store.get_index_text(session, source_id="memory", line_limit=200, line_length=150)
        kw = recording_sdk.last_kwargs("memories.create")
        assert kw["user_id"] == "alice" and kw["customer_id"] == "acme"
        assert "scope" not in kw and "scope_path" not in kw
        assert "scope_path" not in recording_sdk.last_kwargs("fetch")
        assert store.get_owner_id(session) == "/acme/alice"

    def test_write_carries_path_as_scope_and_fetch_as_scope_path(self, recording_sdk):
        store = SynapMemoryStore(recording_sdk, user_id="alice", customer_id="acme",
                                 scope_path=PATH)
        session = make_session()
        store.write_topic(session, make_record(), source_id="memory")
        create = recording_sdk.last_kwargs("memories.create")
        assert create["scope"] == PATH and "scope_path" not in create
        store.get_index_text(session, source_id="memory", line_limit=200, line_length=150)
        fetch = recording_sdk.last_kwargs("fetch")
        assert fetch["scope_path"] == PATH and "scope" not in fetch

    def test_search_transcripts_fetch_carries_path(self, recording_sdk):
        store = SynapMemoryStore(recording_sdk, user_id="alice", scope_path=PATH)
        store.search_transcripts(make_session(), source_id="memory", query="invoices")
        assert recording_sdk.last_kwargs("fetch")["scope_path"] == PATH

    def test_annotation_is_built_from_the_path_in_the_order_given(self, recording_sdk):
        store = SynapMemoryStore(recording_sdk, user_id="alice", customer_id="acme",
                                 scope_path=PATH)
        assert store.get_owner_id(make_session()) == "/acme/payments/alice"
        swapped = {"user": "alice", "team": "payments", "customer": "acme"}
        store2 = SynapMemoryStore(recording_sdk, user_id="alice", scope_path=swapped)
        assert store2.get_owner_id(make_session()) == "/alice/payments/acme"

    def test_path_is_closed_over_not_a_tool_argument(self, recording_sdk):
        for name in ("search_transcripts", "get_index_text", "write_topic"):
            params = inspect.signature(getattr(SynapMemoryStore, name)).parameters
            assert "scope_path" not in params and "scope" not in params
        store = SynapMemoryStore(recording_sdk, user_id="alice", scope_path=PATH)
        with pytest.raises(TypeError):
            store.search_transcripts(make_session(), source_id="m", query="x",
                                     scope_path={"user": "mallory"})
        assert recording_sdk.calls_to("fetch") == []

    def test_path_is_copied(self, recording_sdk):
        path = dict(PATH)
        store = SynapMemoryStore(recording_sdk, user_id="alice", scope_path=path)
        path["user"] = "mallory"
        store.search_transcripts(make_session(), source_id="memory", query="x")
        assert recording_sdk.last_kwargs("fetch")["scope_path"] == PATH


# --- SynapAgentFileStore -------------------------------------------------------------

@requires_harness
class TestFileStore:
    async def test_default_calls_unchanged(self, recording_sdk):
        fs = SynapAgentFileStore(recording_sdk, user_id="alice", customer_id="acme")
        await fs.write("plan.md", "ship on Tuesday")
        await fs.read("MEMORY.md")
        assert recording_sdk.last_kwargs("memories.create") == {
            "document": "ship on Tuesday", "user_id": "alice", "customer_id": "acme",
            "document_type": "document", "mode": "fast"}
        assert "scope_path" not in recording_sdk.last_kwargs("fetch")

    async def test_write_and_read_carry_path(self, recording_sdk):
        fs = SynapAgentFileStore(recording_sdk, user_id="alice", customer_id="acme",
                                 scope_path=PATH)
        await fs.write("plan.md", "ship on Tuesday")
        create = recording_sdk.last_kwargs("memories.create")
        assert create["scope"] == PATH and "scope_path" not in create
        await fs.read("MEMORY.md")
        fetch = recording_sdk.last_kwargs("fetch")
        assert fetch["scope_path"] == PATH and "scope" not in fetch

    def test_path_is_closed_over_not_a_tool_argument(self):
        for name in ("read", "write", "delete"):
            params = inspect.signature(getattr(SynapAgentFileStore, name)).parameters
            assert "scope_path" not in params and "scope" not in params


# --- harness memory provider and factory ----------------------------------------------

@requires_harness
class TestHarnessMemory:
    def test_factory_hands_the_path_to_store_and_provider(self, recording_sdk):
        provider = create_synap_harness_memory(recording_sdk, user_id="alice",
                                               customer_id="acme", scope_path=PATH)
        assert provider.scope_path == PATH and provider.store.scope_path == PATH

    def test_factory_default_has_no_path(self, recording_sdk):
        provider = create_synap_harness_memory(recording_sdk, user_id="alice",
                                               customer_id="acme")
        assert provider.scope_path is None and provider.store.scope_path is None

    async def test_transcript_turns_carry_path(self, recording_sdk):
        provider = create_synap_harness_memory(recording_sdk, user_id="alice",
                                               customer_id="acme", scope_path=PATH)
        msgs = [Message(role="user", contents=["hi"])]
        await provider.save_messages("s1", msgs, state={})
        assert recording_sdk.last_kwargs("instance.send_message")["scope_path"] == PATH
        sdk = _no_stream()
        p2 = create_synap_harness_memory(sdk, user_id="alice", customer_id="acme",
                                         scope_path=PATH)
        await p2.save_messages("s1", msgs, state={})
        assert sdk.last_kwargs("conversation.record_message")["scope_path"] == PATH

    async def test_transcript_default_unchanged(self, recording_sdk):
        provider = create_synap_harness_memory(recording_sdk, user_id="alice",
                                               customer_id="acme")
        await provider.save_messages("s1", [Message(role="user", contents=["hi"])], state={})
        assert "scope_path" not in recording_sdk.last_kwargs("instance.send_message")
        sdk = _no_stream()
        await create_synap_harness_memory(sdk, user_id="alice", customer_id="acme"
                                          ).save_messages(
            "s1", [Message(role="user", contents=["hi"])], state={})
        assert "scope_path" not in sdk.last_kwargs("conversation.record_message")


async def test_old_sdk_is_fine_without_a_path():
    sdk = RecordingSDK(accept_scope_path=False, listening=False)

    async def refuse():
        raise RuntimeError("no stream")

    sdk.instance.listen = refuse
    p = SynapContextProvider(sdk, user_id="alice", conversation_id="conv-1")
    await p.before_run(agent=None, session=None, context=_ctx(), state={})
    await p.after_run(agent=None, session=None, context=_ctx(), state={})
    assert sdk.calls_to("fetch") and sdk.calls_to("conversation.record_message")


# --- validation at construction -----------------------------------------------------

def _build_context(sdk, sp):
    return SynapContextProvider(sdk, user_id="alice", scope_path=sp)


def _build_history(sdk, sp):
    return SynapHistoryProvider(sdk, user_id="alice", scope_path=sp)


def _build_store(sdk, sp):
    return SynapMemoryStore(sdk, user_id="alice", scope_path=sp)


def _build_file_store(sdk, sp):
    return SynapAgentFileStore(sdk, user_id="alice", scope_path=sp)


def _build_factory(sdk, sp):
    return create_synap_harness_memory(sdk, user_id="alice", scope_path=sp)


def _build_harness_provider(sdk, sp):
    from synap_microsoft_agent import SynapMemoryContextProvider
    return SynapMemoryContextProvider(sdk, store=SynapMemoryStore(sdk, user_id="alice"),
                                      user_id="alice", scope_path=sp)


_ENTRY_POINTS = [_build_context, _build_history, _build_store, _build_file_store,
                 _build_factory, _build_harness_provider]
_BAD_PATHS = [
    ({"customer": "acme", "user": ""}, "non-empty string"),
    ({"customer": "acme", "user": None}, "non-empty string"),
    ({"customer": "acme", "user": 7}, "non-empty string"),
    ({"customer": "acme", "user": "   "}, "non-empty string"),
    ({"customer": "acme", "user": "a/b"}, "must not contain '/'"),
    ({"": "acme"}, "level keys"),
    ({"customer": "acme", 3: "x"}, "level keys"),
    (["customer", "acme"], "map level keys"),
]


@requires_harness
@pytest.mark.parametrize("build", _ENTRY_POINTS)
@pytest.mark.parametrize("bad,message", _BAD_PATHS)
def test_a_bad_scope_path_is_refused_at_construction(recording_sdk, build, bad, message):
    with pytest.raises(ValueError, match=message):
        build(recording_sdk, bad)
    assert recording_sdk.calls == []


@requires_harness
@pytest.mark.parametrize("build", _ENTRY_POINTS)
def test_a_good_or_unset_scope_path_is_accepted(recording_sdk, build):
    build(recording_sdk, PATH)
    build(recording_sdk, None)
    build(recording_sdk, {})


@requires_harness
def test_key_order_changes_the_local_partition_label(recording_sdk):
    """Documented: the label is the local record partition key."""
    a = SynapMemoryStore(recording_sdk, user_id="alice",
                         scope_path={"customer": "acme", "user": "alice"})
    b = SynapMemoryStore(recording_sdk, user_id="alice",
                         scope_path={"user": "alice", "customer": "acme"})
    s = make_session()
    assert a.get_owner_id(s) == "/acme/alice" and b.get_owner_id(s) == "/alice/acme"


# --- ids must match the path (alice's path, bob's call) ------------------------------------

from synap_integrations_common import SynapIntegrationError  # noqa: E402

ALICE = {"customer": "acme", "user": "alice"}


@pytest.mark.parametrize("ids", [
    {"user_id": "bob", "customer_id": "rival"},
    {"user_id": "bob", "customer_id": "acme"},
    {"user_id": "alice", "customer_id": "rival"},
])
async def test_report_message_turn_with_other_ids_is_refused_and_sends_nothing(recording_sdk, ids):
    with pytest.raises(SynapIntegrationError):
        await report_message_turn(recording_sdk, role="user", content="hi",
                                  conversation_id="c", scope_path=ALICE, **ids)
    assert recording_sdk.calls == []


async def test_report_message_turn_refuses_before_any_early_return(recording_sdk):
    """Even a turn that would not be sent (empty content) is refused: the check
    does not depend on what happens next."""
    with pytest.raises(SynapIntegrationError):
        await report_message_turn(recording_sdk, role="user", content="",
                                  conversation_id="c", user_id="bob", customer_id="acme",
                                  scope_path=ALICE)


async def test_report_message_turn_with_matching_ids_still_works(recording_sdk):
    await report_message_turn(recording_sdk, role="user", content="hi", conversation_id="c",
                              user_id="alice", customer_id="acme", scope_path=ALICE)
    assert recording_sdk.last_kwargs("instance.send_message")["scope_path"] == ALICE


async def test_report_message_turn_without_a_path_is_not_checked(recording_sdk):
    await report_message_turn(recording_sdk, role="user", content="hi", conversation_id="c",
                              user_id="bob", customer_id="rival")
    k = recording_sdk.last_kwargs("instance.send_message")
    assert k["user_id"] == "bob" and "scope_path" not in k
