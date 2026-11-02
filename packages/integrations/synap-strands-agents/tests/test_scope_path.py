"""``scope_path`` on a deeper scope ladder: reaches every SDK call, and only when set."""

from synap_integrations_common.conversation_id import coerce_conversation_id
from unittest.mock import MagicMock

import pytest
from strands.hooks import BeforeInvocationEvent, MessageAddedEvent

from synap_integrations_common import set_auto_listen
from synap_integrations_common.testing import RecordingSDK, make_ladder
from synap_strands_agents import (
    SynapMemoryStore, SynapShortTermHook, SynapStreamHook, create_synap_tools,
)
from synap_strands_agents import _util

PATH = {"customer": "acme", "team": "payments", "user": "dana"}


@pytest.fixture(autouse=True)
def _clean_ledger():
    _util._written.clear()
    _util._hook_active = False
    _util._warned_unshared = False
    yield
    _util._written.clear()
    _util._hook_active = False
    _util._warned_unshared = False


def _by_name(tools):
    return {getattr(t, "tool_name", None) or t.__name__: t for t in tools}


class TestStore:
    async def test_default_calls_are_unchanged(self):
        sdk = RecordingSDK()
        store = SynapMemoryStore(sdk, "dana", customer_id="acme", conversation_id="conv-1")
        await store.search("q")
        await store.add("note")
        assert sdk.last_kwargs("fetch") == {
            "conversation_id": coerce_conversation_id("conv-1"), "user_id": "dana", "customer_id": "acme",
            "search_query": ["q"], "mode": "accurate", "max_results": 5,
            "include_conversation_context": False}
        assert sdk.last_kwargs("memories.create") == {
            "document": "note", "user_id": "dana", "customer_id": "acme", "metadata": None}

    async def test_search_carries_the_path(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        await SynapMemoryStore(sdk, "dana", scope_path=PATH).search("q")
        assert sdk.last_kwargs("fetch")["scope_path"] == PATH

    async def test_add_carries_the_path_as_scope(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        await SynapMemoryStore(sdk, "dana", scope_path=PATH).add("note")
        kw = sdk.last_kwargs("memories.create")
        assert kw["scope"] == PATH and "scope_path" not in kw

    async def test_add_messages_carries_the_path_on_the_stream(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        store = SynapMemoryStore(sdk, "dana", scope_path=PATH, conversation_id="conv-1")
        await store.add_messages([{"role": "user", "content": [{"text": "hi"}]},
                                  {"role": "assistant", "content": [{"text": "hello"}]}])
        sent = sdk.kwargs_of("instance.send_message")
        assert [k["scope_path"] for k in sent] == [PATH, PATH]

    async def test_add_messages_default_is_unchanged_and_rest_carries_the_path(self):
        sdk = RecordingSDK()
        await SynapMemoryStore(sdk, "dana", conversation_id="conv-1").add_messages(
            [{"role": "user", "content": [{"text": "hi"}]}])
        assert "scope_path" not in sdk.last_kwargs("instance.send_message")

        sdk = RecordingSDK(listening=False, ladder=make_ladder("deep"))
        set_auto_listen(sdk, False)
        await SynapMemoryStore(sdk, "dana", scope_path=PATH, conversation_id="conv-1").add_messages(
            [{"role": "user", "content": [{"text": "hi"}]}])
        assert sdk.last_kwargs("conversation.record_message")["scope_path"] == PATH


class TestTools:
    async def test_default_calls_are_unchanged(self):
        sdk = RecordingSDK()
        tools = _by_name(create_synap_tools(sdk, "dana", customer_id="acme", conversation_id="conv-1"))
        await tools["search_memory"]("q")
        await tools["store_memory"]("note")
        assert sdk.last_kwargs("fetch") == {
            "conversation_id": coerce_conversation_id("conv-1"), "user_id": "dana", "customer_id": "acme",
            "search_query": ["q"], "mode": "accurate", "include_conversation_context": False}
        assert sdk.last_kwargs("memories.create") == {
            "document": "note", "user_id": "dana", "customer_id": "acme"}

    async def test_both_tools_carry_the_path(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        tools = _by_name(create_synap_tools(sdk, "dana", scope_path=PATH))
        await tools["search_memory"]("q")
        await tools["store_memory"]("note")
        assert sdk.last_kwargs("fetch")["scope_path"] == PATH
        kw = sdk.last_kwargs("memories.create")
        assert kw["scope"] == PATH and "scope_path" not in kw

    def test_the_path_is_not_in_any_tool_schema(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        tools = create_synap_tools(sdk, "dana", scope_path=PATH)
        for t in tools:
            props = set(t.tool_spec["inputSchema"]["json"]["properties"])
            assert props <= {"query", "content"}, props
            assert "scope" not in str(t.tool_spec).lower()

    async def test_a_model_supplied_scope_path_is_ignored_or_rejected(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        tools = _by_name(create_synap_tools(sdk, "dana", scope_path=PATH))
        evil = {"customer": "evil", "user": "mallory"}
        for call in (lambda: tools["search_memory"]("q", scope_path=evil),
                     lambda: tools["store_memory"]("n", scope=evil)):
            with pytest.raises(TypeError):
                await call()
        assert sdk.calls == []  # rejected before any SDK call
        await tools["search_memory"]("q")
        await tools["store_memory"]("n")
        assert [k["scope_path"] for k in sdk.kwargs_of("fetch")] == [PATH]
        assert [k["scope"] for k in sdk.kwargs_of("memories.create")] == [PATH]


def _msg(role, text):
    return MessageAddedEvent(agent=MagicMock(), message={"role": role, "content": [{"text": text}]})


class TestStreamHook:
    async def test_default_turn_is_unchanged(self):
        sdk = RecordingSDK()
        hook = SynapStreamHook(sdk, "conv-1", "dana", customer_id="acme")
        await hook._on_message(_msg("user", "hello"))
        assert sdk.last_kwargs("instance.send_message") == {
            "content": "hello", "role": "user", "event_type": "user_message",
            "conversation_id": "conv-1", "user_id": "dana", "customer_id": "acme", "metadata": {}}

    async def test_path_reaches_both_turns_on_the_stream(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        hook = SynapStreamHook(sdk, "conv-1", "dana", scope_path=PATH)
        await hook._on_message(_msg("user", "hello"))
        await hook._on_message(_msg("assistant", "hi"))
        assert [k["scope_path"] for k in sdk.kwargs_of("instance.send_message")] == [PATH, PATH]

    async def test_path_reaches_the_rest_fallback(self):
        sdk = RecordingSDK(listening=False, ladder=make_ladder("deep"))
        set_auto_listen(sdk, False)
        hook = SynapStreamHook(sdk, "conv-1", "dana", scope_path=PATH)
        await hook._on_message(_msg("user", "hello"))
        assert sdk.last_kwargs("conversation.record_message")["scope_path"] == PATH


class TestConversationOnlyReadsStillWork:
    async def test_short_term_hook_takes_no_path(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        hook = SynapShortTermHook(sdk, conversation_id="conv-1")
        agent = MagicMock()
        agent.messages = []
        await hook._inject(BeforeInvocationEvent(
            agent=agent, messages=[{"role": "user", "content": [{"text": "hi"}]}]))
        calls = sdk.calls_to("conversation.context.get_context_for_prompt")
        assert len(calls) == 1 and "scope_path" not in calls[0].kwargs


class TestTheWriteLedgerKeepsWritesApart:
    """The ledger that stops a turn being written twice (hook, then store) must
    only match a write to the SAME conversation at the SAME path."""

    P1 = {"customer": "acme", "team": "payments", "user": "dana"}
    P2 = {"customer": "acme", "team": "support", "user": "dana"}

    @staticmethod
    def _turn(text):
        return [{"role": "assistant", "content": [{"text": text}]}]

    async def test_same_text_at_two_paths_is_written_twice(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        hook = SynapStreamHook(sdk, "conv-1", "dana", scope_path=self.P1)
        await hook._on_message(_msg("assistant", "Sure, done."))
        store = SynapMemoryStore(sdk, "dana", scope_path=self.P2, conversation_id="conv-1")
        await store.add_messages(self._turn("Sure, done."))
        sent = sdk.kwargs_of("instance.send_message")
        assert [k["scope_path"] for k in sent] == [self.P1, self.P2]

    async def test_same_text_in_two_conversations_is_written_twice(self):
        sdk = RecordingSDK()
        hook = SynapStreamHook(sdk, "conv-1", "dana")
        await hook._on_message(_msg("assistant", "ok"))
        store = SynapMemoryStore(sdk, "dana", conversation_id="conv-2")
        await store.add_messages(self._turn("ok"))
        assert [k["conversation_id"] for k in sdk.kwargs_of("instance.send_message")] == [
            "conv-1", "conv-2"]

    async def test_a_genuine_duplicate_is_still_written_once(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        # the key order of the path does not matter
        hook = SynapStreamHook(sdk, "conv-1", "dana", scope_path=self.P1)
        await hook._on_message(_msg("assistant", "Sure, done."))
        store = SynapMemoryStore(sdk, "dana", conversation_id="conv-1",
                                 scope_path=dict(reversed(list(self.P1.items()))))
        await store.add_messages(self._turn("Sure, done."))
        assert len(sdk.calls_to("instance.send_message")) == 1

    async def test_a_genuine_duplicate_without_a_path_is_still_written_once(self):
        sdk = RecordingSDK()
        hook = SynapStreamHook(sdk, "conv-1", "dana")
        await hook._on_message(_msg("assistant", "ok"))
        await SynapMemoryStore(sdk, "dana", conversation_id="conv-1").add_messages(self._turn("ok"))
        assert len(sdk.calls_to("instance.send_message")) == 1


class TestTheLedgerCountsAndTheRetryKeepsItsKey:
    P = {"customer": "acme", "team": "payments", "user": "dana"}

    @staticmethod
    def _turns(*texts):
        return [{"role": "user", "content": [{"text": t}]} for t in texts]

    async def test_two_identical_turns_are_two_writes_not_one(self):
        """Counted, not a set: the hook wrote "ok" twice, so the store skips two."""
        sdk = RecordingSDK()
        hook = SynapStreamHook(sdk, "conv-1", "dana")
        await hook._on_message(_msg("user", "ok"))
        await hook._on_message(_msg("user", "ok"))
        assert len(sdk.calls_to("instance.send_message")) == 2
        store = SynapMemoryStore(sdk, "dana", conversation_id="conv-1")
        await store.add_messages(self._turns("ok", "ok"))
        assert len(sdk.calls_to("instance.send_message")) == 2  # both skipped, neither re-sent
        await store.add_messages(self._turns("ok"))  # a third "ok" is genuinely new
        assert len(sdk.calls_to("instance.send_message")) == 3

    async def test_a_retry_after_a_failed_batch_skips_what_went_through_at_the_same_path(self):
        sdk = RecordingSDK()
        store = SynapMemoryStore(sdk, "dana", conversation_id="conv-1", scope_path=self.P)
        batch = self._turns("one", "two")
        # make the second record fail once, via the store's own record_turn
        from synap_strands_agents import store as store_mod
        calls = {"n": 0}
        orig = store_mod.record_turn

        async def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 2:
                return False
            return await orig(*a, **k)

        store_mod.record_turn = flaky
        try:
            with pytest.raises(Exception):
                await store.add_messages(batch)
        finally:
            store_mod.record_turn = orig
        assert [k["content"] for k in sdk.kwargs_of("instance.send_message")] == ["one"]
        await store.add_messages(batch)
        sent = sdk.kwargs_of("instance.send_message")
        assert [k["content"] for k in sent] == ["one", "two"]  # "one" was not sent again
        assert all(k["scope_path"] == self.P for k in sent)


class TestOneTimeWarningForAnUnsharedConversation:
    async def test_warns_once_when_a_hook_is_active_and_the_store_has_no_conversation_id(self, caplog):
        sdk = RecordingSDK()
        SynapStreamHook(sdk, "conv-1", "dana")
        with caplog.at_level("WARNING", logger="synap_strands_agents._util"):
            store = SynapMemoryStore(sdk, "dana")
            SynapMemoryStore(sdk, "dana")
            await store.add_messages([{"role": "user", "content": [{"text": "hi"}]}])
        hits = [r for r in caplog.records if "conversation_id" in r.getMessage()]
        assert len(hits) == 1

    async def test_warns_when_the_hook_is_built_after_the_store(self, caplog):
        sdk = RecordingSDK()
        store = SynapMemoryStore(sdk, "dana")
        SynapStreamHook(sdk, "conv-1", "dana")
        with caplog.at_level("WARNING", logger="synap_strands_agents._util"):
            await store.add_messages([{"role": "user", "content": [{"text": "hi"}]}])
        assert len([r for r in caplog.records if "conversation_id" in r.getMessage()]) == 1

    async def test_no_warning_with_a_shared_id_or_without_a_hook(self, caplog):
        sdk = RecordingSDK()
        with caplog.at_level("WARNING", logger="synap_strands_agents._util"):
            SynapMemoryStore(sdk, "dana")  # no hook in the process
            SynapStreamHook(sdk, "conv-1", "dana")
            SynapMemoryStore(sdk, "dana", conversation_id="conv-1")
        assert not [r for r in caplog.records if "conversation_id" in r.getMessage()]


class TestIdsMustAgreeWithThePath:
    """The server lets the path's user/customer win over flat ids; refuse a disagreement, no SDK call."""

    BUILD = {
        "store": lambda sdk, **k: SynapMemoryStore(sdk, **k),
        "hook": lambda sdk, **k: SynapStreamHook(sdk, "conv-1", **k),
        "tools": lambda sdk, **k: create_synap_tools(sdk, **k),
    }

    @pytest.mark.parametrize("name", ["store", "hook", "tools"])
    def test_refused_accepted_empty_and_no_path(self, name):
        from synap_integrations_common import SynapIntegrationError
        sdk = RecordingSDK()
        build = self.BUILD[name]
        with pytest.raises(SynapIntegrationError):
            build(sdk, user_id="bob", customer_id="acme", scope_path=PATH)
        with pytest.raises(SynapIntegrationError):
            build(sdk, user_id="dana", customer_id="other", scope_path=PATH)
        build(sdk, user_id="dana", customer_id="acme", scope_path=PATH)   # matching
        build(sdk, user_id="dana", customer_id="", scope_path=PATH)       # empty customer
        build(sdk, user_id="bob", customer_id="other")                     # no path: unchanged
        assert sdk.calls == []


class TestTheConstructorWarns:
    def test_a_store_built_after_a_hook_warns_at_construction_not_only_on_first_write(self, caplog):
        sdk = RecordingSDK()
        SynapStreamHook(sdk, "conv-1", "dana")
        with caplog.at_level("WARNING", logger="synap_strands_agents._util"):
            SynapMemoryStore(sdk, "dana")
        assert len([r for r in caplog.records if "conversation_id" in r.getMessage()]) == 1
