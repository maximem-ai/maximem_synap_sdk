"""``scope_path`` on a deeper scope ladder: reaches every SDK call, and only when set."""

from synap_integrations_common.conversation_id import coerce_conversation_id
from types import SimpleNamespace

import pytest

from synap_integrations_common import set_auto_listen
from synap_integrations_common.testing import RecordingSDK, make_ladder
from synap_smolagents import create_synap_recorder, create_synap_tools, synap_st_instructions

PATH = {"customer": "acme", "team": "payments", "user": "dana"}


def _by_name(tools):
    return {t.name: t for t in tools}


class TestTools:
    def test_default_calls_are_unchanged(self):
        sdk = RecordingSDK()
        tools = _by_name(create_synap_tools(sdk, "dana", customer_id="acme"))
        tools["search_memory"]("q")
        tools["store_memory"]("note")
        assert sdk.last_kwargs("fetch") == {
            "user_id": "dana", "customer_id": "acme", "search_query": ["q"],
            "include_conversation_context": False}
        assert sdk.last_kwargs("memories.create") == {
            "document": "note", "user_id": "dana", "customer_id": "acme"}

    def test_search_memory_carries_the_path(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        _by_name(create_synap_tools(sdk, "dana", customer_id="acme", scope_path=PATH))[
            "search_memory"]("q")
        assert sdk.last_kwargs("fetch")["scope_path"] == PATH

    def test_store_memory_carries_the_path_as_scope(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        _by_name(create_synap_tools(sdk, "dana", customer_id="acme", scope_path=PATH))[
            "store_memory"]("note")
        kw = sdk.last_kwargs("memories.create")
        assert kw["scope"] == PATH and "scope_path" not in kw

    def test_an_empty_path_is_not_sent(self):
        sdk = RecordingSDK()
        _by_name(create_synap_tools(sdk, "dana", scope_path={}))["search_memory"]("q")
        assert "scope_path" not in sdk.last_kwargs("fetch")

    def test_the_path_is_not_in_any_tool_schema(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        tools = _by_name(create_synap_tools(sdk, "dana", scope_path=PATH))
        assert set(tools["search_memory"].inputs) == {"query"}
        assert set(tools["store_memory"].inputs) == {"content"}
        for t in tools.values():
            assert "scope" not in str(t.inputs) and "scope" not in t.description.lower()

    def test_a_model_supplied_scope_path_is_rejected_and_the_developers_is_used(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        tools = _by_name(create_synap_tools(sdk, "dana", scope_path=PATH))
        evil = {"customer": "evil", "user": "mallory"}
        with pytest.raises(TypeError):
            tools["search_memory"](query="q", scope_path=evil)
        with pytest.raises(TypeError):
            tools["store_memory"](content="n", scope=evil)
        assert sdk.calls == []  # rejected before any SDK call
        tools["search_memory"](query="q")
        tools["store_memory"](content="n")
        assert [k["scope_path"] for k in sdk.kwargs_of("fetch")] == [PATH]
        assert [k["scope"] for k in sdk.kwargs_of("memories.create")] == [PATH]


def _no_stream(sdk):
    set_auto_listen(sdk, False)
    return sdk


class TestRecorder:
    def _run(self, sdk, **kw):
        recorder = create_synap_recorder(sdk, "dana", "conv1", customer_id="acme", **kw)
        agent = SimpleNamespace(task="where is my order")
        recorder(SimpleNamespace(model_output="Thought: look", step_number=1, error=None), agent=agent)
        recorder(SimpleNamespace(output="it ships tomorrow"), agent=agent)

    def test_default_rest_turns_are_unchanged(self):
        sdk = _no_stream(RecordingSDK(listening=False))
        self._run(sdk)
        assert sdk.kwargs_of("conversation.record_message") == [
            {"conversation_id": coerce_conversation_id("conv1"), "role": "user", "content": "where is my order",
             "user_id": "dana", "customer_id": "acme"},
            {"conversation_id": coerce_conversation_id("conv1"), "role": "assistant", "content": "it ships tomorrow",
             "user_id": "dana", "customer_id": "acme"},
        ]

    def test_path_reaches_both_turns_over_rest(self):
        sdk = _no_stream(RecordingSDK(listening=False, ladder=make_ladder("deep")))
        self._run(sdk, scope_path=PATH)
        assert [k["scope_path"] for k in sdk.kwargs_of("conversation.record_message")] == [PATH, PATH]

    def test_default_stream_turns_are_unchanged(self):
        sdk = RecordingSDK()
        self._run(sdk)
        for k in sdk.kwargs_of("instance.send_message"):
            assert "scope_path" not in k

    def test_path_reaches_both_turns_on_the_stream(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        self._run(sdk, scope_path=PATH)
        sent = sdk.kwargs_of("instance.send_message")
        assert [k["event_type"] for k in sent] == ["user_message", "assistant_message"]
        assert [k["scope_path"] for k in sent] == [PATH, PATH]
        assert sdk.calls_to("conversation.record_message") == []

    def test_path_reaches_the_step_ingest_as_scope(self):
        sdk = _no_stream(RecordingSDK(listening=False, ladder=make_ladder("deep")))
        self._run(sdk, scope_path=PATH, ingest_steps=True)
        kw = sdk.last_kwargs("memories.create")
        assert kw["scope"] == PATH and "scope_path" not in kw

    def test_default_step_ingest_is_unchanged(self):
        sdk = _no_stream(RecordingSDK(listening=False))
        self._run(sdk, ingest_steps=True)
        assert "scope" not in sdk.last_kwargs("memories.create")

    def test_other_events_never_get_a_path(self):
        # record_tool_call / record_thinking take no scope_path; sending one would be a TypeError
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        recorder = create_synap_recorder(sdk, "dana", "conv1", scope_path=PATH)
        step = SimpleNamespace(
            step_number=1, plan="the plan", model_output="thinking", error=None,
            tool_calls=[SimpleNamespace(name="t", arguments={"a": 1}, id="c1")],
            observations="seen")
        recorder(step, agent=SimpleNamespace(task="x"))
        for name in ("instance.record_thinking", "instance.record_tool_call",
                     "instance.record_tool_result"):
            assert sdk.calls_to(name), name
            for k in sdk.kwargs_of(name):
                assert "scope_path" not in k


class TestConversationOnlyReadsStillWork:
    def test_short_term_instructions_take_no_path(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        create_synap_tools(sdk, "dana", scope_path=PATH)  # a package configured with a path
        out = synap_st_instructions(sdk, "conv1")
        assert out is not None
        call = sdk.calls_to("conversation.context.get_context_for_prompt")[0]
        assert "scope_path" not in call.kwargs


class TestIdsMustAgreeWithThePath:
    BUILD = {
        "tools": lambda sdk, **k: create_synap_tools(sdk, **k),
        "recorder": lambda sdk, **k: create_synap_recorder(sdk, conversation_id="conv-1", **k),
    }

    @pytest.mark.parametrize("name", ["tools", "recorder"])
    def test_refused_accepted_empty_and_no_path(self, name):
        from synap_integrations_common import SynapIntegrationError
        sdk = RecordingSDK()
        build = self.BUILD[name]
        with pytest.raises(SynapIntegrationError):
            build(sdk, user_id="bob", customer_id="acme", scope_path=PATH)
        with pytest.raises(SynapIntegrationError):
            build(sdk, user_id="dana", customer_id="other", scope_path=PATH)
        build(sdk, user_id="dana", customer_id="acme", scope_path=PATH)
        build(sdk, user_id="dana", customer_id="", scope_path=PATH)
        build(sdk, user_id="bob", customer_id="other")
        assert sdk.calls == []
