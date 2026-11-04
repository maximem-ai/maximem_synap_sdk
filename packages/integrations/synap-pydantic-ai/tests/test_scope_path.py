"""``scope_path`` on a deeper scope ladder: reaches every SDK call, and only when set."""

from __future__ import annotations

from synap_integrations_common.conversation_id import coerce_conversation_id
import json

import pytest

from synap_integrations_common.testing import RecordingSDK, make_ladder
from synap_pydantic_ai import SynapDeps, register_synap_tools, synap_run
from synap_pydantic_ai import register_synap_st_system_prompt

from tests.test_deps import _FakeAgent, _FakeRunContext

PATH = {"customer": "acme", "team": "payments", "user": "dana"}


def _registered(sdk, **deps_kw):
    agent = _FakeAgent()
    register_synap_tools(agent)
    search, store = agent.tools[0], agent.tools[1]
    inject = agent.system_prompt_callbacks[0]
    return search, store, inject, _FakeRunContext(
        SynapDeps(sdk=sdk, user_id="dana", customer_id="acme", conversation_id="conv-1", **deps_kw))


class TestTools:
    async def test_default_calls_are_unchanged(self):
        sdk = RecordingSDK()
        search, store, inject, ctx = _registered(sdk)
        await search(ctx, "q")
        await store(ctx, "note")
        await inject(ctx)
        fetches = sdk.kwargs_of("fetch")
        assert fetches[0] == {"conversation_id": coerce_conversation_id("conv-1"), "user_id": "dana", "customer_id": "acme",
                              "search_query": ["q"], "mode": "accurate",
                              "include_conversation_context": False}
        assert fetches[1] == {"conversation_id": coerce_conversation_id("conv-1"), "user_id": "dana", "customer_id": "acme",
                              "include_conversation_context": False}
        assert sdk.last_kwargs("memories.create") == {
            "document": "note", "user_id": "dana", "customer_id": "acme"}

    async def test_search_memory_carries_the_path(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        search, _, _, ctx = _registered(sdk, scope_path=PATH)
        await search(ctx, "q")
        assert sdk.last_kwargs("fetch")["scope_path"] == PATH

    async def test_store_memory_carries_the_path_as_scope(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        _, store, _, ctx = _registered(sdk, scope_path=PATH)
        await store(ctx, "note")
        kw = sdk.last_kwargs("memories.create")
        assert kw["scope"] == PATH and "scope_path" not in kw

    async def test_the_system_prompt_fetch_carries_the_path(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        _, _, inject, ctx = _registered(sdk, scope_path=PATH)
        await inject(ctx)
        assert sdk.last_kwargs("fetch")["scope_path"] == PATH

    def test_an_empty_path_is_normalised_to_none(self):
        assert SynapDeps(sdk=RecordingSDK(), user_id="u", scope_path={}).scope_path is None

    async def test_a_model_supplied_scope_path_is_not_a_parameter(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        search, store, _, ctx = _registered(sdk, scope_path=PATH)
        with pytest.raises(TypeError):
            await search(ctx, "q", scope_path={"customer": "evil", "user": "mallory"})
        with pytest.raises(TypeError):
            await store(ctx, "n", scope={"customer": "evil", "user": "mallory"})
        assert sdk.calls == []


class TestToolSchemaHasNoScopePath:
    async def test_the_model_is_offered_only_query_and_content(self):
        """What the model is actually offered, read through the public
        ``AgentInfo.function_tools`` of a FunctionModel."""
        pydantic_ai = pytest.importorskip("pydantic_ai")
        from pydantic_ai.messages import ModelResponse, TextPart
        from pydantic_ai.models.function import FunctionModel

        offered = {}

        async def respond(messages, info):
            for t in info.function_tools:
                offered[t.name] = t.parameters_json_schema
            return ModelResponse(parts=[TextPart("done")])

        agent = pydantic_ai.Agent(FunctionModel(respond), deps_type=SynapDeps)
        register_synap_tools(agent)
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        await agent.run("hi", deps=SynapDeps(sdk=sdk, user_id="dana", customer_id="acme",
                                              scope_path=PATH))
        assert set(offered) == {"search_memory", "store_memory"}
        assert set(offered["search_memory"]["properties"]) == {"query"}
        assert set(offered["store_memory"]["properties"]) == {"content"}
        assert "scope" not in json.dumps(offered).lower()

    async def test_a_tool_call_the_model_makes_uses_the_developers_path(self):
        pydantic_ai = pytest.importorskip("pydantic_ai")
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
        from pydantic_ai.models.function import FunctionModel

        step = {"n": 0}
        evil = {"customer": "evil", "user": "mallory"}

        async def respond(messages, info):
            step["n"] += 1
            if step["n"] == 1:  # the model tries to pick its own level
                return ModelResponse(parts=[
                    ToolCallPart("search_memory", {"query": "q", "scope_path": evil}),
                    ToolCallPart("store_memory", {"content": "n", "scope": evil})])
            return ModelResponse(parts=[TextPart("done")])

        agent = pydantic_ai.Agent(FunctionModel(respond), deps_type=SynapDeps)
        register_synap_tools(agent)
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        await agent.run("hi", deps=SynapDeps(sdk=sdk, user_id="dana", customer_id="acme",
                                              scope_path=PATH))
        # the framework refuses the call with unknown arguments (it asks the model to retry);
        # whatever happened, nothing the model supplied reached the SDK
        for k in sdk.kwargs_of("fetch"):
            assert k.get("scope_path") == PATH
        for k in sdk.kwargs_of("memories.create"):
            assert k["scope"] == PATH


class _Agent:
    async def run(self, user_prompt=None, *, deps=None, **kw):
        class R:  # a result with the output text
            output = "the answer"
        return R()


class TestTurns:
    async def test_default_turns_unchanged(self):
        sdk = RecordingSDK()
        deps = SynapDeps(sdk=sdk, user_id="u1", customer_id="c1", conversation_id="conv-1")
        await synap_run(_Agent(), "hello", deps=deps)
        for k in sdk.kwargs_of("instance.send_message"):
            assert "scope_path" not in k

    async def test_path_reaches_both_turns_on_the_stream(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        deps = SynapDeps(sdk=sdk, user_id="dana", customer_id="acme",
                         conversation_id="conv-1", scope_path=PATH)
        await synap_run(_Agent(), "hello", deps=deps)
        sent = sdk.kwargs_of("instance.send_message")
        assert [k["event_type"] for k in sent] == ["user_message", "assistant_message"]
        assert [k["scope_path"] for k in sent] == [PATH, PATH]

    async def test_path_reaches_both_turns_over_rest(self):
        from synap_integrations_common import set_auto_listen

        sdk = RecordingSDK(listening=False, ladder=make_ladder("deep"))
        set_auto_listen(sdk, False)
        deps = SynapDeps(sdk=sdk, user_id="dana", customer_id="acme",
                         conversation_id="conv-1", scope_path=PATH)
        await synap_run(_Agent(), "hello", deps=deps)
        assert [k["scope_path"] for k in sdk.kwargs_of("conversation.record_message")] == [PATH, PATH]


class TestConversationOnlyReadsStillWork:
    async def test_short_term_prompt_takes_no_path_and_still_reads(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        agent = _FakeAgent()
        register_synap_st_system_prompt(agent, system="be kind")
        deps = SynapDeps(sdk=sdk, user_id="dana", customer_id="acme",
                         conversation_id="conv-1", scope_path=PATH)
        ctx = _FakeRunContext(deps)
        ctx.messages = []
        await agent.system_prompt_callbacks[0](ctx)
        calls = sdk.calls_to("conversation.context.get_context_for_prompt")
        assert len(calls) == 1 and "scope_path" not in calls[0].kwargs


class TestIdsMustAgreeWithThePath:
    """The server lets the path's user/customer win over flat ids: a disagreement is
    refused before any SDK call (deps are per run, so ids can differ call to call)."""

    def test_construction_refused_accepted_empty_and_no_path(self):
        from synap_integrations_common import SynapIntegrationError
        sdk = RecordingSDK()
        with pytest.raises(SynapIntegrationError):
            SynapDeps(sdk=sdk, user_id="bob", customer_id="acme", scope_path=PATH)
        with pytest.raises(SynapIntegrationError):
            SynapDeps(sdk=sdk, user_id="dana", customer_id="other", scope_path=PATH)
        SynapDeps(sdk=sdk, user_id="dana", customer_id="acme", scope_path=PATH)   # matching
        SynapDeps(sdk=sdk, user_id="dana", customer_id="", scope_path=PATH)       # empty customer
        SynapDeps(sdk=sdk, user_id="bob", customer_id="other")                     # no path
        assert sdk.calls == []

    async def test_reassigned_ids_are_refused_before_each_call(self):
        from synap_integrations_common import SynapIntegrationError
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        search, store, inject, ctx = _registered(sdk, scope_path=PATH)
        ctx.deps.user_id = "bob"
        with pytest.raises(SynapIntegrationError):
            await search(ctx, "q")
        with pytest.raises(SynapIntegrationError):
            await store(ctx, "n")
        assert await inject(ctx) == ""   # a prompt callback never raises; it logs and skips
        assert sdk.calls == []

    async def test_the_prompt_callback_logs_at_error_with_only_fingerprints(self, caplog):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        _, _, inject, ctx = _registered(sdk, scope_path=PATH)
        ctx.deps.user_id = "bob-secret-id"
        with caplog.at_level("ERROR"):
            await inject(ctx)
        assert any(r.levelname == "ERROR" for r in caplog.records)
        assert "bob-secret-id" not in caplog.text

    async def test_synap_run_refuses_before_reporting_or_running(self):
        from synap_integrations_common import SynapIntegrationError
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        deps = SynapDeps(sdk=sdk, user_id="dana", customer_id="acme", scope_path=PATH,
                         conversation_id="conv-1")
        deps.user_id = "bob"
        ran = []

        class A:
            async def run(self, *a, **k):
                ran.append(1)

        with pytest.raises(SynapIntegrationError):
            await synap_run(A(), "hi", deps=deps)
        assert sdk.calls == [] and not ran
