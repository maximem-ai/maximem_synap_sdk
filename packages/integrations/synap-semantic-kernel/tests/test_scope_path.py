"""``scope_path`` on a deeper scope ladder: reaches every SDK call, and only when set."""

from __future__ import annotations

from synap_integrations_common.conversation_id import coerce_conversation_id
import pytest
from semantic_kernel import Kernel
from semantic_kernel.contents import AuthorRole, ChatMessageContent

from synap_integrations_common import set_auto_listen
from synap_integrations_common.testing import RecordingSDK, make_ladder
from synap_semantic_kernel import SynapAgentThread, SynapPlugin, synap_st_chat_message

PATH = {"customer": "acme", "team": "payments", "user": "dana"}


def _plugin(sdk, **kw):
    return SynapPlugin(sdk=sdk, user_id="dana", customer_id="acme", conversation_id="conv-1", **kw)


class TestPlugin:
    @pytest.mark.asyncio
    async def test_default_calls_are_unchanged(self):
        sdk = RecordingSDK()
        p = _plugin(sdk)
        await p.search_memory(query="q")
        await p.store_memory(content="note")
        assert sdk.last_kwargs("fetch") == {
            "conversation_id": coerce_conversation_id("conv-1"), "user_id": "dana", "customer_id": "acme",
            "search_query": ["q"], "mode": "accurate", "include_conversation_context": False}
        assert sdk.last_kwargs("memories.create") == {
            "document": "note", "user_id": "dana", "customer_id": "acme"}

    @pytest.mark.asyncio
    async def test_search_memory_carries_the_path(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        await _plugin(sdk, scope_path=PATH).search_memory(query="q")
        assert sdk.last_kwargs("fetch")["scope_path"] == PATH

    @pytest.mark.asyncio
    async def test_store_memory_carries_the_path_as_scope(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        await _plugin(sdk, scope_path=PATH).store_memory(content="note")
        kw = sdk.last_kwargs("memories.create")
        assert kw["scope"] == PATH and "scope_path" not in kw

    @pytest.mark.asyncio
    async def test_an_empty_path_is_not_sent(self):
        sdk = RecordingSDK()
        await _plugin(sdk, scope_path={}).search_memory(query="q")
        assert "scope_path" not in sdk.last_kwargs("fetch")

    def test_the_kernel_functions_expose_no_scope_parameter(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        kernel = Kernel()
        kernel.add_plugin(_plugin(sdk, scope_path=PATH), plugin_name="synap")
        params = {
            f.name: [p.name for p in f.metadata.parameters]
            for f in kernel.get_plugin("synap").functions.values()
        }
        assert params == {"search_memory": ["query"], "store_memory": ["content"]}
        for f in kernel.get_plugin("synap").functions.values():
            assert "scope" not in str(f.metadata.parameters).lower().replace("synap", "")

    @pytest.mark.asyncio
    async def test_a_model_supplied_scope_path_is_ignored(self):
        from semantic_kernel.functions import KernelArguments

        sdk = RecordingSDK(ladder=make_ladder("deep"))
        kernel = Kernel()
        kernel.add_plugin(_plugin(sdk, scope_path=PATH), plugin_name="synap")
        fn = kernel.get_function("synap", "search_memory")
        evil = {"customer": "evil", "user": "mallory"}
        await fn.invoke(kernel, KernelArguments(query="q", scope_path=evil))
        # the kernel ignores an argument the function does not declare
        assert [k["scope_path"] for k in sdk.kwargs_of("fetch")] == [PATH]


class TestThread:
    @staticmethod
    def _thread(sdk, **kw):
        return SynapAgentThread(sdk, conversation_id="conv-1", user_id="dana",
                                customer_id="acme", **kw)

    @pytest.mark.asyncio
    async def test_default_turn_on_the_stream_is_unchanged(self):
        sdk = RecordingSDK()
        await self._thread(sdk).on_new_message(
            ChatMessageContent(role=AuthorRole.USER, content="hello"))
        assert sdk.last_kwargs("instance.send_message") == {
            "content": "hello", "role": "user", "event_type": "user_message",
            "conversation_id": "conv-1", "user_id": "dana", "customer_id": "acme", "metadata": {}}

    @pytest.mark.asyncio
    async def test_path_reaches_user_and_assistant_turns_on_the_stream(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        t = self._thread(sdk, scope_path=PATH)
        await t.on_new_message(ChatMessageContent(role=AuthorRole.USER, content="hello"))
        await t.on_new_message(ChatMessageContent(role=AuthorRole.ASSISTANT, content="hi"))
        sent = sdk.kwargs_of("instance.send_message")
        assert [k["event_type"] for k in sent] == ["user_message", "assistant_message"]
        assert [k["scope_path"] for k in sent] == [PATH, PATH]
        assert sdk.calls_to("conversation.record_message") == []

    @pytest.mark.asyncio
    async def test_path_reaches_the_rest_fallback(self):
        sdk = RecordingSDK(listening=False, ladder=make_ladder("deep"))
        set_auto_listen(sdk, False)
        t = self._thread(sdk, scope_path=PATH)
        await t.on_new_message(ChatMessageContent(role=AuthorRole.USER, content="hello"))
        await t.on_new_message(ChatMessageContent(role=AuthorRole.ASSISTANT, content="hi"))
        assert [k["scope_path"] for k in sdk.kwargs_of("conversation.record_message")] == [PATH, PATH]


class TestConversationOnlyReadsStillWork:
    @pytest.mark.asyncio
    async def test_short_term_message_takes_no_path(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        msg = await synap_st_chat_message(sdk, "conv-1")()
        assert msg is not None
        call = sdk.calls_to("conversation.context.get_context_for_prompt")[0]
        assert "scope_path" not in call.kwargs

    @pytest.mark.asyncio
    async def test_thread_history_reads_through_the_conversation(self):
        sdk = RecordingSDK(ladder=make_ladder("deep"))
        t = SynapAgentThread(sdk, conversation_id="conv-1", user_id="dana",
                             customer_id="acme", scope_path=PATH)
        [m async for m in t.get_messages()]
        for c in sdk.calls_to("conversation.context.get_context_for_prompt"):
            assert "scope_path" not in c.kwargs


class TestIdsMustAgreeWithThePath:
    BUILD = {
        "plugin": lambda sdk, **k: SynapPlugin(sdk=sdk, **k),
        "thread": lambda sdk, **k: SynapAgentThread(sdk, conversation_id="conv-1", **k),
    }

    @pytest.mark.parametrize("name", ["plugin", "thread"])
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
