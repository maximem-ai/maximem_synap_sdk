"""`scope_path` (a deeper scope ladder's level) through synap-llamaindex.

Every SDK call is recorded with its exact keyword arguments (`RecordingSDK`).
The rule: the keyword is sent ONLY when a path was given, so a call made without
one is exactly what it was before.
"""

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import QueryBundle

from synap_integrations_common.testing import RecordingSDK, make_ladder, recording_sdk  # noqa: F401
from synap_llamaindex import SynapChatMemory, SynapRetriever, synap_st_chat_message

PATH = {"customer": "acme", "team": "payments", "user": "dana"}


# --- SynapRetriever: sdk.fetch -------------------------------------------------

async def test_retriever_default_fetch_kwargs_unchanged(recording_sdk):
    r = SynapRetriever(recording_sdk, user_id="dana", customer_id="acme")
    await r._aretrieve(QueryBundle("q"))
    assert recording_sdk.last_kwargs("fetch") == {
        "conversation_id": None, "user_id": "dana", "customer_id": "acme",
        "search_query": ["q"], "max_results": 20, "types": None, "mode": "accurate",
        "include_conversation_context": False,
    }


async def test_retriever_forwards_scope_path_to_fetch(recording_sdk):
    r = SynapRetriever(recording_sdk, user_id="dana", customer_id="acme", scope_path=PATH)
    await r._aretrieve(QueryBundle("q"))
    kw = recording_sdk.last_kwargs("fetch")
    assert kw["scope_path"] == PATH
    assert kw["user_id"] == "dana" and kw["customer_id"] == "acme"


async def test_retriever_empty_scope_path_is_not_sent(recording_sdk):
    r = SynapRetriever(recording_sdk, user_id="dana", scope_path={})
    await r._aretrieve(QueryBundle("q"))
    assert "scope_path" not in recording_sdk.last_kwargs("fetch")


async def test_retriever_copies_the_path(recording_sdk):
    path = dict(PATH)
    r = SynapRetriever(recording_sdk, user_id="dana", scope_path=path)
    path["user"] = "mallory"
    await r._aretrieve(QueryBundle("q"))
    assert recording_sdk.last_kwargs("fetch")["scope_path"] == PATH


# --- SynapChatMemory: fetch (aget) + turn writes (aput / aset) -------------------

def _memory(sdk, **kw):
    return SynapChatMemory(sdk, conversation_id="conv-1", user_id="dana",
                           customer_id="acme", **kw)


async def test_memory_default_calls_unchanged(recording_sdk):
    m = _memory(recording_sdk)
    await m.aget("what do I like?")
    assert recording_sdk.last_kwargs("fetch") == {
        "conversation_id": "conv-1", "user_id": "dana", "customer_id": "acme",
        "search_query": ["what do I like?"], "include_conversation_context": False,
    }
    await m.aput(ChatMessage(role=MessageRole.USER, content="hello"))
    assert recording_sdk.last_kwargs("instance.send_message") == {
        "content": "hello", "role": "user", "event_type": "user_message",
        "conversation_id": "conv-1", "user_id": "dana", "customer_id": "acme",
        "metadata": {},
    }


async def test_memory_aget_forwards_scope_path_to_fetch(recording_sdk):
    m = _memory(recording_sdk, scope_path=PATH)
    await m.aget("what do I like?")
    assert recording_sdk.last_kwargs("fetch")["scope_path"] == PATH


async def test_memory_aput_stream_turn_carries_scope_path(recording_sdk):
    m = _memory(recording_sdk, scope_path=PATH)
    await m.aput(ChatMessage(role=MessageRole.ASSISTANT, content="hi"))
    assert recording_sdk.last_kwargs("instance.send_message")["scope_path"] == PATH
    assert recording_sdk.calls_to("conversation.record_message") == []


async def test_memory_aput_rest_fallback_turn_carries_scope_path():
    sdk = RecordingSDK(listening=False)
    sdk.instance.listen = _refuse  # the stream will not open: REST is the fallback
    m = _memory(sdk, scope_path=PATH)
    await m.aput(ChatMessage(role=MessageRole.USER, content="hello"))
    kw = sdk.last_kwargs("conversation.record_message")
    assert kw["scope_path"] == PATH and kw["conversation_id"] == "conv-1"


async def test_memory_rest_fallback_default_has_no_scope_path():
    sdk = RecordingSDK(listening=False)
    sdk.instance.listen = _refuse
    m = _memory(sdk)
    await m.aput(ChatMessage(role=MessageRole.USER, content="hello"))
    assert "scope_path" not in sdk.last_kwargs("conversation.record_message")


async def test_memory_aset_records_with_scope_path(recording_sdk):
    m = _memory(recording_sdk, scope_path=PATH)
    await m.aset([ChatMessage(role=MessageRole.USER, content="from history")])
    assert recording_sdk.last_kwargs("instance.send_message")["scope_path"] == PATH


async def test_from_defaults_accepts_scope_path(recording_sdk):
    m = SynapChatMemory.from_defaults(sdk=recording_sdk, conversation_id="conv-1",
                                      user_id="dana", scope_path=PATH)
    await m.aget("q")
    assert recording_sdk.last_kwargs("fetch")["scope_path"] == PATH


async def _refuse():
    raise RuntimeError("stream will not open")


# --- conversation-only reads still work with a configured path ------------------

async def test_conversation_only_reads_take_no_path(recording_sdk):
    m = _memory(recording_sdk, scope_path=PATH)
    await m.aget()  # no input: no fetch, only the conversation read
    ctx_calls = recording_sdk.kwargs_of("conversation.context.get_context_for_prompt")
    assert ctx_calls and all("scope_path" not in k and "scope" not in k for k in ctx_calls)
    assert recording_sdk.calls_to("fetch") == []


async def test_synap_st_chat_message_still_works(recording_sdk):
    msg = await synap_st_chat_message(recording_sdk, "conv-1", system="Be brief.")()
    assert msg is not None and "Be brief." in msg.content
    ctx_calls = recording_sdk.kwargs_of("conversation.context.get_context_for_prompt")
    assert ctx_calls and all("scope_path" not in k for k in ctx_calls)


# --- an SDK that cannot take the keyword is never sent it unset ------------------

async def test_old_sdk_is_fine_without_a_path():
    sdk = RecordingSDK(accept_scope_path=False)
    await SynapRetriever(sdk, user_id="dana")._aretrieve(QueryBundle("q"))
    await _memory(sdk).aget("q")
    await _memory(sdk).aput(ChatMessage(role=MessageRole.USER, content="hello"))
    assert len(sdk.calls_to("fetch")) == 2
