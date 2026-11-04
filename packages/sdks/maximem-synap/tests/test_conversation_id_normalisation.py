"""A conversation id that is not a UUID and the UUID the server derives from it
are ONE conversation, for every local cache.

The stream and the server accept any string as a conversation id and turn one
that is not a UUID into ``uuid5(NAMESPACE_URL, id)``. The REST reads need that
UUID, so a caller sends a turn on the stream under ``pgv2_abc`` and reads under
the UUID it derived. The SDK dropped its cached prompt context and mirrored the
turn into the short-term store under the RAW id and read both under the
DERIVED one: the next read missed the turn just said, for the five minutes the
cached answer lives (found live, 2026-11-02).
"""
from __future__ import annotations

import uuid

import pytest

from maximem_synap import MaximemSynapSDK
from maximem_synap.cache.manager import CacheManager
from maximem_synap.cache.short_term_store import ShortTermContextStore
from maximem_synap.cache.turn_store import TurnStore
from maximem_synap.models.enums import ContextScope
from maximem_synap.utils.validators import validate_conversation_id

RAW = "pgv2_testconv72"
DERIVED = str(uuid.uuid5(uuid.NAMESPACE_URL, RAW))


def _server_coerce(value):
    """The server's rule, written out so the model does not use the code under test."""
    try:
        uuid.UUID(value)
        return value
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


class ServerModel:
    """What the real server does: the stream takes any id, REST needs a UUID."""

    def __init__(self):
        self.messages = {}  # derived uuid -> [{"role","content"}]

    def written(self, conversation_id, role, content):
        key = _server_coerce(conversation_id)
        rows = self.messages.setdefault(key, [])
        rows.append({"role": role, "content": content, "message_id": f"m{len(rows)}"})


class FakeStreamTransport:
    def __init__(self, server):
        self._server = server
        self.sent = []
        self.controls = []

    async def send(self, payload):
        self.sent.append(payload)
        if payload["event_type"] in ("user_message", "assistant_message"):
            self._server.written(payload["conversation_id"], payload["role"], payload["content"])

    async def send_session_control(self, **kwargs):
        self.controls.append(kwargs)
        return True


class FakeController:
    def __init__(self, server):
        self._transport = FakeStreamTransport(server)
        self.is_listening = True


class FakeRest:
    """REST reads: refuse anything that is not a UUID, as the server does."""

    def __init__(self, server):
        self._server = server
        self.reads = 0

    async def get(self, path, **kw):
        self.reads += 1
        conv = path.split("/v1/conversations/")[1].split("/")[0]
        uuid.UUID(conv)  # the REST read needs a UUID
        msgs = self._server.messages.get(conv, [])
        return {
            "available": False,
            "recent_messages": [
                {"role": m["role"], "content": m["content"], "timestamp": "2026-11-02T10:00:00+00:00",
                 "message_id": m["message_id"]}
                for m in msgs
            ],
            "recent_message_count": len(msgs),
            "total_message_count": len(msgs),
            "compacted_message_count": 0,
        }


class _NoAuth:
    async def get_auth_context(self, *_):
        return None


class _NoTelemetry:
    def __getattr__(self, _):
        return lambda *a, **k: None


def make_sdk(tmp_path):
    server = ServerModel()
    sdk = MaximemSynapSDK(api_key="synap_test", _force_new=True)
    sdk._http_transport = FakeRest(server)
    sdk._credential_manager = _NoAuth()
    sdk._telemetry_collector = _NoTelemetry()
    sdk._cache_manager = CacheManager(client_id="c1", instance_id="inst_1", storage_path=str(tmp_path))
    sdk._cache_manager.base_path.mkdir(parents=True, exist_ok=True)
    sdk._initialized = True
    sdk.instance._controller = FakeController(server)
    return sdk, server


async def read_texts(sdk, conversation_id):
    prompt = await sdk.conversation.context.get_context_for_prompt(conversation_id=conversation_id)
    return [m.content for m in prompt.recent_messages]


async def short_term_texts(sdk, conversation_id):
    state = await sdk.conversation.context.fetch(conversation_id=conversation_id)
    cc = state.conversation_context
    return [t.content if hasattr(t, "content") else t["content"] for t in (cc.recent_turns if cc else [])]


# ---------------------------------------------------------------- the bug

@pytest.mark.asyncio
async def test_a_stream_turn_is_in_the_next_prompt_read(tmp_path):
    sdk, _ = make_sdk(tmp_path)
    inst = sdk.instance
    await inst.send_message("seed turn", conversation_id=RAW, user_id="u1")
    assert await read_texts(sdk, DERIVED) == ["seed turn"]  # caches under the UUID
    await inst.send_message("the marker", conversation_id=RAW, user_id="u1")
    assert await read_texts(sdk, DERIVED) == ["seed turn", "the marker"]


@pytest.mark.asyncio
async def test_a_stream_turn_is_in_the_next_short_term_read(tmp_path):
    sdk, _ = make_sdk(tmp_path)
    inst = sdk.instance
    await inst.send_message("seed turn", conversation_id=RAW, user_id="u1")
    await short_term_texts(sdk, DERIVED)
    await inst.send_message("the marker", conversation_id=RAW, user_id="u1")
    assert "the marker" in await short_term_texts(sdk, DERIVED)


@pytest.mark.asyncio
async def test_the_local_mirror_is_one_entry_for_both_spellings(tmp_path):
    sdk, _ = make_sdk(tmp_path)
    await sdk.instance.send_message("seed turn", conversation_id=RAW, user_id="u1")
    assert sdk._st_store.get(RAW) is sdk._st_store.get(DERIVED) is not None
    assert sdk._st_store.size() == 1


@pytest.mark.asyncio
async def test_the_server_count_reaches_the_entry_the_stream_turn_made(tmp_path):
    """ConversationState carries the server's id (the UUID); the turn was mirrored under the raw one."""
    sdk, _ = make_sdk(tmp_path)
    await sdk.instance.send_message("seed turn", conversation_id=RAW, user_id="u1")
    sdk._on_conversation_state(DERIVED, 1)
    assert sdk._st_store.get(RAW).server_message_count == 1


@pytest.mark.asyncio
async def test_a_session_opened_under_one_spelling_ends_under_the_other(tmp_path):
    sdk, _ = make_sdk(tmp_path)
    await sdk.instance.send_message("hi", conversation_id=RAW, user_id="u1")
    opened = list(sdk.instance._open_sessions)
    assert opened == [RAW]  # keys, and so the wire, stay as the caller spelled them
    await sdk.instance.end_session(DERIVED)
    assert sdk.instance._open_sessions == {}
    controls = sdk.instance._controller._transport.controls
    assert [(c["action"], c["conversation_id"]) for c in controls] == [("start", RAW), ("end", DERIVED)]


def test_a_compaction_subscriber_hears_the_bundle_for_either_spelling(tmp_path):
    from maximem_synap.sdk import _dispatch_compaction_subscribers

    sdk, _ = make_sdk(tmp_path)
    heard = []
    sdk.conversation.context.subscribe_to_compaction_updates(RAW, heard.append)
    _dispatch_compaction_subscribers(sdk, DERIVED, {"x": 1})
    assert heard == [{"x": 1}]
    assert sdk.conversation.context.unsubscribe_all_compaction_updates(DERIVED) == 1


def test_the_cache_manager_keys_both_spellings_alike(tmp_path):
    m = CacheManager(client_id="c1", instance_id="inst_1", storage_path=str(tmp_path))
    m.base_path.mkdir(parents=True, exist_ok=True)
    kw = dict(scope=ContextScope.CONVERSATION, context_type="compacted_context", query={"style": "x"})
    m.set(entity_id=DERIVED, value=b"v", **kw)
    assert m.get(entity_id=RAW, **kw) == b"v"
    m.delete(entity_id=RAW, **kw)
    assert m.get(entity_id=DERIVED, **kw) is None


def test_the_short_term_store_keys_both_spellings_alike():
    st = ShortTermContextStore()
    st.append_turn(RAW, "user", "a")
    assert st.get(DERIVED).recent_turns[0]["content"] == "a"
    st.invalidate(DERIVED)
    assert st.get(RAW) is None


@pytest.mark.asyncio
async def test_the_turn_archive_gets_a_streamed_turn_under_the_derived_id(tmp_path):
    sdk, _ = make_sdk(tmp_path)
    await sdk.instance.send_message("seed turn", conversation_id=RAW, user_id="u1")
    assert sdk._turn_store.count(DERIVED) == 1
    assert sdk._turn_store.count(RAW) == 0  # one conversation, one key: the derived one
    sdk.archive_tool_result(RAW, {"r": 1}, tool_name="t")
    assert [t.role for t in sdk._turn_store.turns(DERIVED)] == ["user", "tool"]


# ---------------------------------------------------------------- no change for real UUIDs

from maximem_synap.utils.validators import normalize_conversation_id  # noqa: E402

REAL = [
    "3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
    "3F6B1A2C-4D5E-6F7A-8B9C-0D1E2F3A4B5C",
    "3f6b1a2c4d5e6f7a8b9c0d1e2f3a4b5c",
    "{3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c}",
    "urn:uuid:3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
    str(uuid.uuid4()),
    str(uuid.UUID(int=0)),
]


@pytest.mark.parametrize("value", REAL)
def test_a_real_uuid_in_any_spelling_is_unchanged_and_still_validates(value):
    assert normalize_conversation_id(value) == value
    validate_conversation_id(value)


def test_empty_and_none_pass_through():
    assert normalize_conversation_id(None) is None
    assert normalize_conversation_id("") == ""


# ---------------------------------------------------------------- the same function as the server

ODD = [
    "pgv2_testconv72", "pgv2_" + "a" * 32, "conv_123", "conv-abc-123", "a", " ", " leading", "trailing ",
    "UPPER", "MiXeD_Case", "with space", "with/slash", "with?query=1&x=2", "emoji_\U0001F600", "café",
    "会话-1", "0", "12345", "-" * 5, "x" * 500, "3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5", "3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5cd",
    "g" * 32, "urn:uuid:nope", "{not-a-uuid}", "new\nline", "tab\there", "pgv2_" + "0" * 32, "http://example.com/c/1",
    "session:abc:42", "\u0000nul",
]


@pytest.mark.parametrize("value", ODD)
def test_it_is_the_servers_function(value):
    from synap.cloud.shared.conversation_id_norm import coerce_conversation_id

    assert normalize_conversation_id(value) == coerce_conversation_id(value)
    assert normalize_conversation_id(normalize_conversation_id(value)) == normalize_conversation_id(value)


# ------------------------------------------------ each spelling, in each direction

@pytest.mark.asyncio
async def test_the_prompt_cache_is_dropped_under_the_derived_key(tmp_path):
    """Checked at the call, not through the cache manager's own normalising."""
    sdk, _ = make_sdk(tmp_path)
    seen = []

    class Recorder:
        def delete(self, **kw):
            seen.append(kw["entity_id"])

    sdk._cache_manager = Recorder()
    sdk._invalidate_prompt_cache(RAW)
    assert seen and set(seen) == {DERIVED}


def test_a_value_set_under_the_raw_id_is_read_under_the_derived_one(tmp_path):
    m = CacheManager(client_id="c1", instance_id="inst_1", storage_path=str(tmp_path))
    m.base_path.mkdir(parents=True, exist_ok=True)
    kw = dict(scope=ContextScope.CONVERSATION, context_type="compacted_context", query={"style": "x"})
    m.set(entity_id=RAW, value=b"v", **kw)
    assert m.get(entity_id=DERIVED, **kw) == b"v"


def test_a_server_count_named_with_the_raw_id_reaches_the_derived_entry():
    st = ShortTermContextStore()
    st.append_turn(DERIVED, "user", "a")
    st.note_server_count(RAW, 4)
    assert st.get(DERIVED).server_message_count == 4


@pytest.mark.asyncio
async def test_a_session_opened_under_the_derived_id_ends_under_the_raw_one(tmp_path):
    sdk, _ = make_sdk(tmp_path)
    await sdk.instance.send_message("hi", conversation_id=DERIVED, user_id="u1")
    assert list(sdk.instance._open_sessions) == [DERIVED]
    await sdk.instance.end_session(RAW)
    assert sdk.instance._open_sessions == {}


def test_a_subscriber_for_the_derived_id_hears_a_bundle_named_with_the_raw_one(tmp_path):
    from maximem_synap.sdk import _dispatch_compaction_subscribers

    sdk, _ = make_sdk(tmp_path)
    heard = []
    sdk.conversation.context.subscribe_to_compaction_updates(DERIVED, heard.append)
    _dispatch_compaction_subscribers(sdk, RAW, {"x": 1})
    assert heard == [{"x": 1}]


# ---------------------------------- get_context_for_prompt shows a streamed turn the server has not stored

class LaggingRest(FakeRest):
    """The server stores a streamed turn later than the SDK sends it."""

    def __init__(self, server):
        super().__init__(server)
        self.hidden = 0  # newest N messages the server has not stored yet

    async def get(self, path, **kw):
        result = await super().get(path, **kw)
        if self.hidden:
            msgs = result["recent_messages"][: -self.hidden]
            result["recent_messages"] = msgs
            result["recent_message_count"] = result["total_message_count"] = len(msgs)
        return result


@pytest.mark.asyncio
async def test_prompt_read_shows_a_stream_turn_the_server_has_not_stored(tmp_path):
    sdk, _ = make_sdk(tmp_path)
    sdk._http_transport = LaggingRest(sdk._http_transport._server)
    await sdk.instance.send_message("seed turn", conversation_id=RAW, user_id="u1")
    assert await read_texts(sdk, DERIVED) == ["seed turn"]
    sdk._http_transport.hidden = 1
    await sdk.instance.send_message("the marker", conversation_id=RAW, user_id="u1")
    got = await sdk.conversation.context.get_context_for_prompt(conversation_id=DERIVED)
    assert [m.content for m in got.recent_messages] == ["seed turn", "the marker"]
    assert got.recent_message_count == got.total_message_count == 2


@pytest.mark.asyncio
async def test_prompt_read_does_not_repeat_a_turn_the_server_lists(tmp_path):
    sdk, _ = make_sdk(tmp_path)
    await sdk.instance.send_message("seed turn", conversation_id=RAW, user_id="u1")
    await sdk.instance.send_message("the marker", conversation_id=RAW, user_id="u1")
    got = await sdk.conversation.context.get_context_for_prompt(conversation_id=DERIVED)
    assert [m.content for m in got.recent_messages] == ["seed turn", "the marker"]
    assert got.total_message_count == 2


@pytest.mark.asyncio
async def test_the_cached_answer_is_not_changed_by_the_overlay(tmp_path):
    sdk, _ = make_sdk(tmp_path)
    sdk._http_transport = LaggingRest(sdk._http_transport._server)
    sdk._http_transport.hidden = 1
    await sdk.instance.send_message("only turn", conversation_id=RAW, user_id="u1")
    first = await sdk.conversation.context.get_context_for_prompt(conversation_id=DERIVED)
    assert [m.content for m in first.recent_messages] == ["only turn"]
    sdk._http_transport.hidden = 0
    again = await sdk.conversation.context.get_context_for_prompt(conversation_id=DERIVED)  # cache hit
    assert [m.content for m in again.recent_messages] == ["only turn"]
    assert sdk._http_transport.reads == 1


@pytest.mark.asyncio
async def test_the_kill_switch_leaves_the_server_answer_alone(tmp_path, monkeypatch):
    sdk, _ = make_sdk(tmp_path)
    sdk._http_transport = LaggingRest(sdk._http_transport._server)
    sdk._http_transport.hidden = 1
    await sdk.instance.send_message("only turn", conversation_id=RAW, user_id="u1")
    monkeypatch.setattr(sdk, "_is_st_verbatim_overlay", lambda: False)
    got = await sdk.conversation.context.get_context_for_prompt(conversation_id=DERIVED)
    assert got.recent_messages == []


@pytest.mark.asyncio
async def test_a_different_summary_is_left_alone(tmp_path):
    """Local turns may already be inside a compaction the store has not heard of."""
    sdk, server = make_sdk(tmp_path)

    class Compacted(LaggingRest):
        async def get(self, path, **kw):
            r = await super().get(path, **kw)
            r["compaction_id"] = "server-comp"
            return r

    sdk._http_transport = Compacted(server)
    sdk._http_transport.hidden = 1
    await sdk.instance.send_message("only turn", conversation_id=RAW, user_id="u1")
    got = await sdk.conversation.context.get_context_for_prompt(conversation_id=DERIVED)
    assert got.recent_messages == []
