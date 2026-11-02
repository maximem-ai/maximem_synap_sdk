"""Any string is a conversation id for the developer; REST calls get the UUID the server derives."""
import uuid

import pytest
from maximem_synap.models.errors import InvalidConversationIdError

from synap_integrations_common import (
    coerce_conversation_id, fetch_compacted_history, record_turn, rest_conversation_id,
)
from synap_integrations_common.testing import RecordingSDK

PG = "pgv2_9f8e7d6c5b4a39281716151413121110"
DERIVED = str(uuid.uuid5(uuid.NAMESPACE_URL, PG))


def test_a_uuid_is_unchanged_and_any_other_string_is_derived_the_way_the_server_does():
    u = "123e4567-e89b-12d3-a456-426614174000"
    assert coerce_conversation_id(u) == u
    assert coerce_conversation_id(PG) == DERIVED
    assert coerce_conversation_id(DERIVED) == DERIVED            # idempotent
    assert coerce_conversation_id("") == "" and rest_conversation_id(None) == ""


def test_matches_the_servers_own_function():
    norm = pytest.importorskip("synap.cloud.shared.conversation_id_norm")
    for v in (PG, "chat-42", "x", "123e4567-e89b-12d3-a456-426614174000"):
        assert coerce_conversation_id(v) == norm.coerce_conversation_id(v)


@pytest.mark.asyncio
async def test_premise_the_strict_fake_refuses_what_the_real_sdk_refuses():
    sdk = RecordingSDK(strict_conversation_ids=True)
    with pytest.raises(InvalidConversationIdError):
        await sdk.conversation.record_message(conversation_id=PG, role="user", content="x", user_id="u")
    await sdk.instance.send_message(content="x", role="user", event_type="user_message",
                                    conversation_id=PG)          # the stream takes any string


@pytest.mark.asyncio
async def test_rest_fallback_of_a_turn_uses_the_derived_uuid_and_the_stream_the_raw_id():
    rest = RecordingSDK(listening=False, strict_conversation_ids=True)
    rest.instance.listen = None  # cannot open a stream
    assert await record_turn(rest, role="user", content="hi", conversation_id=PG, user_id="u")
    assert rest.last_kwargs("conversation.record_message")["conversation_id"] == DERIVED

    live = RecordingSDK(strict_conversation_ids=True)
    assert await record_turn(live, role="user", content="hi", conversation_id=PG, user_id="u")
    assert live.last_kwargs("instance.send_message")["conversation_id"] == PG
    assert not live.calls_to("conversation.record_message")


@pytest.mark.asyncio
async def test_compacted_history_reads_the_derived_uuid():
    sdk = RecordingSDK(strict_conversation_ids=True)
    await fetch_compacted_history(sdk, PG)
    assert sdk.last_kwargs("conversation.context.get_context_for_prompt")["conversation_id"] == DERIVED


@pytest.mark.asyncio
async def test_record_turn_via_says_how_the_turn_travelled():
    from synap_integrations_common import record_turn_via

    live = RecordingSDK()
    assert await record_turn_via(live, role="user", content="hi", conversation_id=PG, user_id="u") == "stream"
    down = RecordingSDK(listening=False)
    down.instance.listen = None
    assert await record_turn_via(down, role="user", content="hi", conversation_id=PG, user_id="u") == "rest"
    assert await record_turn_via(down, role="user", content=" ", conversation_id=PG, user_id="u") == "failed"
    down.errors["conversation.record_message"] = RuntimeError("x")
    assert await record_turn_via(down, role="user", content="hi", conversation_id=PG, user_id="u") == "failed"
