"""conversation.ingest_transcript refuses missing fields before posting, and
hands back the server's `warnings` as a typed list.

Found reading the transcript path against the docs (0.5.15):

- An empty conversation_id or user_id, or no transcript, was posted and
  refused by the server (422/400). The JS SDK refuses them before posting.
- The server answers with `warnings` (no event time supplied, or a re-push
  whose existing turns were kept); the response model reached them only
  through `.raw`.
"""
import asyncio

import pytest

from maximem_synap import InvalidInputError, MaximemSynapSDK

REPLY = {
    "conversation_id": "3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
    "external_conversation_id": "call_1",
    "ingestion_id": "6c4a1f9e-2b7d-4e0a-9c1f-1a2b3c4d5e6f",
    "status": "queued",
    "turns_recorded": 1,
    "summary_status": "in_progress",
    "queued_at": "2026-07-15T10:06:00+00:00",
}


class _Transport:
    def __init__(self, reply):
        self.sent = []
        self._reply = reply

    async def post(self, path, **kw):
        self.sent.append(path)
        return dict(self._reply)


class _NoAuth:
    async def get_auth_context(self, *_):
        return None


class _NoTelemetry:
    def __getattr__(self, _):
        return lambda *a, **k: None


def _sdk(reply=REPLY):
    sdk = MaximemSynapSDK(api_key="synap_test", _force_new=True)
    sdk._http_transport = _Transport(reply)
    sdk._credential_manager = _NoAuth()
    sdk._telemetry_collector = _NoTelemetry()
    sdk._initialized = True
    return sdk


@pytest.mark.parametrize("override, message", [
    ({"conversation_id": ""}, "conversation_id is required"),
    ({"user_id": ""}, "user_id is required"),
    ({"transcript": None}, "transcript is required"),
])
def test_a_missing_field_is_refused_before_posting(override, message):
    sdk = _sdk()
    args = {"conversation_id": "call_1", "user_id": "u1", "customer_id": "acme",
            "transcript": "user: hi", **override}
    with pytest.raises(InvalidInputError, match=message):
        asyncio.run(sdk.conversation.ingest_transcript(**args))
    assert sdk._http_transport.sent == []


def test_warnings_are_a_typed_list():
    sdk = _sdk({**REPLY, "warnings": ["no event time supplied"]})
    resp = asyncio.run(sdk.conversation.ingest_transcript(
        conversation_id="call_1", user_id="u1", customer_id="acme", transcript="user: hi",
    ))
    assert resp.warnings == ["no event time supplied"]


def test_warnings_default_to_an_empty_list():
    sdk = _sdk()
    resp = asyncio.run(sdk.conversation.ingest_transcript(
        conversation_id="call_1", user_id="u1", customer_id="acme", transcript="user: hi",
    ))
    assert resp.warnings == []
