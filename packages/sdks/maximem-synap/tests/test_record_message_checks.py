"""conversation.record_message refuses a bad role and missing fields with
InvalidInputError before any request.

Found reading the record path against the docs (0.5.17): a role other than
"user" or "assistant" was a plain ValueError from the controller (the docs and
the JS SDK say InvalidInputError), and an empty conversation_id or user_id, or
no content, was posted for the server to refuse.
"""
import asyncio

import pytest

from maximem_synap import InvalidInputError, MaximemSynapSDK

CONV = "3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c"


class _Transport:
    def __init__(self):
        self.sent = []

    async def post(self, path, **kw):
        self.sent.append(path)
        return {"message_id": "m1", "conversation_id": CONV, "session_id": "s1",
                "recorded_at": "2026-10-23T00:00:00Z"}


class _NoAuth:
    async def get_auth_context(self, *_):
        return None


class _NoTelemetry:
    def __getattr__(self, _):
        return lambda *a, **k: None


def _sdk():
    sdk = MaximemSynapSDK(api_key="synap_test", _force_new=True)
    sdk._http_transport = _Transport()
    sdk._credential_manager = _NoAuth()
    sdk._telemetry_collector = _NoTelemetry()
    sdk._initialized = True
    return sdk


BASE = {"conversation_id": CONV, "role": "user", "content": "hi", "user_id": "u1", "customer_id": "acme"}


@pytest.mark.parametrize("override, message", [
    ({"role": "system"}, "Invalid role 'system'"),
    ({"role": ""}, "role is required"),
    ({"conversation_id": ""}, "conversation_id is required"),
    ({"user_id": ""}, "user_id is required"),
    ({"content": None}, "content is required"),
])
def test_refused_with_invalid_input_before_any_request(override, message):
    sdk = _sdk()
    with pytest.raises(InvalidInputError, match=message):
        asyncio.run(sdk.conversation.record_message(**{**BASE, **override}))
    assert sdk._http_transport.sent == []


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_each_accepted_role_is_sent(role):
    sdk = _sdk()
    asyncio.run(sdk.conversation.record_message(**{**BASE, "role": role}))
    assert sdk._http_transport.sent == ["/v1/conversations/messages"]
