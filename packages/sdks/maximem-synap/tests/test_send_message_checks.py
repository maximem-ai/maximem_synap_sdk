"""instance.send_message refuses bad input with InvalidInputError, carries
scope_path to the stream, and a server error signal does not drop the stream.

Found reading the stream path against the docs (0.5.19):

- An unknown event_type was a plain ValueError (JS: a bare SynapError); a
  missing content was not checked at all.
- The ConversationEvent proto has a `scope` field the server stores as the
  conversation's rung, and no SDK could send it.
- The server sends `signal_type: "error"` for one refused event or
  session_start and keeps the stream open. Python tore the stream down and
  reconnected on every one.
"""
import asyncio
from types import SimpleNamespace

import pytest

from maximem_synap import InvalidInputError, MaximemSynapSDK
from maximem_synap.transport.grpc_client import GRPCTransport

CONV = "3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c"
PATH = {"customer": "acme", "team": "payments"}


class _Stream:
    def __init__(self):
        self.events = []

    async def send(self, payload):
        self.events.append(dict(payload))

    async def send_session_control(self, **kw):
        return True


class _NoTelemetry:
    def __getattr__(self, _):
        return lambda *a, **k: None


def _sdk():
    sdk = MaximemSynapSDK(api_key="synap_test", _force_new=True)
    sdk._telemetry_collector = _NoTelemetry()
    sdk._initialized = True
    stream = _Stream()
    sdk.instance._controller = SimpleNamespace(_transport=stream)
    return sdk, stream


BASE = {"content": "hi", "conversation_id": CONV, "user_id": "u1", "customer_id": "acme"}


@pytest.mark.parametrize("override, message", [
    ({"event_type": "user_msg"}, "unknown event_type"),
    ({"content": None}, "content is required"),
])
def test_refused_with_invalid_input_before_writing(override, message):
    sdk, stream = _sdk()
    with pytest.raises(InvalidInputError, match=message):
        asyncio.run(sdk.instance.send_message(**{**BASE, **override}))
    assert stream.events == []


def test_scope_path_rides_the_event():
    sdk, stream = _sdk()
    asyncio.run(sdk.instance.send_message(**BASE, scope_path=PATH))
    assert stream.events[0]["scope"] == PATH


def test_an_empty_scope_path_is_not_sent():
    sdk, stream = _sdk()
    asyncio.run(sdk.instance.send_message(**BASE, scope_path={}))
    assert "scope" not in stream.events[0]


def test_the_proto_carries_scope():
    """The transport builds ConversationEvent field by field; `scope` was not one."""
    from maximem_synap.transport.proto import synap_service_pb2

    written = []

    class _Writer:
        async def write(self, event):
            written.append(event)

    t = object.__new__(GRPCTransport)
    t._stream = _Writer()
    t._write_lock = asyncio.Lock()
    asyncio.run(GRPCTransport._write_conversation_event(t, {**BASE, "scope": PATH}))
    assert isinstance(written[0], synap_service_pb2.StreamEvent)
    assert dict(written[0].conversation_event.scope) == PATH


def _transport_recording_disconnects():
    t = object.__new__(GRPCTransport)
    seen = []

    async def _disconnect(reason):
        seen.append(reason)

    t._handle_disconnect = _disconnect
    return t, seen


def test_an_error_signal_keeps_the_stream():
    t, seen = _transport_recording_disconnects()

    async def run():
        t._handle_signal(SimpleNamespace(signal_type="error", reason="customer_id is required"))
        await asyncio.sleep(0)

    asyncio.run(run())
    assert seen == []


def test_a_closing_signal_still_reconnects():
    """Control: `closing` is the server ending the stream."""
    t, seen = _transport_recording_disconnects()

    async def run():
        t._handle_signal(SimpleNamespace(signal_type="closing", reason="deploy"))
        await asyncio.sleep(0)

    asyncio.run(run())
    assert seen == ["server_closing"]
