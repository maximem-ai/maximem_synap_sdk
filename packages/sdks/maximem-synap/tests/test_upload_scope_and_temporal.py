"""create_from_file sends scope and omits an absent customer_id; TEMPORAL is the
server's name for the type.

- `ContextType.TEMPORAL` was "temporal"; the server's memory type is
  "temporal_events" (retrieval_manager valid_types), so a types filter built
  from it matched nothing.
- `create_from_file` had no `scope` parameter, so an upload on a nested ladder
  could not name its rung. The route reads it as a JSON string form field.
"""
import asyncio
import json

import pytest

from maximem_synap import ContextType, InvalidInputError, MaximemSynapSDK


class _Transport:
    def __init__(self):
        self.calls = []

    async def post_multipart(self, path, auth_context, data, files=None, correlation_id=None):
        self.calls.append({"path": path, "data": data, "files": files})
        return {
            "ingestion_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
            "document_id": "d1",
            "status": "queued",
            "queued_at": "2026-10-22T00:00:00+00:00",
        }


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


def test_temporal_is_the_servers_type_name():
    assert ContextType.TEMPORAL.value == "temporal_events"


def test_a_temporal_filter_goes_out_as_temporal_events():
    types = [ContextType.FACTS, ContextType.TEMPORAL]
    assert [t.value for t in types] == ["facts", "temporal_events"]


def test_scope_is_sent_as_a_json_string_field():
    sdk = _sdk()
    scope = {"customer": "acme", "team": "payments", "user": "dana"}
    asyncio.run(sdk.memories.create_from_file(
        user_id="dana", customer_id="acme", text="hello", scope=scope,
    ))
    data = sdk._http_transport.calls[0]["data"]
    assert json.loads(data["scope"]) == scope


def test_scope_is_absent_when_not_given_or_empty():
    sdk = _sdk()
    asyncio.run(sdk.memories.create_from_file(user_id="u", customer_id="c", text="t"))
    asyncio.run(sdk.memories.create_from_file(user_id="u", customer_id="c", text="t", scope={}))
    for call in sdk._http_transport.calls:
        assert "scope" not in call["data"]


def test_a_malformed_scope_is_refused_before_any_request():
    sdk = _sdk()
    with pytest.raises(InvalidInputError, match="scope"):
        asyncio.run(sdk.memories.create_from_file(
            user_id="u", customer_id="c", text="t", scope={"customer": 1},  # type: ignore[dict-item]
        ))
    assert sdk._http_transport.calls == []


@pytest.mark.parametrize("customer_id", [None, ""])
def test_no_customer_id_means_no_customer_id_field(customer_id):
    sdk = _sdk()
    asyncio.run(sdk.memories.create_from_file(user_id="u", customer_id=customer_id, text="t"))
    assert "customer_id" not in sdk._http_transport.calls[0]["data"]


def test_a_customer_id_is_still_sent():
    sdk = _sdk()
    asyncio.run(sdk.memories.create_from_file(user_id="u", customer_id="acme", text="t"))
    assert sdk._http_transport.calls[0]["data"]["customer_id"] == "acme"
