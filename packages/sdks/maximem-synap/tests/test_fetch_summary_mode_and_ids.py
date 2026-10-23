"""Summary-mode user fetches never read the anticipation cache, and an empty
user_id or customer_id is refused before any request.

Both were found by reading the fetch path against the docs (0.5.12):

- A summary-mode fetch (context_mode="conversation-summary") is a different
  read: a caller profile and past-conversation summaries, assembled by the
  server. With no search query the anticipation cache answered it with the
  freshest user_summary bundle, so the call-start read came back with
  profile=None, conversations=None and no request sent.
- An empty id reached the server, which answers it with an empty result. The
  JS SDK already refused it.
"""
import asyncio

import pytest

from maximem_synap import InvalidInputError, MaximemSynapSDK

SUMMARY_BUNDLE = {
    "bundle_id": "b-sum",
    "_bundle_type": "user_summary",
    "_anticipation_user_id": "u1",
    "items_by_type": {"facts": [{"item_id": "i1", "content": "Prefers window seats", "scope": "user"}]},
}


class _Transport:
    def __init__(self):
        self.sent = []

    async def post(self, path, **kw):
        self.sent.append((path, kw.get("json")))
        return {
            "context": {},
            "profile": {"overview": "server profile"},
            "conversations": [],
        }


class _NoCache:
    def get(self, **_):
        return None

    def set(self, **_):
        pass


class _NoAuth:
    async def get_auth_context(self, *_):
        return None


class _NoTelemetry:
    def __getattr__(self, _):
        return lambda *a, **k: None


def _sdk():
    sdk = MaximemSynapSDK(api_key="synap_test", _force_new=True)
    sdk._http_transport = _Transport()
    sdk._cache_manager = _NoCache()
    sdk._credential_manager = _NoAuth()
    sdk._telemetry_collector = _NoTelemetry()
    sdk._initialized = True
    sdk._client_id = "cli"
    sdk._anticipation_cache.store(dict(SUMMARY_BUNDLE))
    return sdk


def test_summary_mode_goes_to_the_server_even_with_a_cached_summary():
    sdk = _sdk()
    response = asyncio.run(sdk.user.context.fetch(
        user_id="u1", customer_id="acme", context_mode="conversation-summary",
    ))
    assert len(sdk._http_transport.sent) == 1
    assert sdk._http_transport.sent[0][1]["context_mode"] == "conversation-summary"
    assert response.metadata.source == "cloud"
    assert response.profile is not None


def test_in_conversation_mode_still_reads_the_cache():
    """Control: the cache is skipped for summary mode only."""
    sdk = _sdk()
    response = asyncio.run(sdk.user.context.fetch(user_id="u1", customer_id="acme"))
    assert sdk._http_transport.sent == []
    assert response.metadata.source == "anticipation"


@pytest.mark.parametrize("empty", ["", None])
def test_an_empty_user_id_is_refused_before_any_request(empty):
    sdk = _sdk()
    with pytest.raises(InvalidInputError, match="user_id is required"):
        asyncio.run(sdk.user.context.fetch(user_id=empty, customer_id="acme"))
    assert sdk._http_transport.sent == []


@pytest.mark.parametrize("empty", ["", None])
def test_an_empty_customer_id_is_refused_before_any_request(empty):
    sdk = _sdk()
    with pytest.raises(InvalidInputError, match="customer_id is required"):
        asyncio.run(sdk.customer.context.fetch(customer_id=empty))
    assert sdk._http_transport.sent == []


@pytest.mark.parametrize("empty", ["", None])
def test_an_empty_conversation_id_is_refused_before_any_request(empty):
    """validate_conversation_id checks the format of an id that is there and
    passes an empty one, so it reached the server and came back empty."""
    sdk = _sdk()
    with pytest.raises(InvalidInputError, match="conversation_id is required"):
        asyncio.run(sdk.conversation.context.fetch(conversation_id=empty))
    assert sdk._http_transport.sent == []
