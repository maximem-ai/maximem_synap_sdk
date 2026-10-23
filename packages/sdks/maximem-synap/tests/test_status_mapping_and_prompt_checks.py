"""HTTP 403 and other 4xx map to permanent errors; the conversation-context
reads refuse an empty id and an unknown prompt style before any request.

Found reading get_context_for_prompt against the docs (0.5.14):

- A 403 ("Conversation not found or access denied") had no branch and fell
  through to SynapTransientError, telling the caller to retry a refusal. Every
  other unhandled 4xx did the same. The JS SDK raises AuthenticationError for
  403.
- validate_conversation_id passes an empty id, so get_context_for_prompt,
  get_compacted and get_compaction_status requested /v1/conversations//...
- No SDK checked `style`; the server answered an unknown one with a 500, or
  ignored it when the compaction has a pre-rendered block.
"""
import asyncio

import httpx
import pytest

from maximem_synap import (
    AuthenticationError,
    InvalidInputError,
    MaximemSynapSDK,
    SynapPermanentError,
    SynapTransientError,
)
from maximem_synap.transport.http_client import HTTPTransport

CONV = "3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c"


def _raise_for(status):
    response = httpx.Response(
        status, json={"detail": "x"}, request=httpx.Request("GET", "https://synap.invalid/x"),
    )
    with pytest.raises(Exception) as exc:
        HTTPTransport._handle_response(object.__new__(HTTPTransport), response, "c1")
    return exc.value


def test_a_403_is_an_authentication_error_and_not_retried():
    err = _raise_for(403)
    assert isinstance(err, AuthenticationError)
    assert not isinstance(err, SynapTransientError)


@pytest.mark.parametrize("status", [405, 410, 413, 415, 418])
def test_any_other_4xx_is_permanent(status):
    err = _raise_for(status)
    assert type(err) is SynapPermanentError
    assert not isinstance(err, SynapTransientError)


class _Transport:
    def __init__(self):
        self.sent = []

    async def get(self, path, **kw):
        self.sent.append(path)
        return {}


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


@pytest.mark.parametrize("method", ["get_context_for_prompt", "get_compacted", "get_compaction_status"])
def test_an_empty_conversation_id_is_refused_before_any_request(method):
    sdk = _sdk()
    with pytest.raises(InvalidInputError, match="conversation_id is required"):
        asyncio.run(getattr(sdk.conversation.context, method)(conversation_id=""))
    assert sdk._http_transport.sent == []


@pytest.mark.parametrize("style", ["fancy", "STRUCTURED", ""])
def test_an_unknown_style_is_refused_before_any_request(style):
    sdk = _sdk()
    with pytest.raises(InvalidInputError, match="Invalid style"):
        asyncio.run(sdk.conversation.context.get_context_for_prompt(conversation_id=CONV, style=style))
    assert sdk._http_transport.sent == []


@pytest.mark.parametrize("style", ["structured", "narrative", "bullet_points"])
def test_each_server_style_is_sent(style):
    sdk = _sdk()
    asyncio.run(sdk.conversation.context.get_context_for_prompt(
        conversation_id=CONV, style=style, search_history=False,
    ))
    assert sdk._http_transport.sent == [f"/v1/conversations/{CONV}/context-for-prompt"]
