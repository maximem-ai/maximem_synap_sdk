"""memories.create_from_file refuses a missing or ambiguous source and an empty
user_id with InvalidInputError, before reading or uploading anything.

Found reading the upload path against the docs (0.5.21): no source was a plain
ValueError, several sources were accepted with the first silently winning (the
docs say they are mutually exclusive), and an empty user_id was uploaded. The
JS SDK refused all three.
"""
import asyncio
import io

import pytest

from maximem_synap import InvalidInputError, MaximemSynapSDK


class _Transport:
    def __init__(self):
        self.sent = []

    async def post_multipart(self, path, **kw):
        self.sent.append(path)
        return {"ingestion_id": "6c4a1f9e-2b7d-4e0a-9c1f-1a2b3c4d5e6f", "document_id": "d1",
                "status": "queued", "queued_at": "2026-10-23T00:00:00+00:00"}


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


@pytest.mark.parametrize("kwargs, message", [
    ({"user_id": "u1"}, "One of file_path, file, or text"),
    ({"user_id": "u1", "text": "x", "file": io.BytesIO(b"y")}, "exactly one of"),
    ({"user_id": "u1", "text": "x", "file_path": "/does/not/matter"}, "exactly one of"),
    ({"user_id": "", "text": "x"}, "user_id is required"),
])
def test_refused_before_uploading(kwargs, message):
    sdk = _sdk()
    with pytest.raises(InvalidInputError, match=message):
        asyncio.run(sdk.memories.create_from_file(**kwargs))
    assert sdk._http_transport.sent == []


def test_one_source_uploads():
    sdk = _sdk()
    asyncio.run(sdk.memories.create_from_file(user_id="u1", text="Prefers email."))
    assert sdk._http_transport.sent == ["/api/v1/memories/upload"]
