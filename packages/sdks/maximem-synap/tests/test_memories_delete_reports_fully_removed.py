"""memories.delete returns what the server said about the other stores.

The route answers ``fully_removed``, ``still_present_in`` and
``source_text_kept``; the SDK used to keep only ``memory_id`` and
``deleted_at``, so a caller could not tell that a deleted memory was still in
the search index. ``fully_removed`` is False unless the server said True.
"""
import asyncio

from maximem_synap import MaximemSynapSDK

MID = "6c4a1f9e-2b7d-4e0a-9c1f-1a2b3c4d5e6f"


class _Transport:
    def __init__(self, body):
        self.body = body

    async def delete(self, path, **kw):
        return self.body


class _NoAuth:
    async def get_auth_context(self, *_):
        return None


class _NoTelemetry:
    def __getattr__(self, _):
        return lambda *a, **k: None


def _sdk(body):
    sdk = MaximemSynapSDK(api_key="synap_test", _force_new=True)
    sdk._http_transport = _Transport(body)
    sdk._credential_manager = _NoAuth()
    sdk._telemetry_collector = _NoTelemetry()
    sdk._initialized = True
    return sdk


def test_a_partial_delete_says_so():
    out = asyncio.run(_sdk({
        "memory_id": MID, "deleted_at": "2026-11-03T00:00:00+00:00",
        "fully_removed": False, "still_present_in": ["search index"],
        "source_text_kept": True,
    }).memories.delete(MID))
    assert out["fully_removed"] is False
    assert out["still_present_in"] == ["search index"]
    assert out["source_text_kept"] is True


def test_a_full_delete_says_so():
    out = asyncio.run(_sdk({
        "memory_id": MID, "deleted_at": "2026-11-03T00:00:00+00:00",
        "fully_removed": True, "still_present_in": [], "source_text_kept": False,
    }).memories.delete(MID))
    assert out["fully_removed"] is True
    assert out["still_present_in"] == []


def test_an_older_server_without_the_field_is_not_fully_removed():
    out = asyncio.run(_sdk({
        "memory_id": MID, "deleted_at": "2026-11-03T00:00:00+00:00",
    }).memories.delete(MID))
    assert out["fully_removed"] is False
    assert out["still_present_in"] == []
