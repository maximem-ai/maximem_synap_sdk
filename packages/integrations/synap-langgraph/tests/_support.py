"""Shared fixtures for the scope-ladder / fixed-behaviour tests.

The SDK is a ``RecordingSDK`` (exact keyword arguments of every call, the real
SDK's signatures) plus a ``memories.delete``. Responses are built from the
SDK's real models (Fact, Preference, RankedItem, UnifiedContextResponse), the
same way the characterisation proofs did.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock

from maximem_synap.models.context import (
    ContextForPromptResponse,
    Fact,
    Preference,
    RankedItem,
    RecentMessage,
    UnifiedContextResponse,
)
from synap_integrations_common.testing import RecordingSDK, make_ladder

NOW = datetime(2026, 11, 1, tzinfo=timezone.utc)
SAID = datetime(2023, 6, 10, tzinfo=timezone.utc)
UUIDS = {n: f"00000000-0000-0000-0000-00000000000{n}" for n in range(1, 10)}


def fact(id_, text, scope, conf=0.9, **md):
    return Fact(id=id_, content=text, confidence=conf, source="conv",
                extracted_at=NOW, metadata={"scope": scope, **md})


def pref(id_, text, scope, conf=0.9):
    return Preference(id=id_, category="communication", content=text, strength=conf,
                      extracted_at=NOW, metadata={"scope": scope})


def ranked(id_, rank, category, text, score, scope, **kw):
    md = {"scope": scope, **kw.pop("metadata", {})}
    return RankedItem(id=id_, rank=rank, category=category, content=text,
                      rank_score=score, confidence=0.5, metadata=md, **kw)


def realistic_response() -> UnifiedContextResponse:
    """Ranked best-first: pref(0.91) > policy(0.80) > ride(0.62); the typed
    buckets hold the same items in bucket order, as the server sends them."""
    ride = fact("f1", "The user takes a monthly ride on the 10th", "user", 0.62)
    policy = fact("f2", "Refunds must be filed within 30 days", "client", 0.80)
    email = pref("p1", "Prefers email contact", "user", 0.91)
    return UnifiedContextResponse(
        facts=[ride, policy], preferences=[email],
        items=[
            ranked("p1", 1, "preferences", email.content, 0.91, "user"),
            ranked("f2", 2, "facts", policy.content, 0.80, "client"),
            ranked("f1", 3, "facts", ride.content, 0.62, "user"),
        ],
        scope_map={"f1": "user", "f2": "client", "p1": "user"},
        scopes_queried=["user"], total_items=3,
    )


def tagged(id_, rank, score, sub_ns, key, value='{"v": 1}', scope="user", ident=None):
    """A ranked item written through SynapStore.put, as returned by a server that
    ECHOES caller metadata (today's server does not; see SimSdk). ``ident``: the identity
    entries of a SHARED store (adds the identity tag); None for a single-user store."""
    md = {"lg_store": True, "lg_store_ns": sub_ns, "lg_store_key": key}
    if ident is not None:
        import json
        md["lg_store_id"] = json.dumps(list(ident))
    return ranked(id_, rank, "facts", value, score, scope, metadata=md)


def response_of(*items: RankedItem) -> UnifiedContextResponse:
    return UnifiedContextResponse(
        items=list(items), total_items=len(items),
        scope_map={i.id: i.metadata.get("scope", "") for i in items},
    )


def prompt_ctx(block: Optional[str] = None, turns=()) -> ContextForPromptResponse:
    return ContextForPromptResponse(
        formatted_context=block or "", available=True, context_block=block,
        recent_messages=[RecentMessage(role=r, content=c, timestamp=NOW, message_id=f"m{i}")
                         for i, (r, c) in enumerate(turns)],
    )


class Sdk(RecordingSDK):
    """RecordingSDK with ``memories.delete`` and a response of our choosing."""

    def __init__(self, response: Optional[Any] = None, *, ladder: Any = "b2b", **kw: Any):
        super().__init__(ladder=make_ladder(ladder) if isinstance(ladder, str) else ladder, **kw)
        self.results["fetch"] = response if response is not None else realistic_response()
        self.memories.delete = AsyncMock(return_value={})

    def fails(self, exc: Exception) -> "Sdk":
        self.errors["fetch"] = exc
        return self


class SimSdk:
    """A tiny server (default: it does NOT echo caller metadata, like today's): rows keyed by their identity path; a read returns the
    caller's own level AND every ancestor level; every row that is not at the
    innermost level is labelled ``customer`` (the server's real three-valued
    label), so a team row and a customer row look alike."""

    def __init__(self, ladder: str = "deep", depth: int = 3, *, echo_caller_metadata: bool = False):
        # False (default) is TODAY'S SERVER: an item's metadata is a fixed list
        # (scope, ranking numbers, ...) and the metadata a caller passed to
        # memories.create never comes back. True models a FUTURE server that
        # returns it, which the store's get / delete / tag logic needs.
        self.echo = echo_caller_metadata
        import uuid
        from types import SimpleNamespace
        self._uuid = uuid
        self.ladder = make_ladder(ladder)
        self.depth = depth
        self.rows: list = []          # (id, path tuple, metadata, content)
        self.deleted: list = []
        self.memories = SimpleNamespace(create=self._create, delete=self._delete)

    async def get_scope_ladder(self, refresh=False):
        return self.ladder

    @staticmethod
    def _path(kw, key):
        p = kw.get(key)
        if p:
            return tuple(p.values())
        return tuple(v for v in (kw.get("customer_id"), kw.get("user_id")) if v)

    async def _create(self, **kw):
        self.rows.append((str(self._uuid.uuid4()), self._path(kw, "scope"),
                          dict(kw.get("metadata") or {}), kw["document"]))
        return None

    async def _delete(self, mid):
        self.deleted.append(str(mid))
        self.rows = [r for r in self.rows if r[0] != str(mid)]

    async def fetch(self, **kw):
        me = self._path(kw, "scope_path")
        scopes = kw.get("scopes")
        items = []
        for id_, path, md, content in self.rows:
            if path != me[:len(path)]:
                continue
            label = "user" if len(path) == self.depth else "customer"
            if scopes and label not in scopes:
                continue
            items.append(ranked(id_, len(items) + 1, "facts", content, 1 - len(items) * 0.01,
                                label, metadata=dict(md) if self.echo else {}))
        return response_of(*items)
