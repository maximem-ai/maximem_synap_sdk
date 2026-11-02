"""Shared pytest fixtures and response factories for integration tests.

Each integration previously kept its own ``tests/_helpers.py`` that
re-implemented the same ``mock_sdk`` fixture and the same
``make_fact`` / ``make_preference`` / ``make_episode`` / etc. factories.
That duplication drifted over time and none of the helpers exercised
SDK-failure paths.

To use in an integration's ``tests/conftest.py``::

    from synap_integrations_common.testing import (
        mock_sdk,              # pytest fixture — re-export
        failing_sdk,           # pytest fixture — SDK whose calls raise
        make_unified_response,
        make_fact,
    )

The individual ``make_*`` helpers are plain functions so tests can build
custom response payloads without monkey-patching fixtures.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    from maximem_synap import MaximemSynapSDK
    from maximem_synap.models.context import (
        ContextForPromptResponse,
        Emotion,
        Episode,
        Fact,
        Preference,
        ResponseMetadata,
        TemporalEvent,
        UnifiedContextResponse,
    )
except ImportError:  # pragma: no cover — surfaces clearly in dev setups
    MaximemSynapSDK = None  # type: ignore[assignment]

# The scope ladder arrived in a later SDK release; an older one still imports
# this module, and `make_ladder` then says so instead of failing at import.
try:
    from maximem_synap import ScopeLadder, ScopeLevel
except ImportError:  # pragma: no cover — older SDK
    ScopeLadder = None  # type: ignore[assignment]
    ScopeLevel = None  # type: ignore[assignment]

# Real signatures, so `RecordingSDK` refuses a keyword the installed SDK would.
try:
    from maximem_synap.memories.interface import MemoriesInterface
    from maximem_synap.sdk import ConversationInterface, InstanceInterface
except ImportError:  # pragma: no cover
    MemoriesInterface = ConversationInterface = InstanceInterface = None  # type: ignore[assignment]


def make_metadata(**overrides):
    defaults = {
        "correlation_id": "test-corr",
        "ttl_seconds": 300,
        "source": "cloud",
        "retrieved_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return ResponseMetadata(**defaults)


def make_fact(id="f1", content="User is an engineer", confidence=0.9):
    return Fact(
        id=id,
        content=content,
        confidence=confidence,
        source="test",
        extracted_at=datetime.now(timezone.utc),
    )


def make_preference(id="p1", content="Prefers dark mode", strength=0.8):
    return Preference(
        id=id,
        category="general",
        content=content,
        strength=strength,
        extracted_at=datetime.now(timezone.utc),
    )


def make_episode(id="e1", summary="Had a support call", significance=0.7):
    return Episode(
        id=id,
        summary=summary,
        occurred_at=datetime.now(timezone.utc),
        significance=significance,
    )


def make_emotion(id="em1", emotion_type="frustrated", context="Long wait", intensity=0.6):
    return Emotion(
        id=id,
        emotion_type=emotion_type,
        intensity=intensity,
        detected_at=datetime.now(timezone.utc),
        context=context,
    )


def make_temporal_event(id="t1", content="Trial expires April 15"):
    return TemporalEvent(
        id=id,
        content=content,
        event_date=datetime.now(timezone.utc),
        temporal_category="temporal_fact",
        temporal_confidence=0.9,
    )


def make_unified_response(**overrides):
    defaults = {
        "facts": [make_fact()],
        "preferences": [make_preference()],
        "scope_map": {"f1": "user", "p1": "user"},
        "scopes_queried": ["user"],
        "total_items": 2,
        "formatted_context": "## User Context\n### Facts\n- User is an engineer",
        "metadata": make_metadata(),
    }
    defaults.update(overrides)
    return UnifiedContextResponse(**defaults)


def _build_mock_sdk():
    sdk = MagicMock(spec=MaximemSynapSDK)
    sdk.instance_id = "test-instance"
    sdk._initialized = True

    sdk.fetch = AsyncMock(return_value=make_unified_response())

    sdk.conversation = MagicMock()
    sdk.conversation.record_message = AsyncMock(
        return_value={
            "message_id": "msg-001",
            "conversation_id": "conv-1",
            "session_id": "sess-1",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    sdk.conversation.context = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=ContextForPromptResponse(
            formatted_context="Recent conversation summary",
            available=True,
        )
    )

    sdk.memories = MagicMock()
    create_result = MagicMock()
    create_result.ingestion_id = "ing-001"
    sdk.memories.create = AsyncMock(return_value=create_result)

    sdk.cache = MagicMock()
    return sdk


@pytest.fixture
def mock_sdk():
    """SDK mock pre-wired with successful default responses."""
    return _build_mock_sdk()


@pytest.fixture
def failing_sdk():
    """SDK mock whose every async method raises ``RuntimeError("sdk boom")``.

    Use in tests that assert integrations surface SDK failures instead of
    silently swallowing them.
    """
    sdk = _build_mock_sdk()
    err = RuntimeError("sdk boom")
    sdk.fetch = AsyncMock(side_effect=err)
    sdk.conversation.record_message = AsyncMock(side_effect=err)
    sdk.conversation.context.get_context_for_prompt = AsyncMock(side_effect=err)
    sdk.memories.create = AsyncMock(side_effect=err)
    return sdk


# ---------------------------------------------------------------------------
# Scope ladders
# ---------------------------------------------------------------------------

_ROLE_TENANT = "tenant_of_record"
_ROLE_IDENTITY = "identity"

_LADDER_KINDS: Dict[str, Tuple[Tuple[str, ...], bool]] = {
    "b2b": (("customer", "user"), False),
    "b2c": (("user",), True),
    "deep": (("customer", "team", "user"), False),
}


def make_ladder(
    kind: str = "b2b",
    *,
    keys: Optional[Sequence[str]] = None,
    b2c: Optional[bool] = None,
    ratified: Optional[bool] = None,
    nested: Optional[bool] = None,
    source: str = "server",
):
    """Build a ``maximem_synap.ScopeLadder`` for tests.

    ``kind``: ``"b2b"`` (customer, user: the default), ``"b2c"`` (user alone),
    ``"deep"`` (customer, team, user). ``keys`` overrides the level keys,
    outermost first; the key ``customer`` gets the tenant role, the last key
    the identity role, others none (same as the server). ``b2c`` overrides the
    isolation mode (a deeper ladder on a B2C instance: ``keys=("customer",
    "team", "user"), b2c=True``).

    ``ratified`` / ``nested`` default to what a usable ladder needs: False for
    the default ladder (the server never accepts a path there, so the SDK never
    makes one), True for a deeper one. Pass ``ratified=False`` to test the
    "server would refuse a path" error.
    """
    if ScopeLadder is None:
        raise RuntimeError(
            "make_ladder needs a maximem-synap release that has ScopeLadder")
    if kind not in _LADDER_KINDS:
        raise ValueError(f"unknown ladder kind {kind!r}; use one of {sorted(_LADDER_KINDS)}")
    kind_keys, kind_b2c = _LADDER_KINDS[kind]
    level_keys = tuple(keys) if keys is not None else kind_keys
    is_b2c = kind_b2c if b2c is None else b2c
    default_shapes = (("customer", "user"), ("user",))
    is_default = level_keys in default_shapes
    levels = tuple(
        ScopeLevel(
            key=k,
            label=k.replace("_", " ").title(),
            position=i + 1,
            role=(_ROLE_IDENTITY if i == len(level_keys) - 1
                  else _ROLE_TENANT if k == "customer" else None),
            required=True,
        )
        for i, k in enumerate(level_keys)
    )
    return ScopeLadder(
        levels=levels,
        is_default=is_default,
        ladder_ratified=(not is_default) if ratified is None else ratified,
        nested_scoping_enabled=(not is_default) if nested is None else nested,
        source=source,
        b2c=is_b2c,
    )


# ---------------------------------------------------------------------------
# RecordingSDK
# ---------------------------------------------------------------------------


_DEFAULT_LADDER = object()


@dataclass
class RecordedCall:
    """One call to a recorded SDK method: its exact arguments."""

    name: str
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)


def _real_signature(owner: Any, attr: str, drop: Sequence[str]):
    """The installed SDK's signature of ``owner.attr`` without ``self`` and
    without the parameters in ``drop``; ``None`` if it cannot be read."""
    if owner is None:
        return None
    try:
        params = list(inspect.signature(getattr(owner, attr)).parameters.values())[1:]
    except (AttributeError, TypeError, ValueError):  # pragma: no cover
        return None
    return inspect.Signature([p for p in params if p.name not in drop])


class RecordingSDK:
    """A fake SDK that records the EXACT arguments of every call.

    Recorded (``sdk.calls`` holds ``RecordedCall(name, args, kwargs)`` in call
    order): ``fetch``, ``memories.create``, ``memories.forget``,
    ``conversation.record_message``, ``instance.send_message``,
    ``get_scope_ladder``, and, for the stream helpers,
    ``instance.record_tool_call`` / ``record_tool_result`` /
    ``record_thinking`` / ``end_session`` and
    ``conversation.context.get_context_for_prompt``.

    Each method has the installed SDK's real signature, so a call with a
    keyword the SDK does not take raises ``TypeError`` (and is not recorded),
    as it would for real. ``accept_scope_path=False`` removes ``scope_path``
    (and ``scope`` on ``memories.create`` / ``forget``) to behave like an SDK
    from before the scope ladder.

    The stream is open by default (``listening=True``) so ``report_turn``
    goes through ``instance.send_message``; ``listening=False`` makes
    ``record_turn`` fall back to ``conversation.record_message``.

    ``ladder`` is what ``get_scope_ladder`` returns (default: the B2B ladder;
    a ``ScopeLadder`` from ``make_ladder``; ``None`` for "no ladder", or an exception instance to
    raise). ``results`` / ``errors`` map a call name to the value to return /
    the exception to raise (after recording).

    Example::

        from synap_integrations_common.testing import RecordingSDK, make_ladder

        async def test_deeper_ladder_reaches_every_call():
            sdk = RecordingSDK(ladder=make_ladder("deep"))
            addr = await resolve_address(sdk, values=["acme", "payments", "dana"])
            await sdk.fetch(search_query=["x"], **addr.read_kwargs())
            await sdk.memories.create("note", **addr.write_kwargs())

            assert sdk.last_kwargs("fetch") == {
                "search_query": ["x"], "user_id": "dana", "customer_id": "acme",
                "scope_path": {"customer": "acme", "team": "payments", "user": "dana"},
            }
            assert sdk.last_kwargs("memories.create")["scope"] == {
                "customer": "acme", "team": "payments", "user": "dana"}

    As a pytest fixture, import ``recording_sdk`` into the package's
    ``conftest.py`` (``from synap_integrations_common.testing import
    recording_sdk``) and set ``recording_sdk.ladder = make_ladder("b2c")`` per
    test.
    """

    def __init__(
        self,
        *,
        ladder: Any = _DEFAULT_LADDER,
        listening: bool = True,
        accept_scope_path: bool = True,
        instance_id: str = "test-instance",
    ) -> None:
        self.calls: List[RecordedCall] = []
        if ladder is _DEFAULT_LADDER:
            ladder = make_ladder("b2b") if ScopeLadder is not None else None
        self.ladder = ladder
        self.results: Dict[str, Any] = {}
        self.errors: Dict[str, BaseException] = {}
        self.instance_id = instance_id
        self._initialized = True
        self.cache = MagicMock()

        drop_path = () if accept_scope_path else ("scope_path",)
        drop_scope = () if accept_scope_path else ("scope",)
        if not accept_scope_path and MaximemSynapSDK is None:  # pragma: no cover
            raise RuntimeError("accept_scope_path=False needs the SDK to copy signatures from")

        def method(name: str, owner: Any, attr: str, drop: Sequence[str],
                   default: Callable[[], Any]) -> Callable[..., Any]:
            sig = _real_signature(owner, attr, drop)

            async def call(*args: Any, **kwargs: Any) -> Any:
                if sig is not None:
                    sig.bind(*args, **kwargs)  # TypeError, like the real method
                self.calls.append(RecordedCall(name, args, dict(kwargs)))
                if name in self.errors:
                    raise self.errors[name]
                if name in self.results:
                    return self.results[name]
                return default()

            if sig is not None:
                call.__signature__ = sig  # type: ignore[attr-defined]
            call.__name__ = attr
            return call

        def created() -> Any:
            r = MagicMock()
            r.ingestion_id = "ing-001"
            return r

        def recorded() -> Dict[str, Any]:
            return {"message_id": "msg-001", "conversation_id": "conv-1",
                    "session_id": "sess-1",
                    "recorded_at": datetime.now(timezone.utc).isoformat()}

        def fetched() -> Any:
            return make_unified_response() if MaximemSynapSDK is not None else MagicMock()

        def prompt_ctx() -> Any:
            if MaximemSynapSDK is None:  # pragma: no cover
                return MagicMock()
            return ContextForPromptResponse(
                formatted_context="Recent conversation summary", available=True)

        self.fetch = method("fetch", MaximemSynapSDK, "fetch", drop_path, fetched)

        async def get_scope_ladder(*, refresh: bool = False) -> Any:
            self.calls.append(RecordedCall("get_scope_ladder", (), {"refresh": refresh}
                                           if refresh else {}))
            if isinstance(self.ladder, BaseException):
                raise self.ladder
            return self.ladder

        self.get_scope_ladder = get_scope_ladder

        self.memories = SimpleNamespace(
            create=method("memories.create", MemoriesInterface, "create", drop_scope, created),
            forget=method("memories.forget", MemoriesInterface, "forget", drop_scope,
                          lambda: MagicMock()),
        )
        self.conversation = SimpleNamespace(
            record_message=method("conversation.record_message", ConversationInterface,
                                  "record_message", drop_path, recorded),
            context=SimpleNamespace(
                get_context_for_prompt=method(
                    "conversation.context.get_context_for_prompt", None,
                    "get_context_for_prompt", (), prompt_ctx),
            ),
        )

        async def listen() -> None:
            self.instance.is_listening = True

        async def stop_listening() -> None:
            self.instance.is_listening = False

        self.instance = SimpleNamespace(
            is_listening=listening,
            listen=listen,
            stop_listening=stop_listening,
            send_message=method("instance.send_message", InstanceInterface, "send_message",
                                drop_path, lambda: None),
            record_tool_call=method("instance.record_tool_call", InstanceInterface,
                                    "record_tool_call", (), lambda: None),
            record_tool_result=method("instance.record_tool_result", InstanceInterface,
                                      "record_tool_result", (), lambda: None),
            record_thinking=method("instance.record_thinking", InstanceInterface,
                                   "record_thinking", (), lambda: None),
            end_session=method("instance.end_session", InstanceInterface,
                               "end_session", (), lambda: None),
        )

    # -- reading what was recorded -------------------------------------------

    def calls_to(self, name: str) -> List[RecordedCall]:
        """All recorded calls to ``name`` (e.g. ``"memories.create"``)."""
        return [c for c in self.calls if c.name == name]

    def kwargs_of(self, name: str) -> List[Dict[str, Any]]:
        """The keyword arguments of every call to ``name``, in order."""
        return [c.kwargs for c in self.calls_to(name)]

    def last_kwargs(self, name: str) -> Dict[str, Any]:
        """The keyword arguments of the last call to ``name``; ``AssertionError``
        if it was never called."""
        calls = self.calls_to(name)
        assert calls, f"{name} was never called; recorded: {[c.name for c in self.calls]}"
        return calls[-1].kwargs

    def reset(self) -> None:
        """Forget recorded calls (ladder, results and errors are kept)."""
        self.calls.clear()


@pytest.fixture
def recording_sdk():
    """A ``RecordingSDK`` serving the default B2B ladder. See its docstring."""
    return RecordingSDK()


__all__ = [
    "make_ladder",
    "RecordedCall",
    "RecordingSDK",
    "recording_sdk",
    "make_metadata",
    "make_fact",
    "make_preference",
    "make_episode",
    "make_emotion",
    "make_temporal_event",
    "make_unified_response",
    "mock_sdk",
    "failing_sdk",
]
