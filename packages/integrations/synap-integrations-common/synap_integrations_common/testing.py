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

from datetime import datetime, timezone
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


__all__ = [
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
