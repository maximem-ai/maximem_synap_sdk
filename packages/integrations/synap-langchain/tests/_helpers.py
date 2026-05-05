"""Shared fixtures for synap-langchain tests."""

import sys
import os

# Add SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../synap/sdk/python"))
# Add integration package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Shared integrations utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../synap-integrations-common"))

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from maximem_synap import MaximemSynapSDK
from maximem_synap.models.context import (
    ContextForPromptResponse,
    ContextResponse,
    Fact,
    Preference,
    Episode,
    Emotion,
    TemporalEvent,
    ResponseMetadata,
    UnifiedContextResponse,
)


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
        id=id, content=content, confidence=confidence,
        source="test", extracted_at=datetime.now(timezone.utc),
    )


def make_preference(id="p1", content="Prefers dark mode", strength=0.8):
    return Preference(
        id=id, category="general", content=content, strength=strength,
        extracted_at=datetime.now(timezone.utc),
    )


def make_episode(id="e1", summary="Had a support call", significance=0.7):
    return Episode(
        id=id, summary=summary, occurred_at=datetime.now(timezone.utc),
        significance=significance,
    )


def make_emotion(id="em1", emotion_type="frustrated", context="Long wait", intensity=0.6):
    return Emotion(
        id=id, emotion_type=emotion_type, intensity=intensity,
        detected_at=datetime.now(timezone.utc), context=context,
    )


def make_temporal_event(id="t1", content="Trial expires April 15"):
    return TemporalEvent(
        id=id, content=content, event_date=datetime.now(timezone.utc),
        temporal_category="temporal_fact", temporal_confidence=0.9,
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


@pytest.fixture
def mock_sdk():
    """Create a mock SDK with common methods stubbed."""
    sdk = MagicMock(spec=MaximemSynapSDK)
    sdk.instance_id = "test-instance"
    sdk._initialized = True

    # Default fetch response
    sdk.fetch = AsyncMock(return_value=make_unified_response())

    # Default record_message
    sdk.conversation = MagicMock()
    sdk.conversation.record_message = AsyncMock(return_value={
        "message_id": "msg-001",
        "conversation_id": "conv-1",
        "session_id": "sess-1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    })
    sdk.conversation.context = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=ContextForPromptResponse(
            formatted_context="Recent conversation summary",
            available=True,
        )
    )

    # Default memories.create
    sdk.memories = MagicMock()
    create_result = MagicMock()
    create_result.ingestion_id = "ing-001"
    sdk.memories.create = AsyncMock(return_value=create_result)

    # Cache
    sdk.cache = MagicMock()

    return sdk
