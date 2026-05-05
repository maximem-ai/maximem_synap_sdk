"""Shared fixtures for synap-llamaindex tests."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../synap/sdk/python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../synap-integrations-common"))

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from maximem_synap import MaximemSynapSDK
from maximem_synap.models.context import (
    Fact, Preference, Episode, Emotion, TemporalEvent,
    ResponseMetadata, UnifiedContextResponse, ContextForPromptResponse,
    RecentMessage,
)


def make_metadata(**overrides):
    defaults = {
        "correlation_id": "test-corr", "ttl_seconds": 300,
        "source": "cloud", "retrieved_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return ResponseMetadata(**defaults)


def make_fact(id="f1", content="User is an engineer", confidence=0.9):
    return Fact(id=id, content=content, confidence=confidence,
                source="test", extracted_at=datetime.now(timezone.utc))


def make_preference(id="p1", content="Prefers dark mode", strength=0.8):
    return Preference(id=id, category="general", content=content, strength=strength,
                      extracted_at=datetime.now(timezone.utc))


def make_episode(id="e1", summary="Had a support call", significance=0.7):
    return Episode(id=id, summary=summary, occurred_at=datetime.now(timezone.utc),
                   significance=significance)


def make_unified_response(**overrides):
    defaults = {
        "facts": [make_fact()], "preferences": [make_preference()],
        "scope_map": {"f1": "user", "p1": "user"},
        "scopes_queried": ["user"], "total_items": 2,
        "formatted_context": "## User Context\n### Facts\n- User is an engineer",
        "metadata": make_metadata(),
    }
    defaults.update(overrides)
    return UnifiedContextResponse(**defaults)


@pytest.fixture
def mock_sdk():
    sdk = MagicMock(spec=MaximemSynapSDK)
    sdk.instance_id = "test-instance"
    sdk._initialized = True
    sdk.fetch = AsyncMock(return_value=make_unified_response())
    sdk.conversation = MagicMock()
    sdk.conversation.record_message = AsyncMock(return_value={
        "message_id": "msg-001", "conversation_id": "conv-1",
    })
    sdk.conversation.context = MagicMock()
    sdk.conversation.context.get_context_for_prompt = AsyncMock(
        return_value=ContextForPromptResponse(
            formatted_context="Compacted summary of conversation",
            available=True,
            recent_messages=[
                RecentMessage(role="user", content="Hello", timestamp=datetime.now(timezone.utc), message_id="m1"),
                RecentMessage(role="assistant", content="Hi there!", timestamp=datetime.now(timezone.utc), message_id="m2"),
            ],
            recent_message_count=2,
        )
    )
    sdk.memories = MagicMock()
    sdk.cache = MagicMock()
    return sdk
