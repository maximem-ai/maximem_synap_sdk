import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../synap/sdk/python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../synap-integrations-common"))

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from maximem_synap import MaximemSynapSDK
from maximem_synap.models.context import (
    Fact, ResponseMetadata, UnifiedContextResponse,
)


def _meta():
    return ResponseMetadata(
        correlation_id="t", ttl_seconds=300, source="cloud",
        retrieved_at=datetime.now(timezone.utc),
    )


def _resp(**kw):
    d = {
        "facts": [Fact(
            id="f1", content="User is an engineer", confidence=0.9,
            source="test", extracted_at=datetime.now(timezone.utc),
        )],
        "preferences": [], "scope_map": {"f1": "user"},
        "scopes_queried": ["user"], "total_items": 1,
        "formatted_context": "## User Context\n- User is an engineer",
        "metadata": _meta(),
    }
    d.update(kw)
    return UnifiedContextResponse(**d)


@pytest.fixture
def mock_sdk():
    sdk = MagicMock(spec=MaximemSynapSDK)
    sdk.instance_id = "test"
    sdk._initialized = True
    sdk.fetch = AsyncMock(return_value=_resp())
    sdk.memories = MagicMock()
    r = MagicMock()
    r.ingestion_id = "ing-001"
    sdk.memories.create = AsyncMock(return_value=r)
    return sdk
