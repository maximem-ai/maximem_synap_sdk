"""Shared helpers for the harness-surface tests.

The harness API landed in ``agent-framework`` 1.13, well after this package's
``>=1.0`` floor. These tests skip cleanly on an older install rather than
failing, so a contributor working on the SDK surfaces is not blocked by a
version they do not need.
"""

from __future__ import annotations

from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    from agent_framework import (
        AgentSession,
        MemoryTopicRecord,
    )

    HARNESS_AVAILABLE = True
except ImportError:  # pragma: no cover — depends on the installed version
    AgentSession = None  # type: ignore[assignment]
    MemoryTopicRecord = None  # type: ignore[assignment]
    HARNESS_AVAILABLE = False

requires_harness = pytest.mark.skipif(
    not HARNESS_AVAILABLE,
    reason="needs the MAF harness API (agent-framework>=1.13)",
)


def make_session(session_id: str = "sess-1", **state: Any):
    """Return an ``AgentSession`` carrying ``state``."""
    session = AgentSession(session_id=session_id)
    session.state.update(state)
    return session


def make_record(
    topic: str = "deployment workflow",
    summary: str = "How this person ships software.",
    memories: Optional[List[str]] = None,
    updated_at: str = "2026-08-07T00:00:00+00:00",
):
    """Return a ``MemoryTopicRecord`` in the shape MAF's extraction produces."""
    # `is None` rather than a falsy check, so a caller can ask for a record
    # with no memory lines yet — which is a real state, not a missing argument.
    if memories is None:
        memories = ["Tags a release; Actions promotes the build."]
    return MemoryTopicRecord(
        topic=topic,
        summary=summary,
        memories=memories,
        updated_at=updated_at,
        session_ids=["sess-1"],
    )


def status_with(memory_ids: List[str]) -> Any:
    """Return a ``status()`` result carrying ``memory_ids``."""
    status = MagicMock()
    status.memory_ids = memory_ids
    return status


def wire_ingestion(sdk: Any, memory_ids: Optional[List[str]] = None) -> None:
    """Give ``sdk.memories`` the create/status/delete chain ``delete`` walks.

    The shared ``mock_sdk`` fixture wires ``create`` only, because no other
    integration needed the rest. ``delete`` here resolves an ingestion id to
    memory ids and deletes each, so all three have to be present.
    """
    sdk.memories.status = AsyncMock(return_value=status_with(memory_ids or ["mem-1"]))
    sdk.memories.delete = AsyncMock(return_value={"deleted": True})
