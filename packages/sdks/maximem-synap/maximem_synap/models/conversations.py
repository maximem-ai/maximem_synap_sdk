"""Conversation-ingest SDK models.

Client-side mirrors of the server contract for the one-shot transcript
ingest endpoint (``POST /v1/conversations/ingest`` — spec
``Synap_Async_Integration_and_User_Profile_Spec_v1`` §4.1). The response
model follows the ``models/context.py`` convention: ``extra="allow"`` plus a
``.raw`` escape hatch and an explicit ``from_cloud_response`` factory (never
``model_validate`` on the wire dict) so a newer server that adds fields never
breaks an older SDK.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel


def _parse_dt(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp (tolerating a trailing ``Z``)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return None


class TranscriptTurn(BaseModel):
    """A single transcript turn — the preferred, typed ``transcript`` form.

    Passing a ``List[TranscriptTurn]`` (rather than a plain string) preserves
    per-turn timestamps and speaker labels. ``role`` is constrained to the
    storage enum; the server maps any diarized label that is not
    ``user``/``human`` to ``assistant`` when a string transcript is split, so
    a typed turn always lands inside this ``Literal``.
    """

    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[datetime] = None
    speaker: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}

    def __init__(self, **data):
        raw_data = data.pop("raw_data", data.copy())
        super().__init__(**data)
        object.__setattr__(self, "_raw_data", raw_data)

    @property
    def raw(self) -> Dict[str, Any]:
        return getattr(self, "_raw_data", {})


class TranscriptIngestResponse(BaseModel):
    """Response from :meth:`sdk.conversation.ingest_transcript`.

    Mirrors the server's ``TranscriptIngestResponse``
    (``routes/sdk_conversations.py``) exactly. ``ingestion_id`` is always set
    on a 2xx (never null, even on the ``duplicate`` branch) and is the handle
    for ``sdk.memories.status()`` / ``wait_for_completion()``.
    """

    conversation_id: str            # coerced (UUID form)
    external_conversation_id: str   # the raw client-supplied id, echoed back
    ingestion_id: UUID
    status: Literal["queued", "duplicate"]
    turns_recorded: int
    summary_status: Literal["in_progress", "already_compacted", "skipped"]
    queued_at: datetime

    model_config = {"extra": "allow"}

    def __init__(self, **data):
        raw_data = data.pop("raw_data", data.copy())
        super().__init__(**data)
        object.__setattr__(self, "_raw_data", raw_data)

    @property
    def raw(self) -> Dict[str, Any]:
        """Raw response dict for fields not yet in the typed model."""
        return getattr(self, "_raw_data", {})

    @classmethod
    def from_cloud_response(cls, data: Dict[str, Any]) -> "TranscriptIngestResponse":
        """Build from the raw ``POST /v1/conversations/ingest`` response."""
        return cls(
            conversation_id=data["conversation_id"],
            external_conversation_id=data.get(
                "external_conversation_id", data["conversation_id"]
            ),
            ingestion_id=UUID(str(data["ingestion_id"])),
            status=data["status"],
            turns_recorded=int(data.get("turns_recorded", 0)),
            summary_status=data["summary_status"],
            queued_at=_parse_dt(data.get("queued_at")) or datetime.now(),
            raw_data=data,
        )


# Public alias for the accepted ``transcript`` argument type.
Transcript = Union[str, List[TranscriptTurn]]
