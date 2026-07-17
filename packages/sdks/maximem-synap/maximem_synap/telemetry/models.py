"""Telemetry event models."""

from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum
from pydantic import BaseModel, Field


class TelemetryEventType(str, Enum):
    """Types of telemetry events."""
    SDK_INIT = "sdk_init"
    SDK_SHUTDOWN = "sdk_shutdown"

    # Context fetch events
    FETCH_CONVERSATION_CONTEXT = "fetch_conversation_context"
    FETCH_USER_CONTEXT = "fetch_user_context"
    FETCH_CUSTOMER_CONTEXT = "fetch_customer_context"
    FETCH_CLIENT_CONTEXT = "fetch_client_context"

    # Compaction
    COMPACT_CONTEXT = "compact_context"

    # Cache events
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"

    # Listening events
    LISTEN_START = "listen_start"
    LISTEN_RECONNECT = "listen_reconnect"
    LISTEN_DISCONNECT = "listen_disconnect"

    # HTTP events
    HTTP_REQUEST = "http_request"

    # Memory operations
    MEMORY_CREATE = "memory_create"
    MEMORY_BATCH_CREATE = "memory_batch_create"
    MEMORY_GET = "memory_get"
    MEMORY_UPDATE = "memory_update"
    MEMORY_DELETE = "memory_delete"
    MEMORY_STATUS = "memory_status"

    # Conversation ingest (one-shot transcript push)
    CONVERSATION_INGEST = "conversation_ingest"

    # User profile
    USER_PROFILE_GET = "user_profile_get"

    # Errors
    ERROR = "error"


class TelemetryEvent(BaseModel):
    """A single telemetry event.

    IMPORTANT: No PII fields - no user_id, no content, no query text.
    """
    event_type: TelemetryEventType
    instance_id: str
    client_id: str
    correlation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())

    # Performance
    latency_ms: Optional[int] = None

    # Status
    status: str = "success"  # "success", "error", "timeout"
    error_code: Optional[str] = None

    # Context (no PII)
    scope: Optional[str] = None  # "client", "customer", "user", "conversation"
    cache_status: Optional[str] = None  # "hit", "miss"
    attempt: Optional[int] = None  # Retry attempt number

    # HTTP specific
    http_method: Optional[str] = None
    http_path: Optional[str] = None
    http_status_code: Optional[int] = None

    # Additional metadata (must not contain PII)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TelemetryBatch(BaseModel):
    """Batch of telemetry events for transmission."""
    events: list[TelemetryEvent]
    sdk_version: str
    batch_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())
