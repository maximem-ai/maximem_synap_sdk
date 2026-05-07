"""Models for memory operations."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from uuid import UUID


class DocumentType(str, Enum):
    """Type of document being ingested."""
    CONVERSATION = "ai-chat-conversation"
    DOCUMENT = "document"
    EMAIL = "email"
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"
    MEETING_TRANSCRIPT = "meeting-transcript"


class IngestMode(str, Enum):
    """Ingestion processing mode."""
    FAST = "fast"           # Quick processing, less thorough
    LONG_RANGE = "long-range"  # Full pipeline, more thorough


class IngestStatus(str, Enum):
    """Status of an ingestion job."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"


class MergeStrategy(str, Enum):
    """Strategy for memory updates."""
    REPLACE = "replace"
    APPEND = "append"
    SMART_MERGE = "smart-merge"


class CreateMemoryRequest(BaseModel):
    """Request to create/ingest a memory."""
    # Content
    document: Union[str, Dict[str, Any]]
    document_type: DocumentType = DocumentType.CONVERSATION

    # Optional identifiers
    document_id: Optional[str] = None  # Client-provided ID

    # Temporal
    document_created_at: Optional[datetime] = None  # DCT

    # Scope (optional — server derives the effective scope level from which
    # IDs are passed and the instance's user_context_isolation. B2C: either
    # ID → user-scope; neither → client-scope. B2B: both → user; customer
    # only → customer; neither → client; user_id without customer_id → 400.)
    user_id: Optional[str] = None
    customer_id: Optional[str] = None

    # Processing options
    mode: IngestMode = IngestMode.LONG_RANGE

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateMemoryResponse(BaseModel):
    """Response from memory creation."""
    ingestion_id: UUID
    document_id: str
    status: IngestStatus
    queued_at: datetime
    error_message: Optional[str] = None


class BatchCreateRequest(BaseModel):
    """Request to create multiple memories."""
    documents: List[CreateMemoryRequest]
    fail_fast: bool = False


class BatchCreateResponse(BaseModel):
    """Response from batch creation."""
    batch_id: UUID
    total: int
    succeeded: int
    failed: int
    results: List[CreateMemoryResponse]


class MemoryStatusResponse(BaseModel):
    """Status of an ingestion job."""
    ingestion_id: UUID
    document_id: str
    status: IngestStatus
    queued_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    memories_created: int = 0
    memory_ids: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None


class UpdateMemoryRequest(BaseModel):
    """Request to update a memory."""
    document: Union[str, Dict[str, Any]]
    merge_strategy: MergeStrategy = MergeStrategy.SMART_MERGE
    document_type: Optional[DocumentType] = None
    metadata: Optional[Dict[str, Any]] = None


class Memory(BaseModel):
    """A memory object."""
    memory_id: UUID
    memory_type: str
    content: str
    confidence: float
    category: Optional[str] = None
    subcategory: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    source_document_id: Optional[str] = None
