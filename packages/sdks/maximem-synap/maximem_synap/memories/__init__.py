"""Memory operations interface and models."""

from .interface import MemoriesInterface
from .models import (
    DocumentType,
    IngestMode,
    IngestStatus,
    MergeStrategy,
    CreateMemoryRequest,
    CreateMemoryResponse,
    AttachmentResult,
    BatchCreateRequest,
    BatchCreateResponse,
    MemoryStatusResponse,
    UpdateMemoryRequest,
    Memory,
    ForgetMode,
    ForgetStatus,
    ForgetCandidate,
    ForgetResponse,
)
from .parts import OBJECTIVES, FilePart, Objective, TextPart

__all__ = [
    "MemoriesInterface",
    "DocumentType",
    "IngestMode",
    "IngestStatus",
    "MergeStrategy",
    "CreateMemoryRequest",
    "CreateMemoryResponse",
    "AttachmentResult",
    "BatchCreateRequest",
    "BatchCreateResponse",
    "MemoryStatusResponse",
    "UpdateMemoryRequest",
    "Memory",
    "ForgetMode",
    "ForgetStatus",
    "ForgetCandidate",
    "ForgetResponse",
    "TextPart",
    "FilePart",
    "OBJECTIVES",
    "Objective",
]
