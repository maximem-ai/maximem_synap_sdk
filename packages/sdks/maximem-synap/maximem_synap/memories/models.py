"""Models for memory operations."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator
from uuid import UUID

from ..models.errors import InvalidInputError
from .parts import FilePart, TextPart


class DocumentType(StrEnum):
    """The document types the server knows (D8), for documentation and
    autocomplete.

    ``document_type`` is a plain string on the wire: any value is sent as
    given, and the server matches free text against the type descriptions. A
    declared type is authoritative. ``pdf``, ``image`` and ``audio`` are older
    values, read by the server as a format hint with the type detected.
    """
    CONVERSATION = "ai-chat-conversation"
    DOCUMENT = "document"
    EMAIL = "email"
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"
    MEETING_TRANSCRIPT = "meeting-transcript"
    SUPPORT_CONVERSATION = "support-conversation"
    CALL_TRANSCRIPT = "call-transcript"
    TEAM_CHAT = "team-chat"
    GROUP_CHAT = "group-chat"
    TICKET = "ticket"
    KNOWLEDGE_ARTICLE = "knowledge-article"
    POLICY = "policy"
    CONTRACT = "contract"
    FINANCIAL_RECORD = "financial-record"
    REPORT = "report"
    PROFILE_RECORD = "profile-record"
    MEMORY_RECORD = "memory-record"


#: ``document_format`` values (D9): for text whose format cannot be seen.
DOCUMENT_FORMATS = ("text", "markdown", "html", "csv", "json")

#: The Documents fields on a create, in the order they are sent. Each is sent
#: only when set, after ``objective``, so a caller who sets none of them sends
#: exactly the body older SDKs sent.
DOCUMENT_CREATE_FIELDS = (
    "document_format", "utility", "plan", "conversation_id", "source", "participants",
    "on_conflict",
)

#: ``on_conflict`` values for memory records (D155).
ON_CONFLICT_VALUES = ("supersede", "keep_both", "reject")


def check_document_fields(
    document_format: Optional[str] = None,
    utility: Any = None,
    plan: Any = None,
    conversation_id: Optional[str] = None,
    source: Any = None,
    participants: Any = None,
    on_conflict: Optional[str] = None,
) -> None:
    """Refuse a malformed Documents field before anything is sent.

    Only the shape is checked here. What the objects hold (the utility enums,
    a participant naming exactly one of user_id, customer_id or role, unknown
    keys) is checked by the server, which answers 422.
    """
    if document_format is not None and document_format not in DOCUMENT_FORMATS:
        raise InvalidInputError(
            f"Invalid document_format {document_format!r}. "
            f"Expected one of: {', '.join(DOCUMENT_FORMATS)}"
        )
    for name, value in (("utility", utility), ("plan", plan), ("source", source),
                        ("participants", participants)):
        if value is not None and not isinstance(value, dict):
            raise InvalidInputError(f"{name} must be a dict")
    if conversation_id is not None and (
        not isinstance(conversation_id, str) or not 1 <= len(conversation_id) <= 200
    ):
        raise InvalidInputError("conversation_id must be a string of 1 to 200 characters")
    if on_conflict is not None and on_conflict not in ON_CONFLICT_VALUES:
        raise InvalidInputError(
            f"Invalid on_conflict {on_conflict!r}. Expected one of: {', '.join(ON_CONFLICT_VALUES)}"
        )


def document_fields_wire(values: Dict[str, Any]) -> Dict[str, Any]:
    """The Documents fields that are set, in wire order, JSON-ready.

    Datetimes inside ``source`` are written as pydantic writes
    ``document_created_at``.
    """
    from pydantic import TypeAdapter

    to_json = TypeAdapter(Any).dump_python

    return {
        name: to_json(values[name], mode="json")
        for name in DOCUMENT_CREATE_FIELDS
        if values.get(name) is not None
    }


class IngestMode(StrEnum):
    """Ingestion processing mode."""
    FAST = "fast"           # Quick processing, less thorough
    LONG_RANGE = "long-range"  # Full pipeline, more thorough


class IngestStatus(StrEnum):
    """Status of an ingestion job."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"


class MergeStrategy(StrEnum):
    """Strategy for memory updates."""
    REPLACE = "replace"
    APPEND = "append"
    SMART_MERGE = "smart-merge"


#: What ``document`` was typed as before parts: validated with it, so a value
#: that was accepted (or refused) before still is.
_LEGACY_DOCUMENT = TypeAdapter(Union[str, Dict[str, Any]])


class CreateMemoryRequest(BaseModel):
    """Request to create/ingest a memory. Also one item of ``batch_create``.

    ``document`` takes what ``memories.create`` takes: a string, a dict, or a
    list of strings, ``TextPart`` and ``FilePart``. A batch with any file in it
    is sent as multipart.

    The fields after ``metadata`` (``objective`` and the Documents fields) are
    left out of ``model_dump``: they are sent only when set, so an item that
    sets none of them is the body older SDKs sent. ``to_wire`` builds it.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Content: a string, a dict, a TextPart or FilePart, or a list of them.
    # Typed Any and checked below, because pydantic would otherwise try to
    # build a FilePart out of a dict document.
    document: Any

    @field_validator("document", mode="before")
    @classmethod
    def _document_shape(cls, value: Any) -> Any:
        if isinstance(value, (TextPart, FilePart)):
            return value
        if isinstance(value, (list, tuple)):
            return list(value)
        # Anything else is validated exactly as before parts existed.
        return _LEGACY_DOCUMENT.validate_python(value)
    # A plain string (D8). See DocumentType for the values the server knows.
    document_type: str = DocumentType.CONVERSATION.value

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

    # The rung this write belongs to, named in full, one entry per level down
    # to it: {"customer": "acme", "team": "payments", "user": "dana"}.
    #
    # Only needed by an account whose scope ladder has a level between its
    # customer and its user levels. `user_id` and `customer_id` cannot say
    # which one a write belongs to, so on such a ladder the server refuses the
    # write rather than guessing: a wrong guess at or above the customer level
    # crosses a tenant boundary.
    #
    # Omit it and nothing changes. Sending one to an account without nested
    # scoping is an error rather than ignored, because the caller asked to be
    # narrowed and ignoring them widens.
    scope: Optional[Dict[str, str]] = None

    # Processing options
    mode: IngestMode = IngestMode.LONG_RANGE

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Sent only when set (see the class docstring).
    objective: Optional[str] = Field(default=None, exclude=True)
    document_format: Optional[str] = Field(default=None, exclude=True)   # D9
    utility: Optional[Dict[str, Any]] = Field(default=None, exclude=True)  # D31
    plan: Optional[Dict[str, Any]] = Field(default=None, exclude=True)     # D34
    conversation_id: Optional[str] = Field(default=None, exclude=True)   # D183
    source: Optional[Dict[str, Any]] = Field(default=None, exclude=True)   # D21, D86
    participants: Optional[Dict[str, Any]] = Field(default=None, exclude=True)  # D21
    # Memory records only (D155): supersede (the server's default), keep_both
    # or reject. Sent only when set.
    on_conflict: Optional[str] = Field(default=None, exclude=True)

    @field_validator("document_type", mode="before")
    @classmethod
    def _type_as_string(cls, value: Any) -> Any:
        return value.value if isinstance(value, DocumentType) else value

    @field_validator("utility", "plan", "source", "participants", mode="before")
    @classmethod
    def _objects_are_dicts(cls, value: Any, info: Any) -> Any:
        if value is not None and not isinstance(value, dict):
            raise InvalidInputError(f"{info.field_name} must be a dict")
        return value

    @model_validator(mode="after")
    def _document_fields(self) -> "CreateMemoryRequest":
        check_document_fields(
            self.document_format, self.utility, self.plan, self.conversation_id,
            self.source, self.participants, self.on_conflict,
        )
        return self

    def has_files(self) -> bool:
        """Whether ``document`` carries a file, so the item needs multipart."""
        doc = self.document
        if isinstance(doc, FilePart):
            return True
        return isinstance(doc, (list, tuple)) and any(isinstance(p, FilePart) for p in doc)

    def to_wire(self, document: Any = None) -> Dict[str, Any]:
        """The JSON body: every legacy field, nulls included, then ``objective``
        and the Documents fields that are set.

        ``document`` replaces the document in the body (the joined text of a
        text-only list, or a manifest's parts).
        """
        body = self.model_dump(mode="json", exclude={"document"})
        body = {"document": self.document if document is None else document, **body}
        if self.objective is not None:
            body["objective"] = self.objective
        body.update(document_fields_wire(self.__dict__))
        return body

    @model_validator(mode="after")
    def _client_self_scope(self) -> "CreateMemoryRequest":
        """`scope={"client": "self"}` names the caller's own client level.

        A write with no ids already lands there, so the path is sent as null
        and the body is the one an unscoped write sends. With a user or
        customer id it contradicts itself and is refused. "self" next to other
        entries is refused for the same reason. Every other use of a "client"
        key goes to the server untouched.
        """
        scope = self.scope
        if not scope or "self" not in scope.values():
            return self
        if scope == {"client": "self"}:
            if self.user_id is not None or self.customer_id is not None:
                raise InvalidInputError(
                    'scope={"client": "self"} names the client level, so it '
                    "cannot be combined with user_id or customer_id"
                )
            self.scope = None
            return self
        if len(scope) > 1:
            raise InvalidInputError(
                '"self" can only be used alone, as scope={"client": "self"}; '
                f"got {scope}"
            )
        return self


class AttachmentResult(BaseModel):
    """What happened to one file sent to `memories.create`.

    `status` is "read" when the file was read into the memory, or "failed"
    when it could not be; the rest of the document is still ingested and
    `error` says why.
    """
    filename: str
    media_type: str
    status: str
    error: Optional[str] = None
    # Which document a file became (D10, D171). None from a server that
    # predates documents.
    document_key: Optional[str] = None
    document_type: Optional[str] = None
    type_source: Optional[str] = None
    document_id: Optional[str] = None
    version_status: Optional[str] = None


class CreateMemoryResponse(BaseModel):
    """Response from memory creation.

    The document fields are None from a server that predates them, and on a
    chat create except `document_type`. They are plain strings, so a value
    added later reaches the caller instead of failing the parse.
    """
    ingestion_id: UUID
    document_id: str
    status: IngestStatus
    queued_at: datetime
    error_message: Optional[str] = None
    # One per file sent, in order. None when the document had no files.
    attachments: Optional[List[AttachmentResult]] = None
    # Stable across versions: the address for the documents routes (T12).
    document_key: Optional[str] = None
    # The resolved type (D11), and where it came from: caller, call-shape,
    # format, rules, model, default or pending (T4).
    document_type: Optional[str] = None
    type_source: Optional[str] = None
    # D91. version_status: created, unchanged, new_version, stale or pending.
    version: Optional[int] = None
    version_status: Optional[str] = None
    # accepted, searchable, processed or failed (D137).
    processing_stage: Optional[str] = None

    @classmethod
    def from_wire(cls, result: Dict[str, Any]) -> "CreateMemoryResponse":
        """Parse a create result (`/create`, `/create-multipart`, a batch item)."""
        attachments = result.get("attachments")
        return cls(
            ingestion_id=UUID(result["ingestion_id"]),
            document_id=result["document_id"],
            status=IngestStatus(result["status"]),
            queued_at=datetime.fromisoformat(result["queued_at"]),
            error_message=result.get("error_message"),
            attachments=(
                [AttachmentResult(**a) for a in attachments]
                if attachments is not None else None
            ),
            document_key=result.get("document_key"),
            document_type=result.get("document_type"),
            type_source=result.get("type_source"),
            version=result.get("version"),
            version_status=result.get("version_status"),
            processing_stage=result.get("processing_stage"),
        )


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
    # Memory records only (T79): one entry per record, in order: index, id,
    # memory_id, outcome (stored | duplicate | rejected_conflict), perspective,
    # conflicting_memory_id, category_flag, subject, consolidation (pending |
    # done) and consolidation_run_id. None for any other ingestion.
    records: Optional[List[Dict[str, Any]]] = None


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


class ForgetMode(StrEnum):
    """How `memories.forget` acts on what it finds."""
    PREVIEW = "preview"
    DIRECT = "direct"


class ForgetStatus(StrEnum):
    """Outcome of a `memories.forget` call."""
    NOTHING_MATCHES = "nothing_matches"
    PREVIEW = "preview"
    DELETED = "deleted"
    PARTIAL = "partial"


class ForgetCandidate(BaseModel):
    """Something the description matched: a memory, or a copy of its text kept elsewhere."""
    memory_id: str
    content: str
    memory_type: Optional[str] = None
    # "memory", or where a copy lives: "chunk", "sentence", "entity", "triple",
    # "profile" or "candidate".
    store: str = "memory"


class ForgetResponse(BaseModel):
    """Result of `memories.forget`.

    Check `status == "deleted"` before telling an end user something was
    removed. "partial" means the memories are hidden but a named store may
    still hold the text (see `still_present_in`).
    """
    status: ForgetStatus
    request_id: Optional[str] = None
    candidates: List[ForgetCandidate] = Field(default_factory=list)
    deleted_ids: List[str] = Field(default_factory=list)
    failed_ids: List[str] = Field(default_factory=list)
    still_present_in: List[str] = Field(default_factory=list)
    may_be_incomplete: bool = False
    # A `direct` request that named its `conversation_id` stays open until the
    # batch of that conversation that holds it has been saved and checked,
    # whether or not the search found something now. False for a request that
    # did not name a conversation, and from servers that predate it.
    retry_scheduled: bool = False
