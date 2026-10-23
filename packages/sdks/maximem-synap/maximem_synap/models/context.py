"""Synap SDK Data Models.

All models use Pydantic for validation and serialization.
Response models include both typed access and raw escape hatch.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from pydantic import BaseModel, Field

from .enums import CompactionLevel


def _parse_iso_datetime(value) -> Optional[datetime]:
    """Parse an ISO 8601 string to datetime, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


_CONFIDENCE_STRINGS: dict = {
    "explicit": 1.0,
    "inferred": 0.75,
    "assumed": 0.5,
}


def _coerce_float(value, default: float = 0.0) -> float:
    """Coerce a value to float, mapping known semantic strings."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        mapped = _CONFIDENCE_STRINGS.get(value.lower())
        if mapped is not None:
            return mapped
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    return default


# Context Items
class Fact(BaseModel):
    """A factual piece of information about an entity."""

    id: str
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    extracted_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    event_date: Optional[datetime] = None
    # The day the user SAID this, which is not the day the thing happened.
    #
    # `event_date` is now empty unless somebody actually stated when the event
    # took place, so without this field beside it an item whose turn stated no
    # event date arrives with no machine-readable date at all. On the
    # 500-question benchmark that was 2,017 of 10,443 served items (19.3%),
    # where the day they were said reached the caller only inside the content
    # string as "(said ...)".
    #
    # Empty when the client never told the server when the conversation
    # happened. It is never the server's own ingest or extraction clock.
    said_at: Optional[datetime] = None
    # The rest of the event window. `event_date` is its START: "last month"
    # is 2023-04-01 to 2023-04-30 at precision "month", and reading only the
    # start makes it look like the 1st. `event_precision` also carries the one
    # distinction nothing else does: "approximate" has no end, so "around
    # November 20" and "on November 20" differ only by this label.
    event_end: Optional[datetime] = None
    event_precision: Optional[str] = None
    valid_until: Optional[datetime] = None
    temporal_category: Optional[str] = None
    temporal_confidence: float = 0.0
    source_evidence: Optional[List[str]] = None
    # The search queries of the fetch that found this item, in the order they
    # were sent. Empty when the fetch named no query, when the server predates
    # the field, and for an item served from the prefetch cache.
    matched_queries: List[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class Preference(BaseModel):
    """A preference or behavioral pattern."""

    id: str
    category: str
    content: str
    strength: float = Field(ge=0.0, le=1.0)  # How strong is this preference
    source: str = ""
    extracted_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    event_date: Optional[datetime] = None
    # The day the user SAID this, as distinct from when the event happened.
    # See the note on `Fact.said_at`.
    said_at: Optional[datetime] = None
    # The rest of the event window. `event_date` is its START: "last month"
    # is 2023-04-01 to 2023-04-30 at precision "month", and reading only the
    # start makes it look like the 1st. `event_precision` also carries the one
    # distinction nothing else does: "approximate" has no end, so "around
    # November 20" and "on November 20" differ only by this label.
    event_end: Optional[datetime] = None
    event_precision: Optional[str] = None
    valid_until: Optional[datetime] = None
    temporal_category: Optional[str] = None
    temporal_confidence: float = 0.0
    source_evidence: Optional[List[str]] = None
    # The search queries of the fetch that found this item, in the order they
    # were sent. Empty when the fetch named no query, when the server predates
    # the field, and for an item served from the prefetch cache.
    matched_queries: List[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class Episode(BaseModel):
    """A memorable event or interaction."""

    id: str
    summary: str
    occurred_at: datetime
    significance: float = Field(ge=0.0, le=1.0)
    participants: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    event_date: Optional[datetime] = None
    # The day the user SAID this, as distinct from when the event happened.
    # See the note on `Fact.said_at`.
    said_at: Optional[datetime] = None
    # The rest of the event window. `event_date` is its START: "last month"
    # is 2023-04-01 to 2023-04-30 at precision "month", and reading only the
    # start makes it look like the 1st. `event_precision` also carries the one
    # distinction nothing else does: "approximate" has no end, so "around
    # November 20" and "on November 20" differ only by this label.
    event_end: Optional[datetime] = None
    event_precision: Optional[str] = None
    valid_until: Optional[datetime] = None
    temporal_category: Optional[str] = None
    temporal_confidence: float = 0.0
    source_evidence: Optional[List[str]] = None
    # The search queries of the fetch that found this item, in the order they
    # were sent. Empty when the fetch named no query, when the server predates
    # the field, and for an item served from the prefetch cache.
    matched_queries: List[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class Emotion(BaseModel):
    """Detected emotional state."""

    id: str
    emotion_type: str  # e.g., "frustrated", "satisfied", "confused"
    intensity: float = Field(ge=0.0, le=1.0)
    detected_at: datetime
    context: str  # What triggered this
    metadata: Dict[str, Any] = Field(default_factory=dict)
    event_date: Optional[datetime] = None
    # The day the user SAID this, as distinct from when the event happened.
    # See the note on `Fact.said_at`.
    said_at: Optional[datetime] = None
    # The rest of the event window. `event_date` is its START: "last month"
    # is 2023-04-01 to 2023-04-30 at precision "month", and reading only the
    # start makes it look like the 1st. `event_precision` also carries the one
    # distinction nothing else does: "approximate" has no end, so "around
    # November 20" and "on November 20" differ only by this label.
    event_end: Optional[datetime] = None
    event_precision: Optional[str] = None
    valid_until: Optional[datetime] = None
    temporal_category: Optional[str] = None
    temporal_confidence: float = 0.0
    source_evidence: Optional[List[str]] = None
    # The search queries of the fetch that found this item, in the order they
    # were sent. Empty when the fetch named no query, when the server predates
    # the field, and for an item served from the prefetch cache.
    matched_queries: List[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class TemporalEvent(BaseModel):
    """A time-bound event or fact with explicit temporal boundaries."""

    id: str
    content: str
    # Optional for the same reason it is optional on `Fact`: the server now
    # leaves it empty unless somebody actually stated when the event happened.
    # It was required here, and the parser below dropped any item that arrived
    # without one, which was harmless only while the server always filled the
    # field with the conversation's own date. On the benchmark instance 917 of
    # 5,345 stored temporal events (17%) have no stated event date, so keeping
    # it required would have deleted every one of them from a typed caller's
    # response while raw HTTP and the JS SDK still saw them.
    event_date: Optional[datetime] = None
    # The day the user SAID this, as distinct from when the event happened.
    # See the note on `Fact.said_at`.
    said_at: Optional[datetime] = None
    # The rest of the event window. `event_date` is its START: "last month"
    # is 2023-04-01 to 2023-04-30 at precision "month", and reading only the
    # start makes it look like the 1st. `event_precision` also carries the one
    # distinction nothing else does: "approximate" has no end, so "around
    # November 20" and "on November 20" differ only by this label.
    event_end: Optional[datetime] = None
    event_precision: Optional[str] = None
    valid_until: Optional[datetime] = None
    temporal_category: str  # "perpetual" | "temporal_fact" | "episode"
    temporal_confidence: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    source: str = ""
    extracted_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_evidence: Optional[List[str]] = None
    # The search queries of the fetch that found this item, in the order they
    # were sent. Empty when the fetch named no query, when the server predates
    # the field, and for an item served from the prefetch cache.
    matched_queries: List[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


# Keys the parsers below map onto typed fields. Everything else on a served
# item is passed through untouched, so a field the server adds is reachable as
# an attribute the day it ships.
#
# ⚠ This exists so that adding a served field does not force an SDK release.
# `event_end` and `event_precision` were computed by the server, dropped by its
# own payload, and then dropped again here: the parsers build each model from a
# hand-picked list of keys, so `extra: "allow"` on the model was never reached.
# A client could not read "sometime in April" even after the server sent it.
_CONSUMED_KEYS = frozenset({
    "id", "content", "confidence", "source", "extracted_at", "metadata",
    "event_date", "said_at", "valid_until", "temporal_category",
    "temporal_confidence", "source_evidence", "matched_queries", "strength",
    "category", "occurred_at", "participants", "emotion_type", "intensity",
    "detected_at", "context", "trigger", "valid_from",
    "event_end", "event_precision",
})


def _extras(item: Dict[str, Any], model: type) -> Dict[str, Any]:
    """Whatever the server sent that this SDK version has no field for.

    The model's own declared fields are excluded as well as the source keys
    above, which makes this self-maintaining: a hand-written exclusion list
    missed `significance` on Episode and the parser then received it twice.
    Any key the model already has a field for cannot collide, and a genuinely
    new server key still comes through.
    """
    if not isinstance(item, dict):
        return {}
    skip = _CONSUMED_KEYS | set(getattr(model, "model_fields", {}))
    return {k: v for k, v in item.items() if k not in skip}


# Response Metadata
class ResponseMetadata(BaseModel):
    """Metadata about a context response."""

    correlation_id: str
    ttl_seconds: int
    source: str  # "cache" or "cloud"
    retrieved_at: datetime
    compaction_applied: Optional[CompactionLevel] = None


class ConversationContextModel(BaseModel):
    """Current session context from compaction + raw buffer.

    Provides the narrative summary, current state, key extractions,
    and recent turns as a coherent block (not atomized into items).
    """
    summary: Optional[str] = None
    current_state: Dict[str, Any] = Field(default_factory=dict)
    key_extractions: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    recent_turns: List[Dict[str, Any]] = Field(default_factory=list)
    compaction_id: Optional[str] = None
    compacted_at: Optional[str] = None
    conversation_id: Optional[str] = None


# User Profile (C3) + Conversation Summary (C4) typed models
class ProfileAttributeModel(BaseModel):
    """A single critical-attribute value on a caller profile (spec §5.2).

    Typed + raw escape hatch; unknown fields tolerated.
    """

    value: Any = None
    confidence: Optional[float] = None
    updated_at: Optional[str] = None
    source_conversation_id: Optional[str] = None

    model_config = {"extra": "allow"}

    def __init__(self, **data):
        raw_data = data.pop("raw_data", data.copy())
        super().__init__(**data)
        object.__setattr__(self, "_raw_data", raw_data)

    @property
    def raw(self) -> Dict[str, Any]:
        return getattr(self, "_raw_data", {})

    @classmethod
    def from_cloud_response(cls, data: Any) -> "ProfileAttributeModel":
        if not isinstance(data, dict):
            return cls(value=data, raw_data={"value": data})
        return cls(
            value=data.get("value"),
            confidence=data.get("confidence"),
            updated_at=data.get("updated_at"),
            source_conversation_id=data.get("source_conversation_id"),
            raw_data=data,
        )


class UserProfileModel(BaseModel):
    """A caller profile document (spec §5.2): client-defined critical
    attributes + a free-text overview + stable extras.

    Typed access to ``attributes``/``overview``, plus a ``.raw`` escape hatch
    for the full document (including ``_meta``).
    """

    attributes: Dict[str, ProfileAttributeModel] = Field(default_factory=dict)
    overview: Optional[str] = None
    extras: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    def __init__(self, **data):
        raw_data = data.pop("raw_data", data.copy())
        super().__init__(**data)
        object.__setattr__(self, "_raw_data", raw_data)

    @property
    def raw(self) -> Dict[str, Any]:
        return getattr(self, "_raw_data", {})

    @classmethod
    def from_cloud_response(cls, data: Any) -> "UserProfileModel":
        """Build from the ``profile`` document.

        Tolerates both the flat document (``{attributes, overview, ...}``) and
        the storage-wrapped shape (``{"profile": {attributes, ...}}``).
        """
        if not isinstance(data, dict):
            return cls(raw_data={} if data is None else {"value": data})
        doc = data
        if "attributes" not in doc and isinstance(doc.get("profile"), dict):
            doc = doc["profile"]
        attributes: Dict[str, ProfileAttributeModel] = {}
        for name, attr in (doc.get("attributes") or {}).items():
            attributes[name] = ProfileAttributeModel.from_cloud_response(attr)
        return cls(
            attributes=attributes,
            overview=doc.get("overview"),
            extras=doc.get("extras") or {},
            meta=doc.get("_meta") or doc.get("meta") or {},
            raw_data=data,
        )


class ConversationSummaryModel(BaseModel):
    """A previous-conversation summary (spec §6.3 ConversationSummaryPayload).

    Returned by ``context_mode="conversation-summary"`` fetches: what a prior
    call was about and how it progressed.
    """

    conversation_id: str
    external_conversation_id: Optional[str] = None
    conversation_type: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    message_count: int = 0
    summary_status: str = "pending"  # available | pending | failed
    summary: Optional[Dict[str, Any]] = None
    classification: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    compaction_version: Optional[int] = None
    compacted_at: Optional[datetime] = None

    model_config = {"extra": "allow"}

    def __init__(self, **data):
        raw_data = data.pop("raw_data", data.copy())
        super().__init__(**data)
        object.__setattr__(self, "_raw_data", raw_data)

    @property
    def raw(self) -> Dict[str, Any]:
        return getattr(self, "_raw_data", {})

    @classmethod
    def from_cloud_response(cls, data: Dict[str, Any]) -> "ConversationSummaryModel":
        return cls(
            conversation_id=str(data.get("conversation_id", "")),
            external_conversation_id=data.get("external_conversation_id"),
            conversation_type=data.get("conversation_type"),
            started_at=_parse_iso_datetime(data.get("started_at")),
            ended_at=_parse_iso_datetime(data.get("ended_at")),
            last_message_at=_parse_iso_datetime(data.get("last_message_at")),
            message_count=int(data.get("message_count", 0) or 0),
            summary_status=data.get("summary_status", "pending"),
            summary=data.get("summary"),
            classification=data.get("classification"),
            analysis=data.get("analysis"),
            compaction_version=data.get("compaction_version"),
            compacted_at=_parse_iso_datetime(data.get("compacted_at")),
            raw_data=data,
        )

    def overview_text(self) -> Optional[str]:
        """Best-effort narrative overview from the ``summary`` payload."""
        if not isinstance(self.summary, dict):
            return None
        for key in ("narrative_summary", "overview", "narrative", "summary"):
            val = self.summary.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None

    def outcome_text(self) -> Optional[str]:
        """Best-effort outcome/classification label for prompt rendering."""
        if not isinstance(self.classification, dict):
            return None
        parts = [
            self.classification.get("primary_category"),
            self.classification.get("subcategory"),
            self.classification.get("objective"),
        ]
        joined = " / ".join(p for p in parts if isinstance(p, str) and p.strip())
        return joined or None


# Response Wrappers (Hybrid: typed + raw)
class RetrievalHealthModel(BaseModel):
    """Why a fetch may be thinner than the stores can support.

    `degraded` says something went wrong, `level` says how thin the answer is
    ("partial", "minimal", "failed"), and `code` says whether retrying is worth
    it ("timeout", "overloaded", "unavailable", "internal"). The server does not
    name which of its services broke, by design.

    Absent on a healthy fetch, so `response.retrieval is None` means fine.
    """

    degraded: bool = True
    level: Optional[str] = None
    code: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


# The two response shapes a context fetch can ask for (`response_shape`).
RESPONSE_SHAPES = ("by_type", "ranked")

# The memory types, in the order the by-type response lists them.
MEMORY_TYPES = ("facts", "preferences", "episodes", "emotions", "temporal_events")

# The type of a chunk in a ranked response: a quote from a source document,
# not an extracted memory. A by-type response files it under "facts".
EXCERPT_CATEGORY = "excerpts"


class RankedItem(BaseModel):
    """One item of a ranked response (`response_shape="ranked"`).

    The same fields for every type, which is what makes one list possible:
    `category` says which type it is, `rank` where it stands (1 is best), and
    `rank_score` the number it was ranked by. The typed views on the response
    (`facts`, `preferences`, ...) hold the same items as Fact, Preference and
    the rest, for code written against those.
    """

    id: str
    rank: int
    # "facts" | "preferences" | "episodes" | "emotions" | "temporal_events" |
    # "excerpts"
    category: str
    content: str
    # Higher is better; sorting on it descending reproduces the order. None
    # only for an item a server that predates ranking sent.
    rank_score: Optional[float] = None
    confidence: float = 0.0
    source: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    event_date: Optional[datetime] = None
    # The day the user SAID this. See the note on `Fact.said_at`.
    said_at: Optional[datetime] = None
    event_end: Optional[datetime] = None
    event_precision: Optional[str] = None
    valid_until: Optional[datetime] = None
    temporal_category: Optional[str] = None
    temporal_confidence: float = 0.0
    source_evidence: Optional[List[str]] = None
    matched_queries: List[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


# How each type is named on a line of the ranked prompt list
# (`UnifiedContextResponse.format_for_prompt`).
_RANKED_LABELS = {
    "facts": "fact",
    "preferences": "preference",
    "episodes": "episode",
    "emotions": "emotion",
    "temporal_events": "temporal event",
    EXCERPT_CATEGORY: "excerpt",
}


def _wire_rank_score(item: Dict[str, Any]) -> Optional[float]:
    """An item's rank score: `metadata.rank_score` on the HTTP response, a
    top-level `rank_score` on a prefetched (gRPC) item."""
    md = item.get("metadata")
    raw = md.get("rank_score") if isinstance(md, dict) else None
    if raw is None:
        raw = item.get("rank_score")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _wire_category(item: Dict[str, Any], fallback: str = "") -> str:
    """An item's type. A chunk is an excerpt whatever list it came in: a server
    or a prefetched bundle that predates ranking files it under facts."""
    if item.get("source") == "chunks":
        return EXCERPT_CATEGORY
    return item.get("category") or item.get("context_type") or fallback


def rank_items(by_type: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merge per-type item lists into one list in rank order, numbered from 1.

    For the cases where the server's own order is not available: a server that
    predates `response_shape` (and returned the five lists), and a prefetched
    cache hit. Sorted on rank score, highest first; items without one keep
    their relative order after the scored ones. Each returned dict is a copy
    carrying `rank` and `category`, so the input is not changed.
    """
    flat: List[Dict[str, Any]] = []
    for ctx_type in list(MEMORY_TYPES) + [t for t in by_type if t not in MEMORY_TYPES]:
        for it in by_type.get(ctx_type) or []:
            if isinstance(it, dict):
                # A prefetched item names its id `item_id`; the typed parsers
                # read `id`, and would otherwise build every view item with "".
                flat.append(dict(it, category=_wire_category(it, ctx_type),
                                 id=it.get("id") or it.get("item_id") or ""))
    flat.sort(key=lambda it: (_wire_rank_score(it) is None, -(_wire_rank_score(it) or 0.0)))
    for n, it in enumerate(flat, start=1):
        it["rank"] = n
    return flat


def _ranked_item_from(r: Dict[str, Any]) -> RankedItem:
    return RankedItem(
        id=r.get("id") or r.get("item_id") or "",
        rank=int(r.get("rank") or 0),
        category=_wire_category(r),
        content=r.get("content", ""),
        rank_score=_wire_rank_score(r),
        confidence=_coerce_float(r.get("confidence"), default=0.0),
        source=r.get("source", ""),
        metadata=r.get("metadata") or {},
        event_date=_parse_iso_datetime(r.get("event_date")),
        said_at=_parse_iso_datetime(r.get("said_at")),
        event_end=_parse_iso_datetime(r.get("event_end")),
        event_precision=r.get("event_precision"),
        valid_until=_parse_iso_datetime(r.get("valid_until")),
        temporal_category=r.get("temporal_category"),
        temporal_confidence=_coerce_float(r.get("temporal_confidence"), default=0.0),
        source_evidence=r.get("source_evidence"),
        matched_queries=list(r.get("matched_queries") or []),
        **{k: v for k, v in _extras(r, RankedItem).items()
           if k not in ("item_id", "context_type", "rank_score")},
    )


# One parser per memory type, shared by the by-type lists and by the views a
# ranked response builds (`ContextResponse.items`), so an item reads the same
# whichever shape it arrived in.
def _fact_from(f: Dict[str, Any]) -> Fact:
    return Fact(
        id=f.get("id", ""),
        content=f.get("content", ""),
        confidence=_coerce_float(f.get("confidence"), default=0.0),
        source=f.get("source", ""),
        extracted_at=f.get("extracted_at") or datetime.now(timezone.utc),
        metadata=f.get("metadata", {}),
        event_date=_parse_iso_datetime(f.get("event_date")),
        said_at=_parse_iso_datetime(f.get("said_at")),
        event_end=_parse_iso_datetime(f.get("event_end")),
        event_precision=f.get("event_precision"),
        valid_until=_parse_iso_datetime(f.get("valid_until")),
        temporal_category=f.get("temporal_category"),
        temporal_confidence=_coerce_float(f.get("temporal_confidence"), default=0.0),
        source_evidence=f.get("source_evidence"),
        matched_queries=list(f.get("matched_queries") or []),
        **_extras(f, Fact),
    )


def _preference_from(p: Dict[str, Any]) -> Preference:
    return Preference(
        id=p.get("id", ""),
        category=p.get("category", ""),
        content=p.get("content", ""),
        strength=_coerce_float(p.get("strength", p.get("confidence")), default=0.0),
        extracted_at=p.get("extracted_at") or datetime.now(timezone.utc),
        metadata=p.get("metadata", {}),
        event_date=_parse_iso_datetime(p.get("event_date")),
        said_at=_parse_iso_datetime(p.get("said_at")),
        event_end=_parse_iso_datetime(p.get("event_end")),
        event_precision=p.get("event_precision"),
        valid_until=_parse_iso_datetime(p.get("valid_until")),
        temporal_category=p.get("temporal_category"),
        temporal_confidence=_coerce_float(p.get("temporal_confidence"), default=0.0),
        source_evidence=p.get("source_evidence"),
        matched_queries=list(p.get("matched_queries") or []),
        **_extras(p, Preference),
    )


def _episode_from(e: Dict[str, Any]) -> Episode:
    return Episode(
        id=e.get("id", ""),
        summary=e.get("summary", e.get("content", "")),
        occurred_at=e.get("occurred_at") or e.get("extracted_at") or datetime.now(timezone.utc),
        significance=_coerce_float(e.get("significance", e.get("confidence")), default=0.0),
        participants=e.get("participants", []),
        metadata=e.get("metadata", {}),
        event_date=_parse_iso_datetime(e.get("event_date")),
        said_at=_parse_iso_datetime(e.get("said_at")),
        event_end=_parse_iso_datetime(e.get("event_end")),
        event_precision=e.get("event_precision"),
        valid_until=_parse_iso_datetime(e.get("valid_until")),
        temporal_category=e.get("temporal_category"),
        temporal_confidence=_coerce_float(e.get("temporal_confidence"), default=0.0),
        source_evidence=e.get("source_evidence"),
        matched_queries=list(e.get("matched_queries") or []),
        **_extras(e, Episode),
    )


def _emotion_from(em: Dict[str, Any]) -> Emotion:
    return Emotion(
        id=em.get("id", ""),
        emotion_type=em.get("emotion_type", em.get("category", "")),
        intensity=_coerce_float(em.get("intensity", em.get("confidence")), default=0.0),
        detected_at=em.get("detected_at") or em.get("extracted_at") or datetime.now(timezone.utc),
        context=em.get("context", em.get("content", "")),
        metadata=em.get("metadata", {}),
        event_date=_parse_iso_datetime(em.get("event_date")),
        said_at=_parse_iso_datetime(em.get("said_at")),
        event_end=_parse_iso_datetime(em.get("event_end")),
        event_precision=em.get("event_precision"),
        valid_until=_parse_iso_datetime(em.get("valid_until")),
        temporal_category=em.get("temporal_category"),
        temporal_confidence=_coerce_float(em.get("temporal_confidence"), default=0.0),
        source_evidence=em.get("source_evidence"),
        matched_queries=list(em.get("matched_queries") or []),
        **_extras(em, Emotion),
    )


def _temporal_event_from(t: Dict[str, Any]) -> TemporalEvent:
    return TemporalEvent(
        id=t.get("id", ""),
        content=t.get("content", ""),
        event_date=_parse_iso_datetime(t.get("event_date")),
        said_at=_parse_iso_datetime(t.get("said_at")),
        event_end=_parse_iso_datetime(t.get("event_end")),
        event_precision=t.get("event_precision"),
        valid_until=_parse_iso_datetime(t.get("valid_until")),
        temporal_category=t.get("temporal_category", ""),
        temporal_confidence=_coerce_float(t.get("temporal_confidence"), default=0.0),
        confidence=_coerce_float(t.get("confidence"), default=0.0),
        source=t.get("source", ""),
        extracted_at=_parse_iso_datetime(t.get("extracted_at")),
        metadata=t.get("metadata", {}),
        source_evidence=t.get("source_evidence"),
        matched_queries=list(t.get("matched_queries") or []),
        **_extras(t, TemporalEvent),
    )



class ContextResponse(BaseModel):
    """Response from context fetch operations.

    Provides typed access to common fields and raw escape hatch.
    """

    facts: List[Fact] = Field(default_factory=list)
    preferences: List[Preference] = Field(default_factory=list)
    episodes: List[Episode] = Field(default_factory=list)
    emotions: List[Emotion] = Field(default_factory=list)
    temporal_events: List[TemporalEvent] = Field(default_factory=list)
    # `response_shape="ranked"` only: every item once, best first, across
    # types. Empty for a by-type fetch, which is the default. In a ranked
    # response the five lists above are views of these same items, by type.
    items: List[RankedItem] = Field(default_factory=list)
    # `response_shape="ranked"` only: the chunks, quotes from source documents.
    # A by-type response files them under `facts` instead.
    excerpts: List[Fact] = Field(default_factory=list)
    conversation_context: Optional[ConversationContextModel] = None
    # C4 conversation-summary sections (populated only in summary mode).
    profile: Optional["UserProfileModel"] = None
    conversations: Optional[List["ConversationSummaryModel"]] = None
    metadata: ResponseMetadata
    # Why this response may be thinner than the stores can support. None on a
    # healthy fetch.
    #
    # ⚠ This had no typed slot, and that is how a real failure stayed invisible.
    # On 2026-10-12 a retrieval lost its embedding call, the server returned 200
    # with ZERO items for a user holding 122 memories, and said so in the
    # response's `retrieval` block. The server-side field was added after an
    # identical incident in September precisely so a client could see it. But
    # the factory below built this model from a hand-picked list of keys, so
    # `extra: "allow"` was never reached and the block was dropped. A caller
    # could not tell "this user has no relevant memories" from "our retrieval
    # broke", and scored the question as a legitimate blank.
    retrieval: Optional[RetrievalHealthModel] = None

    # Raw response for forward compatibility
    model_config = {"extra": "allow"}
    
    # Private storage for raw data (not a field)
    def __init__(self, **data):
        # Extract raw data before calling parent init
        raw_data = data.pop("raw_data", data.copy())
        super().__init__(**data)
        object.__setattr__(self, "_raw_data", raw_data)

    @property
    def raw(self) -> Dict[str, Any]:
        """Access raw response for fields not yet in typed model."""
        return getattr(self, "_raw_data", {})

    @classmethod
    def from_cloud_response(
        cls, data: Dict[str, Any], metadata: ResponseMetadata,
        response_shape: str = "by_type",
    ) -> "ContextResponse":
        """Factory to create from cloud API response.

        Maps generic ContextItem fields from the cloud API to type-specific
        SDK model fields (e.g. confidence -> strength for Preference).

        ``response_shape`` is what the caller asked for. A ranked response
        carries ``items``; the typed lists are then built from it. When
        "ranked" was asked for and the data has no ``items`` (a server or a
        cache entry that predates the parameter), the five lists are merged
        into ``items`` here by rank score, so the caller gets the shape it
        asked for either way.
        """
        raw_source = data
        ranked_wire = data.get("items")
        if response_shape == "ranked" or isinstance(ranked_wire, list):
            if not isinstance(ranked_wire, list):
                ranked_wire = rank_items({t: data.get(t) or [] for t in MEMORY_TYPES})
            data = dict(data)
            for t in MEMORY_TYPES:
                data[t] = [it for it in ranked_wire if _wire_category(it) == t]
            ranked = [_ranked_item_from(it) for it in ranked_wire]
            excerpts = [_fact_from(it) for it in ranked_wire
                        if _wire_category(it) == EXCERPT_CATEGORY]
        else:
            ranked, excerpts = [], []
        facts = [_fact_from(f) for f in data.get("facts", [])]
        preferences = [_preference_from(p) for p in data.get("preferences", [])]
        episodes = [_episode_from(e) for e in data.get("episodes", [])]
        emotions = [_emotion_from(em) for em in data.get("emotions", [])]
        temporal_events = [_temporal_event_from(t) for t in data.get("temporal_events", [])]

        conv_ctx = None
        conv_ctx_data = data.get("conversation_context")
        if conv_ctx_data and isinstance(conv_ctx_data, dict):
            conv_ctx = ConversationContextModel(**conv_ctx_data)

        # C4 conversation-summary siblings (spec §6.3). Built via explicit
        # factories — never model_validate on the wire dict — so an older SDK
        # tolerates a newer server. Absent keys leave these None (regression:
        # a normal fetch is unchanged).
        profile = None
        prof_data = data.get("profile")
        if isinstance(prof_data, dict):
            profile = UserProfileModel.from_cloud_response(prof_data)

        conversations = None
        conv_list = data.get("conversations")
        if isinstance(conv_list, list):
            conversations = [
                ConversationSummaryModel.from_cloud_response(c)
                for c in conv_list
                if isinstance(c, dict)
            ]

        return cls(
            facts=facts,
            preferences=preferences,
            episodes=episodes,
            emotions=emotions,
            temporal_events=temporal_events,
            items=ranked,
            excerpts=excerpts,
            conversation_context=conv_ctx,
            profile=profile,
            conversations=conversations,
            metadata=metadata,
            retrieval=(RetrievalHealthModel(**data["retrieval"])
                       if isinstance(data.get("retrieval"), dict) else None),
            raw_data=raw_source,
        )


class CompactionResponse(BaseModel):
    """Response from context compaction."""

    compacted_context: str
    original_token_count: int
    compacted_token_count: int
    compression_ratio: float
    level_applied: CompactionLevel
    metadata: ResponseMetadata

    compaction_id: Optional[str] = None
    strategy_used: Optional[str] = None
    validation_score: Optional[float] = None
    validation_passed: Optional[bool] = None
    facts: List[Dict[str, Any]] = Field(default_factory=list)
    decisions: List[Dict[str, Any]] = Field(default_factory=list)
    preferences: List[Dict[str, Any]] = Field(default_factory=list)
    current_state: Optional[Dict[str, Any]] = None
    quality_warning: Optional[bool] = None

    model_config = {"extra": "allow"}
    
    def __init__(self, **data):
        raw_data = data.pop("raw_data", data.copy())
        super().__init__(**data)
        object.__setattr__(self, "_raw_data", raw_data)

    @property
    def raw(self) -> Dict[str, Any]:
        return getattr(self, "_raw_data", {})


class CompactionTriggerResponse(BaseModel):
    """Response from triggering an async compaction."""

    compaction_id: str
    conversation_id: str
    status: str
    trigger_type: str
    initiated_at: datetime
    estimated_completion_seconds: Optional[int] = 60

    previous_context: Optional[Dict[str, Any]] = None
    previous_context_age_seconds: Optional[int] = None
    previous_compaction_id: Optional[str] = None


class CompactionStatusResponse(BaseModel):
    """Response from checking compaction status."""

    conversation_id: str
    status: str
    compaction_id: Optional[str] = None
    completed_at: Optional[datetime] = None
    compression_ratio: Optional[float] = None
    validation_score: Optional[float] = None
    estimated_completion_seconds: Optional[int] = None
    error_message: Optional[str] = None
    latest_version: Optional[int] = None
    latest_created_at: Optional[datetime] = None


class RecentMessage(BaseModel):
    """A recent un-compacted conversation message."""
    role: str
    content: str
    timestamp: datetime
    message_id: str


class HistoryExcerpt(BaseModel):
    """An earlier turn, verbatim or cut to its matching sentences, found by
    the local history search."""

    position: Optional[int] = None  # the turn number, when known
    role: str
    text: str


class ContextForPromptResponse(BaseModel):
    """Response from get_context_for_prompt()."""

    formatted_context: Optional[str] = None
    available: bool = False
    is_stale: bool = False
    compression_ratio: Optional[float] = None
    validation_score: Optional[float] = None
    compaction_age_seconds: Optional[int] = None
    quality_warning: bool = False

    recent_messages: List[RecentMessage] = []
    recent_message_count: int = 0
    compacted_message_count: int = 0
    total_message_count: int = 0

    # The framed compaction block alone (it is also inside formatted_context),
    # for a framework that keeps its own message list: put it in a system
    # message ahead of recent_messages, in place of the turns it covers.
    context_block: Optional[str] = None
    compaction_id: Optional[str] = None
    # Earlier turns matched to the latest user message by the local search,
    # also appended to formatted_context.
    history_excerpts: List[HistoryExcerpt] = []
    # The same excerpts rendered as appended to formatted_context, for a
    # framework that puts context_block in a system message.
    history_block: Optional[str] = None


# Unified cross-scope response

class UnifiedContextResponse(BaseModel):
    """Result of a cross-scope fetch via sdk.fetch().

    Contains all memory items from all queried scopes, deduplicated
    by item ID, with scope attribution and a ready-to-use formatted
    string for LLM prompt injection.
    """

    facts: List[Fact] = Field(default_factory=list)
    preferences: List[Preference] = Field(default_factory=list)
    episodes: List[Episode] = Field(default_factory=list)
    emotions: List[Emotion] = Field(default_factory=list)
    temporal_events: List[TemporalEvent] = Field(default_factory=list)
    # `response_shape="ranked"` only: every item once, best first across
    # scopes and types, renumbered from 1. Empty for a by-type fetch.
    items: List[RankedItem] = Field(default_factory=list)
    # `response_shape="ranked"` only: the chunks (quotes from source documents).
    excerpts: List[Fact] = Field(default_factory=list)

    scope_map: Dict[str, str] = Field(default_factory=dict)

    conversation_context: Optional[ContextForPromptResponse] = None

    # C4 conversation-summary sections, carried from the user-scope sub-fetch.
    profile: Optional["UserProfileModel"] = None
    conversations: Optional[List["ConversationSummaryModel"]] = None

    formatted_context: Optional[str] = None

    # Metadata
    scopes_queried: List[str] = Field(default_factory=list)
    total_items: int = 0
    metadata: Optional[ResponseMetadata] = None

    @staticmethod
    def merge(
        scope_results: List[Tuple[str, "ContextResponse"]],
    ) -> "UnifiedContextResponse":
        """Merge multiple scope results, deduplicating by item ID.

        Items are kept in order of scope_results (first scope wins on
        duplicate IDs). Each item is attributed to its source scope
        in scope_map.

        Ranked responses (``response_shape="ranked"``) also have their
        ``items`` merged into one list ordered by ``rank_score``. Scores
        compare across scopes: each item is scored on its own against the
        same queries, not relative to the rest of its scope's batch.

        Args:
            scope_results: List of (scope_name, ContextResponse) tuples.
                           Failed scopes should be filtered out before calling.

        Returns:
            UnifiedContextResponse with merged, deduplicated items.
        """
        seen_ids: set = set()
        facts: List[Fact] = []
        preferences: List[Preference] = []
        episodes: List[Episode] = []
        emotions: List[Emotion] = []
        temporal_events: List[TemporalEvent] = []
        excerpts: List[Fact] = []
        ranked: List[RankedItem] = []
        ranked_ids: set = set()
        scope_map: Dict[str, str] = {}
        scopes_queried: List[str] = []
        first_metadata: Optional[ResponseMetadata] = None
        profile: Optional["UserProfileModel"] = None
        conversations: Optional[List["ConversationSummaryModel"]] = None

        for scope_name, response in scope_results:
            scopes_queried.append(scope_name)
            if first_metadata is None:
                first_metadata = response.metadata

            # Carry C4 summary sections up from whichever scope produced them
            # (the user-scope sub-fetch — see the sdk.fetch fan-out rule).
            if profile is None and getattr(response, "profile", None) is not None:
                profile = response.profile
            if conversations is None and getattr(response, "conversations", None) is not None:
                conversations = response.conversations

            for fact in response.facts:
                if fact.id not in seen_ids:
                    seen_ids.add(fact.id)
                    facts.append(fact)
                    scope_map[fact.id] = scope_name

            for pref in response.preferences:
                if pref.id not in seen_ids:
                    seen_ids.add(pref.id)
                    preferences.append(pref)
                    scope_map[pref.id] = scope_name

            for ep in response.episodes:
                if ep.id not in seen_ids:
                    seen_ids.add(ep.id)
                    episodes.append(ep)
                    scope_map[ep.id] = scope_name

            for em in response.emotions:
                if em.id not in seen_ids:
                    seen_ids.add(em.id)
                    emotions.append(em)
                    scope_map[em.id] = scope_name

            for te in response.temporal_events:
                if te.id not in seen_ids:
                    seen_ids.add(te.id)
                    temporal_events.append(te)
                    scope_map[te.id] = scope_name

            for ex in response.excerpts:
                if ex.id not in seen_ids:
                    seen_ids.add(ex.id)
                    excerpts.append(ex)
                    scope_map[ex.id] = scope_name

            for it in response.items:
                if it.id not in ranked_ids:
                    ranked_ids.add(it.id)
                    ranked.append(it)

        # One order across scopes. Python's sort is stable, so equal scores keep
        # scope order (the first scope wins, as it does for duplicates above),
        # and items without a score go last in the order they came.
        ranked.sort(key=lambda it: (it.rank_score is None, -(it.rank_score or 0.0)))
        ranked = [it.model_copy(update={"rank": n}) for n, it in enumerate(ranked, start=1)]

        total = (len(facts) + len(preferences) + len(episodes) + len(emotions)
                 + len(temporal_events) + len(excerpts))

        return UnifiedContextResponse(
            facts=facts,
            preferences=preferences,
            episodes=episodes,
            emotions=emotions,
            temporal_events=temporal_events,
            items=ranked,
            excerpts=excerpts,
            scope_map=scope_map,
            scopes_queried=scopes_queried,
            total_items=total,
            metadata=first_metadata,
            profile=profile,
            conversations=conversations,
        )

    def format_for_prompt(
        self,
        include_scope: bool = False,
        include_conversation_context: bool = True,
    ) -> str:
        """Format all context into a string suitable for LLM system prompt injection.

        Args:
            include_scope: If True, annotate each item with its source scope.
            include_conversation_context: If True, include compacted history + recent messages.

        Returns:
            Formatted string ready for LLM prompt injection.
        """
        sections: List[str] = []

        # A ranked response is printed as one numbered list in rank order, which
        # is the point of asking for it: the typed sections below would split
        # the order back up by type. A by-type response prints as it always did.
        if self.items:
            sections.append(self._format_ranked_items(include_scope))
        typed = not self.items

        if typed and self.facts:
            lines = []
            for f in self.facts:
                line = f"- {f.content}"
                if include_scope:
                    line += f" [scope: {self.scope_map.get(f.id, 'unknown')}]"
                lines.append(line)
            sections.append("### Facts\n" + "\n".join(lines))

        if typed and self.preferences:
            lines = []
            for p in self.preferences:
                line = f"- {p.content}"
                extras = []
                if include_scope:
                    extras.append(f"scope: {self.scope_map.get(p.id, 'unknown')}")
                if p.strength and p.strength >= 0.7:
                    extras.append(f"strength: {p.strength:.1f}")
                if extras:
                    line += f" [{', '.join(extras)}]"
                lines.append(line)
            sections.append("### Preferences\n" + "\n".join(lines))

        if typed and self.episodes:
            lines = []
            for e in self.episodes:
                line = f"- {e.summary}"
                if include_scope:
                    line += f" [scope: {self.scope_map.get(e.id, 'unknown')}]"
                lines.append(line)
            sections.append("### Episodes\n" + "\n".join(lines))

        if typed and self.emotions:
            lines = []
            for em in self.emotions:
                line = f"- {em.emotion_type}: {em.context}"
                extras = []
                if include_scope:
                    extras.append(f"scope: {self.scope_map.get(em.id, 'unknown')}")
                if em.intensity and em.intensity >= 0.5:
                    extras.append(f"intensity: {em.intensity:.1f}")
                if extras:
                    line += f" [{', '.join(extras)}]"
                lines.append(line)
            sections.append("### Emotions\n" + "\n".join(lines))

        if typed and self.temporal_events:
            lines = []
            for te in self.temporal_events:
                line = f"- {te.content}"
                extras = []
                if include_scope:
                    extras.append(f"scope: {self.scope_map.get(te.id, 'unknown')}")
                if te.valid_until:
                    extras.append(f"valid until: {te.valid_until.strftime('%Y-%m-%d')}")
                if extras:
                    line += f" [{', '.join(extras)}]"
                lines.append(line)
            sections.append("### Temporal Events\n" + "\n".join(lines))

        # Add conversation context if available
        if include_conversation_context and self.conversation_context:
            if self.conversation_context.formatted_context:
                sections.append(
                    "### Conversation History\n" + self.conversation_context.formatted_context
                )

        # Assemble top-level blocks. The "## User Context" block is emitted
        # byte-for-byte as before; the C4 sections append after it only when
        # present, so output is unchanged whenever profile/conversations are
        # absent (regression-tested).
        blocks: List[str] = []
        if sections:
            blocks.append("## User Context\n" + "\n\n".join(sections))

        profile_block = self._format_profile_block()
        if profile_block:
            blocks.append(profile_block)

        conversations_block = self._format_conversations_block()
        if conversations_block:
            blocks.append(conversations_block)

        if not blocks:
            return ""

        return "\n\n".join(blocks)

    def _format_ranked_items(self, include_scope: bool) -> str:
        """``### Memories`` as one numbered list, best first.

        One line per item: ``1. [fact] content``. The type is on the line because
        the list mixes types, and an excerpt is a quote rather than an extracted
        memory. With ``include_scope`` the scope follows the type:
        ``1. [fact, scope: user] content``.
        """
        lines = []
        for n, it in enumerate(self.items, start=1):
            label = _RANKED_LABELS.get(it.category, it.category or "item")
            if include_scope:
                label += f", scope: {self.scope_map.get(it.id, 'unknown')}"
            lines.append(f"{n}. [{label}] {it.content}")
        return "### Memories\n" + "\n".join(lines)

    def _format_profile_block(self) -> Optional[str]:
        """Render the ``## Caller Profile`` section, or None when empty."""
        profile = self.profile
        if profile is None:
            return None
        lines: List[str] = []
        for name, attr in (profile.attributes or {}).items():
            value = attr.value if attr is not None else None
            if isinstance(value, (list, tuple)):
                value_str = ", ".join(str(v) for v in value)
            elif value is None:
                value_str = ""
            else:
                value_str = str(value)
            lines.append(f"- {name}: {value_str}".rstrip())
        body_parts: List[str] = []
        if lines:
            body_parts.append("\n".join(lines))
        if profile.overview:
            body_parts.append(profile.overview)
        if not body_parts:
            return None
        return "## Caller Profile\n" + "\n\n".join(body_parts)

    def _format_conversations_block(self) -> Optional[str]:
        """Render the ``## Previous Conversations`` section, or None."""
        conversations = self.conversations
        if not conversations:
            return None
        call_blocks: List[str] = []
        for conv in conversations:
            when = conv.started_at or conv.last_message_at or conv.ended_at
            when_str = when.strftime("%Y-%m-%d") if when else "unknown date"
            header = f"### Call on {when_str}"
            if conv.conversation_type:
                header += f" ({conv.conversation_type})"
            call_lines = [header]
            overview = conv.overview_text()
            if overview:
                call_lines.append(f"Overview: {overview}")
            outcome = conv.outcome_text()
            if outcome:
                call_lines.append(f"Outcome: {outcome}")
            if overview is None and outcome is None:
                # No compaction yet / failed — still tell the model the call
                # happened so it can acknowledge the prior contact.
                call_lines.append(f"Status: {conv.summary_status}")
            call_blocks.append("\n".join(call_lines))
        return "## Previous Conversations\n" + "\n\n".join(call_blocks)


# Backward compatibility - deprecated dataclass-style models
class ContextItem:
    """Individual context item within a bundle.

    DEPRECATED: Use Fact, Preference, Episode, or Emotion instead.
    """

    def __init__(
        self,
        context_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        confidence_score: float = 1.0,
        timestamp: Optional[datetime] = None,
    ):
        self.context_type = context_type
        self.content = content
        self.metadata = metadata or {}
        self.confidence_score = confidence_score
        self.timestamp = timestamp


class ContextBundle:
    """Bundle of context items returned by the SDK.

    DEPRECATED: Use ContextResponse instead.
    """

    def __init__(
        self,
        items: Optional[List[ContextItem]] = None,
        total_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ):
        self.items = items or []
        self.total_count = total_count
        self.metadata = metadata or {}
        self.correlation_id = correlation_id
