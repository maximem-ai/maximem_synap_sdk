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
    valid_until: Optional[datetime] = None
    temporal_category: Optional[str] = None
    temporal_confidence: float = 0.0
    source_evidence: Optional[List[str]] = None


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
    valid_until: Optional[datetime] = None
    temporal_category: Optional[str] = None
    temporal_confidence: float = 0.0
    source_evidence: Optional[List[str]] = None


class Episode(BaseModel):
    """A memorable event or interaction."""

    id: str
    summary: str
    occurred_at: datetime
    significance: float = Field(ge=0.0, le=1.0)
    participants: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    event_date: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    temporal_category: Optional[str] = None
    temporal_confidence: float = 0.0
    source_evidence: Optional[List[str]] = None


class Emotion(BaseModel):
    """Detected emotional state."""

    id: str
    emotion_type: str  # e.g., "frustrated", "satisfied", "confused"
    intensity: float = Field(ge=0.0, le=1.0)
    detected_at: datetime
    context: str  # What triggered this
    metadata: Dict[str, Any] = Field(default_factory=dict)
    event_date: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    temporal_category: Optional[str] = None
    temporal_confidence: float = 0.0
    source_evidence: Optional[List[str]] = None


class TemporalEvent(BaseModel):
    """A time-bound event or fact with explicit temporal boundaries."""

    id: str
    content: str
    event_date: datetime
    valid_until: Optional[datetime] = None
    temporal_category: str  # "perpetual" | "temporal_fact" | "episode"
    temporal_confidence: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    source: str = ""
    extracted_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_evidence: Optional[List[str]] = None


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
class ContextResponse(BaseModel):
    """Response from context fetch operations.

    Provides typed access to common fields and raw escape hatch.
    """

    facts: List[Fact] = Field(default_factory=list)
    preferences: List[Preference] = Field(default_factory=list)
    episodes: List[Episode] = Field(default_factory=list)
    emotions: List[Emotion] = Field(default_factory=list)
    temporal_events: List[TemporalEvent] = Field(default_factory=list)
    conversation_context: Optional[ConversationContextModel] = None
    # C4 conversation-summary sections (populated only in summary mode).
    profile: Optional["UserProfileModel"] = None
    conversations: Optional[List["ConversationSummaryModel"]] = None
    metadata: ResponseMetadata

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
        cls, data: Dict[str, Any], metadata: ResponseMetadata
    ) -> "ContextResponse":
        """Factory to create from cloud API response.

        Maps generic ContextItem fields from the cloud API to type-specific
        SDK model fields (e.g. confidence -> strength for Preference).
        """
        facts = []
        for f in data.get("facts", []):
            facts.append(Fact(
                id=f.get("id", ""),
                content=f.get("content", ""),
                confidence=_coerce_float(f.get("confidence"), default=0.0),
                source=f.get("source", ""),
                extracted_at=f.get("extracted_at") or datetime.now(timezone.utc),
                metadata=f.get("metadata", {}),
                event_date=_parse_iso_datetime(f.get("event_date")),
                valid_until=_parse_iso_datetime(f.get("valid_until")),
                temporal_category=f.get("temporal_category"),
                temporal_confidence=_coerce_float(f.get("temporal_confidence"), default=0.0),
                source_evidence=f.get("source_evidence"),
            ))

        preferences = []
        for p in data.get("preferences", []):
            preferences.append(Preference(
                id=p.get("id", ""),
                category=p.get("category", ""),
                content=p.get("content", ""),
                strength=_coerce_float(p.get("strength", p.get("confidence")), default=0.0),
                extracted_at=p.get("extracted_at") or datetime.now(timezone.utc),
                metadata=p.get("metadata", {}),
                event_date=_parse_iso_datetime(p.get("event_date")),
                valid_until=_parse_iso_datetime(p.get("valid_until")),
                temporal_category=p.get("temporal_category"),
                temporal_confidence=_coerce_float(p.get("temporal_confidence"), default=0.0),
                source_evidence=p.get("source_evidence"),
            ))

        episodes = []
        for e in data.get("episodes", []):
            episodes.append(Episode(
                id=e.get("id", ""),
                summary=e.get("summary", e.get("content", "")),
                occurred_at=e.get("occurred_at") or e.get("extracted_at") or datetime.now(timezone.utc),
                significance=_coerce_float(e.get("significance", e.get("confidence")), default=0.0),
                participants=e.get("participants", []),
                metadata=e.get("metadata", {}),
                event_date=_parse_iso_datetime(e.get("event_date")),
                valid_until=_parse_iso_datetime(e.get("valid_until")),
                temporal_category=e.get("temporal_category"),
                temporal_confidence=_coerce_float(e.get("temporal_confidence"), default=0.0),
                source_evidence=e.get("source_evidence"),
            ))

        emotions = []
        for em in data.get("emotions", []):
            emotions.append(Emotion(
                id=em.get("id", ""),
                emotion_type=em.get("emotion_type", em.get("category", "")),
                intensity=_coerce_float(em.get("intensity", em.get("confidence")), default=0.0),
                detected_at=em.get("detected_at") or em.get("extracted_at") or datetime.now(timezone.utc),
                context=em.get("context", em.get("content", "")),
                metadata=em.get("metadata", {}),
                event_date=_parse_iso_datetime(em.get("event_date")),
                valid_until=_parse_iso_datetime(em.get("valid_until")),
                temporal_category=em.get("temporal_category"),
                temporal_confidence=_coerce_float(em.get("temporal_confidence"), default=0.0),
                source_evidence=em.get("source_evidence"),
            ))

        temporal_events = []
        for t in data.get("temporal_events", []):
            evt_date = _parse_iso_datetime(t.get("event_date"))
            if evt_date is None:
                continue
            temporal_events.append(TemporalEvent(
                id=t.get("id", ""),
                content=t.get("content", ""),
                event_date=evt_date,
                valid_until=_parse_iso_datetime(t.get("valid_until")),
                temporal_category=t.get("temporal_category", ""),
                temporal_confidence=_coerce_float(t.get("temporal_confidence"), default=0.0),
                confidence=_coerce_float(t.get("confidence"), default=0.0),
                source=t.get("source", ""),
                extracted_at=_parse_iso_datetime(t.get("extracted_at")),
                metadata=t.get("metadata", {}),
                source_evidence=t.get("source_evidence"),
            ))

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
            conversation_context=conv_ctx,
            profile=profile,
            conversations=conversations,
            metadata=metadata,
            raw_data=data,
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

        total = len(facts) + len(preferences) + len(episodes) + len(emotions) + len(temporal_events)

        return UnifiedContextResponse(
            facts=facts,
            preferences=preferences,
            episodes=episodes,
            emotions=emotions,
            temporal_events=temporal_events,
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

        if self.facts:
            lines = []
            for f in self.facts:
                line = f"- {f.content}"
                if include_scope:
                    line += f" [scope: {self.scope_map.get(f.id, 'unknown')}]"
                lines.append(line)
            sections.append("### Facts\n" + "\n".join(lines))

        if self.preferences:
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

        if self.episodes:
            lines = []
            for e in self.episodes:
                line = f"- {e.summary}"
                if include_scope:
                    line += f" [scope: {self.scope_map.get(e.id, 'unknown')}]"
                lines.append(line)
            sections.append("### Episodes\n" + "\n".join(lines))

        if self.emotions:
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

        if self.temporal_events:
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
