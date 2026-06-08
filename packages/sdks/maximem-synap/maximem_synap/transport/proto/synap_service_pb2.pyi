from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StreamEvent(_message.Message):
    __slots__ = ("conversation_event", "heartbeat_ping", "session_control", "context_used", "context_assembled")
    CONVERSATION_EVENT_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_PING_FIELD_NUMBER: _ClassVar[int]
    SESSION_CONTROL_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_USED_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_ASSEMBLED_FIELD_NUMBER: _ClassVar[int]
    conversation_event: ConversationEvent
    heartbeat_ping: HeartbeatPing
    session_control: SessionControl
    context_used: ContextUsedEvent
    context_assembled: ContextAssembledEvent
    def __init__(self, conversation_event: _Optional[_Union[ConversationEvent, _Mapping]] = ..., heartbeat_ping: _Optional[_Union[HeartbeatPing, _Mapping]] = ..., session_control: _Optional[_Union[SessionControl, _Mapping]] = ..., context_used: _Optional[_Union[ContextUsedEvent, _Mapping]] = ..., context_assembled: _Optional[_Union[ContextAssembledEvent, _Mapping]] = ...) -> None: ...

class ContextUsedEvent(_message.Message):
    __slots__ = ("bundle_id", "conversation_id", "user_id", "customer_id", "served_item_ids", "timestamp_ms", "scope", "source_bundle_ids")
    BUNDLE_ID_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    CUSTOMER_ID_FIELD_NUMBER: _ClassVar[int]
    SERVED_ITEM_IDS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_BUNDLE_IDS_FIELD_NUMBER: _ClassVar[int]
    bundle_id: str
    conversation_id: str
    user_id: str
    customer_id: str
    served_item_ids: _containers.RepeatedScalarFieldContainer[str]
    timestamp_ms: int
    scope: str
    source_bundle_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, bundle_id: _Optional[str] = ..., conversation_id: _Optional[str] = ..., user_id: _Optional[str] = ..., customer_id: _Optional[str] = ..., served_item_ids: _Optional[_Iterable[str]] = ..., timestamp_ms: _Optional[int] = ..., scope: _Optional[str] = ..., source_bundle_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class ContextAssembledEvent(_message.Message):
    __slots__ = ("correlation_id", "conversation_id", "user_id", "customer_id", "final_item_ids", "final_total_tokens", "compaction_id", "recent_turn_count", "compaction_end_timestamp", "assembly_source", "assembly_duration_ms", "cache_hit", "timestamp_ms", "sdk_version")
    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    CUSTOMER_ID_FIELD_NUMBER: _ClassVar[int]
    FINAL_ITEM_IDS_FIELD_NUMBER: _ClassVar[int]
    FINAL_TOTAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPACTION_ID_FIELD_NUMBER: _ClassVar[int]
    RECENT_TURN_COUNT_FIELD_NUMBER: _ClassVar[int]
    COMPACTION_END_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    ASSEMBLY_SOURCE_FIELD_NUMBER: _ClassVar[int]
    ASSEMBLY_DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    CACHE_HIT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    SDK_VERSION_FIELD_NUMBER: _ClassVar[int]
    correlation_id: str
    conversation_id: str
    user_id: str
    customer_id: str
    final_item_ids: _containers.RepeatedScalarFieldContainer[str]
    final_total_tokens: int
    compaction_id: str
    recent_turn_count: int
    compaction_end_timestamp: str
    assembly_source: str
    assembly_duration_ms: int
    cache_hit: bool
    timestamp_ms: int
    sdk_version: str
    def __init__(self, correlation_id: _Optional[str] = ..., conversation_id: _Optional[str] = ..., user_id: _Optional[str] = ..., customer_id: _Optional[str] = ..., final_item_ids: _Optional[_Iterable[str]] = ..., final_total_tokens: _Optional[int] = ..., compaction_id: _Optional[str] = ..., recent_turn_count: _Optional[int] = ..., compaction_end_timestamp: _Optional[str] = ..., assembly_source: _Optional[str] = ..., assembly_duration_ms: _Optional[int] = ..., cache_hit: bool = ..., timestamp_ms: _Optional[int] = ..., sdk_version: _Optional[str] = ...) -> None: ...

class ConversationEvent(_message.Message):
    __slots__ = ("event_type", "conversation_id", "user_id", "role", "content", "customer_id", "session_id", "metadata", "timestamp_ms", "tool_name", "tool_args_json", "search_queries", "context_types")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    CUSTOMER_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    TOOL_NAME_FIELD_NUMBER: _ClassVar[int]
    TOOL_ARGS_JSON_FIELD_NUMBER: _ClassVar[int]
    SEARCH_QUERIES_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_TYPES_FIELD_NUMBER: _ClassVar[int]
    event_type: str
    conversation_id: str
    user_id: str
    role: str
    content: str
    customer_id: str
    session_id: str
    metadata: _containers.ScalarMap[str, str]
    timestamp_ms: int
    tool_name: str
    tool_args_json: str
    search_queries: _containers.RepeatedScalarFieldContainer[str]
    context_types: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, event_type: _Optional[str] = ..., conversation_id: _Optional[str] = ..., user_id: _Optional[str] = ..., role: _Optional[str] = ..., content: _Optional[str] = ..., customer_id: _Optional[str] = ..., session_id: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ..., timestamp_ms: _Optional[int] = ..., tool_name: _Optional[str] = ..., tool_args_json: _Optional[str] = ..., search_queries: _Optional[_Iterable[str]] = ..., context_types: _Optional[_Iterable[str]] = ...) -> None: ...

class HeartbeatPing(_message.Message):
    __slots__ = ("timestamp_ms",)
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    timestamp_ms: int
    def __init__(self, timestamp_ms: _Optional[int] = ...) -> None: ...

class SessionControl(_message.Message):
    __slots__ = ("action", "session_id", "conversation_id", "user_id", "customer_id")
    ACTION_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    CUSTOMER_ID_FIELD_NUMBER: _ClassVar[int]
    action: str
    session_id: str
    conversation_id: str
    user_id: str
    customer_id: str
    def __init__(self, action: _Optional[str] = ..., session_id: _Optional[str] = ..., conversation_id: _Optional[str] = ..., user_id: _Optional[str] = ..., customer_id: _Optional[str] = ...) -> None: ...

class StreamResponse(_message.Message):
    __slots__ = ("context_bundle", "heartbeat_pong", "signal")
    CONTEXT_BUNDLE_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_PONG_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_FIELD_NUMBER: _ClassVar[int]
    context_bundle: ContextBundleProto
    heartbeat_pong: HeartbeatPong
    signal: StreamSignal
    def __init__(self, context_bundle: _Optional[_Union[ContextBundleProto, _Mapping]] = ..., heartbeat_pong: _Optional[_Union[HeartbeatPong, _Mapping]] = ..., signal: _Optional[_Union[StreamSignal, _Mapping]] = ...) -> None: ...

class HeartbeatPong(_message.Message):
    __slots__ = ("timestamp_ms",)
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    timestamp_ms: int
    def __init__(self, timestamp_ms: _Optional[int] = ...) -> None: ...

class StreamSignal(_message.Message):
    __slots__ = ("signal_type", "reason", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SIGNAL_TYPE_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    signal_type: str
    reason: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, signal_type: _Optional[str] = ..., reason: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ContextBundleProto(_message.Message):
    __slots__ = ("bundle_id", "decision_id", "items_by_type", "total_tokens", "token_budget", "budget_exceeded", "retrieval_mode", "sources_queried", "degradation_level", "warnings", "created_at", "retrieval_time_ms", "cache_hit", "search_queries", "anticipation_user_id", "anticipation_customer_id", "anticipation_conversation_id", "search_keywords", "bundle_type", "conversation_context", "bundle_confidence", "origin_pattern_id", "ttl_hint_seconds")
    class ItemsByTypeEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ContextItemList
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ContextItemList, _Mapping]] = ...) -> None: ...
    BUNDLE_ID_FIELD_NUMBER: _ClassVar[int]
    DECISION_ID_FIELD_NUMBER: _ClassVar[int]
    ITEMS_BY_TYPE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_BUDGET_FIELD_NUMBER: _ClassVar[int]
    BUDGET_EXCEEDED_FIELD_NUMBER: _ClassVar[int]
    RETRIEVAL_MODE_FIELD_NUMBER: _ClassVar[int]
    SOURCES_QUERIED_FIELD_NUMBER: _ClassVar[int]
    DEGRADATION_LEVEL_FIELD_NUMBER: _ClassVar[int]
    WARNINGS_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    RETRIEVAL_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    CACHE_HIT_FIELD_NUMBER: _ClassVar[int]
    SEARCH_QUERIES_FIELD_NUMBER: _ClassVar[int]
    ANTICIPATION_USER_ID_FIELD_NUMBER: _ClassVar[int]
    ANTICIPATION_CUSTOMER_ID_FIELD_NUMBER: _ClassVar[int]
    ANTICIPATION_CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    SEARCH_KEYWORDS_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_TYPE_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_PATTERN_ID_FIELD_NUMBER: _ClassVar[int]
    TTL_HINT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    bundle_id: str
    decision_id: str
    items_by_type: _containers.MessageMap[str, ContextItemList]
    total_tokens: int
    token_budget: int
    budget_exceeded: bool
    retrieval_mode: str
    sources_queried: _containers.RepeatedScalarFieldContainer[str]
    degradation_level: str
    warnings: _containers.RepeatedScalarFieldContainer[str]
    created_at: str
    retrieval_time_ms: int
    cache_hit: bool
    search_queries: _containers.RepeatedScalarFieldContainer[str]
    anticipation_user_id: str
    anticipation_customer_id: str
    anticipation_conversation_id: str
    search_keywords: _containers.RepeatedScalarFieldContainer[str]
    bundle_type: str
    conversation_context: ConversationContextProto
    bundle_confidence: float
    origin_pattern_id: str
    ttl_hint_seconds: int
    def __init__(self, bundle_id: _Optional[str] = ..., decision_id: _Optional[str] = ..., items_by_type: _Optional[_Mapping[str, ContextItemList]] = ..., total_tokens: _Optional[int] = ..., token_budget: _Optional[int] = ..., budget_exceeded: bool = ..., retrieval_mode: _Optional[str] = ..., sources_queried: _Optional[_Iterable[str]] = ..., degradation_level: _Optional[str] = ..., warnings: _Optional[_Iterable[str]] = ..., created_at: _Optional[str] = ..., retrieval_time_ms: _Optional[int] = ..., cache_hit: bool = ..., search_queries: _Optional[_Iterable[str]] = ..., anticipation_user_id: _Optional[str] = ..., anticipation_customer_id: _Optional[str] = ..., anticipation_conversation_id: _Optional[str] = ..., search_keywords: _Optional[_Iterable[str]] = ..., bundle_type: _Optional[str] = ..., conversation_context: _Optional[_Union[ConversationContextProto, _Mapping]] = ..., bundle_confidence: _Optional[float] = ..., origin_pattern_id: _Optional[str] = ..., ttl_hint_seconds: _Optional[int] = ...) -> None: ...

class ConversationContextProto(_message.Message):
    __slots__ = ("summary", "current_state_json", "key_extractions_json", "recent_turns", "compaction_id", "compacted_at", "conversation_id")
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    CURRENT_STATE_JSON_FIELD_NUMBER: _ClassVar[int]
    KEY_EXTRACTIONS_JSON_FIELD_NUMBER: _ClassVar[int]
    RECENT_TURNS_FIELD_NUMBER: _ClassVar[int]
    COMPACTION_ID_FIELD_NUMBER: _ClassVar[int]
    COMPACTED_AT_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    summary: str
    current_state_json: str
    key_extractions_json: str
    recent_turns: _containers.RepeatedCompositeFieldContainer[RecentTurnProto]
    compaction_id: str
    compacted_at: str
    conversation_id: str
    def __init__(self, summary: _Optional[str] = ..., current_state_json: _Optional[str] = ..., key_extractions_json: _Optional[str] = ..., recent_turns: _Optional[_Iterable[_Union[RecentTurnProto, _Mapping]]] = ..., compaction_id: _Optional[str] = ..., compacted_at: _Optional[str] = ..., conversation_id: _Optional[str] = ...) -> None: ...

class RecentTurnProto(_message.Message):
    __slots__ = ("role", "content", "timestamp")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    role: str
    content: str
    timestamp: str
    def __init__(self, role: _Optional[str] = ..., content: _Optional[str] = ..., timestamp: _Optional[str] = ...) -> None: ...

class ContextItemList(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[ContextItemProto]
    def __init__(self, items: _Optional[_Iterable[_Union[ContextItemProto, _Mapping]]] = ...) -> None: ...

class ContextItemProto(_message.Message):
    __slots__ = ("item_id", "content", "context_type", "source", "similarity_score", "relevance_score", "confidence", "scope", "entity_id", "created_at", "event_date", "valid_until", "temporal_category", "temporal_confidence")
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_SCORE_FIELD_NUMBER: _ClassVar[int]
    RELEVANCE_SCORE_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_ID_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    EVENT_DATE_FIELD_NUMBER: _ClassVar[int]
    VALID_UNTIL_FIELD_NUMBER: _ClassVar[int]
    TEMPORAL_CATEGORY_FIELD_NUMBER: _ClassVar[int]
    TEMPORAL_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    item_id: str
    content: str
    context_type: str
    source: str
    similarity_score: float
    relevance_score: float
    confidence: float
    scope: str
    entity_id: str
    created_at: str
    event_date: str
    valid_until: str
    temporal_category: str
    temporal_confidence: float
    def __init__(self, item_id: _Optional[str] = ..., content: _Optional[str] = ..., context_type: _Optional[str] = ..., source: _Optional[str] = ..., similarity_score: _Optional[float] = ..., relevance_score: _Optional[float] = ..., confidence: _Optional[float] = ..., scope: _Optional[str] = ..., entity_id: _Optional[str] = ..., created_at: _Optional[str] = ..., event_date: _Optional[str] = ..., valid_until: _Optional[str] = ..., temporal_category: _Optional[str] = ..., temporal_confidence: _Optional[float] = ...) -> None: ...

class TelemetryEvent(_message.Message):
    __slots__ = ("event_type", "instance_id", "client_id", "correlation_id", "timestamp_ms", "latency_ms", "status", "error_code", "scope", "cache_status", "attempt", "http_method", "http_path", "http_status_code", "metadata", "sdk_version", "batch_id")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    CACHE_STATUS_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    HTTP_METHOD_FIELD_NUMBER: _ClassVar[int]
    HTTP_PATH_FIELD_NUMBER: _ClassVar[int]
    HTTP_STATUS_CODE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    SDK_VERSION_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    event_type: str
    instance_id: str
    client_id: str
    correlation_id: str
    timestamp_ms: int
    latency_ms: int
    status: str
    error_code: str
    scope: str
    cache_status: str
    attempt: int
    http_method: str
    http_path: str
    http_status_code: int
    metadata: _containers.ScalarMap[str, str]
    sdk_version: str
    batch_id: str
    def __init__(self, event_type: _Optional[str] = ..., instance_id: _Optional[str] = ..., client_id: _Optional[str] = ..., correlation_id: _Optional[str] = ..., timestamp_ms: _Optional[int] = ..., latency_ms: _Optional[int] = ..., status: _Optional[str] = ..., error_code: _Optional[str] = ..., scope: _Optional[str] = ..., cache_status: _Optional[str] = ..., attempt: _Optional[int] = ..., http_method: _Optional[str] = ..., http_path: _Optional[str] = ..., http_status_code: _Optional[int] = ..., metadata: _Optional[_Mapping[str, str]] = ..., sdk_version: _Optional[str] = ..., batch_id: _Optional[str] = ...) -> None: ...

class TelemetryAck(_message.Message):
    __slots__ = ("status", "events_received")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    EVENTS_RECEIVED_FIELD_NUMBER: _ClassVar[int]
    status: str
    events_received: int
    def __init__(self, status: _Optional[str] = ..., events_received: _Optional[int] = ...) -> None: ...
