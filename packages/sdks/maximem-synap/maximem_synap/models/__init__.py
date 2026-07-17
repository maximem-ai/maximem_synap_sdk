"""Models package for MaximemSynap SDK."""

# Enums
from .enums import (
    AGGRESSIVE,
    BALANCED,
    CONSERVATIVE,
    CompactionLevel,
    ContextScope,
    ContextType,
    LogLevel,
)

# Config models
from .config import CacheConfig, RetryPolicy, SDKConfig, TimeoutConfig

# Context models
from .context import (
    CompactionResponse,
    CompactionStatusResponse,
    CompactionTriggerResponse,
    ContextBundle,
    ContextItem,
    ContextResponse,
    ConversationSummaryModel,
    Emotion,
    Episode,
    Fact,
    Preference,
    ProfileAttributeModel,
    ResponseMetadata,
    UnifiedContextResponse,
    UserProfileModel,
)

# Conversation-ingest models
from .conversations import (
    TranscriptIngestResponse,
    TranscriptTurn,
)

# Request/Response envelopes
from .requests import RequestEnvelope, ResponseEnvelope

# Errors
from .errors import (
    AgentUnavailableError,
    AuthenticationError,
    ConflictError,
    ConnectionError,
    ContextNotFoundError,
    InsufficientCreditsError,
    InvalidConversationIdError,
    InvalidInputError,
    InvalidInstanceIdError,
    ListeningAlreadyActiveError,
    NetworkTimeoutError,
    PermanentError,
    RateLimitError,
    SDKError,
    ServiceUnavailableError,
    SessionExpiredError,
    SynapError,
    SynapPermanentError,
    SynapTransientError,
    TranscriptConflictError,
    TransientError,
)

__all__ = [
    # Enums
    "ContextScope",
    "ContextType",
    "CompactionLevel",
    "LogLevel",
    "AGGRESSIVE",
    "BALANCED",
    "CONSERVATIVE",
    # Config
    "CacheConfig",
    "TimeoutConfig",
    "RetryPolicy",
    "SDKConfig",
    # Context
    "Fact",
    "Preference",
    "Episode",
    "Emotion",
    "ResponseMetadata",
    "ContextResponse",
    "CompactionResponse",
    "CompactionTriggerResponse",
    "CompactionStatusResponse",
    "ContextBundle",
    "ContextItem",
    "UnifiedContextResponse",
    "UserProfileModel",
    "ProfileAttributeModel",
    "ConversationSummaryModel",
    # Conversation ingest
    "TranscriptTurn",
    "TranscriptIngestResponse",
    # Requests
    "RequestEnvelope",
    "ResponseEnvelope",
    # Auth
    # Errors (new)
    "SynapError",
    "SynapTransientError",
    "SynapPermanentError",
    "NetworkTimeoutError",
    "RateLimitError",
    "ServiceUnavailableError",
    "InsufficientCreditsError",
    "InvalidInputError",
    "InvalidInstanceIdError",
    "InvalidConversationIdError",
    "AuthenticationError",
    "ContextNotFoundError",
    "ConflictError",
    "TranscriptConflictError",
    "SessionExpiredError",
    "ListeningAlreadyActiveError",
    "AgentUnavailableError",
    # Backward compatibility aliases
    "SDKError",
    "TransientError",
    "PermanentError",
    "ConnectionError",
]
