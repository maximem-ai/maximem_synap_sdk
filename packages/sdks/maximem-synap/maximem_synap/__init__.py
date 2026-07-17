"""Synap SDK - Agentic Context Management."""

from ._version import __version__
from .sdk import MaximemSynapSDK

# Re-export commonly used models
from .models.enums import CompactionLevel, ContextScope, ContextType, LogLevel
from .models.context import (
    CompactionResponse,
    CompactionStatusResponse,
    CompactionTriggerResponse,
    ContextBundle,
    ContextForPromptResponse,
    ConversationSummaryModel,
    RecentMessage,
    ContextItem,
    ContextResponse,
    Emotion,
    Episode,
    Fact,
    Preference,
    ProfileAttributeModel,
    ResponseMetadata,
    TemporalEvent,
    UnifiedContextResponse,
    UserProfileModel,
)
from .models.conversations import (
    TranscriptIngestResponse,
    TranscriptTurn,
)
from .models.config import CacheConfig, RetryPolicy, SDKConfig, TimeoutConfig
from .models.errors import (
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
    ListeningNotActiveError,
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
from .memories.models import (
    BatchCreateRequest,
    BatchCreateResponse,
    CreateMemoryRequest,
    CreateMemoryResponse,
    DocumentType,
    IngestMode,
    IngestStatus,
    Memory,
    MemoryStatusResponse,
    MergeStrategy,
    UpdateMemoryRequest,
)

__all__ = [
    # Main class
    "__version__",
    "MaximemSynapSDK",
    
    # Config
    "SDKConfig",
    "TimeoutConfig",
    "RetryPolicy",
    "CacheConfig",  # Deprecated but kept for compatibility
    
    # Response models
    "ContextResponse",
    "UnifiedContextResponse",
    "CompactionResponse",
    "CompactionTriggerResponse",
    "CompactionStatusResponse",
    "ContextForPromptResponse",
    "RecentMessage",
    "Fact",
    "Preference",
    "Episode",
    "Emotion",
    "ResponseMetadata",
    # User profile + conversation-summary models
    "UserProfileModel",
    "ProfileAttributeModel",
    "ConversationSummaryModel",
    # Conversation ingest
    "TranscriptTurn",
    "TranscriptIngestResponse",
    
    # Backward compatibility models (deprecated)
    "ContextBundle",
    "ContextItem",
    
    # Enums
    "ContextScope",
    "ContextType",
    "CompactionLevel",
    "LogLevel",
    "DocumentType",
    "IngestMode",
    "IngestStatus",
    "MergeStrategy",
    
    # Exceptions (new hierarchy)
    "SynapError",
    "SynapTransientError",
    "SynapPermanentError",
    "NetworkTimeoutError",
    "RateLimitError",
    "ServiceUnavailableError",
    "AgentUnavailableError",
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
    "ListeningNotActiveError",
    
    # Backward compatibility error aliases
    "SDKError",
    "TransientError",
    "PermanentError",
    "ConnectionError",
    
    # Memory models
    "Memory",
    "CreateMemoryRequest",
    "CreateMemoryResponse",
    "BatchCreateRequest",
    "BatchCreateResponse",
    "MemoryStatusResponse",
    "UpdateMemoryRequest",
]
