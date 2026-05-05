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
    RecentMessage,
    ContextItem,
    ContextResponse,
    Emotion,
    Episode,
    Fact,
    Preference,
    ResponseMetadata,
    TemporalEvent,
)
from .models.config import CacheConfig, RetryPolicy, SDKConfig, TimeoutConfig
from .models.errors import (
    AgentUnavailableError,
    AuthenticationError,
    ConnectionError,
    ContextNotFoundError,
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
    "InvalidInputError",
    "InvalidInstanceIdError",
    "InvalidConversationIdError",
    "AuthenticationError",
    "ContextNotFoundError",
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
