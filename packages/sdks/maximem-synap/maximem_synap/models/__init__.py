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
    Emotion,
    Episode,
    Fact,
    Preference,
    ResponseMetadata,
    UnifiedContextResponse,
)

# Request/Response envelopes
from .requests import RequestEnvelope, ResponseEnvelope

# Errors
from .errors import (
    AgentUnavailableError,
    AuthenticationError,
    ConnectionError,
    ContextNotFoundError,
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
    "InvalidInputError",
    "InvalidInstanceIdError",
    "InvalidConversationIdError",
    "AuthenticationError",
    "ContextNotFoundError",
    "SessionExpiredError",
    "ListeningAlreadyActiveError",
    "AgentUnavailableError",
    # Backward compatibility aliases
    "SDKError",
    "TransientError",
    "PermanentError",
    "ConnectionError",
]
