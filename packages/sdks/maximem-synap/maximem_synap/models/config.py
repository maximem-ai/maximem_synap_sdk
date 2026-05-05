"""Configuration models for MaximemSynap SDK."""

from typing import List, Optional

from pydantic import BaseModel, Field


class TimeoutConfig(BaseModel):
    """Timeout configuration for SDK operations."""

    connect: float = 5.0  # TCP connection timeout
    read: float = 30.0  # Response read timeout
    write: float = 10.0  # Request write timeout
    stream_idle: float = 60.0  # gRPC stream idle timeout


class RetryPolicy(BaseModel):
    """Retry policy configuration."""

    max_attempts: int = 3
    backoff_base: float = 1.0  # Base delay in seconds
    backoff_max: float = 10.0  # Max delay cap
    backoff_jitter: bool = True  # Add randomness to prevent thundering herd
    retryable_errors: List[str] = Field(
        default_factory=lambda: [
            "NetworkTimeoutError",
            "RateLimitError",
            "ServiceUnavailableError",
            "SynapTransientError",
        ]
    )


class SDKConfig(BaseModel):
    """Full SDK configuration."""

    api_base_url: Optional[str] = None  # Override default API base URL
    grpc_host: Optional[str] = None  # Override default gRPC host
    grpc_port: Optional[int] = None  # Override default gRPC port (e.g. 50051)
    grpc_use_tls: bool = True  # Set False for plaintext connections (testing)
    storage_path: Optional[str] = None  # Override default cache path (~/.synap/)
    cache_backend: Optional[str] = "sqlite"  # "sqlite" or None
    session_timeout_minutes: int = Field(default=30, ge=5, le=1440)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    retry_policy: Optional[RetryPolicy] = Field(default_factory=RetryPolicy)
    log_level: str = "WARNING"


# Backward compatibility - deprecated dataclass-style models
class CacheConfig:
    """Client-side cache configuration.

    DEPRECATED: Use SDKConfig.cache_backend instead.
    """

    def __init__(
        self,
        enabled: bool = True,
        ttl_seconds: int = 300,
        max_entries: int = 1000,
        storage_path: Optional[str] = None,
    ):
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.storage_path = storage_path
