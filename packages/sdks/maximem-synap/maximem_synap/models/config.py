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
    grpc_use_tls: Optional[bool] = None  # None=use transport default (TLS on); False=plaintext
    storage_path: Optional[str] = None  # Override default cache path (~/.synap/)
    cache_backend: Optional[str] = "sqlite"  # "sqlite" or None
    # Accepted and range-checked so existing configs keep working, but nothing
    # reads it: there is no session, and the API key is sent on every request.
    session_timeout_minutes: int = Field(default=30, ge=5, le=1440)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    retry_policy: Optional[RetryPolicy] = Field(default_factory=RetryPolicy)
    log_level: str = "WARNING"
    # SDK-authoritative short-term context. When true the SDK serves
    # get_compacted/get_context_for_prompt from its local ShortTermContextStore
    # (the last compaction it was sent, plus the turns since), and falls back
    # to REST whenever that store does not hold the whole conversation: the
    # server reports its message count on every write and every compaction, and
    # the store only answers when it accounts for all of them. Against a server
    # that sends no counts it never answers locally.
    #
    # Defaults OFF, and should stay off for an application that runs more than
    # one process against the same conversation. A process learns the server's
    # count only when it writes. If it reads before it writes (build the prompt,
    # then record the turn), it cannot know that another process recorded a
    # turn in between, and it answers without that turn. Turn it on where one
    # process owns a conversation for its whole life, such as a voice agent.
    # SYNAP_SDK_ST_AUTHORITATIVE overrides this in either direction. See
    # docs/internal/sdk_authoritative_short_term_context_plan.md.
    sdk_st_authoritative: bool = False
    # Local embedding model for history search after compaction and for
    # matching fetch queries to prefetched items (cache/encoders.py): "auto"
    # uses the default multilingual model when maximem-synap[local-search] is
    # installed; "onnx:<repo>@<rev>/<file>", "fastembed:<name>" or
    # "model2vec:<id>" picks another (history search only: anticipation
    # matching keeps to the model its thresholds were tuned on); None keeps
    # keywords only. Runs on this machine; downloaded once into
    # ~/.synap/models, loaded in the background.
    history_search_model: Optional[str] = "auto"
    # Read-your-writes verbatim short-term overlay (correctness, not an
    # optimization). When true: a turn the SDK has just emitted via
    # record_message/send_message is overlaid onto the next fetch's
    # conversation_context.recent_turns, even with sdk_st_authoritative off and
    # before any compaction has landed — so a fast follow-up reflects the
    # just-said turn instead of an async-lagged server buffer. Independent of
    # sdk_st_authoritative (that flag governs the *skip server ST assembly*
    # cost optimization; this one governs *freshness*). Defaults ON; the
    # overlay is a no-op when the local store is cold, and can be disabled as a
    # kill-switch via config or SYNAP_ST_VERBATIM_OVERLAY=0. See
    # docs/internal/short_term_context_freshness_plan.md.
    st_verbatim_overlay: bool = True
    # The model the agent runs on, and optionally its context window. Sent
    # with every recorded turn so the server sizes compaction to the agent's
    # window (it compacts at ~11.7% of it: 15K tokens on a 128K model). A name
    # is enough; the server knows common models' windows. Unset = 128K. See
    # agent_model.py; record_message / send_message take per-call overrides.
    agent_model: Optional[str] = None
    agent_context_window: Optional[int] = None


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
