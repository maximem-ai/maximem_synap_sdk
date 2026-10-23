"""Enumerations for MaximemSynap SDK."""

from enum import StrEnum


class ContextScope(StrEnum):
    """Scope levels for context storage."""

    CLIENT = "client"
    CUSTOMER = "customer"
    USER = "user"
    CONVERSATION = "conversation"


class ContextType(StrEnum):
    """Types of context that can be retrieved."""

    FACTS = "facts"
    PREFERENCES = "preferences"
    EPISODES = "episodes"
    EMOTIONS = "emotions"
    TEMPORAL = "temporal_events"
    ALL = "all"


class CompactionLevel(StrEnum):
    """Levels of context compaction."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ADAPTIVE = "adaptive"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


class LogLevel(StrEnum):
    """SDK logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class RetrievalMode(StrEnum):
    """Retrieval mode for context fetch operations."""

    FAST = "fast"
    ACCURATE = "accurate"


# Backward compatibility aliases for CompactionLevel
AGGRESSIVE = CompactionLevel.HIGH
BALANCED = CompactionLevel.MEDIUM
CONSERVATIVE = CompactionLevel.LOW
