# MaximemSynap Python SDK

Python client library for the MaximemSynap context management system.

## Installation

```bash
pip install maximem-synap
```

## Quick Start

```python
from maximem_synap import MaximemSynapSDK, ContextType, CompactionLevel

# Initialize SDK (reads SYNAP_API_KEY from your environment)
sdk = MaximemSynapSDK()

# Fetch conversation context
context = sdk.conversation.fetch(
    conversation_id="conv-123",
    search_query="user preferences",
    max_results=10,
    types=[ContextType.FACTS, ContextType.PREFERENCES]
)

# Access context items
for item in context.items:
    print(f"{item.context_type}: {item.content}")

# Compact conversation context
compacted = sdk.conversation.compact(
    conversation_id="conv-123",
    compaction_level=CompactionLevel.BALANCED
)

# Fetch user context
user_context = sdk.user.fetch(
    user_id="user-456",
    conversation_id="conv-123",  # Optional
    max_results=20
)

# Listen for real-time updates
sdk.instance.listen()
# ... your application logic ...
sdk.instance.stop()

# Cleanup
sdk.shutdown()
```

## Configuration

```python
from maximem_synap import CacheConfig, TimeoutConfig

# Configure caching
cache_config = CacheConfig(
    enabled=True,
    ttl_seconds=300,
    max_entries=1000
)

# Configure timeouts
timeout_config = TimeoutConfig(
    connect_timeout_ms=5000,
    read_timeout_ms=30000,
    total_timeout_ms=60000
)

# Apply configuration
sdk.configure(
    cache=cache_config,
    log_level="DEBUG",
    timeouts=timeout_config
)
```

## Environment Variables

- `SYNAP_API_KEY`: Your Synap API key

## Context Controllers

### Conversation Context

```python
# Fetch conversation context
context = sdk.conversation.fetch(
    conversation_id="conv-123",
    search_query="optional search",
    max_results=10,
    types=[ContextType.FACTS]
)

# Compact conversation context
compacted = sdk.conversation.compact(
    conversation_id="conv-123",
    compaction_level=CompactionLevel.ADAPTIVE
)
```

### User Context

```python
context = sdk.user.fetch(
    user_id="user-456",
    conversation_id="conv-123",  # Optional
    types=[ContextType.PREFERENCES, ContextType.EMOTIONS]
)
```

### Customer Context

```python
context = sdk.customer.fetch(
    customer_id="cust-789",
    conversation_id="conv-123"  # Optional
)
```

### Client Context

```python
context = sdk.client.fetch(
    client_id="client-abc",
    conversation_id="conv-123"  # Optional
)
```

## Error Handling

```python
from maximem_synap import (
    SDKError,
    AgentUnavailableError,
    ContextNotFoundError,
    AuthenticationError
)

try:
    context = sdk.conversation.fetch(conversation_id="conv-123")
except AgentUnavailableError as e:
    # Retryable error
    print(f"Agent unavailable: {e}")
except ContextNotFoundError as e:
    # Non-retryable error
    print(f"Context not found: {e}")
except AuthenticationError as e:
    # Authentication failed
    print(f"Auth error: {e}")
except SDKError as e:
    # Base exception
    print(f"SDK error: {e}, retryable: {e.retryable}")
```

## Context Types

- `ContextType.FACTS`: Factual information
- `ContextType.PREFERENCES`: User preferences
- `ContextType.EPISODES`: Conversation episodes
- `ContextType.EMOTIONS`: Emotional context
- `ContextType.TEMPORAL`: Time-based context
- `ContextType.ALL`: All context types

## Compaction Levels

- `CompactionLevel.AGGRESSIVE`: Maximum compression
- `CompactionLevel.BALANCED`: Balanced approach
- `CompactionLevel.CONSERVATIVE`: Minimal compression
- `CompactionLevel.ADAPTIVE`: Adaptive based on context

## License

MIT
