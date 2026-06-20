# SDK setup — install, init, lifecycle, errors

The same `MaximemSynapSDK` is the foundation of every framework integration. Get this right once and everything else slots in.

## Prerequisites — do this in the dashboard, not in code

The user must do these manually at `https://synap.maximem.ai`:

1. Create a **Client** (their organization). Get the `cli_<hex16>`.
2. Create an **Instance** for the agent they're building. Get the `inst_<hex16>`.
3. (Recommended) Upload a **Use-Case Markdown** describing what the agent does — Synap uses this to auto-generate an optimized MACA. Template + guidance: `reference/use-case-markdown.md`.
4. Generate an **API Key** under the instance's API Keys section. **The key is shown once.** Format: `synap_...`.

Never attempt to provision instances or keys from code in this skill.

## Install

**Python:**

```bash
pip install maximem-synap
```

gRPC streaming ships in the box (`grpcio` is a core dependency, on by default) — there is no
separate `[grpc]` extra. Verify:

```bash
python -c "import maximem_synap; print(maximem_synap.__version__)"
```

Python 3.11+ required.

**TypeScript / Node:**

```bash
npm install @maximem/synap-js-sdk
```

The JS SDK is a thin wrapper that spawns the Python SDK as a subprocess, so the host needs
**Python 3.11+ on `PATH` in addition to Node 18+**. It does **not** run on Edge Runtime,
Cloudflare Workers, Bun, Deno Deploy, or Node-only Lambda runtimes.

## Environment variables (the canonical pattern)

```bash
export SYNAP_INSTANCE_ID="inst_a1b2c3d4e5f67890"
export SYNAP_API_KEY="synap_..."
```

With these set, the SDK auto-loads them — no constructor args needed.

## Basic init

**Python:**

```python
from maximem_synap import MaximemSynapSDK

sdk = MaximemSynapSDK()              # reads env vars
await sdk.initialize()
# ... use sdk ...
await sdk.shutdown()
```

Or pass explicitly:

```python
sdk = MaximemSynapSDK(
    instance_id="inst_a1b2c3d4e5f67890",
    api_key="synap_...",
)
await sdk.initialize()
```

**TypeScript:**

```typescript
import { createClient } from "@maximem/synap-js-sdk";

const sdk = createClient({ apiKey: process.env.SYNAP_API_KEY! });
await sdk.init();             // note: init(), not initialize()
// ... use sdk ...
await sdk.shutdown();
```

The JS API is flat and camelCase, and is **not** identical to Python: there is no
`MaximemSynapSDK` class and no `sdk.memories` / `sdk.conversation` namespaces. Write with
`sdk.addMemory({ userId, customerId, messages, mode })`; read with
`sdk.fetchUserContext({ userId, searchQuery, mode })`,
`sdk.fetchCustomerContext({ customerId, ... })`, `sdk.fetchClientContext({ ... })`, or
`sdk.getContextForPrompt({ conversationId })`. Full example: `examples/typescript-minimal.ts`.

`init()` validates the API key, starts the Python bridge, opens the connection, and sets up the local cache.

**Calling any SDK method before `await sdk.initialize()` raises `AuthenticationError`.**

## Singleton behavior

By design, constructing `MaximemSynapSDK` twice with the same `instance_id` returns the same instance:

```python
a = MaximemSynapSDK(instance_id="inst_...", api_key="...")
b = MaximemSynapSDK(instance_id="inst_...", api_key="...")
assert a is b   # True
```

This prevents duplicate connections from multi-module imports. Don't fight it. For tests, pass `_force_new=True`.

## Production-grade init with config

```python
from maximem_synap import (
    MaximemSynapSDK,
    SDKConfig,
    TimeoutConfig,
    RetryPolicy,
)

config = SDKConfig(
    storage_path="/var/lib/myapp/synap",     # cache location
    cache_backend="sqlite",                  # "sqlite" (default), or None to disable caching
    session_timeout_minutes=60,
    timeouts=TimeoutConfig(
        connect=10.0,
        read=30.0,
        write=15.0,
        stream_idle=120.0,
    ),
    retry_policy=RetryPolicy(
        max_attempts=3,
        backoff_base=1.5,
        backoff_max=30.0,
        backoff_jitter=True,
    ),
    log_level="WARNING",
)

sdk = MaximemSynapSDK(
    instance_id=os.environ["SYNAP_INSTANCE_ID"],
    api_key=os.environ["SYNAP_API_KEY"],
    config=config,
)
await sdk.initialize()
```

`configure()` after construction works **only before** `initialize()`:

```python
sdk = MaximemSynapSDK()
sdk.configure(log_level="DEBUG")    # OK — applied before initialize()
await sdk.initialize()
sdk.configure(log_level="INFO")     # raises InvalidInputError("Cannot reconfigure after initialization")
```

## Lifecycle — `try / finally` is the safe pattern

```python
sdk = MaximemSynapSDK()
try:
    await sdk.initialize()
    # application logic
finally:
    await sdk.shutdown()
```

**Always shut down.** `shutdown()` flushes pending telemetry and closes gRPC streams. Skipping it loses recent ingestion telemetry and leaves connections lingering.

For long-running servers (FastAPI, etc.), initialize on startup and shut down on shutdown:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    sdk = MaximemSynapSDK()
    await sdk.initialize()
    app.state.synap = sdk
    yield
    await sdk.shutdown()

app = FastAPI(lifespan=lifespan)
```

For serverless (Lambda, Cloud Run), keep the SDK module-level and initialize lazily on first invocation. Cold starts will pay the connection cost; warm invocations reuse it.

## Error handling

Import error types from the top-level `maximem_synap` package:

```python
from maximem_synap import (
    AuthenticationError,        # bad/missing API key, expired key
    NetworkTimeoutError,        # could not reach Synap Cloud
    ServiceUnavailableError,    # Synap returned 5xx
    InvalidConversationIdError, # conversation_id is not a valid UUID
    InvalidInputError,          # other bad inputs (parent of the above)
    SynapError,                 # base class — catch-all
)
```

Framework integration packages additionally raise `SynapIntegrationError`
(from `synap_integrations_common`) on write failures — catch that in framework code.

Every error carries a `correlation_id` — log it. Synap support can trace a request from this ID.

**Pattern: degrade gracefully on read, surface on write.** Every official integration follows this; do the same in custom code.

```python
# READ path — never block the agent on a memory failure
try:
    context = await sdk.conversation.context.fetch(
        conversation_id=conv_id,
        search_query=[query],
        mode="fast",
    )
except SynapError as e:
    logger.warning("Synap read failed (corr=%s): %s", e.correlation_id, e)
    context = None  # agent proceeds with no memory

# WRITE path — surface failure so caller knows persistence was lost
await sdk.memories.create(...)   # let it raise
```

## Retries

`RetryPolicy` retries transient network failures and 5xx responses with exponential backoff. It does **not** retry 4xx (those are your bug, not a transient issue). Default is 3 attempts.

For ingestion at high throughput, prefer `batch_create()` over many `create()` calls — see `reference/ingestion.md`.

## Cache

The SDK caches retrieval responses locally (default backend: SQLite). Cached responses include a TTL on `context.metadata.ttl_seconds`. Subsequent fetches inside the TTL come from cache (`context.metadata.source == "cache"`). To force a fresh fetch, you generally need a different `search_query` or to wait for the TTL.

For testing, point `storage_path` at a tempdir or disable caching with `cache_backend=None`.

## Live doc references

- Initialization: `https://docs.maximem.ai/sdk/initialization`
- Configuration: `https://docs.maximem.ai/sdk/configuration`
- Authentication: `https://docs.maximem.ai/setup/authentication`
- Installation: `https://docs.maximem.ai/setup/installation`
- Error handling: `https://docs.maximem.ai/sdk/error-handling`

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.2.4 (JS) — verified 2026-06-17. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
