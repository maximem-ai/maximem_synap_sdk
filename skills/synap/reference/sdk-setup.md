# SDK setup — install, init, lifecycle, errors

> **Read this next:** [Installation](https://docs.maximem.ai/setup/installation.md); [Authentication](https://docs.maximem.ai/setup/authentication.md); [SDK initialization](https://docs.maximem.ai/sdk/initialization.md); [SDK configuration](https://docs.maximem.ai/sdk/configuration.md); [initialize](https://docs.maximem.ai/sdk-reference/lifecycle/initialize.md); [configure](https://docs.maximem.ai/sdk-reference/lifecycle/configure.md); [shutdown](https://docs.maximem.ai/sdk-reference/lifecycle/shutdown.md); [Errors](https://docs.maximem.ai/sdk-reference/errors.md); [Error handling](https://docs.maximem.ai/sdk/error-handling.md); [Agent integration walkthrough](https://docs.maximem.ai/setup/agent-integration.md). **Checkpoint:** open the method's page before writing each call, re-read it after and compare every parameter, scope rule and error; if the docs and this file disagree, the docs win and tell the human.

The same `MaximemSynapSDK` is the foundation of every framework integration. Get this right once and everything else slots in.

## Prerequisites: an instance and an API key

Before any code runs you need a **Client** (workspace, `cli_<hex16>`), an **Instance** (`inst_<hex16>`) and an
**API key** (`synap_...`, shown once). The onboarding procedure in `SKILL.md` creates the instance and key
through the CLI (`reference/dashboard-setup.md`); a developer with no coding agent can create them by hand in the dashboard at
`https://synap.maximem.ai`. The use-case file that seeds the memory architecture is described in
`reference/use-case-markdown.md`. Never provision from application code.

## While the memory architecture is building: `instance_not_ready`

For 15 to 20 minutes after `instances create`, the instance rejects runtime writes and context fetches with
HTTP 409 `{"error":"instance_not_ready","message":"...","state":"building","retry_after_seconds":N,"instance_id":"..."}`
and a `Retry-After` header. This applies to the user's own app too, not just to you.

**One rule, two SDK generations. Handle BOTH:**

- Newer SDKs raise `InstanceNotReadyError` (a subclass of `ConflictError`) with `retry_after_seconds` (the SDK
  tells you how long to wait), `from_cooldown` (true when the SDK refused locally, no network call, and the
  figure is the remaining wait), `instance_id` and `state`.
- Older SDKs, including the currently published 0.5.16, raise a plain `ConflictError` whose message begins
  "This instance's memory architecture is still being built"; the code and `retry_after_seconds` are dropped.

Catch the typed class first, then check the `ConflictError` message. Any other `ConflictError` (for example
"compaction already in progress") is a real conflict. The SDK never retries this error inside a call, so
**never loop inside the call**: a read degrades to no memory (log it, continue), a write is retried later.

```python
from maximem_synap import ConflictError

try:  # newer SDKs
    from maximem_synap import InstanceNotReadyError
except ImportError:  # older SDKs (0.5.16 and before)
    InstanceNotReadyError = None

NOT_READY_TEXT = "memory architecture is still being built"
DEFAULT_WAIT = 30  # seconds, when the SDK gives no figure


def not_ready_wait(exc: BaseException):
    """Seconds to wait if `exc` means "instance still building", else None."""
    if InstanceNotReadyError is not None and isinstance(exc, InstanceNotReadyError):  # typed first
        return exc.retry_after_seconds or DEFAULT_WAIT
    if isinstance(exc, ConflictError) and NOT_READY_TEXT in str(exc).lower():  # then the message
        return DEFAULT_WAIT
    return None
```

Async (the SDK's native form). A read degrades, a write is parked for a later retry:

```python
async def recall(sdk, user_id, query):
    try:
        return await sdk.user.context.fetch(user_id=user_id, search_query=[query])
    except ConflictError as e:               # InstanceNotReadyError is a ConflictError too
        if not_ready_wait(e) is None:
            raise                            # a real conflict: do not hide it
        log.warning("memory not ready (%s); continuing without memory", e)
        return None                          # degrade: the agent answers without memory

async def remember(sdk, user_id, text, retry_queue):
    try:
        await sdk.memories.create(document=text, document_type="ai-chat-conversation", user_id=user_id)
    except ConflictError as e:
        wait = not_ready_wait(e)
        if wait is None:
            raise
        retry_queue.put((time.monotonic() + wait, user_id, text))   # retry later, once, after `wait`
```

Sync (a script, CLI or worker thread with no running event loop; in a web framework use its own loop instead of
`asyncio.run`):

```python
def recall_sync(sdk, user_id, query):
    try:
        return asyncio.run(sdk.user.context.fetch(user_id=user_id, search_query=[query]))
    except ConflictError as e:
        if not_ready_wait(e) is None:
            raise
        log.warning("memory not ready (%s); continuing without memory", e)
        return None

def remember_sync(sdk, user_id, text, retry_queue):
    try:
        asyncio.run(sdk.memories.create(document=text, document_type="ai-chat-conversation", user_id=user_id))
    except ConflictError as e:
        wait = not_ready_wait(e)
        if wait is None:
            raise
        retry_queue.append((time.monotonic() + wait, user_id, text))
```

Tell the human how long you are waiting ("the memory architecture is still being built, retrying in 30 s").
An in-process retry queue is lost on a serverless cold start; there, re-send from a durable source or accept the
loss and say so. Add `customer_id` to these calls on a B2B instance only. Error-handling patterns:
[Error handling](https://docs.maximem.ai/sdk/error-handling.md). How the JS SDK surfaces this is not confirmed in
this skill: check the status code and message rather than assuming an error class.
`scripts/verify_synap.py` already does the Python detection and a bounded retry.

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

The JS SDK is **pure TypeScript**: no Python, no subprocess, and zero runtime
dependencies. It needs **Node 20+** and nothing else. gRPC is reached through a
lazy `import()`, so importing the SDK on Edge Runtime, Cloudflare Workers, Bun
or Deno is safe; only the live stream (`instance.listen()`) needs a Node
runtime, because grpc-js is built on `node:http2`.

## Environment variables (the canonical pattern)

```bash
export SYNAP_API_KEY="synap_..."          # required
export SYNAP_INSTANCE_ID="inst_..."       # optional; resolved from the key
```

### Pointing at something other than production

Four more, and they are the difference between testing against staging and
silently writing to production. Leave them unset for production.

```bash
export SYNAP_BASE_URL="https://<host>"    # the REST API
export SYNAP_GRPC_HOST="<host>"           # the live stream
export SYNAP_GRPC_PORT="443"
export SYNAP_GRPC_USE_TLS="true"
```

⚠ **An API key belongs to exactly one instance, and an instance lives in one
deployment.** So pointing at a different host also means using that
deployment's key: a production key against a staging host authenticates to
nothing. `verify_synap.py` now fails on that rather than printing green, but
the quiet version of this mistake is writing real customer turns into the
wrong environment, which nothing catches.

Set all four together or none of them. `SYNAP_BASE_URL` alone leaves the
stream pointed at production while reads go elsewhere, which looks like
anticipation silently not working.

With this set, the SDK auto-loads it — no constructor args needed. The instance is
resolved from the API key during `initialize()`, so `SYNAP_INSTANCE_ID` is **not**
required.

Set it only when a key has access to more than one instance and you want to pin a
specific one:

```bash
export SYNAP_INSTANCE_ID="inst_a1b2c3d4e5f67890"   # optional
```

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
import { SynapClient } from "@maximem/synap-js-sdk";

const sdk = new SynapClient({ apiKey: process.env.SYNAP_API_KEY! });
await sdk.init();             // note: init(), not initialize()
// ... use sdk ...
await sdk.shutdown();
```

The class is `SynapClient`, imported from `@maximem/synap-js-sdk`. There is no
`createClient` factory and no `MaximemSynapSDK` class.

**Two surfaces, both live.** The namespaced one mirrors Python method for
method: `sdk.conversation`, `sdk.memories`, `sdk.user`, `sdk.customer`,
`sdk.client`, `sdk.instance`, `sdk.credits` and `sdk.cache` all exist. Prefer
it. The flat camelCase one is deprecated but supported:
`sdk.addMemory({ userId, customerId, messages, mode })`,
`sdk.fetchUserContext({ userId, searchQuery, mode })`,
`sdk.fetchCustomerContext({ customerId, ... })`, `sdk.fetchClientContext({ ... })`,
`sdk.getContextForPrompt({ conversationId })`. They return different shapes on
purpose: namespaced gives raw snake_case, camelCase gives a normalised object.

`customerId` and `fetchCustomerContext` are B2B only: on a B2C instance
(`user_context_isolation = "equals_customer"`) send `userId` alone, since a
`customerId` comes back as HTTP 400. Full example:
`examples/typescript-minimal.ts`.

`init()` validates the API key, opens the connection and sets up the local cache.

**It is pure TypeScript.** No Python, no subprocess, and zero runtime
dependencies. Node 20+. gRPC is reached through a lazy `import()`, so importing
the SDK on Edge or in a Worker is safe; only the live stream needs Node.

**Calling any SDK method before `await sdk.initialize()` raises `AuthenticationError`.**

## Singleton behavior

By design, constructing `MaximemSynapSDK` twice with the same **API key** returns the same live SDK:

```python
a = MaximemSynapSDK(api_key="sk-...")
b = MaximemSynapSDK(api_key="sk-...")
# One live SDK: b shares a's connections, caches and credentials.
# Note: `a is b` is False. They share internal state, they are not the same
# object, so use the SDK rather than identity-checking it.
```

This prevents duplicate connections from multi-module imports. Don't fight it. For tests, pass `_force_new=True`.

Different API keys give you different SDKs, so serving several tenants from one process is supported. Two keys issued against the *same* instance are the exception: each key is its own identity, so you get two SDKs, two Listen streams and two caches for one instance. `initialize()` warns when it sees that. Use one key per instance per process.

Passing `instance_id` explicitly still keys on the id, so two keys under one id share a single SDK on the first key. You do not need it: your API key already identifies your instance.

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
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
