# Migrating from 0.3.x to 0.4

`@maximem/synap-js-sdk` 0.4 is a **native TypeScript SDK**. It no longer spawns
a Python subprocess.

## The short version

```bash
npm install @maximem/synap-js-sdk
```

Then delete any `npx synap-js-sdk setup` step from your install or CI scripts.
Most code needs no other change.

## Breaking changes

### There is no setup step

0.3.x had two: `npx synap-js-sdk setup` (built a Python virtualenv) and
`npx synap-js-sdk setup-ts` (installed `typescript` + `@types/node` and wrote a
tsconfig). Neither exists in 0.4. Installing the package is the whole
installation, and types ship inside it.

### Python is no longer required, or used

0.3.x shipped a JSON-RPC bridge: Node spawned a Python interpreter and talked to
the real SDK over stdin/stdout. That required Python 3.11+ on every host, made
the package unusable anywhere without a process spawn, and meant every call
crossed a serialised pipe.

0.4 talks to the API directly.

- **Remove** `npx synap-js-sdk setup` from postinstall and CI.
- `~/.synap-js-sdk/.venv` is now orphaned. It is safe to delete.
- These environment variables no longer do anything: `SYNAP_JS_SDK_HOME`,
  `SYNAP_PYTHON_BIN`, `SYNAP_PYTHON_BOOTSTRAP`, `SYNAP_PY_SDK_PACKAGE`,
  `SYNAP_PY_SDK_VERSION`.
- The `synap-js-sdk` command still exists and still exits 0, so an un-updated
  script will not break your build. It prints a notice and does nothing.

### Node 20 is the minimum

Node 18 is end-of-life, and `globalThis.crypto` is only unflagged from Node 19.

### `listen()` is now opt-in

0.3.x started a gRPC anticipation stream **automatically** during init. Every
user had one whether they wanted it or not.

In 0.4 you ask for it explicitly. Be aware of what that means for your bill:
the anticipation stream is what lets the SDK answer some retrievals from a
locally cached bundle. Without it, those turns become billed cloud fetches. If
you were relying on that behavior, opt back in.

> The stream needs the optional peers: `npm install @grpc/grpc-js
> @grpc/proto-loader`. It is Node-only. Edge runtimes and Workers cannot host
> gRPC at all, so importing the SDK there stays safe and only the stream is
> unavailable.

### The local cache no longer persists across restarts

The Python SDK cached to SQLite, so a restarted process kept its cache. The JS
SDK caches in memory.

This is a **billing** change, not only a latency one: a cache miss is a metered
retrieval. Long-lived servers are largely unaffected. Short-lived processes and
serverless functions will see more cloud fetches. A pluggable cache backend is
available if you want to add Redis.

### Deleting by user is gone

```js
// 0.3.x -- silently did nothing in serverless, and reported success
await client.deleteMemory({ userId: 'u1' });

// 0.4
await client.memories.delete(memoryId);
```

The old form tracked memory ids in a process-local dictionary. In any
serverless or multi-process deployment that dictionary was empty, so the call
deleted nothing and returned `{ success: true, deletedCount: 0 }`. It now
throws instead of quietly lying.

### Timestamps are proper ISO-8601

The bridge stringified datetimes through Python's `str()`, producing
`2026-08-12 10:00:00+00:00` -- a space instead of the `T`. Native JS emits
`2026-08-12T10:00:00+00:00`. This is a fix, but it is still a change if you
were parsing those strings.

### True concurrency

The bridge serialised every call through one pipe. Concurrent calls are now
genuinely concurrent, which is faster and also means server rate limits the
pipe used to mask can now be reached. If you fan out hard, cap your own
concurrency.

## What did NOT change

Your call sites, in almost every case.

```js
// Unchanged
const ctx = await client.user.context.fetch({ user_id: 'u1', search_query: ['email'] });
await client.memories.create({ document: 'text', user_id: 'u1' });
```

- Both the namespaced (`client.user.context.fetch`) and legacy camelCase
  (`client.fetchUserContext`) surfaces still work.
- **They still return different shapes**, exactly as before: namespaced returns
  raw `snake_case`, camelCase returns normalised `camelCase`. Converging them
  would break one set of callers, so it is deferred.
- Both `user_id` and `userId` spellings are still accepted everywhere.
- `client.init()` still exists as a no-op.

## New in 0.4

- **Runs where Python could not**: Edge, Workers and browsers, for the HTTP
  paths. Only the gRPC stream is Node-only.
- **Real types**, generated from the source rather than hand-written.
- **A typed error taxonomy** mirroring the Python SDK. Every error carries a
  stable `.code`, which is what you should branch on:

  ```js
  import { isSynapError } from '@maximem/synap-js-sdk';

  try { await client.memories.create({ document }); }
  catch (e) {
    if (isSynapError(e) && e.code === 'insufficient_credits') { /* top up */ }
  }
  ```

  Prefer `.code` over `instanceof` at a package boundary: with dual ESM/CJS a
  consumer can end up holding two copies of a class. (`instanceof` is made to
  work across copies too, but `.code` is the documented contract.)
- **Explicit lifecycle.** Call `await client.shutdown()` when you are done. The
  SDK does not register `process.on('exit')` handlers: they leak a listener per
  client, never fire in Lambda, and `process` does not exist in Workers.

## Staying on 0.3.x

```bash
npm install @maximem/synap-js-sdk@legacy
```

0.3.x remains installable under the `legacy` dist-tag and is not deprecated.
