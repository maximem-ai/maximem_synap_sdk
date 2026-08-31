# @maximem/synap-js-sdk

Native TypeScript SDK for [Synap](https://synap.maximem.ai) context and memory
management.

**No Python required.** 0.4 replaced the Python-subprocess bridge with a native
client. Upgrading from 0.3.x? See [MIGRATING.md](./MIGRATING.md).

## Install

```bash
npm install @maximem/synap-js-sdk
```

Requires Node.js 20+. **That is the whole installation.** There is no setup
command, and TypeScript needs no extra step: types ship in the package.

Verified against TypeScript 5.7 and 7.0 with `moduleResolution` set to `node`,
`node16`, `nodenext` and `bundler`.

## Quick start

```ts
import { SynapClient } from '@maximem/synap-js-sdk';

const client = new SynapClient({ apiKey: process.env.SYNAP_API_KEY });

// Store
await client.memories.create({
  document: 'The user prefers aisle seats on long-haul flights.',
  user_id: 'user-123',
  // Optional. Defaults to 'ai-chat-conversation'. It selects the extraction
  // path, so setting it wrongly stores less from the same text.
  document_type: 'ai-chat-conversation',
});

// Retrieve
const context = await client.user.context.fetch({
  user_id: 'user-123',
  search_query: ['seat preference'],
});

await client.shutdown();
```

## Scopes

Scopes are **client > customer > user**. A narrower request also matches
broader-scope memories.

```ts
await client.user.context.fetch({ user_id, customer_id });
await client.customer.context.fetch({ customer_id });
await client.client.context.fetch({});
```

`conversation_id` is **not** a scope tier. It groups turns within a scope, and
narrows a fetch rather than selecting a different one.

## Errors

Every error carries a stable `.code`. Branch on that rather than on the class,
because a dual ESM/CJS dependency graph can hand you two copies of the same
class:

```ts
import { isSynapError } from '@maximem/synap-js-sdk';

try {
  await client.memories.create({ document, user_id });
} catch (e) {
  if (!isSynapError(e)) throw e;
  switch (e.code) {
    case 'insufficient_credits': /* top up */ break;
    case 'rate_limit':           /* e.retryAfterSeconds */ break;
    case 'authentication':       /* bad key */ break;
    default: throw e;
  }
}
```

Transient errors are retried automatically. Ingestion is deliberately **not**
retried when a failure leaves the outcome unknown, because a retry would store
and bill twice.

## Runtime support

| Runtime | HTTP context + memories | gRPC anticipation stream |
|---|---|---|
| Node.js 20+ | Yes | Yes (opt-in) |
| Bun | Yes | Unverified |
| Deno 2 | Yes | Unverified |
| Vercel Edge | Yes | No |
| Cloudflare Workers | Yes | No |
| Browser | Yes (do not ship an API key) | No |

Edge and Workers cannot run gRPC at all: it needs raw TCP and `node:http2`,
which they do not provide. Bun and Deno have both, so gRPC is plausible on each,
but **bidirectional streaming has not been tested there**, so those cells stay
unverified rather than claimed either way.

Importing the SDK on Edge is safe: gRPC lives behind the
`@maximem/synap-js-sdk/grpc` subpath and is only ever loaded lazily.

## Configuration

| Option | Default | Notes |
|---|---|---|
| `apiKey` | `SYNAP_API_KEY` | Required. |
| `clientId` | `SYNAP_CLIENT_ID` | Skips a `whoami` round trip. |
| `baseUrl` | prod | |
| `heartbeat` | `false` | Keeps the connection warm. Worth it for a long-lived process, pointless in serverless. |
| `timeouts` | 5s connect / 30s read | |
| `retryPolicy` | 3 attempts | |

Environment flags are read at call time, so you can toggle them at runtime:
`SYNAP_SDK_CACHE_RECALL_BYPASS`, `SYNAP_SDK_CACHE_HONOR_TTL_HINT`,
`SYNAP_SDK_CACHE_INVALIDATE_ON_WRITE`, `SYNAP_SDK_CACHE_COVERAGE_MIN`,
`SYNAP_SDK_CACHE_MAX_ENTRY_AGE`.

## Development

```bash
npm install
npm test
npm run build
npm run typecheck
npm run sync-behavior   # after changing synap/sdk/CONTRACT/**
```

Retrieval behavior (recall-bypass patterns, BM25 thresholds, the stemmer) is
**not defined in this package**. It lives in `synap/sdk/CONTRACT/`, shared
byte-identically with the Python SDK, and both SDKs run the same conformance
corpus. Never edit the vendored copy in `src/behavior/`.
