# @maximem/synap-js-sdk

Node.js wrapper for the Synap Python SDK.

## Prerequisites

- Node.js 18+
- Python 3.8+

## Install

```bash
npm install @maximem/synap-js-sdk
```

## Setup (JS Runtime)

Install the Python runtime used by the wrapper:

```bash
npx synap-js-sdk setup --sdk-version 0.2.0
```

If you want the latest Python SDK version, omit `--sdk-version` and pass `--upgrade`.

## Verify Runtime

macOS/Linux:

```bash
~/.synap-js-sdk/.venv/bin/python -c "import maximem_synap; print(maximem_synap.__version__)"
```

Windows:

```powershell
$env:USERPROFILE\.synap-js-sdk\.venv\Scripts\python.exe -c "import maximem_synap; print(maximem_synap.__version__)"
```

## Required Environment Variable

- `SYNAP_API_KEY` (from the Synap dashboard — Instances → Generate API Key)

The SDK uses the default Synap cloud endpoints automatically. `SYNAP_BASE_URL`,
`SYNAP_GRPC_HOST`, `SYNAP_GRPC_PORT`, and `SYNAP_GRPC_TLS` are only needed for
advanced overrides such as local development or custom environments.

## Quick Start (JavaScript)

```js
const { createClient } = require('@maximem/synap-js-sdk');

const synap = createClient({
  apiKey: process.env.SYNAP_API_KEY,
});

async function run() {
  await synap.init();

  await synap.addMemory({
    userId: 'user-123',
    customerId: 'customer-456',
    conversationId: 'conv-123',
    messages: [{ role: 'user', content: 'My name is Alex and I live in Austin.' }],
  });

  const context = await synap.fetchUserContext({
    userId: 'user-123',
    customerId: 'customer-456',
    conversationId: 'conv-123',
    searchQuery: ['Where does the user live?'],
    maxResults: 10,
  });

  console.log(context.facts);

  const promptContext = await synap.getContextForPrompt({
    conversationId: 'conv-123',
    style: 'structured',
  });

  console.log(promptContext.formattedContext);
  await synap.shutdown();
}

run().catch(console.error);
```

## TypeScript Extension Setup

Add TypeScript support to an existing JS project:

```bash
npx synap-js-sdk setup-ts
```

This command can:
- install `typescript` and `@types/node`
- generate `tsconfig.json` (if missing)
- generate `src/synap.ts` typed wrapper (if missing)

## Single-Flow Setup (JS + TS)

```bash
npm install @maximem/synap-js-sdk && npx synap-js-sdk setup --sdk-version 0.2.0 && npx synap-js-sdk setup-ts
```

## API Notes

- `addMemory()` now requires `customerId` to match the Python SDK's explicit ingestion scope.
- `fetchUserContext()`, `fetchCustomerContext()`, and `fetchClientContext()` expose the structured Python `ContextResponse` surface in JS/TS.
- `getContextForPrompt()` exposes compacted context plus recent un-compacted messages.
- `searchMemory()` and `getMemories()` remain convenience helpers built on top of user-scoped context fetches.
- Temporal fields are exposed in JS/TS as `eventDate`, `validUntil`, `temporalCategory`, `temporalConfidence`, plus top-level `temporalEvents`.

## CLI Commands

```bash
synap-js-sdk setup [options]
synap-js-sdk setup-ts [options]
```
