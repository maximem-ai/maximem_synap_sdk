# Vercel AI SDK

```bash
npm install @maximem/synap-vercel-adk
```

**TypeScript only.** Wraps any Vercel AI SDK model with automatic Synap context — works with `generateText`, `streamText`, `generateObject`, `streamObject`, and any provider (`@ai-sdk/openai`, `@ai-sdk/anthropic`, `@ai-sdk/google`, etc.).

| Export | Purpose |
| --- | --- |
| `createSynap` | Async factory that initializes the Synap provider |
| `SynapProvider` | Provider class with `wrap` and `listen` methods |

## Quick start

```typescript
import { generateText } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { createSynap } from "@maximem/synap-vercel-adk";

const synap = await createSynap({
  apiKey: process.env.SYNAP_API_KEY!,   // falls back to SYNAP_API_KEY; no instanceId (resolved from the key)
  // optional: baseUrl, grpcHost, grpcPort, grpcUseTls
});

const model = synap.wrap(anthropic("claude-sonnet-4-6"), {
  userId: "alice",
  customerId: "acme",   // B2B only; omit on B2C
});

const { text } = await generateText({
  model,
  messages: [{ role: "user", content: "What do you remember about my account?" }],
});
```

`synap.wrap()` returns a standard Vercel AI SDK `LanguageModel` — pass it anywhere you'd use a plain model. Nothing else changes.

**Scoping:** `customerId` is B2B only, and required there. On a B2C instance (`user_context_isolation = "equals_customer"`) pass `userId` alone: a `customerId` comes back as HTTP 400. `GET /api/v1/auth/whoami` tells you which mode the instance is in.

## What happens under the hood

On every call to `generateText` / `streamText` / `generateObject`:

1. **Before** — fetch user's Synap context, inject as system message
2. **Generate** — proxy to wrapped model unchanged
3. **After** — ingest completed user + assistant turn asynchronously

## Works with any provider

```typescript
import { openai } from "@ai-sdk/openai";
import { google } from "@ai-sdk/google";
import { anthropic } from "@ai-sdk/anthropic";

const gptWithMemory    = synap.wrap(openai("gpt-4o"),                 { userId: "alice" });
const geminiWithMemory = synap.wrap(google("gemini-2.0-flash"),        { userId: "alice" });
const claudeWithMemory = synap.wrap(anthropic("claude-sonnet-4-6"),    { userId: "alice" });
```

## Streaming

```typescript
const { textStream } = await streamText({
  model: synap.wrap(openai("gpt-4o"), { userId: "alice" }),
  messages: [{ role: "user", content: "Summarize my recent priorities." }],
});

for await (const chunk of textStream) {
  process.stdout.write(chunk);
}
```

No special handling needed — context is injected before the stream starts; ingestion happens on stream completion.

## Per-request scoping

Wrap fresh per request to scope per-user without any global state. This is the B2C shape; on a B2B instance pass the tenant's `customerId` alongside `userId`:

```typescript
async function handleChat(userId: string, message: string) {
  const model = synap.wrap(openai("gpt-4o"), { userId });
  const { text } = await generateText({
    model,
    messages: [{ role: "user", content: message }],
  });
  return text;
}
```

## The live stream — wire this, it is not advanced

`synap.listen()` opens the gRPC stream. Once it is open the middleware reports
turns, tool calls and reasoning for you, and Synap pushes predicted context
back before the next request arrives, so the fetch resolves from memory instead
of over the network. Reported turns also become long-term memory on their own.

```typescript
export const synap = await createSynap({ apiKey: process.env.SYNAP_API_KEY! });

await synap.listen();          // once, at startup. Returns Promise<void>.

// on shutdown
await synap.stopListening();
// synap.isListening → boolean
```

`listen()` does not return a stop function; `stopListening()` is a separate
method. It is Node-only and silently no-ops in Edge Runtime and on serverless,
where nothing can hold a connection, and the provider falls back to HTTP
context fetching. That is the one case where skipping it is correct.

In a long-lived Node server, call it. Without it every request pays the full
HTTP context fetch (roughly 50-200ms) instead of a cache hit (under 1ms), and
nothing becomes memory unless you write it yourself.

This package is the most complete of the five: the middleware reports turns,
tool calls and reasoning, so nothing is left for you beyond opening the stream.
See `reference/streaming.md`.

## Live doc

`https://docs.maximem.ai/integrations/vercel-adk`

---
*Accurate as of `maximem-synap` 0.5.1 (Python) · `@maximem/synap-js-sdk` 0.5.1 (JS) — verified 2026-09-25. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
