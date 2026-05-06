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
  apiKey: process.env.SYNAP_API_KEY!,
  instanceId: process.env.SYNAP_INSTANCE_ID!,
});

const model = synap.wrap(anthropic("claude-sonnet-4-6"), {
  userId: "alice",
  customerId: "acme",   // optional
});

const { text } = await generateText({
  model,
  messages: [{ role: "user", content: "What do you remember about my account?" }],
});
```

`synap.wrap()` returns a standard Vercel AI SDK `LanguageModel` — pass it anywhere you'd use a plain model. Nothing else changes.

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

Wrap fresh per request to scope per-user without any global state:

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

## Anticipation stream (advanced)

`synap.listen()` opens a gRPC stream that pre-fetches context speculatively before the user's request arrives. Reduces perceived latency in long-lived server processes where you can predict who's about to send a message:

```typescript
const stop = synap.listen({ userId: "alice" });

// later:
stop();
```

Most users don't need this. Reach for it only when you can demonstrably predict the next sender and the read latency is the bottleneck.

## Live doc

`https://docs.maximem.ai/integrations/vercel-adk`
