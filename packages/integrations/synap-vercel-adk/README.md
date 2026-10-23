# @maximem/synap-vercel-adk

Vercel AI SDK middleware that wraps any language model with **Synap context** — automatically fetching user memory, preferences, and conversation history and injecting them into every LLM call.

```bash
npm install @maximem/synap-vercel-adk
```

---

## How it works

Synap maintains long-term memory about your users: facts, preferences, past episodes, emotional context, and conversation history. This package sits between your app and any LLM (Anthropic, OpenAI, Google, etc.) and:

1. **Fetches** the relevant context for the current user before the LLM call
2. **Injects** it as a `<synap_context>` block in the system prompt
3. **Calls** your underlying model with the enriched prompt
4. **Records** the new user message and the model's reply as conversation turns after the response

The developer writes zero context management code.

---

## Prerequisites

- Synap API key from the [Synap dashboard](https://dashboard.maximem.ai)
- The `ai` package (`npm install ai`)
- Any Vercel AI SDK provider (`@ai-sdk/anthropic`, `@ai-sdk/openai`, etc.)

---

## Quick start

```typescript
import { createSynap } from '@maximem/synap-vercel-adk';
import { anthropic } from '@ai-sdk/anthropic';
import { generateText } from 'ai';

const synap = await createSynap({
  apiKey: process.env.SYNAP_API_KEY!,
});

const { text } = await generateText({
  model: synap.wrap(anthropic('claude-sonnet-4-6'), { userId: 'user_123' }),
  messages: [{ role: 'user', content: 'What do you know about me?' }],
});
```

That's it. Synap context is fetched and injected, and the new turn is recorded automatically. The gRPC stream opens on the first call; you do not have to call `listen()`.

---

## Next.js / streaming chat

**`app/lib/synap.ts`** — initialize once at app startup:

```typescript
import { createSynap } from '@maximem/synap-vercel-adk';

export const synap = await createSynap({
  apiKey: process.env.SYNAP_API_KEY!,
});

// Optional: the gRPC stream opens on the first model call by itself.
// Call this to have it ready earlier. Node.js only, a safe no-op in Edge Runtime.
await synap.listen();
```

**`app/api/chat/route.ts`** — one route handler, zero boilerplate:

```typescript
import { streamText } from 'ai';
import { anthropic } from '@ai-sdk/anthropic';
import { synap } from '@/lib/synap';

export async function POST(req: Request) {
  const { messages, userId, conversationId } = await req.json();

  return streamText({
    model: synap.wrap(anthropic('claude-sonnet-4-6'), { userId, conversationId }),
    messages,
  }).toDataStreamResponse();
}
```

**`app/page.tsx`** — full streaming chat UI:

```tsx
'use client';
import { useChat } from 'ai/react';

export default function Chat() {
  const { messages, input, handleSubmit, handleInputChange } = useChat();

  return (
    <div>
      {messages.map(m => (
        <div key={m.id}>
          <strong>{m.role}:</strong> {m.content}
        </div>
      ))}
      <form onSubmit={handleSubmit}>
        <input value={input} onChange={handleInputChange} placeholder="Say something..." />
        <button type="submit">Send</button>
      </form>
    </div>
  );
}
```

---

## Context scopes

Synap stores context at four scopes. Pass whichever identifiers are relevant:

```typescript
// User scope — long-term facts, preferences, episodes for a specific user
synap.wrap(model, { userId: 'user_123' })

// Conversation scope — session context, recent turns, compaction summary
synap.wrap(model, { userId: 'user_123', conversationId: 'conv_abc' })

// Customer scope — org-level context shared across all users in a company
synap.wrap(model, { userId: 'user_123', customerId: 'acme_corp' })
```

All three can be combined — Synap fetches and merges context from all applicable scopes.

---

## gRPC stream (opens by default)

The first model call opens a persistent bidirectional gRPC stream to Synap. Turns, tool calls, tool results and reasoning go on it, and the server pushes context bundles for your users *before* they send a message, so when the request arrives the context is already in the in-memory cache and no HTTP call is needed.

```
Without the stream:  request → HTTP fetch context → inject → LLM   (~50-200ms overhead)
With the stream:     request → cache hit → inject → LLM             (<1ms overhead)
```

- The first call does not wait for it: the stream opens alongside the context fetch, with a 10s limit. If it cannot open (Edge Runtime, blocked network, server down), turns are recorded over REST instead and nothing is retried for 60s. Turns are never sent both ways.
- An open stream does not keep a script alive: the process exits when your work is done.
- To turn it off, set `SYNAP_AUTO_LISTEN=false` (or pass `autoListen: false` to `createSynap`). Everything then goes over REST.
- `listen()` is optional. Use it at startup to open the stream before the first request:

```typescript
// instrumentation.ts (Next.js)
export async function register() {
  if (process.env.NEXT_RUNTIME === 'nodejs') {
    const { synap } = await import('@/lib/synap');
    await synap.listen();
  }
}
```

---

## Conversation history and compaction

The AI SDK has no compaction: you send the whole `messages` list on every call. Synap compacts a conversation server-side as it grows. When you pass a `conversationId` and Synap has compacted that conversation, the model sees Synap's compaction in place of the earlier turns in your `messages`:

```
system  (your system prompt + <synap_context>)
system  <synap_conversation_history> …compaction… </synap_conversation_history>
user / assistant   the turns after the compaction
user    (the current turn, with any tool calls and results, untouched)
```

Until Synap has compacted the conversation, your messages are sent as you gave them. If the history fetch fails, the same. This needs `injectContext` (the default) and a `conversationId`; there is no separate switch.

Each call records only the new user message and the reply as turns, so Synap always has the conversation it compacts. Your resent history is not re-ingested.

---

## API reference

### `createSynap(options)`

Creates and initializes a `SynapProvider`. Returns a promise.

```typescript
const synap = await createSynap({
  apiKey?: string;          // falls back to SYNAP_API_KEY env var
  baseUrl?: string;         // advanced: override API base URL
  grpcHost?: string;        // advanced: override gRPC host
  grpcPort?: number;        // advanced: override gRPC port
  grpcUseTls?: boolean;     // advanced: override gRPC TLS
  autoListen?: boolean;     // open the gRPC stream on first use. default: true
});
```

**Credential resolution order:**
1. `apiKey` option
2. `SYNAP_API_KEY` environment variable

---

### `synap.wrap(model, options?)`

Wraps any Vercel AI SDK `LanguageModelV1` (AI SDK 3 and 4) with Synap context middleware. AI SDK 5 models (`LanguageModelV2`) are not supported: context is injected, but replies and tool calls are not recorded.

```typescript
synap.wrap(model, {
  userId?: string;           // user to fetch context for
  customerId?: string;       // customer/org scope
  conversationId?: string;   // session scope
  contextTypes?: Array<'facts' | 'preferences' | 'episodes' | 'emotions' | 'temporal_events'>;
  maxContextResults?: number; // default: 10
  writeMemory?: boolean;      // record the turn (user message + reply) after the response. default: true
  injectContext?: boolean;    // inject context into prompt. default: true
})
```

Works with `generateText`, `streamText`, `generateObject`, `streamObject` — any Vercel AI SDK function.

---

### `synap.listen()`

Opens the gRPC stream now instead of on the first call. Optional. Node.js only, safe to call anywhere. Models you wrapped earlier use it too.

### `synap.stopListening()`

Gracefully closes the gRPC stream. It stays closed (turns go over REST) until you call `listen()` again.

### `synap.isListening`

`boolean` — whether the gRPC stream is currently connected.

### `synap.cacheSize`

`number` — current number of entries in the anticipation cache (diagnostic).

---

## Environment variables

| Variable | Description |
|---|---|
| `SYNAP_API_KEY` | Your Synap API key |
| `SYNAP_AUTO_LISTEN` | `false` stops the gRPC stream opening on first use. Default: on |

---

## Works with any provider

```typescript
import { openai } from '@ai-sdk/openai';
import { anthropic } from '@ai-sdk/anthropic';
import { google } from '@ai-sdk/google';

synap.wrap(openai('gpt-4o'), { userId })
synap.wrap(anthropic('claude-sonnet-4-6'), { userId })
synap.wrap(google('gemini-2.0-flash'), { userId })
```

---

## License

MIT © [Maximem](https://maximem.ai)
