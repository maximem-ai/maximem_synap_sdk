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
4. **Writes** the conversation turn back to Synap memory after the response

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

That's it. Synap context is fetched, injected, and the conversation is written back to memory automatically.

---

## Next.js / streaming chat

**`app/lib/synap.ts`** — initialize once at app startup:

```typescript
import { createSynap } from '@maximem/synap-vercel-adk';

export const synap = await createSynap({
  apiKey: process.env.SYNAP_API_KEY!,
});

// Optional: open the gRPC stream for real-time context anticipation
// Node.js only — safe no-op in Edge Runtime
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

## gRPC anticipation stream (optional)

When `synap.listen()` is called, the provider opens a persistent bidirectional gRPC stream to Synap. The server proactively pushes context bundles for your users *before* they send a message — so when the request arrives, context is already in the in-memory cache and no HTTP call is needed.

```
Without listen():   request → HTTP fetch context → inject → LLM   (~50-200ms overhead)
With listen():      request → cache hit → inject → LLM             (<1ms overhead)
```

`listen()` is Node.js only. In Edge Runtime or Vercel Serverless, it silently no-ops and the provider falls back to HTTP context fetching automatically.

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
});
```

**Credential resolution order:**
1. `apiKey` option
2. `SYNAP_API_KEY` environment variable

---

### `synap.wrap(model, options?)`

Wraps any Vercel AI SDK `LanguageModelV1` with Synap context middleware.

```typescript
synap.wrap(model, {
  userId?: string;           // user to fetch context for
  customerId?: string;       // customer/org scope
  conversationId?: string;   // session scope
  contextTypes?: Array<'facts' | 'preferences' | 'episodes' | 'emotions' | 'temporal_events'>;
  maxContextResults?: number; // default: 10
  writeMemory?: boolean;      // write turn to memory after response. default: true
  injectContext?: boolean;    // inject context into prompt. default: true
})
```

Works with `generateText`, `streamText`, `generateObject`, `streamObject` — any Vercel AI SDK function.

---

### `synap.listen()`

Opens the gRPC anticipation stream. Node.js only, safe to call anywhere.

### `synap.stopListening()`

Gracefully closes the gRPC stream.

### `synap.isListening`

`boolean` — whether the gRPC stream is currently connected.

### `synap.cacheSize`

`number` — current number of entries in the anticipation cache (diagnostic).

---

## Environment variables

| Variable | Description |
|---|---|
| `SYNAP_API_KEY` | Your Synap API key |

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
