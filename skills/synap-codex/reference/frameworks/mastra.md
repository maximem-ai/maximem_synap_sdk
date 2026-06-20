# Mastra

```bash
npm install @maximem/synap-mastra @mastra/core zod
```

**TypeScript only.** For the Mastra agent framework.

| Export | Purpose |
| --- | --- |
| `SynapMemory` | Extends `MastraMemory` to persist agent memory in Synap |
| `synapSearchTool` | Factory returning a Mastra-compatible search tool |
| `synapStoreTool` | Factory returning a Mastra-compatible store tool |

## Quick start

```typescript
import { Agent } from "@mastra/core";
import { openai } from "@ai-sdk/openai";
import { SynapMemory, synapSearchTool, synapStoreTool } from "@maximem/synap-mastra";

// `sdk` is a constructed Synap SDK instance, wired once at startup (see reference/sdk-setup.md).
// SynapMemory expects an SDK exposing fetch / conversation.record_message /
// conversation.context.get_context_for_prompt / memories.create — confirm the exact wrapper
// against the live doc linked at the bottom of this file.

const agent = new Agent({
  name: "MemoryAgent",
  instructions:
    "You are an agent with persistent memory. Use synapSearch to recall context and synapStore to remember new information.",
  model: openai("gpt-4o"),

  memory: new SynapMemory({
    sdk,
    userId: "alice",
    customerId: "acme",   // optional
  }),

  tools: {
    synapSearch: synapSearchTool({ sdk, userId: "alice", customerId: "acme" }),
    synapStore: synapStoreTool({ sdk, userId: "alice", customerId: "acme" }),
  },
});

const result = await agent.generate("What do you remember about my project deadlines?");
console.log(result.text);
```

## SynapMemory — automatic memory

```typescript
const memory = new SynapMemory({
  sdk,
  userId: "alice",
  customerId: "acme",          // optional — scopes to org
  mode: "accurate",            // "accurate" (default) | "fast"
  injectSystemContext: true,   // default; set false for recall-only (no context preamble)
});
// The conversation/thread is supplied by Mastra per call (threadId), not in the constructor.
```

Methods Mastra calls automatically (you don't call these directly):

```
getSystemMessage()   // fetches Synap context (sdk.fetch) → injected as a system preamble
recall()             // loads recent thread messages via conversation.context.get_context_for_prompt
saveMessages()       // persists each user/assistant turn via conversation.record_message
```

Thread metadata is kept in-process; working memory and delete are best-effort no-ops (Synap has no delete API) and warn once.

## Tools — explicit control

Use as alternative or complement to `SynapMemory`:

```typescript
const agent = new Agent({
  tools: {
    synapSearch: synapSearchTool({
      sdk,
      userId: "alice",
      maxResults: 5,
      mode: "accurate",
    }),
    synapStore: synapStoreTool({
      sdk,
      userId: "alice",
    }),
  },
});
```

Schemas:

```typescript
// synapSearchTool
z.object({
  query: z.string().describe("What to search for in memory"),
  maxResults: z.number().optional().default(5),
})

// synapStoreTool
z.object({
  content: z.string().describe("The information to remember"),
  memoryType: z.string().optional().default("fact"),
})
```

## Memory vs. tools

| | `SynapMemory` | Tools |
| --- | --- | --- |
| Context injection | Automatic on every `generate` | On-demand when model calls the tool |
| Memory storage | Automatic after every response | On-demand when model calls the tool |
| Best for | Always-on memory | Agents that decide when memory matters |

**Use both** for maximum flexibility — `SynapMemory` for automatic recall, `synapStore` for explicit bookmarking by the model.

## Live doc

`https://docs.maximem.ai/integrations/mastra`

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.2.4 (JS) — verified 2026-06-17. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
