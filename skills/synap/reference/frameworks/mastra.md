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
  customerId: "acme",      // optional — scopes to org
  conversationId: "t-001",  // optional — scopes to session
  maxResults: 8,
  mode: "fast",            // "fast" | "accurate"
});
```

Methods overridden from `MastraMemory`:

```
remember(message)            // ingest into Synap
recall(query, options)       // semantic search → MemoryMessage[]
getMessages(threadId)        // retrieve thread from Synap
```

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
