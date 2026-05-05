# @maximem/synap-mastra

Synap memory integration for the [Mastra ADK](https://mastra.ai/docs) (Agent Development Kit).

Drop Synap directly into `new Agent({ memory, tools })` — two plug points, composable or used independently.

## Install

```bash
npm install @maximem/synap-mastra @mastra/core zod
```

## 1. `SynapMemory` — Agent memory

`SynapMemory` extends `MastraMemory` so you can pass it straight to `new Agent({ memory })`:

```ts
import { Agent } from "@mastra/core/agent";
import { SynapMemory } from "@maximem/synap-mastra";

const agent = new Agent({
  name: "support-agent",
  instructions: "You are a helpful assistant.",
  model: /* your model */,
  memory: new SynapMemory({ sdk, userId: "alice", customerId: "acme" }),
});
```

Synap-backed methods:

- `recall({ threadId })` → loads prior turns from `sdk.conversation.context.get_context_for_prompt`
- `saveMessages({ messages })` → persists each turn via `sdk.conversation.record_message`
- `getSystemMessage({ threadId })` → fetches Synap context and returns it as a system-message preamble the Agent injects before the turn

Thread metadata is kept in-process via a Map — adequate for single-process apps; multi-process thread persistence is a v0.2 concern.

Working memory, message deletion, and multi-process thread sharing are not supported in v0.1. Calls log once and no-op or return `null` honestly.

## 2. `synapSearchTool` / `synapStoreTool` — Agent tools

Register them in the Agent's `tools` map so the model can read/write memory explicitly via tool calls:

```ts
import { synapSearchTool, synapStoreTool } from "@maximem/synap-mastra";

const agent = new Agent({
  name: "support-agent",
  instructions: "...",
  model: /* your model */,
  tools: {
    synapSearch: synapSearchTool({ sdk, userId: "alice" }),
    synapStore: synapStoreTool({ sdk, userId: "alice" }),
  },
});
```

Use one or both surfaces; they compose.

## Error policy

- **`SynapMemory.recall` / `getSystemMessage`** — read-side, degrade gracefully (empty recall / null system message) with an `ERROR` log on failure.
- **`SynapMemory.saveMessages`** — write-side, throws on SDK failure (silent drops would hide ingestion outages).
- **`synapSearchTool`** — returns `{ available: false }` on SDK failure; the agent loop keeps going.
- **`synapStoreTool`** — throws on SDK failure so the tool call surfaces as an error to the agent.
