# @maximem/synap-mastra

Synap memory integration for the [Mastra ADK](https://mastra.ai/docs) (Agent Development Kit).

Drop Synap directly into `new Agent({ memory, tools })` — two plug points, composable or used independently.

## Install

```bash
npm install @maximem/synap-mastra @maximem/synap-js-sdk @mastra/core zod
```

## Initialize the SDK

```ts
import { createClient } from "@maximem/synap-js-sdk";

// Reads SYNAP_API_KEY from the environment; pass `apiKey` to override.
const sdk = createClient();
await sdk.init();
```

You don't need to provide an instance ID — every Synap API key is bound to exactly one instance, and the SDK resolves it server-side on `init()`. Requires Node 18+ and Python 3.11+ on the host.

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

## 3. The live stream — anticipation

Synap has a bidirectional `Listen` stream that carries five per-turn events. Two of them —
`user_message` and `assistant_message` — are persisted as conversation. The other three
(`agent_thinking`, `tool_call`, `tool_result`) have no other durable home: they are what lets the
anticipation agent predict the next turn, so that the next fetch is not a cold retrieval.

`SynapMemory` reports all five. Start a stream and it happens on its own:

```ts
await sdk.instance.listen();   // your app opens this once

const agent = new Agent({
  name: "support-agent",
  model: /* your model */,
  memory: new SynapMemory({ sdk, userId: "alice" }),
});
```

What each part of a saved message becomes:

| Mastra message part | Synap event |
|---|---|
| role `user`, `text` part | `user_message` |
| role `assistant`, `text` part | `assistant_message` |
| `reasoning` part | `agent_thinking` |
| `tool-invocation` part | `tool_call` (with Mastra's own `toolCallId`) |
| the same part once `state` is `result` or `output-error` | `tool_result`, under the same id |
| role `system` | nothing — the instructions are not a turn |

**Stream first, REST second, never both.** The server persists a turn it receives on the stream, so
when the stream takes a turn `saveMessages` does *not* also call `conversation.record_message`.

**Nothing changes if you never call `listen()`.** Every report is a no-op without an open stream, and
`saveMessages` is byte for byte the REST write it always was.

Two options, both on by default:

```ts
new SynapMemory({
  sdk,
  userId: "alice",
  // A tool result is usually your own customer's data. It is an anticipation hint
  // and never becomes a long-term memory, but it does leave your process.
  reportToolResults: false,
  reportThoughts: false,
});
```

## Error policy

- **`SynapMemory.recall` / `getSystemMessage`** — read-side, degrade gracefully (empty recall / null system message) with an `ERROR` log on failure.
- **`SynapMemory.saveMessages`** — write-side, throws on SDK failure (silent drops would hide ingestion outages).
- **`synapSearchTool`** — returns `{ available: false }` on SDK failure; the agent loop keeps going.
- **`synapStoreTool`** — throws on SDK failure so the tool call surfaces as an error to the agent.
- **Stream reporting** — never throws and never blocks the Agent. A failed report falls back to the
  REST write; a failed reasoning or tool event is simply not reported.
