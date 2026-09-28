# @maximem/synap-eve

Synap memory integration for [Vercel eve](https://vercel.com/eve) agents.

eve's built-in durability (Vercel Workflows) persists a single session's turn state so it survives
crashes and redeploys — but that is short-term session state, not cross-session memory. This package
adds **Synap** as the durable, cross-session memory layer, through two native eve extension points.

> Requires `eve >= 0.25.0` and `zod >= 3.23`. Tested against `eve@0.25.1` (AI SDK v7).

## Install

```bash
npm install @maximem/synap-eve
# peers: eve, zod
```

## Surfaces

### 1. Memory tools — `agent/tools/`

The filename is the model-facing tool name. Point the factories at a configured Synap SDK:

```ts
// agent/tools/synap_search.ts
import { createSynapSearchTool } from "@maximem/synap-eve";
import { sdk } from "../lib/synap.js";
export default createSynapSearchTool({ sdk });
```

```ts
// agent/tools/synap_store.ts
import { createSynapStoreTool } from "@maximem/synap-eve";
import { sdk } from "../lib/synap.js";
export default createSynapStoreTool({ sdk });
```

`synap_search` lets the model pull long-term context on demand; `synap_store` persists an explicit
fact or preference for future sessions.

### 2. Short-term context — `agent/instructions/`

A per-turn resolver that injects Synap's compacted short-term summary into the system prompt. It
**augments** `instructions.md`; it never replaces it.

```ts
// agent/instructions/synap.ts
import { createSynapInstructions } from "@maximem/synap-eve";
import { sdk } from "../lib/synap.js";
export default createSynapInstructions({ sdk });
```

### 3. The live stream — `agent/hooks/`

Synap has a bidirectional `Listen` stream that carries five per-turn events. Two of them —
`user_message` and `assistant_message` — are persisted as conversation. The other three
(`agent_thinking`, `tool_call`, `tool_result`) have no other durable home: they are what lets the
anticipation agent predict the next turn, so that the next fetch is not a cold retrieval.

```ts
// agent/hooks/synap.ts
import { createSynapStreamHooks } from "@maximem/synap-eve";
import { sdk } from "../lib/synap.js";
export default createSynapStreamHooks({ sdk });
```

Your app opens the stream once, wherever it configures the SDK:

```ts
await sdk.instance.listen();
```

What each eve stream event becomes:

| eve event | Synap event |
|---|---|
| `message.received` | `user_message` |
| `message.completed` | `assistant_message` (one per completed assistant message, including a reply given before a tool call) |
| `reasoning.completed` | `agent_thinking`, with eve's own `stepIndex` |
| `actions.requested` | `tool_call` per action, with eve's own `callId` — tool calls, subagent and remote-agent calls, and `load-skill` |
| `action.result` | `tool_result`, under the same `callId` |
| `session.completed` | closes the Synap session |

**Nothing changes if you never call `listen()`.** Every handler is a no-op without an open stream.
There is no REST fallback here and that is deliberate: this package has never written conversation
turns, and adding a write would change what happens for someone who has not opted into streaming.

Three options, all on by default:

```ts
export default createSynapStreamHooks({
  sdk,
  // A tool result is usually your own customer's data. It is an anticipation hint
  // and never becomes a long-term memory, but it does leave your process.
  reportToolResults: false,
  reportThoughts: false,
  // Turn this off when several eve sessions share one explicit conversationId.
  endSessionOnComplete: false,
});
```

## Identity

All three surfaces auto-scope from the eve session — `user_id` from
`ctx.session.auth.current.principalId` and `conversation_id` from `ctx.session.id`. On an
unauthenticated channel (`auth.current` is `null`), pass an explicit `userId`:

```ts
export default createSynapStoreTool({ sdk, userId: "alice", customerId: "acme" });
```

## Error policy

| Operation | Behavior |
|---|---|
| Reads (`synap_search`, instructions recall) | Degrade — log at ERROR, return empty / `null` so the turn proceeds |
| Writes (`synap_store`) | Raise `SynapIntegrationError` — ingestion outages are surfaced to the agent |
| Deletes | Not supported (Synap has no public delete API) |
| Stream hooks | Never raise — a hook runs inside eve's own accepted-event path, and no context is worth ending a run for |

## License

Apache-2.0
