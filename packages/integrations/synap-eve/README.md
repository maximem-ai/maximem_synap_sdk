# @maximem/synap-eve

Synap memory integration for [Vercel eve](https://vercel.com/eve) agents.

eve's built-in durability (Vercel Workflows) persists a single session's turn state so it survives
crashes and redeploys — but that is short-term session state, not cross-session memory. This package
adds **Synap** as the durable, cross-session memory layer, through two native eve extension points.

> Requires `eve >= 0.25.0` and `zod >= 3.23`. Tested against `eve@0.25.1` and `eve@0.82.0` (AI SDK v7).

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

### 2. Synap's compaction: `agent/instructions/` and `agent.ts`

A per-turn resolver that adds Synap's compaction (the summary of the turns the conversation has
outgrown, plus earlier turns found for the user's message) to the system prompt. It **augments**
`instructions.md`; it never replaces it, and it adds nothing until Synap has compacted the
conversation. It does not repeat the recent turns, which eve's own history already holds.

```ts
// agent/instructions/synap.ts
import { createSynapInstructions } from "@maximem/synap-eve";
import { sdk } from "../lib/synap.js";
export default createSynapInstructions({ sdk });
```

eve compacts on its own when the conversation passes `compaction.thresholdPercent` of the context
window (default 0.9), by asking a model to summarize the older turns. `createSynapCompactionModel`
wraps your model so that this one call is answered with Synap's compaction: the checkpoint eve
stores is Synap's compaction instead of an LLM-written summary. Every other call goes to the model
you pass in, and so does the compaction when Synap has none yet.

Set the wrapped model as the agent's **`model`**, and leave `compaction.model` unset:

```ts
// agent/agent.ts
import { defineAgent } from "eve";
import { createSynapCompactionModel } from "@maximem/synap-eve";
import { sdk } from "./lib/synap.js";

export default defineAgent({
  model: createSynapCompactionModel({
    sdk,
    model: openai("gpt-5"), // an AI SDK model object, not a "provider/model" string
    conversationId: "support-alice",
  }),
  modelContextWindowTokens: 400_000, // a model object has no gateway metadata to read
  compaction: {
    thresholdPercent: 0.6, // hand over sooner than the 0.9 default
  },
});
```

Why not `compaction.model`: eve compiles a model object in `agent.ts` into a reference to the
module, and at runtime turns that reference back into a model by loading `agent.ts` and taking its
`model`. A model object in `compaction.model` is therefore replaced by the agent's own model when
eve compacts, and is never called. This is the same in eve 0.25.1 and 0.82.0. With no
`compaction.model`, eve compacts with the agent's model, which is the wrapper.

What this cannot do, from eve's source: eve decides when to compact, so until it does the
model sees eve's own turn-by-turn history; hooks only observe events after they are recorded, so
the stream hooks below cannot supply or veto a summary; and a model call carries no session id, so
the Synap conversation is named in `agent.ts` (use the same `conversationId` you give the hooks).

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

The first event opens the stream, so there is no `sdk.instance.listen()` to call. Set
`SYNAP_AUTO_LISTEN=false` (or call `setAutoListen(sdk, false)`) to keep it closed; a stream you
opened yourself is used either way, and it does not keep a plain `node` script alive once your work
is done.

What each eve stream event becomes:

| eve event | Synap event |
|---|---|
| `message.received` | `user_message` |
| `message.completed` | `assistant_message` (one per completed assistant message, including a reply given before a tool call) |
| `reasoning.completed` | `agent_thinking`, with eve's own `stepIndex` |
| `actions.requested` | `tool_call` per action, with eve's own `callId` — tool calls, subagent and remote-agent calls, and `load-skill` |
| `action.result` | `tool_result`, under the same `callId` |
| `session.completed` | closes the Synap session |

**When the stream cannot open, the turns are still written.** `message.received` and
`message.completed` fall back to REST (`sdk.conversation.record_message`, under the eve session id
or your `conversationId`), so Synap keeps the conversation for compaction and history search. A turn
is never written both ways. Reasoning, tool calls, tool results and the session end are stream-only
and a no-op without an open stream.

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

The hooks, instructions and tools auto-scope from the eve session: `user_id` from
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
