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

## Identity

Both surfaces auto-scope from the eve session — `user_id` from `ctx.session.auth.current.principalId`
and `conversation_id` from `ctx.session.id`. On an unauthenticated channel (`auth.current` is `null`),
pass an explicit `userId`:

```ts
export default createSynapStoreTool({ sdk, userId: "alice", customerId: "acme" });
```

## Error policy

| Operation | Behavior |
|---|---|
| Reads (`synap_search`, instructions recall) | Degrade — log at ERROR, return empty / `null` so the turn proceeds |
| Writes (`synap_store`) | Raise `SynapIntegrationError` — ingestion outages are surfaced to the agent |
| Deletes | Not supported (Synap has no public delete API) |

## License

Apache-2.0
