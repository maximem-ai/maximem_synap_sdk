# @maximem/synap-claude-agent

Synap memory integration for Anthropic's [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview) (TypeScript).

Python sibling published as `synap-claude-agent`.

## Install

```bash
npm install @maximem/synap-claude-agent @anthropic-ai/claude-agent-sdk zod
```

## Two plug points

### 1. Hooks — automatic context injection

```ts
import { query } from "@anthropic-ai/claude-agent-sdk";
import { createSynapHooks } from "@maximem/synap-claude-agent";

const sdk = /* your Synap SDK instance */;

for await (const message of query({
  prompt: "What did I tell you about my trial?",
  options: {
    hooks: createSynapHooks({ sdk, userId: "alice", customerId: "acme" }),
  },
})) {
  console.log(message);
}
```

### 2. MCP tools — explicit read/write

```ts
import { createSynapMcpServer } from "@maximem/synap-claude-agent";

const options = {
  mcpServers: { synap: createSynapMcpServer({ sdk, userId: "alice" }) },
  allowedTools: ["mcp__synap__synap_search", "mcp__synap__synap_remember"],
};
```

Use both together for automatic context injection plus explicit agent read/write.

## Error policy

- **Hooks** never throw — SDK failures log and fall through (no context injected, no prompt recorded).
- **`synap_search`** returns a "no context available" message on SDK failure — keeps the agent loop alive.
- **`synap_remember`** returns `isError: true` on ingestion failure — silent drops would hide outages, so failures surface to the agent explicitly.
