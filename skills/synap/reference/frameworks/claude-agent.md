# Claude Agent SDK

Available in **Python and TypeScript**.

```bash
# Python
pip install synap-claude-agent

# TypeScript
npm install @maximem/synap-claude-agent @anthropic-ai/claude-agent-sdk zod
```

For Anthropic's official Claude Agent SDK.

| Export | Lang | Purpose |
| --- | --- | --- |
| `create_synap_hooks` / `createSynapHooks` | Py + TS | Hooks dict for automatic context injection + recording |
| `create_synap_mcp_server` / `createSynapMcpServer` | Py + TS | MCP server exposing `synap_search` and `synap_remember` tools |
| `buildSynapTools` | TS | Raw tool definitions for manual composition |

## Hooks — automatic memory (zero tool calls)

Synap injects context before each turn and records after. The model never sees the tools.

**Python:**

```python
import asyncio
from anthropic.claude_agent_sdk import query, ClaudeAgentOptions
from synap_claude_agent import create_synap_hooks

hooks = create_synap_hooks(
    sdk=sdk,
    user_id="alice",
    customer_id="acme",          # optional
    conversation_id="conv-001",  # optional; auto-generated if omitted
)

async def main():
    async for message in query(
        prompt="What did I tell you about my trial account?",
        options=ClaudeAgentOptions(hooks=hooks),
    ):
        print(message)

asyncio.run(main())
```

**TypeScript:**

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";
import { createSynapHooks } from "@maximem/synap-claude-agent";

const hooks = createSynapHooks({
  sdk,
  userId: "alice",
  customerId: "acme",         // optional
  conversationId: "conv-001",  // optional
});

for await (const message of query({
  prompt: "What did I tell you about my trial account?",
  options: { hooks },
})) {
  console.log(message);
}
```

How hooks work: `before_query` fetches context and prepends it as a system message; `after_turn` ingests the completed user + assistant turn. Step 1 failures degrade gracefully; step 2 surfaces.

## MCP server — explicit memory tools

Use when the model should decide when to search/store. Server exposes `synap_search` and `synap_remember`.

**Python:**

```python
from anthropic.claude_agent_sdk import query, ClaudeAgentOptions
from synap_claude_agent import create_synap_hooks, create_synap_mcp_server

hooks = create_synap_hooks(sdk=sdk, user_id="alice")
mcp_server = create_synap_mcp_server(sdk=sdk, user_id="alice")

async for message in query(
    prompt="Search your memory for anything about my project deadlines.",
    options=ClaudeAgentOptions(
        hooks=hooks,
        mcp_servers={"synap": mcp_server},
    ),
):
    print(message)
```

**TypeScript:**

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";
import { createSynapHooks, createSynapMcpServer } from "@maximem/synap-claude-agent";

const hooks = createSynapHooks({ sdk, userId: "alice" });
const mcpServer = createSynapMcpServer({ sdk, userId: "alice" });

for await (const message of query({
  prompt: "Search your memory for anything about my project deadlines.",
  options: {
    hooks,
    mcpServers: { synap: mcpServer },
  },
})) {
  console.log(message);
}
```

## Hooks vs. MCP server — which to use

| | Hooks | MCP server |
| --- | --- | --- |
| Context injection | Automatic, every turn | On-demand via tool call |
| Memory storage | Automatic, every turn | On-demand via tool call |
| Model awareness | Model doesn't see tools | Model decides when to use them |
| Best for | Production agents — memory always relevant | Research / agentic exploration where the model should reason about memory |

**Use both together** for maximum coverage: hooks handle automatic ingestion, MCP tools let the model query memory explicitly when it wants more.

## TypeScript: raw tools

For manual composition without the full MCP server:

```typescript
import { buildSynapTools } from "@maximem/synap-claude-agent";

const tools = buildSynapTools({ sdk, userId: "alice", customerId: "acme" });
// [synapSearchTool, synapRememberTool] — raw Anthropic tool definitions
```

## Live doc

`https://docs.maximem.ai/integrations/claude-agent`

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.3.0 (JS) — verified 2026-06-20. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
