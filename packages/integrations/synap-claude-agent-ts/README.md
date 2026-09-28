# @maximem/synap-claude-agent

Synap memory integration for Anthropic's [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview) (TypeScript).

Python sibling published as `synap-claude-agent`.

## Install

```bash
npm install @maximem/synap-claude-agent @maximem/synap-js-sdk @anthropic-ai/claude-agent-sdk zod
```

## Initialize the SDK

```ts
import { createClient } from "@maximem/synap-js-sdk";

// Reads SYNAP_API_KEY from the environment; pass `apiKey` to override.
const sdk = createClient();
await sdk.init();
```

You don't need to provide an instance ID — every Synap API key is bound to exactly one instance, and the SDK resolves it server-side on `init()`. Requires Node 18+ and Python 3.11+ on the host.

## Two plug points

### 1. Hooks — automatic context injection

```ts
import { query } from "@anthropic-ai/claude-agent-sdk";
import { createSynapHooks } from "@maximem/synap-claude-agent";

for await (const message of query({
  prompt: "What did I tell you about my trial?",
  options: {
    hooks: createSynapHooks({ sdk, userId: "alice", customerId: "acme" }),
  },
})) {
  console.log(message);
}
```

`createSynapHooks` installs six hooks, not one. Five of them only do anything
while a Synap stream is open; see the next section.

### 1b. The live stream — what the agent is doing, as it does it

Synap's `Listen` stream carries five per-turn events. Open one and the hooks
above report four of them with no further wiring:

```ts
await sdk.instance.listen();   // that is the whole opt-in
```

| Synap event | Hook | Persisted? |
| --- | --- | --- |
| `user_message` | `UserPromptSubmit` | yes, and feeds long-term extraction |
| `tool_call` | `PreToolUse` | no, anticipation only |
| `tool_result` | `PostToolUse`, `PostToolUseFailure` | no, anticipation only |
| `assistant_message` | `Stop` | yes, and **this is the event that wakes the anticipation agent** |
| end of session | `SessionEnd` | — |

Two things worth knowing:

- **`agent_thinking` is not reported**, because the Claude Agent SDK exposes no
  hook carrying the model's reasoning text. Four of the five events is what the
  framework makes available, and we would rather say so than invent a signal.
- **Stream first, REST only as a fallback, never both.** The server persists a
  turn it receives on the stream itself, so the hook skips
  `sdk.conversation.record_message` when the stream accepted the turn and falls
  back to it when there is no stream or the send failed.

If you never call `listen()`, behaviour is exactly what it was before: the
prompt is recorded over REST and nothing else is sent. `recordUserPrompts:
false` turns off both halves of the conversation on both paths; it does not
affect tool events, which are never persisted.

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
