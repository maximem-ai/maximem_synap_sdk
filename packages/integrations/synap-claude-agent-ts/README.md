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

`createSynapHooks` installs seven hooks, not one. The first prompt opens
Synap's live stream by itself; see the next section.

### 1b. The live stream — what the agent is doing, as it does it

Synap's `Listen` stream carries five per-turn events. The hooks above open the
stream on the first event (no `sdk.instance.listen()` call needed) and report
four of them:

| Synap event | Hook | Persisted? |
| --- | --- | --- |
| `user_message` | `UserPromptSubmit` | yes, and feeds long-term extraction |
| `tool_call` | `PreToolUse` | no, anticipation only |
| `tool_result` | `PostToolUse`, `PostToolUseFailure` | no, anticipation only |
| `assistant_message` | `Stop` | yes, and **this is the event that wakes the anticipation agent** |
| end of session | `SessionEnd` | — |

Things worth knowing:

- **`agent_thinking` is not reported**, because the Claude Agent SDK exposes no
  hook carrying the model's reasoning text.
- **Stream first, REST only as a fallback, never both.** The server persists a
  turn it receives on the stream itself, so a turn goes over REST
  (`sdk.conversation.record_message`) only when the stream cannot open. The
  assistant's reply is written the same way.
- **Opt out** with `SYNAP_AUTO_LISTEN=false` (or `setAutoListen(sdk, false)`):
  turns then go over REST and the stream-only events are not sent. A stream you
  opened with `listen()` yourself is used either way. The stream does not keep a
  plain `node script.js` alive once your work is done.
- `recordUserPrompts: false` stops the user's prompt being recorded. The
  assistant's reply and the tool events are not affected by it.

### 1c. Synap's compaction

Claude Code owns the transcript and runs its own compaction, and the hook
surface cannot replace either: `PreCompact` and `PostCompact` receive input and
return nothing the CLI reads, and `additionalContext` only ever adds a message.
What `createSynapHooks` does instead is add Synap's compaction block as
`additionalContext`: on the first prompt (a new process on an existing Synap
conversation), whenever Synap compacts again, and from a `SessionStart` hook
with `source: "compact"`, which fires right after the CLI replaced its history
with its own summary and discarded what was added before. Only the compaction
block goes in, never the turns after it, which the transcript already holds.
`createSynapShortTermHook` injects the same block, once per compaction.

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
