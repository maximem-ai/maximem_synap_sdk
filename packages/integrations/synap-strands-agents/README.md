# maximem-synap-strands-agents

[Maximem Synap](https://maximem.ai) memory for [Strands Agents](https://strandsagents.com/).
Four surfaces, each mapped onto a native Strands extension point.

```bash
pip install maximem-synap-strands-agents
```

| Surface | Class / function | Strands extension point | What it does |
|---|---|---|---|
| Long-term memory | `SynapMemoryStore` | `MemoryStore` → `MemoryManager` | Synap as the agent's memory backend: recall, prompt injection, server-side extraction |
| Short-term context | `SynapShortTermHook` | `HookProvider` + model-call middleware | Sends the model Synap's compaction in place of the earlier turns once Synap has compacted the conversation |
| Explicit tools | `create_synap_tools` | `@tool` | `search_memory` / `store_memory` for agents not using `MemoryManager` |
| Turns and anticipation | `SynapStreamHook` | `HookProvider` (message + tool-call events) | Writes turns, tool calls and results on Synap's gRPC stream (opened on first use) |

All four take an already-constructed `MaximemSynapSDK` — the app owns the SDK, its
credentials, and (for streaming) its connection lifecycle.

## Long-term memory — `SynapMemoryStore`

Register Synap as a Strands `MemoryStore`; `MemoryManager` then handles recall, automatic
prompt injection, and extraction.

```python
from strands import Agent
from strands.memory import MemoryManager
from maximem_synap import MaximemSynapSDK
from synap_strands_agents import SynapMemoryStore

sdk = MaximemSynapSDK(api_key="sk-...")
store = SynapMemoryStore(sdk, user_id="alice", customer_id="acme")

agent = Agent(memory_manager=MemoryManager(stores=[store]))
```

`search` reads Synap's long-term layer (`sdk.fetch`) and returns structured `MemoryEntry`
objects. Writes go through `sdk.memories.create`: `add` for single facts, and `add_messages`
for conversation batches. `add_messages` records the batch as conversation turns, not as a
transcript through `memories.create`: Synap extracts from the conversation it keeps, so a
transcript on top of the turns `SynapStreamHook` already wrote would be extracted twice. Turns the
hook wrote are skipped one for one, so the two compose; without the hook, this is what gets the
conversation to Synap. Pass the hook's `conversation_id` as the store's so both write to one
conversation.

## Short-term context — `SynapShortTermHook`

Strands keeps the conversation in `agent.messages` and sends all of it. The hook installs
model-call middleware (`InvokeModelStage`) that, once Synap has compacted the conversation, sends
the model Synap's compaction block and the turns after it **in place of the turns before the
current one**. That is per model call: `agent.messages` is not rewritten, Strands' own conversation
manager stays installed (it sees a short history and does not fire), and it is what the model is
sent whenever Synap has nothing.

```python
from synap_strands_agents import SynapShortTermHook

agent = Agent(
    system_prompt="You are a support agent.",
    hooks=[SynapShortTermHook(sdk, conversation_id="conv_abc")],
)
```

`conversation_id` is required and explicit. SDK failures are swallowed by default
(`on_error="fallback"`), or set `on_error="raise"` for strict environments.

> **Note.** Where the framework sends no history (an agent in a restarted process) and Synap has
> no compaction yet, the hook folds Synap's recent-conversation block into the first user message,
> so the model still knows what came before. Strands has no ephemeral per-call injection hook for
> user code, so that block stays in the message; the stream hook never records it as something the
> user said. With a history, or a compaction, nothing is folded. `SynapStreamHook` installs the same
> middleware, so either hook is enough.

## Explicit tools — `create_synap_tools`

For agents that want direct control instead of `MemoryManager`:

```python
from synap_strands_agents import create_synap_tools

agent = Agent(
    system_prompt="You are a helpful assistant.",
    tools=create_synap_tools(sdk, user_id="alice", customer_id="acme"),
)
```

Adds `search_memory(query)` and `store_memory(content)`.

## Turns and anticipation: `SynapStreamHook`

Writes the agent's turns, tool calls and tool results to Synap. Nothing to set up: the **first turn
opens the gRPC stream** (`sdk.instance.listen()`), and everything after goes on it. If the stream
cannot open, the turns are written over REST (`conversation.record_message`) instead, never both;
tool events are stream-only. `SYNAP_AUTO_LISTEN=false` (or `set_auto_listen(sdk, False)`) stays on
REST and never opens a stream; a stream you opened yourself is used as is.

```python
from synap_strands_agents import SynapStreamHook

hook = SynapStreamHook(sdk, conversation_id="conv_abc", user_id="alice")
agent = Agent(hooks=[hook, SynapShortTermHook(sdk, conversation_id="conv_abc")])
```

> **Run on one event loop.** The stream's background tasks bind to the loop that opened it; run the
> agent on one asyncio loop (a caller that wraps each turn in its own `asyncio.run()` stays on REST).
> Beyond the turns, the hook feeds real-time *signal*; it does not store long-term memories
> (that's `SynapMemoryStore` / the tools).

## Error policy

- **Reads** (`search`, `search_memory`) degrade gracefully — log at ERROR, return empty.
- **Writes** (`add`, `add_messages`, `store_memory`) raise `SynapIntegrationError`.
- **Stream sends** log and never raise — they run inside Strands' event loop and must not abort a turn.

## Not in scope: `SessionManager`

This integration does not implement Strands' `SessionManager` / snapshot storage. That persists
opaque conversation snapshots for replay, which is a different concern from Synap's semantic
memory. Use `FileSessionManager` / `S3SessionManager` for durable sessions *and* a
`SynapMemoryStore` for memory — they compose.

## License

Apache-2.0
