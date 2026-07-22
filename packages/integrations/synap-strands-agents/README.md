# maximem-synap-strands-agents

[Maximem Synap](https://maximem.ai) memory for [Strands Agents](https://strandsagents.com/).
Four surfaces, each mapped onto a native Strands extension point.

```bash
pip install maximem-synap-strands-agents
```

| Surface | Class / function | Strands extension point | What it does |
|---|---|---|---|
| Long-term memory | `SynapMemoryStore` | `MemoryStore` → `MemoryManager` | Synap as the agent's memory backend: recall, prompt injection, server-side extraction |
| Short-term context | `SynapShortTermHook` | `HookProvider` (`BeforeInvocationEvent`) | Injects Synap's working-memory summary before each turn |
| Explicit tools | `create_synap_tools` | `@tool` | `search_memory` / `store_memory` for agents not using `MemoryManager` |
| Anticipation feed | `SynapStreamHook` | `HookProvider` (message + tool-call events) | Feeds turns and tool intent onto Synap's gRPC Listen stream |

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
for conversation batches — the latter ingests the assembled transcript under a **stable
`document_id`** so `MemoryManager`'s periodic extraction updates one document instead of
duplicating. (Recorded conversation messages, by contrast, feed only short-term compaction — so
they are deliberately *not* the write path here.)

## Short-term context — `SynapShortTermHook`

Strands' `system_prompt` is static, so working-memory context is injected via a hook. It folds
Synap's short-term summary into the current turn's first user message.

```python
from synap_strands_agents import SynapShortTermHook

agent = Agent(
    system_prompt="You are a support agent.",
    hooks=[SynapShortTermHook(sdk, conversation_id="conv_abc")],
)
```

`conversation_id` is required and explicit. Empty context is a no-op; SDK failures are swallowed
by default (`on_error="fallback"`), or set `on_error="raise"` for strict environments.

> **Note.** Strands exposes no ephemeral per-call injection hook to user code, so the injected
> context becomes part of the conversation the agent persists. The hook folds one context block
> into each turn's user message (with an idempotency guard) rather than adding throwaway turns.

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

## Anticipation feed — `SynapStreamHook`

Makes the agent a participant in Synap's real-time anticipation pipeline by feeding its turns and
tool-call intent onto the gRPC Listen stream. The **app owns the stream lifecycle**; the hook only
feeds an already-open stream and no-ops when none is active.

```python
from synap_strands_agents import SynapStreamHook

await sdk.instance.listen()                      # app opens the stream
hook = SynapStreamHook(sdk, conversation_id="conv_abc", user_id="alice")
agent = Agent(hooks=[hook, SynapShortTermHook(sdk, conversation_id="conv_abc")])
# ... run the agent ...
await sdk.instance.stop_listening()              # app closes it on shutdown
```

> **Run on one event loop.** The stream's background tasks bind to the loop `listen()` ran on;
> construct the SDK, call `listen()`, and run the agent on that same asyncio loop. The hook feeds
> real-time *signal* — it does not durably store memories (that's `SynapMemoryStore` / the tools).

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
