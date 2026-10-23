# synap-agno

Synap integration for [Agno](https://docs.agno.com) — backs Agno's user memories with Synap's semantic memory store.

## Install

```bash
pip install synap-agno
```

Requires `agno>=2.0`, `maximem-synap>=0.2.0`.

## Quickstart

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
import asyncio

from maximem_synap import MaximemSynapSDK
from synap_agno import SynapDb, create_synap_hooks

sdk = MaximemSynapSDK(api_key="sk-...")
asyncio.run(sdk.initialize())  # required: without it every Synap call fails

agent = Agent(
    db=SynapDb(sdk, customer_id="acme"),
    model=OpenAIChat(id="gpt-4o-mini"),
    update_memory_on_run=True,  # Agno 2.5+; enable_user_memories was removed in Agno 3.0
    **create_synap_hooks(sdk, customer_id="acme"),
)

agent.run("Remember that I prefer tea over coffee", user_id="alice")
agent.run("What do you remember about me?", user_id="alice")
```

For async agents (`await agent.arun(...)`) use `AsyncSynapDb` instead of `SynapDb`.

Synap retrieval answers a question; it cannot list everything stored for a user.
`get_user_memories(user_id, search_content="...")` returns what matches that topic; with no
search term it asks `default_query` and returns a best-effort sample.

## Reporting the turn

`create_synap_hooks` returns the `pre_hooks` / `post_hooks` an `Agent` takes. With
`sdk.instance.listen()` running they put the whole turn on Synap's live stream, which
is what the anticipation agent reads. The first run opens that stream by itself, so there is
no `listen()` to call (`SYNAP_AUTO_LISTEN=false` turns that off):

| Event | Source |
|-------|--------|
| user turn | pre-hook, `run_input.input_content_string()` |
| assistant turn | post-hook, `run_output.content` |
| tool call | post-hook, `run_output.tools[].tool_call_id` / `.tool_name` / `.tool_args` |
| tool result | post-hook, `run_output.tools[].result`, under the same `tool_call_id` |
| reasoning | post-hook, `run_output.reasoning_content` and `run_output.reasoning_steps[]` |

Notes:

- **The stream opens by default.** The first hook call opens `sdk.instance.listen()` and the
  turns, tool calls, tool results and reasoning go on it. When the stream cannot carry them
  (it will not open, or `SYNAP_AUTO_LISTEN=false`), the user and assistant turns go over REST
  (`conversation.record_message`) under Agno's `session_id` (or your `conversation_id=`);
  never both. Tool calls, tool results and reasoning need the stream.
- **Synap's compaction replaces the history Agno sends.** The pre-hook reads Synap's view of the
  conversation; once Synap has compacted it, the model sees the compaction block and the turns
  after it instead of the earlier turns Agno stored (the stored session is not changed).
  Until then, and whenever Synap cannot answer, Agno's own history (and its `compaction=`, if
  you set one) is sent as before. This needs `db=` on the agent, as Agno's history does, and
  the hooks on it.
- **Tool events land at the end of the run, not during it.** Agno's `tool_hooks` fire
  around each call, but the arguments Agno passes them carry no call id
  (`function_call` is bound to the next link in the middleware chain, not to the
  `FunctionCall` that holds `call_id`). `run_output.tools` carries Agno's own
  `tool_call_id` on both the call and the result, so that is what gets used.
- **`run()` and `arun()`.** `create_synap_hooks` returns plain functions, which Agno
  runs on both paths, bridged to the async SDK. If every call site is `arun()`, use
  `create_synap_async_hooks` instead: the hooks are awaited natively and no event
  loop is patched. Agno's sync `run()` **skips** coroutine hooks with a warning.
- **Scope.** `conversation_id` defaults to Agno's `session_id`; `user_id` comes off
  the run, so one agent serving many users reports each under their own scope.

## Scope

Agno 2.x unifies every persistence concern (sessions, traces, evals, metrics, knowledge, culture, memories) under a single `BaseDb` with 46+ abstract methods. Synap natively backs only **user memories**, so `SynapDb`:

- Extends Agno's `InMemoryDb`
- Overrides user-memory methods (`upsert_user_memory`, `get_user_memory`, `get_user_memories`, `get_all_memory_topics`) to route through Synap
- Leaves sessions, traces, evals, metrics, knowledge, and culture in-process (inherited from `InMemoryDb`)

Need durable sessions or traces? Use `SqliteDb` / `PostgresDb` from Agno directly — this package is scoped to memory specifically.

## Error policy

- **Reads** (`get_user_memory`, `get_user_memories`, `get_all_memory_topics`) degrade gracefully — SDK failures log at `ERROR` and return empty results.
- **Writes** (`upsert_user_memory`, `upsert_memories`) surface `SynapIntegrationError` so ingestion outages are observable.
- **Deletes** (`delete_user_memory`, `delete_user_memories`) call `memories.delete` for the Synap ids `get_user_memories` reports. Any other id (Agno assigns its own when it saves a memory) and `clear_memories` warn once and no-op. A failed delete is logged, not raised.
- **Stats** (`get_user_memory_stats`) warns once and returns `([], 0)` — Synap doesn't expose aggregate counts.
