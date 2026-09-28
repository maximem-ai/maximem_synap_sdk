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
from maximem_synap import MaximemSynapSDK
from synap_agno import SynapDb, create_synap_hooks

sdk = MaximemSynapSDK(api_key="sk-...")

agent = Agent(
    db=SynapDb(sdk, customer_id="acme"),
    model=OpenAIChat(id="gpt-4o-mini"),
    enable_user_memories=True,
    **create_synap_hooks(sdk, customer_id="acme"),
)

agent.run("Remember that I prefer tea over coffee", user_id="alice")
agent.run("What do you remember about me?", user_id="alice")
```

## Reporting the turn

`create_synap_hooks` returns the `pre_hooks` / `post_hooks` an `Agent` takes. With
`sdk.instance.listen()` running they put the whole turn on Synap's live stream, which
is what the anticipation agent reads:

| Event | Source |
|-------|--------|
| user turn | pre-hook, `run_input.input_content_string()` |
| assistant turn | post-hook, `run_output.content` |
| tool call | post-hook, `run_output.tools[].tool_call_id` / `.tool_name` / `.tool_args` |
| tool result | post-hook, `run_output.tools[].result`, under the same `tool_call_id` |
| reasoning | post-hook, `run_output.reasoning_content` and `run_output.reasoning_steps[]` |

Notes:

- **Silent without a stream, and never over REST.** No `listen()`, no events. This
  package has never recorded conversation turns over the REST API and these hooks do
  not start: writing the turn twice would extract it twice.
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
- **Deletes** (`delete_user_memory`, `delete_user_memories`, `clear_memories`) warn once and no-op — Synap has no public delete API. Same contract used by [synap-crewai](../synap-crewai/).
- **Stats** (`get_user_memory_stats`) warns once and returns `([], 0)` — Synap doesn't expose aggregate counts.
