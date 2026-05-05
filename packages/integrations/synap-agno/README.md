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
from synap_agno import SynapDb

sdk = MaximemSynapSDK(api_key="sk-...")

agent = Agent(
    db=SynapDb(sdk, customer_id="acme"),
    model=OpenAIChat(id="gpt-4o-mini"),
    enable_user_memories=True,
)

agent.run("Remember that I prefer tea over coffee", user_id="alice")
agent.run("What do you remember about me?", user_id="alice")
```

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
