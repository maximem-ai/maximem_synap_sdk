# CrewAI

`pip install synap-crewai`

Backs CrewAI's unified `Memory` system with Synap. Single class to swap in.

| Class | Purpose |
| --- | --- |
| `SynapStorageBackend` | Implements CrewAI's `StorageBackend` protocol |

## Quick start

```python
from crewai import Agent, Crew, Task
from crewai.memory import Memory
from synap_crewai import SynapStorageBackend

backend = SynapStorageBackend(
    sdk=sdk,
    user_id="alice",
    customer_id="acme",   # optional
)

memory = Memory(storage=backend)

crew = Crew(
    agents=[your_agent],
    tasks=[your_task],
    memory=memory,
)

result = crew.kickoff(inputs={"topic": "quarterly planning"})
```

## How it maps

| CrewAI op | Synap behavior |
| --- | --- |
| `save(value, metadata)` | Ingests memory fragment with optional metadata tags |
| `search(query, limit, score_threshold)` | Semantic search, ranked results |
| `list_records(limit)` | Recent memories via broad search |
| `count()` | Approximate count via broad search |
| `delete()` | No-op with warning (Synap deletion is explicit and permanent) |

## Async crews (CrewAI 0.100+)

`asearch` is exposed for native async use:

```python
results = await backend.asearch("project deadlines", limit=5)
```

The sync `search` wraps `asearch` via an event-loop bridge, so it works in both sync and async crews.

## Live doc

`https://docs.maximem.ai/integrations/crewai`

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.3.0 (JS) — verified 2026-06-20. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
