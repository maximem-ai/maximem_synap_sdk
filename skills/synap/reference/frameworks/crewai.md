# CrewAI

> **Read this next:** [This integration's docs page](https://docs.maximem.ai/integrations/crewai.md); [memories.create](https://docs.maximem.ai/sdk-reference/memories/create.md); [context fetch](https://docs.maximem.ai/sdk-reference/context/fetch.md); [listen](https://docs.maximem.ai/sdk-reference/instance/listen.md); [Errors](https://docs.maximem.ai/sdk-reference/errors.md); [Graceful degradation](https://docs.maximem.ai/patterns/graceful-degradation.md). **Checkpoint:** open the method's page before writing each call, re-read it after and compare every parameter, scope rule and error; if the docs and this file disagree, the docs win and tell the human.

`pip install maximem-synap-crewai`

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
    customer_id="acme",   # B2B only; omit on B2C
)

memory = Memory(storage=backend)

crew = Crew(
    agents=[your_agent],
    tasks=[your_task],
    memory=memory,
)

result = crew.kickoff(inputs={"topic": "quarterly planning"})
```

**Scoping:** `customer_id` is B2B only, and required there. On a B2C instance (`user_context_isolation = "equals_customer"`) pass `user_id` alone: a `customer_id` comes back as HTTP 400. `GET /api/v1/auth/whoami` tells you which mode the instance is in.

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
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
