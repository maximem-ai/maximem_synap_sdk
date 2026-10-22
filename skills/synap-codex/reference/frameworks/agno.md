# Agno

> **Read this next:** [This integration's docs page](https://docs.maximem.ai/integrations/agno.md); [memories.create](https://docs.maximem.ai/sdk-reference/memories/create.md); [context fetch](https://docs.maximem.ai/sdk-reference/context/fetch.md); [listen](https://docs.maximem.ai/sdk-reference/instance/listen.md); [Errors](https://docs.maximem.ai/sdk-reference/errors.md); [Graceful degradation](https://docs.maximem.ai/patterns/graceful-degradation.md). **Checkpoint:** open the method's page before writing each call, re-read it after and compare every parameter, scope rule and error; if the docs and this file disagree, the docs win and tell the human.

`pip install maximem-synap-agno`

For agno-agi/agno. A drop-in `InMemoryDb` subclass — minimum-friction install.

| Class | Purpose |
| --- | --- |
| `SynapDb` | Extends Agno's `InMemoryDb` to persist user memories in Synap |

## Quick start

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from synap_agno import SynapDb

db = SynapDb(sdk=sdk, customer_id="acme")   # B2B only; on B2C use SynapDb(sdk=sdk)

agent = Agent(
    db=db,
    model=OpenAIChat(id="gpt-4o-mini"),
    enable_user_memories=True,
)

agent.run("Remember that I prefer async communication", user_id="alice")
agent.run("What are my communication preferences?", user_id="alice")
```

`enable_user_memories=True` is what makes Agno call into the db's memory methods. Without it, `SynapDb` is dormant.

**Scoping:** `customer_id` is B2B only, and required there. On a B2C instance (`user_context_isolation = "equals_customer"`) pass `user_id` alone: a `customer_id` comes back as HTTP 400. `GET /api/v1/auth/whoami` tells you which mode the instance is in.

## How it overrides

`SynapDb` overrides exactly four methods:

| Method | Behavior |
| --- | --- |
| `upsert_user_memory` | Writes a new/updated memory to Synap |
| `get_user_memory` | Fetches a specific memory by ID |
| `get_user_memories` | Semantic search over user's memories |
| `get_all_memory_topics` | Unique memory topics via broad search |

All other `InMemoryDb` behavior (sessions, tool calls, non-memory storage) is inherited unchanged.

## Multi-user — single SynapDb

`user_id` is passed per-call by Agno's runtime, so one `SynapDb` instance serves all users:

```python
db = SynapDb(sdk=sdk, customer_id="acme")   # B2B only; on B2C use SynapDb(sdk=sdk)

for user_id in ["alice", "bob", "carol"]:
    agent.run("What do you remember about me?", user_id=user_id)
```

## Live doc

`https://docs.maximem.ai/integrations/agno`

---
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
