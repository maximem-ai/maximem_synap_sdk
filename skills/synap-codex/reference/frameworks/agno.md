# Agno

`pip install synap-agno`

For agno-agi/agno. A drop-in `InMemoryDb` subclass — minimum-friction install.

| Class | Purpose |
| --- | --- |
| `SynapDb` | Extends Agno's `InMemoryDb` to persist user memories in Synap |

## Quick start

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from synap_agno import SynapDb

db = SynapDb(sdk=sdk, customer_id="acme")   # customer_id optional

agent = Agent(
    db=db,
    model=OpenAIChat(id="gpt-4o-mini"),
    enable_user_memories=True,
)

agent.run("Remember that I prefer async communication", user_id="alice")
agent.run("What are my communication preferences?", user_id="alice")
```

`enable_user_memories=True` is what makes Agno call into the db's memory methods. Without it, `SynapDb` is dormant.

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
db = SynapDb(sdk=sdk, customer_id="acme")

for user_id in ["alice", "bob", "carol"]:
    agent.run("What do you remember about me?", user_id=user_id)
```

## Live doc

`https://docs.maximem.ai/integrations/agno`

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.2.4 (JS) — verified 2026-06-17. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
