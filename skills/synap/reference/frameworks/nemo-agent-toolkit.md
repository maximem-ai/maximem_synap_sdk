# NVIDIA NeMo Agent Toolkit (NAT)

> **Read this next:** [This integration's docs page](https://docs.maximem.ai/integrations/nemo-agent-toolkit.md); [memories.create](https://docs.maximem.ai/sdk-reference/memories/create.md); [context fetch](https://docs.maximem.ai/sdk-reference/context/fetch.md); [listen](https://docs.maximem.ai/sdk-reference/instance/listen.md); [Errors](https://docs.maximem.ai/sdk-reference/errors.md); [Graceful degradation](https://docs.maximem.ai/patterns/graceful-degradation.md). **Checkpoint:** open the method's page before writing each call, re-read it after and compare every parameter, scope rule and error; if the docs and this file disagree, the docs win and tell the human.

`pip install maximem-synap-nemo-agent-toolkit`

| Export | Purpose |
| --- | --- |
| `SynapMemoryEditor` | Implements `nat.memory.interfaces.MemoryEditor` |
| `@register_memory` | Registers `SynapMemoryEditor` in NAT's memory registry |
| `synap_memory_client` | Factory that returns a ready `SynapMemoryEditor` (manages the SDK internally) |

## Quick start

```python
from synap_nemo_agent_toolkit import SynapMemoryEditor

editor = SynapMemoryEditor(
    sdk=sdk,
    customer_id="acme",   # B2B only; omit on B2C
    mode="accurate",       # "fast" or "accurate"
)

# Store memories
await editor.add_items([
    MemoryItem(user_id="alice", memory="Prefers concise bullet-point summaries", tags=["preference"]),
    MemoryItem(user_id="alice", memory="Working on Q3 roadmap planning", tags=["project"]),
])

# Search
results = await editor.search("communication preferences", top_k=5, user_id="alice")
for item in results:
    print(item.memory, item.score)
```

**Scoping:** `customer_id` is B2B only, and required there. On a B2C instance (`user_context_isolation = "equals_customer"`) pass `user_id` alone: a `customer_id` comes back as HTTP 400. `GET /api/v1/auth/whoami` tells you which mode the instance is in.

## MemoryEditor protocol

```
add_items(items)              # batch-ingest MemoryItem objects
search(query, top_k, user_id) # semantic search, scored MemoryItem list
update_items(items)           # update existing memories by ID
get_items(user_id, limit)     # all memories for a user
```

## YAML pipeline configuration

NAT supports declaring memory backends in YAML. Register `SynapMemoryEditor`:

```python
from synap_nemo_agent_toolkit import register_memory

@register_memory("synap")
class SynapMemoryEditor(SynapMemoryEditor):
    pass
```

Then in the NAT YAML:

```yaml
memory:
  type: synap
  config:
    api_key: ${SYNAP_API_KEY}
    # instance_id: ${SYNAP_INSTANCE_ID}   # optional — resolved from the key if omitted
    mode: accurate
    customer_id: acme   # B2B only; drop this line on B2C
```

## Factory shortcut

For programmatic setup outside YAML — when you don't want to manage the SDK lifecycle yourself:

```python
from synap_nemo_agent_toolkit import synap_memory_client

editor = synap_memory_client(
    api_key=os.environ["SYNAP_API_KEY"],
    customer_id="acme",   # B2B only; omit on B2C
    mode="accurate",
)
```

`synap_memory_client` initializes the SDK internally — useful for embedding in NAT pipelines that don't otherwise own an SDK.

## Live doc

`https://docs.maximem.ai/integrations/nemo-agent-toolkit`

---
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
