# NVIDIA NeMo Agent Toolkit (NAT)

`pip install synap-nemo-agent-toolkit`

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
    customer_id="acme",   # optional
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
    instance_id: ${SYNAP_INSTANCE_ID}
    api_key: ${SYNAP_API_KEY}
    mode: accurate
    customer_id: acme
```

## Factory shortcut

For programmatic setup outside YAML — when you don't want to manage the SDK lifecycle yourself:

```python
from synap_nemo_agent_toolkit import synap_memory_client

editor = synap_memory_client(
    instance_id=os.environ["SYNAP_INSTANCE_ID"],
    api_key=os.environ["SYNAP_API_KEY"],
    customer_id="acme",
    mode="accurate",
)
```

`synap_memory_client` initializes the SDK internally — useful for embedding in NAT pipelines that don't otherwise own an SDK.

## Live doc

`https://docs.maximem.ai/integrations/nemo-agent-toolkit`

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.3.0 (JS) — verified 2026-06-20. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
