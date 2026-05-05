# synap-nemo-agent-toolkit

Synap memory plugin for **NVIDIA NeMo Agent Toolkit (NAT)**.

Implements `nat.memory.interfaces.MemoryEditor` so that NAT workflows can
use Synap as long-term memory out of the box — either programmatically,
or via a single YAML line:

```yaml
memory:
  synap:
    _type: synap_memory
    customer_id: "acme"          # optional scope
    mode: "accurate"              # or "fast"
```

## Install

```bash
pip install synap-nemo-agent-toolkit
```

Depends on `maximem-synap>=0.2.0`, `synap-integrations-common>=0.1.0`,
and `nvidia-nat-core>=1.0`.

## Use (programmatic)

```python
from maximem_synap import MaximemSynapSDK
from nat.memory.models import MemoryItem
from synap_nemo_agent_toolkit import SynapMemoryEditor

sdk = MaximemSynapSDK(api_key="...")
await sdk.initialize()

editor = SynapMemoryEditor(sdk=sdk, customer_id="acme", mode="accurate")

# Write
await editor.add_items([
    MemoryItem(
        user_id="alice",
        memory="Prefers tea over coffee",
        tags=["beverage", "preference"],
    ),
])

# Read — user_id is required on every search
hits = await editor.search("beverage preference", top_k=5, user_id="alice")
for hit in hits:
    print(hit.memory, hit.tags, hit.similarity_score)
```

## Use (YAML / NAT workflow)

Set `SYNAP_API_KEY` (optionally `SYNAP_INSTANCE_ID`) in the environment
and declare the memory provider in the workflow config:

```yaml
memory:
  synap:
    _type: synap_memory
    customer_id: "acme"
    mode: "accurate"
    # Optional — falls back to SYNAP_API_KEY env if omitted
    # api_key: "${SYNAP_API_KEY}"
```

NAT's plugin loader picks up `synap_memory` via the `nat.components`
entry-point registered in this package's `pyproject.toml`.

## Error policy

| Operation       | Synap failure          | Why                                       |
|-----------------|------------------------|-------------------------------------------|
| `add_items`     | `SynapIntegrationError`| Writes must be observable                 |
| `search`        | Returns `[]`           | A read blip must never crash an agent turn |
| `remove_items`  | Warn once, no-op       | Synap has no public delete API            |

Every write is tagged with an internal marker so `search` filters out
non-NAT memories that happen to share the same user scope — the
`MemoryItem` stream stays NAT-owned.

## License

Apache-2.0.
