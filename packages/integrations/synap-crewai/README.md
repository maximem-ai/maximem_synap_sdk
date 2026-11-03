# synap-crewai

Synap memory for [CrewAI](https://github.com/crewAIInc/crewAI). `SynapStorageBackend` plugs into CrewAI's unified `Memory` so records are stored in and recalled from Synap; `SynapMemory` makes recall work against it; `SynapCrewAIListener` reports the run (turns, tool calls, reasoning) on Synap's live stream.

## Install

```bash
pip install maximem-synap-crewai
```

(import name: `synap_crewai`; requires Python 3.11+, `maximem-synap`, `maximem-synap-integrations-common` and `crewai>=1.10`.)

## Quickstart

```python
from crewai.memory.types import MemoryRecord
from synap_crewai import SynapMemory, SynapStorageBackend

backend = SynapStorageBackend(sdk, user_id="alice", customer_id="acme")  # sdk: a MaximemSynapSDK
memory = SynapMemory(storage=backend)      # pass to Crew(memory=memory)

backend.save([MemoryRecord(content="Isha is vegetarian", scope="/diet",
                           categories=["preference"], importance=0.8)])
hits = backend.search([0.0], metadata_filter={"_query_text": "What does Isha eat?"})
# hits[0][0].scope == "/acme/alice"  (the annotation: /customer/user)
```

This example was run against a recording stand-in for the SDK; with a real `MaximemSynapSDK` the same calls go to Synap.

## Deeper scope ladders

If your account's scope ladder has more levels than customer and user, pass the address of the level to read and write at. It is sent only when set; without it every call is exactly what it was.

```python
path = {"customer": "acme", "team": "payments", "user": "dana"}  # from sdk.get_scope_ladder() / ScopeAddress
backend = SynapStorageBackend(sdk, user_id="dana", customer_id="acme", scope_path=path)
listener = SynapCrewAIListener(sdk, user_id="dana", conversation_id="c1",
                               customer_id="acme", scope_path=path)
```

- The ids you pass must agree with the path's `user` and `customer` levels (an empty `customer_id` is fine). A mismatch raises `SynapIntegrationError` at construction, because the server would otherwise let the path win and file the memory under the path's user.
- **The annotation drops the keys.** `MemoryRecord.scope` is built from the path's *values* in the order you give them (`/acme/payments/dana`), not from the level keys. Without a path it is `/customer/user` as before. The order only affects this annotation; the server orders levels by their ladder position.
- A path value containing `/` would make the annotation ambiguous, so it is rejected.
