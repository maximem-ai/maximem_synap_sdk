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

Depends on `maximem-synap>=0.7.1`, `maximem-synap-integrations-common>=0.3.1`,
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

## Report the whole turn to the anticipation agent

`_type: synap_stream` is a NAT telemetry exporter. Switch it on and every
event of every run is reported on Synap's live gRPC stream, so the
anticipation agent can see what the agent is doing while it does it. Nothing
else to set up: the stream is opened for you when the workflow starts
(`SYNAP_AUTO_LISTEN=false` keeps everything on REST and never opens one).

```yaml
general:
  telemetry:
    tracing:
      synap:
        _type: synap_stream
        customer_id: "acme"        # B2B instances only
```

| NAT event        | Synap event                    |
|------------------|--------------------------------|
| `WORKFLOW_START` | user turn                      |
| `WORKFLOW_END`   | assistant turn                 |
| `TOOL_START`     | tool call                      |
| `TOOL_END`       | tool result (same call id)     |
| `LLM_END`        | reasoning, except the last one |

The turn is the workflow run, not the model call, so a three-tool turn is
still one user message and one assistant message. The last LLM output of a
run is the answer and goes out as the assistant turn, so it is never also
reported as a thought.

The user and assistant turns go on the stream; if it cannot carry them they are
written over REST with `conversation.record_message` (needs a conversation id,
NAT's own or the configured `conversation_id`). Never both, since the server
persists a streamed turn itself. Tool calls, tool results and reasoning are
stream-only. To write long-term memories, use `SynapMemoryEditor` above.

> **Turns need a hook.** NAT has no always-on callback for a conversation; the
> telemetry exporter is the way to read one. Without `synap_stream`, turns
> reach Synap only through `synap_memory` under NAT's `auto_memory_agent`, which
> hands every user and assistant message to the editor: the editor records
> those as turns (and skips the ones the exporter already wrote) rather than
> ingesting a transcript document on top of them. A plain `add_items` with a
> `memory` string is a fact and is still written with `memories.create`.

To report from your own code instead of from YAML, construct the exporter
with the SDK you use elsewhere. No `listen()` needed; the first report opens it:

```python
from synap_nemo_agent_toolkit import SynapStreamExporter

exporter = SynapStreamExporter(sdk=sdk, user_id="alice", customer_id="acme")
async with exporter.start():
    ...  # run the workflow
```

## Synap's compaction for the agent

NAT's agents are handed the conversation as a `ChatRequest` and keep the last
`max_history` messages of it. `_type: synap_history` is a function middleware:
once Synap has compacted the conversation, the agent is handed the compaction
block and the turns after it **in place of the messages before the current one**.
The agent's own trimming stays in place (it now sees a short list, and it is what
the agent has when Synap has nothing).

```yaml
middleware:
  synap_history:
    _type: synap_history

functions:
  my_agent:
    _type: tool_calling_agent
    llm_name: nim_llm
    tool_names: [calculator]
    middleware: [synap_history]
```

The conversation id is NAT's own for the run, else the configured
`conversation_id`; with neither the input is left alone. `synap_short_term` (a
function returning Synap's context for a prompt template) now returns the
compaction block alone once there is one, never the recent turns the agent
already has.

## One SDK for all of them

`synap_memory`, `synap_short_term`, `synap_stream` and `synap_history` share one
Synap SDK per set of credentials (API key and instance id), so the stream the
exporter opens, and what comes back on it, serves the others' reads. The last
component to shut down stops the stream and closes the SDK.

## Deeper scope ladders

On an account whose scope ladder has more levels than customer and user, give the editor and the stream exporter
the level as `scope_path` (level key to value), in code or in YAML. It is sent with every read, write and recorded
turn, and only when set. It is fixed when the component is built; a `scope_path` passed to `search` is ignored. A path that includes `user` or `customer` fixes them: a configured `customer_id`/`user_id`, or a per-call `user_id`, that disagrees with it is refused with `SynapIntegrationError` before anything is sent (the exporter skips such events and logs at ERROR).

```yaml
memory:
  synap:
    _type: synap_memory
    scope_path: {customer: acme, team: payments, user: alice}
```

```python
editor = SynapMemoryEditor(sdk=sdk, scope_path={"customer": "acme", "team": "payments", "user": "alice"})
```

`maximem_synap.ScopeLadder.address(values)` builds `user_id`, `customer_id` and `scope_path` from level values.
Without `scope_path` nothing changes.

**Identity must match the path.** `search(user_id=...)`, each `MemoryItem.user_id` and the user NAT puts on each workflow event are supplied per call, while `scope_path` is fixed when the editor or exporter is built. The server lets the path win over flat ids, so a call for another user would read or write the path's user's memory. With a `scope_path` set, a `user` or `customer` id that differs from the path's `user` / `customer` level is refused: `search` and `add_items` raise `SynapIntegrationError` and nothing is sent, and the exporter skips the event and logs an error (it runs inside the workflow, where raising would end the run). Build one editor and one exporter per identity; do not share a path-scoped one across users. Messages carry short fingerprints, never the ids.

**Needs a newer SDK.** A path needs a `maximem-synap` release that has `get_scope_ladder` and `scope_path` on `fetch`, `send_message` and `record_message` (and `scope` on `memories.create`). With an older SDK, a turn that carries a path is dropped and an error is logged: it is never filed at a wider scope. Leave `scope_path` unset on an older SDK.

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
