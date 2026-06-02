# synap-langgraph

Synap memory integration for [LangGraph](https://langchain-ai.github.io/langgraph/).

## Install

```bash
pip install synap-langgraph
```

Requires `langgraph>=1.0`, `maximem-synap>=0.2.0`.

## What's in the box

- **`SynapStore`** — implements LangGraph's `BaseStore` for cross-thread long-term memory. Semantic search via `store.search(namespace, query=...)` routes to `sdk.fetch(...)`, so your graph nodes get Synap-powered recall out of the box.
  - **User or customer scope.** Pass `user_id` for private per-user memory, or just a `customer_id` (no `user_id`) for a **customer-wide shared pool** visible to every user in the deployment.
  - **All memory types.** Reads surface facts *and* preferences (plus episodes / emotions / temporal events), so stated preferences aren't dropped.
  - **Anticipation (optional).** Construct with `include_conversation_context=True` and feed turns via `store.record_message(conversation_id, role, content)` so just-stated context is in play on the next read. (This lives alongside the `BaseStore` API — anticipation has no key/value analogue.)

- **`SynapCheckpointSaver`** — implements `BaseCheckpointSaver` with **best-effort fuzzy retrieval**. Checkpoint writes succeed durably; reads use `sdk.fetch` which is semantic-search shaped rather than exact KV. Use for observability/audit and demo flows. For production checkpoint durability, pair with `SqliteSaver` / `PostgresSaver`.

- **`create_synap_node`** — re-exported from `synap-langchain` for users who discovered our LangGraph support through the LangChain package. This is the canonical home.

## Quickstart

```python
from langgraph.graph import StateGraph, START, END
from maximem_synap import MaximemSynapSDK
from synap_langgraph import SynapStore, SynapCheckpointSaver

sdk = MaximemSynapSDK(api_key="sk-...")

store = SynapStore(sdk, user_id="alice", customer_id="acme")
saver = SynapCheckpointSaver(sdk, user_id="alice", customer_id="acme")

graph = StateGraph(MyState)
# ... add nodes / edges ...
app = graph.compile(checkpointer=saver, store=store)

# Store usage inside a node:
async def remember(state, runtime):
    await runtime.store.aput(
        ("alice", "preferences"),
        "language",
        {"value": "English"},
    )
```

## Error policy

- **Writes** (`SynapStore.put`, `SynapCheckpointSaver.put`, `put_writes`) surface SDK failures as `SynapIntegrationError`. Silent drops would hide ingestion outages.
- **Reads** (`get`, `search`, `get_tuple`, `list`) degrade gracefully — they log at `ERROR` and return `None`/`[]` rather than crashing the graph.
- **Deletes** (`SynapStore.delete`, `SynapCheckpointSaver.delete_thread`) warn once and no-op — Synap has no public delete API.

> **Note on exact key lookups.** `get`/`search` match memories by custom metadata markers. On instances that strip custom metadata during extraction (e.g. MACA atomization), exact-key `get` is unreliable and a one-time warning is logged — semantic `search` remains the reliable path. Job/document-level attribution (mapping fragments back to a source id) is not done in the store; build it in app code from the ids returned at write time.

## When to use which checkpointer

| Goal | Saver |
| ---- | ----- |
| Durable thread checkpoints, exact restore | LangGraph's `SqliteSaver` or `PostgresSaver` |
| Thread state surfaced in Synap for observability/audit/cross-thread analysis | `SynapCheckpointSaver` |
| Both | Use Sqlite/Postgres as primary; layer `SynapCheckpointSaver` for the Synap view |

Cross-thread long-term memory (`BaseStore`) maps cleanly to Synap — use `SynapStore` as your default.
