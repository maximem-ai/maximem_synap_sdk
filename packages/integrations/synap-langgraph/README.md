# synap-langgraph

Synap memory integration for [LangGraph](https://langchain-ai.github.io/langgraph/).

## Install

```bash
pip install synap-langgraph
```

Requires `langgraph>=1.0`, `maximem-synap>=0.2.0`.

## What's in the box

- **`SynapStore`** — implements LangGraph's `BaseStore` for cross-thread long-term memory. Semantic search via `store.search(namespace, query=...)` routes to `sdk.fetch(...)`, so your graph nodes get Synap-powered recall out of the box.

- **`SynapCheckpointSaver`** — implements `BaseCheckpointSaver` with **best-effort fuzzy retrieval**. Checkpoint writes succeed durably; reads use `sdk.fetch` which is semantic-search shaped rather than exact KV. Use for observability/audit and demo flows. For production checkpoint durability, pair with `SqliteSaver` / `PostgresSaver`.

- **`create_synap_node`** — re-exported from `synap-langchain` for users who discovered our LangGraph support through the LangChain package. This is the canonical home.

- **`synap_st_prompt`** — short-term conversation context as a `prompt` callable for `create_react_agent`. Prepends Synap's compacted summary + recent turns above your system prompt at every LLM step.

- **`create_synap_st_node`** — same short-term context, exposed as a `StateGraph` node that writes the ST string into state for your LLM node to consume.

## Short-term context (compacted conversation, on every LLM step)

LangGraph's built-in memory truncates recent turns to a token budget. Synap's short-term context is the **compacted summary + recent turns** maintained per conversation by the Synap server — a richer, more token-efficient view of "what happened so far." Drop it into a prebuilt agent or a custom graph; both helpers serve from the SDK's local cache when warm (near-zero overhead) and fall back to the Synap server when cold.

The SDK helper they both wrap is `sdk.conversation.context.get_context_for_prompt(conv_id, style=...)`, which is cache-first whenever the `SYNAP_SDK_ST_AUTHORITATIVE` flag is on.

### A) Prebuilt agent — one-line drop-in

```python
from langgraph.prebuilt import create_react_agent  # or langchain.agents.create_agent
from maximem_synap import MaximemSynapSDK
from synap_langgraph import synap_st_prompt

sdk = MaximemSynapSDK(api_key="sk-...")

agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-20241022",
    tools=[...],
    prompt=synap_st_prompt(
        sdk,
        conversation_id="conv_abc123",       # required, explicit
        system="You are a helpful agent.",   # your own instructions
        style="narrative",                   # default; also "structured" | "bullet_points"
    ),
)
```

What the model sees at every step (system message content):

```
<synap_short_term_context>
... compacted summary + recent turns from Synap ...
</synap_short_term_context>

You are a helpful agent.
```

### B) Custom graph — write ST into state, consume in your own LLM node

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from synap_langgraph import create_synap_st_node

class State(TypedDict):
    messages: Annotated[list, add_messages]
    synap_st: str   # populated by the node

async def my_llm_node(state: State):
    system_text = f"You are a helpful agent.\n\n{state['synap_st']}".strip()
    ...   # build your prompt template with system_text and invoke the model

graph = StateGraph(State)
graph.add_node("st", create_synap_st_node(sdk, conversation_id="conv_abc123"))
graph.add_node("llm", my_llm_node)
graph.add_edge(START, "st")
graph.add_edge("st", "llm")
graph.add_edge("llm", END)
```

### Which to use

| You're building... | Use |
| --- | --- |
| A prebuilt React-style agent (`create_react_agent` / `create_agent`) | **`synap_st_prompt`** — drop into `prompt=` |
| A custom `StateGraph` with multi-LLM / conditional routing / per-step prompt composition | **`create_synap_st_node`** — read `state["synap_st"]` in your nodes |
| Both | They compose — adapter Option A is sugar over Option B + a SystemMessage prepend |

### Error policy

- SDK failures **never crash the graph** by default (`on_error="fallback"`): logged at `ERROR` via `SynapIntegrationError`'s log path, then the helper degrades to your bare system prompt (or empty state slot).
- Pass `on_error="raise"` for strict environments that want the failure surfaced as `SynapIntegrationError`.
- An empty short-term result (no compaction yet **and** no recent turns) is a legitimate empty case, not a failure — the user's system prompt is preserved as-is.

### Conversation ID

Always explicit. We deliberately do **not** infer it from LangGraph's `thread_id` because the two namespaces can diverge — your thread might span multiple Synap conversations, or vice versa. For multi-conversation agents, construct one prompt callable per conversation inside your per-run setup.

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

## When to use which checkpointer

| Goal | Saver |
| ---- | ----- |
| Durable thread checkpoints, exact restore | LangGraph's `SqliteSaver` or `PostgresSaver` |
| Thread state surfaced in Synap for observability/audit/cross-thread analysis | `SynapCheckpointSaver` |
| Both | Use Sqlite/Postgres as primary; layer `SynapCheckpointSaver` for the Synap view |

Cross-thread long-term memory (`BaseStore`) maps cleanly to Synap — use `SynapStore` as your default.
