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
  - **Anticipation (optional).** Construct with `include_conversation_context=True` and feed turns via `store.record_message(conversation_id, role, content)` so just-stated context is in play on the next read. The turn goes out on the live stream when one is open and over REST only when there is none, never both. (This lives alongside the `BaseStore` API — anticipation has no key/value analogue.)

- **`SynapCheckpointSaver`**: **deprecated; it cannot restore what it saves.** Checkpoints go through `sdk.memories.create`, which extracts facts from them; `fetch` returns those facts, never the checkpoint, so a thread always resumes from scratch (and the base64 blob lands in the user's memory extraction). Constructing it warns. Use `SqliteSaver` / `PostgresSaver` for checkpoints and `SynapStore` for long-term memory.

- **`create_synap_node`** — re-exported from `synap-langchain` for users who discovered our LangGraph support through the LangChain package. This is the canonical home.

- **`synap_st_prompt`**: Synap's compaction as the `prompt` callable of `create_react_agent`. Once Synap has compacted the conversation, the model sees the compaction above your system prompt, then the turns after it, then the current turn; the turns it replaces are the ones your graph's state holds. Until then your messages go through as they are.

- **`create_synap_st_node`**: the same compaction, exposed as a `StateGraph` node that writes the string into state for your LLM node to consume (empty until the conversation has been compacted).

- **`SynapLangGraphCallbackHandler`** — reports the whole turn on Synap's live gRPC stream so the anticipation agent can see it: the user turn, the reasoning, every tool call and result, and the assistant turn. Pass it like any LangChain callback.

## Reporting a run on the live stream

LangGraph runs on LangChain's callback machinery, so a Synap handler in the run config reaches the chat model and the tool nodes inside your compiled graph.

```python
from langchain_core.messages import HumanMessage
from synap_langgraph import SynapLangGraphCallbackHandler

handler = SynapLangGraphCallbackHandler(
    sdk=sdk,
    conversation_id="conv_abc123",
    user_id="user_456",
    customer_id="cust_789",
)

await app.ainvoke(
    {"messages": [HumanMessage("where is my order QX41RT")]},
    config={"callbacks": [handler]},
)
```

Build one handler per conversation. What it reports:

| Event | Where it comes from |
|---|---|
| user turn | `on_chat_model_start`, once per turn (a ReAct loop calls the model again after every tool result with the same human message still in the list) |
| reasoning | the AIMessage's `reasoning` content blocks in `on_llm_end` — `on_agent_action` is an `AgentExecutor` callback and never fires in a graph |
| tool call | `on_tool_start`, with the parsed argument dict |
| tool result | `on_tool_end`, tied to the call by the shared `run_id` |
| assistant turn | `on_llm_end` |

**The stream opens itself.** The first turn calls `sdk.instance.listen()` for you; you do not. The user turn, the assistant turn, tool calls, results and reasoning then go on it. A turn goes over REST only when the stream cannot carry it (never both: the server persists a streamed turn itself). `SYNAP_AUTO_LISTEN=false`, or `synap_integrations_common.set_auto_listen(sdk, False)`, turns the auto-open off. The same rule applies to `SynapStore.record_message`, which works for B2C stores built with `user_id` alone. None of these calls can raise into your graph.

A run is one user turn and one assistant turn: the question the run was given and the answer it ended with, not every model call in the tool loop. Tool callbacks of a *sync* tool run on a different event loop from the stream's and are not reported; write the tool as `async def` to get tool events.

## Short-term context (compacted conversation, on every LLM step)

LangGraph's built-in memory truncates recent turns to a token budget. Synap's short-term context is the **compaction** the Synap server maintains per conversation: a richer, more token-efficient view of "what happened so far" than the turns it replaces. Drop it into a prebuilt agent or a custom graph; both helpers serve from the SDK's local cache when warm and fall back to the Synap server when cold. Neither adds the recent turns: your graph already sends them, and adding them again showed each one twice.

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

What the model sees at every step once the conversation has been compacted (system message content, then the turns Synap holds after the compaction, then the current turn):

```
<synap_short_term_context>
... Synap's compaction ...
</synap_short_term_context>

You are a helpful agent.
```

Before the first compaction, and whenever Synap cannot answer, the model sees only `You are a helpful agent.` and your messages. The graph's own state is never changed. For `langchain.agents.create_agent` (which has no `prompt=`), use `synap_langchain.SynapHistoryMiddleware`.

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
| A prebuilt `create_react_agent` | **`synap_st_prompt`**: drop into `prompt=` |
| A `langchain.agents.create_agent` agent | `synap_langchain.SynapHistoryMiddleware` |
| A custom `StateGraph` with multi-LLM / conditional routing / per-step prompt composition | **`create_synap_st_node`** — read `state["synap_st"]` in your nodes |
| Both | They compose — adapter Option A is sugar over Option B + a SystemMessage prepend |

### Error policy

- SDK failures **never crash the graph** by default (`on_error="fallback"`): logged at `ERROR` via `SynapIntegrationError`'s log path, then the helper degrades to your bare system prompt (or empty state slot).
- Pass `on_error="raise"` for strict environments that want the failure surfaced as `SynapIntegrationError`.
- No compaction yet is a legitimate empty case, not a failure; the user's system prompt and your messages are passed through as-is.

### Conversation ID

Always explicit. We deliberately do **not** infer it from LangGraph's `thread_id` because the two namespaces can diverge — your thread might span multiple Synap conversations, or vice versa. For multi-conversation agents, construct one prompt callable per conversation inside your per-run setup.

## Quickstart

```python
from langgraph.checkpoint.memory import InMemorySaver  # or SqliteSaver / PostgresSaver
from langgraph.graph import StateGraph, START, END
from maximem_synap import MaximemSynapSDK
from synap_langgraph import SynapStore

sdk = MaximemSynapSDK(api_key="sk-...")

store = SynapStore(sdk, user_id="alice", customer_id="acme")

graph = StateGraph(MyState)
# ... add nodes / edges ...
app = graph.compile(checkpointer=InMemorySaver(), store=store)

# Store usage inside a node:
async def remember(state, runtime):
    await runtime.store.aput(
        ("alice", "preferences"),
        "language",
        {"value": "English"},
    )
```

## Error policy

- **Writes** (`SynapStore.put`) surface SDK failures as `SynapIntegrationError`. Silent drops would hide ingestion outages.
- **Reads** (`get`, `search`) degrade gracefully; they log at `ERROR` and return `None`/`[]` rather than crashing the graph.
- **Deletes** (`SynapStore.delete`) warn once and no-op; Synap has no public delete API.

> **Note on metadata-stripping backends.** `get`/`search` identify items by custom metadata markers. On instances that atomize content during extraction (e.g. MACA), those markers are stripped, so:
> - **`search`** falls back to returning the **scope-filtered** results Synap ranked (semantic retrieval still works). Scope (`user_id`/`customer_id`) is enforced at the fetch layer, but sub-namespace isolation *within* a scope is not. A one-time warning is logged. Set `SynapStore(..., semantic_fallback=False)` for strict namespace semantics (search returns `[]` when markers are absent).
> - **`get`** (exact key) returns `None` — there's no reliable way to resolve an exact key without the markers. Use `search` as the read path.
>
> Job/document-level attribution (mapping fragments back to a source id) is not done in the store; build it in app code.

## When to use which checkpointer

| Goal | Saver |
| ---- | ----- |
| Durable thread checkpoints, exact restore | LangGraph's `SqliteSaver` or `PostgresSaver` |
| Thread state in Synap | not possible: `SynapCheckpointSaver` is deprecated because Synap returns extracted facts, not the checkpoint |

Cross-thread long-term memory (`BaseStore`) maps cleanly to Synap — use `SynapStore` as your default.
