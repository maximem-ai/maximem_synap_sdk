# LangGraph

> **Read this next:** [This integration's docs page](https://docs.maximem.ai/integrations/langgraph.md); [memories.create](https://docs.maximem.ai/sdk-reference/memories/create.md); [context fetch](https://docs.maximem.ai/sdk-reference/context/fetch.md); [listen](https://docs.maximem.ai/sdk-reference/instance/listen.md); [Errors](https://docs.maximem.ai/sdk-reference/errors.md); [Graceful degradation](https://docs.maximem.ai/patterns/graceful-degradation.md). **Checkpoint:** open the method's page before writing each call, re-read it after and compare every parameter, scope rule and error; if the docs and this file disagree, the docs win and tell the human.

`pip install maximem-synap-langgraph`

Two pieces: a checkpointer for thread-level state and a `BaseStore` for cross-thread long-term memory. Use either or both depending on what your graph needs.

| Class | Purpose |
| --- | --- |
| `SynapCheckpointSaver` | Thread-level checkpoint persistence with fuzzy/semantic retrieval |
| `SynapStore` | Cross-thread long-term memory via LangGraph's `BaseStore` |

## SynapCheckpointSaver

Replaces `MemorySaver` / `SqliteSaver` so threads survive restarts and can be retrieved by similarity:

```python
from langgraph.graph import StateGraph
from synap_langgraph import SynapCheckpointSaver

saver = SynapCheckpointSaver(sdk=sdk, user_id="alice")

graph = StateGraph(...)
# add nodes / edges
app = graph.compile(checkpointer=saver)

config = {"configurable": {"thread_id": "thread-001"}}
result = await app.ainvoke({"messages": [HumanMessage("Hello")]}, config=config)
```

## SynapStore — cross-thread memory

Implements `BaseStore`, so it works with anything that accepts a store:

```python
from synap_langgraph import SynapStore, SynapCheckpointSaver

store = SynapStore(sdk=sdk, user_id="alice", customer_id="acme")
saver = SynapCheckpointSaver(sdk=sdk, user_id="alice")

app = graph.compile(checkpointer=saver, store=store)
```

**Scoping:** `customer_id` is B2B only, and required there. On a B2C instance (`user_context_isolation = "equals_customer"`) pass `user_id` alone: a `customer_id` comes back as HTTP 400. `GET /api/v1/auth/whoami` tells you which mode the instance is in.

Inside graph nodes, use the store via the `store` kwarg LangGraph injects:

```python
async def my_node(state, config, *, store):
    # cross-thread retrieval
    memories = await store.asearch(("user", "alice"), query="project preferences")

    # cross-thread write
    await store.aput(
        ("user", "alice"),
        key="pref-001",
        value={"content": "Prefers async communication", "type": "preference"},
    )
    return state
```

## Both together

```python
from synap_langgraph import SynapStore, SynapCheckpointSaver

store = SynapStore(sdk=sdk, user_id="alice", customer_id="acme")
saver = SynapCheckpointSaver(sdk=sdk, user_id="alice")

app = graph.compile(checkpointer=saver, store=store)

config = {"configurable": {"thread_id": "session-42"}}
async for event in app.astream({"messages": [HumanMessage("Hi")]}, config=config):
    print(event)
```

## When to use which

- **Need conversations to survive restart** → checkpointer.
- **Agents need to remember things from other threads** (e.g. "what did we discuss in last week's session") → store.
- **Both** is the production default for any meaningful agent.

## Live doc

`https://docs.maximem.ai/integrations/langgraph`

---
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
