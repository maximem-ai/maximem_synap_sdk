# LangGraph

`pip install synap-langgraph`

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
