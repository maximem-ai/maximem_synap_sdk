# LangChain

> **Read this next:** [This integration's docs page](https://docs.maximem.ai/integrations/langchain.md); [memories.create](https://docs.maximem.ai/sdk-reference/memories/create.md); [context fetch](https://docs.maximem.ai/sdk-reference/context/fetch.md); [listen](https://docs.maximem.ai/sdk-reference/instance/listen.md); [Errors](https://docs.maximem.ai/sdk-reference/errors.md); [Graceful degradation](https://docs.maximem.ai/patterns/graceful-degradation.md). **Checkpoint:** open the method's page before writing each call, re-read it after and compare every parameter, scope rule and error; if the docs and this file disagree, the docs win and tell the human.

`pip install maximem-synap-langchain`

Four drop-in components that together cover most LangChain memory needs. Pick the ones you actually need; you don't have to use all four.

| Class | Purpose |
| --- | --- |
| `SynapChatMessageHistory` | `BaseChatMessageHistory` for `RunnableWithMessageHistory` |
| `SynapCallbackHandler` | Auto-records every LLM turn via callbacks — no app code change |
| `SynapRetriever` | Returns Synap memories as LangChain `Document`s for RAG chains |
| `SynapSearchTool`, `SynapStoreTool` | Agent-callable tools for explicit memory ops |

## SynapChatMessageHistory — persistent chat history

The cleanest way to give an existing chain memory across sessions:

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from synap_langchain import SynapChatMessageHistory

def get_history(session_id: str):
    return SynapChatMessageHistory(
        sdk=sdk,
        conversation_id=session_id,    # must be UUID
        user_id="alice",
        customer_id="acme",            # B2B only; omit on B2C
    )

chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_session_history=get_history,
)

response = await chain_with_history.ainvoke(
    {"question": "What did we discuss last time?"},
    config={"configurable": {"session_id": "conv-123"}},
)
```

## SynapCallbackHandler — auto-ingest every turn

Drop into any chain or agent without touching application logic:

```python
from synap_langchain import SynapCallbackHandler

handler = SynapCallbackHandler(
    sdk=sdk,
    conversation_id="conv-123",
    user_id="alice",
)

response = await chain.ainvoke(
    {"question": "Remind me of my project deadlines."},
    config={"callbacks": [handler]},
)
```

Ingestion failures are logged at `ERROR` and **do not raise** — the chain always completes. This is the standard read-side-graceful behavior; it differs slightly here because callback writes piggy-back on a successful chain run.

## SynapRetriever — memories as `Document`s

Use as a standard LangChain retriever in RAG pipelines or `ConversationalRetrievalChain`:

```python
from synap_langchain import SynapRetriever

retriever = SynapRetriever(
    sdk=sdk,
    user_id="alice",
    customer_id="acme",   # B2B only; omit on B2C
    max_results=8,
    mode="fast",        # or "accurate"
)

docs = await retriever.aget_relevant_documents("project deadlines")
# doc.page_content = memory text
# doc.metadata = {"confidence": 0.92, "type": "fact", ...}
```

## Tools for explicit agent control

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from synap_langchain import SynapSearchTool, SynapStoreTool

tools = [
    SynapSearchTool(sdk=sdk, user_id="alice", customer_id="acme"),
    SynapStoreTool(sdk=sdk, user_id="alice", customer_id="acme"),
]

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
```

**Scoping:** `customer_id` is B2B only, and required there. On a B2C instance (`user_context_isolation = "equals_customer"`) pass `user_id` alone: a `customer_id` comes back as HTTP 400. `GET /api/v1/auth/whoami` tells you which mode the instance is in. Every component on this page takes the pair the same way.

## When to use which

- **Chat-only app, just need persistence** → `SynapChatMessageHistory`.
- **Existing chain you don't want to refactor** → `SynapCallbackHandler`.
- **RAG over memories** (e.g. Q&A "what do you know about me") → `SynapRetriever`.
- **Agent should explicitly decide when to remember/recall** → tools.
- **All of the above** → combine them; they don't conflict.

## Live doc

`https://docs.maximem.ai/integrations/langchain`

---
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
