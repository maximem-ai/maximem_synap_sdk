# LlamaIndex

> **Read this next:** [This integration's docs page](https://docs.maximem.ai/integrations/llamaindex.md); [memories.create](https://docs.maximem.ai/sdk-reference/memories/create.md); [context fetch](https://docs.maximem.ai/sdk-reference/context/fetch.md); [listen](https://docs.maximem.ai/sdk-reference/instance/listen.md); [Errors](https://docs.maximem.ai/sdk-reference/errors.md); [Graceful degradation](https://docs.maximem.ai/patterns/graceful-degradation.md). **Checkpoint:** open the method's page before writing each call, re-read it after and compare every parameter, scope rule and error; if the docs and this file disagree, the docs win and tell the human.

`pip install maximem-synap-llamaindex`

| Class | Purpose |
| --- | --- |
| `SynapChatMemory` | `BaseMemory` implementation for chat engines |
| `SynapRetriever` | Returns `NodeWithScore` for RAG pipelines |

## SynapChatMemory — drop-in for chat engines

```python
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from synap_llamaindex import SynapChatMemory

memory = SynapChatMemory(
    sdk=sdk,
    conversation_id="conv-001",   # UUID
    user_id="alice",
    customer_id="acme",           # B2B only; omit on B2C
)

chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=your_retriever,
    memory=memory,
)

response = await chat_engine.achat("What were my action items from last week?")
```

`get()` loads prior messages from Synap; `put()` writes new turns back. Failed reads return empty buffer; failed writes raise so callers know persistence failed.

**Scoping:** `customer_id` is B2B only, and required there. On a B2C instance (`user_context_isolation = "equals_customer"`) pass `user_id` alone: a `customer_id` comes back as HTTP 400. `GET /api/v1/auth/whoami` tells you which mode the instance is in.

## SynapRetriever — for RAG pipelines

```python
from synap_llamaindex import SynapRetriever

retriever = SynapRetriever(
    sdk=sdk,
    user_id="alice",
    customer_id="acme",   # B2B only; omit on B2C
    max_results=6,
    mode="accurate",   # "fast" or "accurate"
)

nodes = await retriever.aretrieve("What are the user's project preferences?")
# node.text = memory text
# node.score = relevance
```

Compose with `RouterRetriever` or `QueryFusionRetriever` to blend Synap memories with document retrieval — useful when the agent needs both org/user memory **and** document RAG.

```python
from llama_index.core.retrievers import QueryFusionRetriever

fusion = QueryFusionRetriever(
    [synap_retriever, document_retriever],
    similarity_top_k=10,
)
```

## Live doc

`https://docs.maximem.ai/integrations/llamaindex`

---
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
