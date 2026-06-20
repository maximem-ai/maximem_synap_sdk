# LlamaIndex

`pip install synap-llamaindex`

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
    customer_id="acme",           # optional
)

chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=your_retriever,
    memory=memory,
)

response = await chat_engine.achat("What were my action items from last week?")
```

`get()` loads prior messages from Synap; `put()` writes new turns back. Failed reads return empty buffer; failed writes raise so callers know persistence failed.

## SynapRetriever — for RAG pipelines

```python
from synap_llamaindex import SynapRetriever

retriever = SynapRetriever(
    sdk=sdk,
    user_id="alice",
    customer_id="acme",
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
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.2.4 (JS) — verified 2026-06-17. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
