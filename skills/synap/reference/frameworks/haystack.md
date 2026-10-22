# Haystack

> **Read this next:** [This integration's docs page](https://docs.maximem.ai/integrations/haystack.md); [memories.create](https://docs.maximem.ai/sdk-reference/memories/create.md); [context fetch](https://docs.maximem.ai/sdk-reference/context/fetch.md); [listen](https://docs.maximem.ai/sdk-reference/instance/listen.md); [Errors](https://docs.maximem.ai/sdk-reference/errors.md); [Graceful degradation](https://docs.maximem.ai/patterns/graceful-degradation.md). **Checkpoint:** open the method's page before writing each call, re-read it after and compare every parameter, scope rule and error; if the docs and this file disagree, the docs win and tell the human.

`pip install maximem-synap-haystack`

| Class | Purpose |
| --- | --- |
| `SynapRetriever` | Pipeline component that fetches memories as `Document`s |
| `SynapMemoryWriter` | Pipeline component that records turns back to Synap |

## SynapRetriever

```python
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from synap_haystack import SynapRetriever

retriever = SynapRetriever(
    sdk=sdk,
    user_id="alice",
    customer_id="acme",   # B2B only; omit on B2C
    max_results=6,
    mode="fast",          # "fast" or "accurate"
)

pipeline = Pipeline()
pipeline.add_component("retriever", retriever)
pipeline.add_component("prompt_builder", PromptBuilder(template=your_template))
pipeline.connect("retriever.documents", "prompt_builder.documents")

result = pipeline.run({"retriever": {"query": "project deadlines"}})
```

Each `Document` returned has:

- `content` — memory text
- `meta["type"]` — memory type (`"fact"`, `"preference"`, etc.)
- `meta["confidence"]` — relevance score

**Scoping:** `customer_id` is B2B only, and required there. On a B2C instance (`user_context_isolation = "equals_customer"`) pass `user_id` alone: a `customer_id` comes back as HTTP 400. `GET /api/v1/auth/whoami` tells you which mode the instance is in.

## SynapMemoryWriter

Place at the end of your pipeline after the LLM response:

```python
from synap_haystack import SynapRetriever, SynapMemoryWriter

writer = SynapMemoryWriter(
    sdk=sdk,
    conversation_id="conv-001",   # UUID
    user_id="alice",
    customer_id="acme",   # B2B only; omit on B2C
)

pipeline = Pipeline()
pipeline.add_component("retriever", retriever)
pipeline.add_component("llm", your_llm)
pipeline.add_component("writer", writer)

pipeline.connect("retriever.documents", "llm.documents")
pipeline.connect("llm.replies", "writer.replies")
```

Write failures raise `SynapIntegrationError` — Haystack's component error handling will surface them.

## Full RAG-with-memory pipeline

```python
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from synap_haystack import SynapRetriever, SynapMemoryWriter

retriever = SynapRetriever(sdk=sdk, user_id="alice")
writer = SynapMemoryWriter(sdk=sdk, conversation_id="conv-001", user_id="alice")

template = """
Given this context about the user:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}
Answer: {{ query }}
"""

pipeline = Pipeline()
pipeline.add_component("retriever", retriever)
pipeline.add_component("prompt", PromptBuilder(template=template))
pipeline.add_component("llm", OpenAIGenerator(model="gpt-4o"))
pipeline.add_component("writer", writer)

pipeline.connect("retriever.documents", "prompt.documents")
pipeline.connect("prompt.prompt", "llm.prompt")
pipeline.connect("llm.replies", "writer.replies")

result = pipeline.run({
    "retriever": {"query": "What are my priorities?"},
    "prompt": {"query": "What are my priorities?"},
})
```

## Live doc

`https://docs.maximem.ai/integrations/haystack`

---
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
