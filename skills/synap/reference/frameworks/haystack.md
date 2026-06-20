# Haystack

`pip install synap-haystack`

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
    customer_id="acme",   # optional
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

## SynapMemoryWriter

Place at the end of your pipeline after the LLM response:

```python
from synap_haystack import SynapRetriever, SynapMemoryWriter

writer = SynapMemoryWriter(
    sdk=sdk,
    conversation_id="conv-001",   # UUID
    user_id="alice",
    customer_id="acme",
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
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.3.0 (JS) — verified 2026-06-20. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
