<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner-light.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/banner-dark.png">
    <img src="assets/banner-light.png" alt="Maximem Synap — AI Agents Forget. Synap Makes Them Remember." width="100%" />
  </picture>
</p>

<p align="center">
  <a href="https://www.maximem.ai/docs"><strong>Docs</strong></a> ·
  <a href="https://synap.maximem.ai"><strong>Dashboard</strong></a> ·
  <a href="https://www.maximem.ai/blog/synap-benchmark-results"><strong>Benchmarks</strong></a> ·
  <a href="https://www.maximem.ai/synap"><strong>Website</strong></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/maximem-synap"><img src="https://img.shields.io/pypi/v/maximem-synap?style=flat-square&color=blue&label=pypi" alt="PyPI" /></a>
  <a href="https://pypi.org/project/maximem-synap"><img src="https://img.shields.io/pypi/dm/maximem-synap?style=flat-square&color=blue" alt="PyPI Downloads" /></a>
  <a href="https://www.npmjs.com/package/@maximem/synap-js-sdk"><img src="https://img.shields.io/npm/v/@maximem/synap-js-sdk?style=flat-square&color=blue&label=npm" alt="npm" /></a>
  <a href="https://pypi.org/project/maximem-synap"><img src="https://img.shields.io/pypi/pyversions/maximem-synap?style=flat-square" alt="Python versions" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="License" /></a>
  <a href="https://x.com/maximem_ai"><img src="https://img.shields.io/badge/follow-%40maximem__ai-1DA1F2?style=flat-square&logo=x&logoColor=white" alt="Twitter" /></a>
  <a href="https://www.linkedin.com/company/maximem-ai"><img src="https://img.shields.io/badge/LinkedIn-Maximem%20AI-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn" /></a>
</p>

---

## The memory layer for production AI agents

Your AI agents forget everything between conversations. Synap fixes that — with a production-grade memory layer built for applications that serve real users at scale. **#1 on [LongMemEval](https://www.maximem.ai/blog/synap-benchmark-results) with 90.2% accuracy**, sub-15ms anticipatory retrieval, and native integrations with every major AI framework.

<p align="center">
  <strong>LangChain · LlamaIndex · CrewAI · Haystack · Google ADK · AutoGen · OpenAI Agents · Semantic Kernel · Pydantic AI</strong>
</p>

---

## #1 on LongMemEval

Synap leads every major AI memory benchmark — tested across all systems on identical hardware with an open-source evaluation harness.

<p align="center">
  <img src="https://qirwvzn87kgnbyp6.public.blob.vercel-storage.com/assets/images/blogs/1775874451091-rkiulltbv8.jpeg" alt="Memory system performance comparison" width="720" />
</p>

| System | LongMemEval |
|---|---|
| **Synap** | **90.2%** |
| SuperMemory | 71.3% |
| Zep | 63.8% * |
| Mem0 | 57.5% |

*Synap, SuperMemory, and Mem0 were run through the same [open-source evaluation harness](https://github.com/gauravmaximem/memory_and_context_eval_harness) on identical hardware and configs. Zep's score is self-reported — we could not reproduce it independently.*

> **"Longer conversations make Synap better, not worse."** Richer entity graphs and stronger pattern recognition at scale.

Full methodology and reproduction instructions → [maximem.ai/blog/synap-benchmark-results](https://www.maximem.ai/blog/synap-benchmark-results)

---

## Install

```bash
# Python
pip install maximem-synap

# JavaScript / TypeScript
npm install @maximem/synap-js-sdk
```

---

## 60-second quickstart

Your agent forgets. Synap remembers — across conversations, sessions, and devices.

```python
import asyncio
from maximem_synap import MaximemSynapSDK

sdk = MaximemSynapSDK(api_key="your-api-key")

async def main():
    await sdk.initialize()

    # Monday's standup
    await sdk.conversation.record_message(
        conversation_id="mon-standup",
        user_id="alice",
        role="user",
        content="I'm migrating our auth service to OAuth2 this sprint.",
    )

    # Friday — completely different conversation, same user
    context = await sdk.fetch(
        conversation_id="fri-review",
        user_id="alice",
        search_query=["what is alice working on?"],
    )

    print(context.formatted_context)
    # → "Alice is migrating the auth service to OAuth2 this sprint."

asyncio.run(main())
```

<details>
<summary><strong>JavaScript / TypeScript</strong></summary>

```javascript
const { createClient } = require('@maximem/synap-js-sdk');

const client = createClient({ apiKey: 'your-api-key' });
await client.init();

// Record
await client.conversation.recordMessage({
    conversationId: 'mon-standup',
    userId: 'alice',
    role: 'user',
    content: "I'm migrating our auth service to OAuth2 this sprint.",
});

// Fetch later — anywhere
const context = await client.fetchUserContext({
    userId: 'alice',
    query: 'what is alice working on?',
});

console.log(context.formattedContext);
```

</details>

---

## What makes Synap different

### 🎯 Anticipatory Retrieval

Synap **pre-fetches context before your agent requests it**. 15ms P50 latency in production. For voice AI agents, this is the difference between natural conversation and awkward pauses.

### 🔗 Entity Resolution

When a user says *"my manager"* in turn 3 and *"Sarah"* in turn 12, Synap resolves them automatically. Cross-session, cross-conversation, without the agent doing any work.

### ⏳ Temporal Awareness

Context from 30 minutes ago and context from 30 days ago should not carry equal weight. Synap applies temporal decay and relevance scoring so your agent surfaces the right information at the right time.

### 🧠 Conscious Forgetting

When a user says *"ignore what I said about the budget,"* Synap processes that as a retraction — not just more context to store. Contradiction handling is built into the pipeline.

### 🏗️ Custom Memory Architectures

No universal memory model. Synap builds customized memory architectures per use case. Customer support agents and voice AI agents need different context strategies. Synap handles both.

### 🏢 Multi-Tenant Scoping

Built for B2B from day one. Memory is scoped across a four-level hierarchy:

```
client          → shared knowledge across your entire platform
  └── customer  → per-company context (multi-tenant B2B)
        └── user       → per-user memory and preferences
              └── conversation → in-session history
```

One `fetch()` call merges all relevant scopes in parallel.

---

## Framework integrations

Nine installable packages — not code snippets. Deep framework surfaces with callbacks, graph nodes, retrievers, memories, and plugins.

### LangChain

```bash
pip install maximem-synap synap-langchain
```

```python
from maximem_synap import MaximemSynapSDK
from synap_langchain import SynapChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory

sdk = MaximemSynapSDK(api_key="your-api-key")
await sdk.initialize()

chain = RunnableWithMessageHistory(
    ChatOpenAI(),
    lambda session_id: SynapChatMessageHistory(
        sdk=sdk, conversation_id=session_id, user_id="alice",
    ),
)
```

### CrewAI

```bash
pip install maximem-synap synap-crewai
```

```python
from synap_crewai import SynapStorageBackend

crew = Crew(
    agents=[...], tasks=[...],
    memory=True,
    storage=SynapStorageBackend(sdk=sdk, user_id="alice"),
)
```

### LlamaIndex

```bash
pip install maximem-synap synap-llamaindex
```

```python
from synap_llamaindex import SynapChatMemory

memory = SynapChatMemory(sdk=sdk, user_id="alice")
agent = ReActAgent.from_tools(tools, memory=memory)
```

### All integrations

| Package | Framework | Install |
|---|---|---|
| [synap-langchain](packages/integrations/synap-langchain/) | LangChain / LangGraph | `pip install synap-langchain` |
| [synap-llamaindex](packages/integrations/synap-llamaindex/) | LlamaIndex | `pip install synap-llamaindex` |
| [synap-crewai](packages/integrations/synap-crewai/) | CrewAI | `pip install synap-crewai` |
| [synap-haystack](packages/integrations/synap-haystack/) | Haystack | `pip install synap-haystack` |
| [synap-google-adk](packages/integrations/synap-google-adk/) | Google ADK | `pip install synap-google-adk` |
| [synap-autogen](packages/integrations/synap-autogen/) | AutoGen | `pip install synap-autogen` |
| [synap-openai-agents](packages/integrations/synap-openai-agents/) | OpenAI Agents SDK | `pip install synap-openai-agents` |
| [synap-semantic-kernel](packages/integrations/synap-semantic-kernel/) | Semantic Kernel | `pip install synap-semantic-kernel` |
| [synap-pydantic-ai](packages/integrations/synap-pydantic-ai/) | Pydantic AI | `pip install synap-pydantic-ai` |

*Vercel AI SDK coming soon.*

---

## Deep dives

Understand the system before building on it:

- 📘 **[Why we built Synap](https://www.maximem.ai/blog/why-we-built-synap)** — the problem with current AI memory systems
- ⚙️ **[How Synap works under the hood](https://www.maximem.ai/blog/how-maximem-synap-works)** — architecture, retrieval pipeline, and design decisions
- 📊 **[Benchmark results](https://www.maximem.ai/blog/synap-benchmark-results)** — 90.2% on LongMemEval, methodology, and reproducibility

---

## Agent skills

Drop-in instructions for coding agents (Claude Code, Cursor, etc.) that teach them how to wire Synap into your codebase.

- **[Maximem Synap skill](skills/synap/)** — covers SDK setup, scoping (User/Customer/Client), ingestion, retrieval, and one-page wiring guides for all 18+ supported frameworks.

---

## Requirements

- **Python SDK**: Python 3.9+
- **JavaScript SDK**: Node 18+ (Python 3.9+ for the bridge layer)
- A Synap API key — [get one at maximem.ai](https://www.maximem.ai/synap)

---

## Resources & community

- 📖 [Documentation](https://www.maximem.ai/docs)
- 🚀 [Dashboard](https://synap.maximem.ai)
- 𝕏 [Twitter / X](https://x.com/maximem_ai)
- 💼 [LinkedIn](https://www.linkedin.com/company/maximem-ai)

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the fork-first workflow, branch conventions, and how to add a new framework integration.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://www.maximem.ai"><strong>Maximem AI</strong></a>
</p>
