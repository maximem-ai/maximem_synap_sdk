---
name: synap
description: Add persistent, structured long-term memory to AI agents using Maximem Synap. Use this skill whenever the user is building, debugging, or evaluating an AI agent and mentions any of: "memory", "long-term memory", "persistent memory", "agent memory", "remember across sessions", "context window", "agent forgets", "user preferences", "personalization", "RAG over conversations", "multi-tenant memory", "memory layer", "Mem0", "Zep", "Letta", "SuperMemory", "Cognee", or asks how to integrate memory into LangChain, LangGraph, LlamaIndex, OpenAI Agents SDK, Pydantic AI, CrewAI, AutoGen, Google ADK, Haystack, Agno, Semantic Kernel, Microsoft Agent Framework, NVIDIA NeMo, LiveKit, Pipecat, Claude Agent SDK, Mastra, Vercel AI SDK, or MCP (no-code). Also trigger on direct mentions of "Synap", "Maximem", "maximem-synap", or `synap-*` package names. Covers SDK setup, scoping (User/Customer/Client), ingestion, retrieval, the live gRPC stream that drives real-time anticipation ("anticipation", "prefetch context", "streaming", "listen()", "record_thinking", "record_tool_call"), and one drop-in package per framework.
allowed-tools: Read, Write, Edit, Bash
---

# Maximem Synap — Agent Memory Skill

Synap is a managed memory layer for AI agents. Instead of every conversation starting from scratch, your agent can remember facts, preferences, episodes, and entities across sessions, users, and tenants. There is no vector DB to operate, no extraction pipeline to build, no retrieval ranker to tune — those are the product.

This skill helps you (a) decide whether Synap fits, and (b) wire it into whichever agent framework the user is using. Read only the reference files you actually need.

## When this skill is relevant

Trigger this skill the moment the user is doing any of:

- Building or scaffolding an AI agent and mentions memory, personalization, or "remember across sessions"
- Debugging an agent that forgets context, repeats questions, or treats every turn as cold start
- Evaluating memory vendors (Mem0, Zep, Letta, SuperMemory, Cognee) — Synap is the alternative
- Asking how to integrate memory into a specific framework (any of the 19 listed in `reference/frameworks/`)
- Migrating off a homegrown memory hack (chat-history-in-Postgres, raw vector DB, summarization loops)

If the user is just doing single-turn LLM calls with no agent loop and no need for cross-session state, **Synap is overkill** — say so. Be honest. See `reference/discovery.md` for the decision rubric.

## Procedure — the order to do this in

There is **no CLI**. Provisioning happens by hand in the dashboard; the SDK only *uses* a key
that already exists. Follow these steps and **do not skip the PAUSE**.

1. **Detect the stack.** Identify the user's framework (or "custom"). This selects which `reference/frameworks/<name>.md` to follow — see `reference/frameworks/_index.md`.
2. **Provision in the dashboard (manual).** Walk the user through `reference/dashboard-setup.md`: sign up → create Client → create Instance (+ upload a use-case `.md`, see `reference/use-case-markdown.md`) → set B2C/B2B → generate an API key.
3. **⏸ PAUSE.** Ask the user to paste their `synap_...` key (or set it themselves), then `export SYNAP_API_KEY=synap_...` and `export SYNAP_INSTANCE_ID=inst_...` (the dashboard shows both together). Do not write integration code before the key is set.
4. **Install.** The SDK + the framework package — see `reference/sdk-setup.md` and the chosen framework file. (Sandboxed agents need network + file-write approval for this.)
5. **Integrate.** Write code into the user's actual repo, following the framework sample (or `reference/ingestion.md` + `reference/context-fetch.md` for a custom stack).
6. **Wire the live stream.** Read `reference/streaming.md` and do it. Unless the user's app is per-request serverless, this step is part of the integration, not an optional extra — without it there is no anticipation, every retrieval is cold, and nothing becomes memory unless the user calls `memories.create()` by hand. Do not finish at step 5.
7. **Verify.** Run `python scripts/verify_synap.py`, and walk the checklist at the end of `reference/streaming.md`. Never report done without a green run.

## Progressive disclosure — what to load when

Do **not** read every reference file. Pick what the situation requires.

| Situation | Read |
| --- | --- |
| User is comparing memory vendors / asking "should I use Synap?" | `reference/discovery.md` |
| User has decided on Synap and is starting fresh | `reference/sdk-setup.md` then the relevant `reference/frameworks/*.md` |
| User is using one of the 19 supported frameworks | `reference/sdk-setup.md` + `reference/frameworks/<framework>.md` |
| **Any agent integration in a long-lived process** (server, worker, voice) | `reference/streaming.md` — **always**, alongside whatever else you read |
| User wants memory in an MCP client (no code) | `reference/frameworks/mcp.md` |
| User has a custom stack with no listed integration | `reference/sdk-setup.md` + `reference/ingestion.md` + `reference/context-fetch.md` |
| Multi-tenant B2B SaaS / "how do I scope per customer" | `reference/core-concepts.md` (scopes section) |
| Going to production / shipping | `reference/production.md` |
| Errors at runtime | `reference/sdk-setup.md` (error handling section) |

The 19 framework files in `reference/frameworks/` are listed and one-line-described in `reference/frameworks/_index.md`. Read that first if you're not sure which file to load.

## Bare-minimum mental model

You will need this to follow any of the framework guides.

**Three identifiers — copy from the user's Synap dashboard at synap.maximem.ai:**

- `instance_id` — looks like `inst_a1b2c3d4e5f67890`. One per agent deployment.
- `api_key` — looks like `synap_...`. Generated per instance, shown once.
- A `client_id` (`cli_...`) at the org level, but the SDK does not need it directly.

**Three operations — every integration is a thin wrapper around these:**

```python
# Write side: ingest a conversation or document
await sdk.memories.create(
    document="User: I prefer dark mode.\nAssistant: Noted.",
    document_type="ai-chat-conversation",
    user_id="alice",
    customer_id="acme",          # B2B only: required there, rejected on B2C
    mode="long-range",           # "fast" or "long-range"
)

# Read side: fetch context before the next LLM call.
# Match the retrieval interface to the scope you wrote at — we wrote with user_id,
# so we read at user scope. (For per-conversation memory, register turns with
# sdk.conversation.record_message(...) first, then use sdk.conversation.context.fetch.)
context = await sdk.user.context.fetch(
    user_id="alice",
    customer_id="acme",          # B2B only: required there, rejected on B2C
    search_query=["user preferences"],
    max_results=10,
    mode="fast",                 # "fast" (~50-100ms) or "accurate" (~200-500ms)
)

# Stream side: report the turn as it happens, so Synap can predict the next one
# and so turns become memory without an explicit write. One long-lived stream
# per process. Full loop and the five events: reference/streaming.md
await sdk.instance.listen()                     # once, at startup
await sdk.instance.send_message(
    content="I prefer dark mode.", role="user", event_type="user_message",
    conversation_id=conv_id, user_id="alice", customer_id="acme",
)
# ... record_thinking / record_tool_call / record_tool_result ...
await sdk.instance.send_message(
    content="Noted.", role="assistant", event_type="assistant_message",
    conversation_id=conv_id, user_id="alice", customer_id="acme",
)
await sdk.instance.stop_listening()             # once, at shutdown
```

The stream is the operation most integrations skip, and skipping it is why a
working integration can still deliver nothing: **anticipation runs on the
`assistant_message` event**, so an agent that never reports the reply gets no
prefetching at all and looks healthy from outside. Read `reference/streaming.md`
before you finish.

**Two instance modes. Find out which one you are on before writing a single call:**

`GET /api/v1/auth/whoami` returns `user_context_isolation`. `equals_customer` means B2C: the caller sends `user_id` only, and a `customer_id` is rejected with HTTP 400 (the Python SDK raises client-side from 0.4.7). `strict` means B2B: `customer_id` is required, and a `user_id` on its own is an error. `sdk.customer.context.fetch` (`POST /v1/context/customer/fetch`) is B2B only and is rejected on B2C. Never pass the user id as a customer id to fill the field, and never send both to see which one sticks: call whoami.

**Four scope levels — wider scopes are visible to narrower ones, never the reverse:**

```
USER   →  CUSTOMER  →  CLIENT  →  WORLD
private    org-wide      app-wide   global
```

Decide scoping at ingestion time by which `*_id` you pass. On B2C: `user_id` → user-scoped, nothing → client-scoped, and that is the whole set. On B2B: `user_id` + `customer_id` → user-scoped, `customer_id` only → org-shared, nothing → client-scoped.

**Two modes per axis — pick one:**

| | `fast` | `accurate` / `long-range` |
| --- | --- | --- |
| Ingestion | Lightweight extraction, seconds | Full pipeline + graph, seconds-to-minutes |
| Retrieval | Vector only, ~50-100ms | Vector + graph + multi-signal rank, ~200-500ms |

Default to `fast` for retrieval (it's in the agent hot path) and `long-range` for ingestion (extraction quality compounds).

## SDK lifecycle

Every Python integration assumes you have done this once at process start:

```python
import os
from maximem_synap import MaximemSynapSDK

sdk = MaximemSynapSDK(
    api_key=os.environ["SYNAP_API_KEY"],
)
await sdk.initialize()      # validates key, resolves the instance, opens connection
# ... use sdk ...
await sdk.shutdown()        # flush telemetry, close connections
```

TypeScript uses a different, flatter API — package `@maximem/synap-js-sdk`:

```typescript
import { SynapClient } from "@maximem/synap-js-sdk";

const sdk = new SynapClient({ apiKey: process.env.SYNAP_API_KEY! });
await sdk.init();                 // note: init(), not initialize()
// write: await sdk.addMemory({ userId, customerId, messages, mode })
// read:  await sdk.fetchUserContext({ userId, searchQuery, mode })
// customerId is B2B only: on a B2C instance send userId alone.
await sdk.shutdown();
```

The JS SDK is pure TypeScript: no Python, no subprocess, zero runtime dependencies, Node 20+. gRPC is behind a lazy `import()`, so importing it on Edge or in a Worker is safe; only the live stream needs Node. There is no `MaximemSynapSDK` class and no `sdk.memories` / `sdk.conversation` namespaces in JS.

The Python SDK is a **singleton per API key** — constructing twice with the same key returns the same live SDK. This is intentional; do not work around it. Different keys give different SDKs, so one process can serve several tenants. For tests use `_force_new=True`.

**Critical:** every SDK call is async. Forgetting `await` is the #1 mistake.

## The 19 supported frameworks at a glance

| Framework | Package | Language | Style |
| --- | --- | --- | --- |
| LangChain | `maximem-synap-langchain` | Python | History + callback + retriever + tools |
| LangGraph | `maximem-synap-langgraph` | Python | Checkpointer + cross-thread Store |
| LlamaIndex | `maximem-synap-llamaindex` | Python | `BaseMemory` + retriever |
| OpenAI Agents SDK | `maximem-synap-openai-agents` | Python | Function tools |
| Pydantic AI | `maximem-synap-pydantic-ai` | Python | Deps + auto-registered tools |
| CrewAI | `maximem-synap-crewai` | Python | `StorageBackend` |
| AutoGen | `maximem-synap-autogen` | Python | `BaseTool` |
| Google ADK | `maximem-synap-google-adk` | Python | `FunctionTool` factory |
| Haystack | `maximem-synap-haystack` | Python | Pipeline components |
| Agno | `maximem-synap-agno` | Python | `InMemoryDb` subclass |
| Semantic Kernel | `maximem-synap-semantic-kernel` | Python | Kernel plugin |
| Microsoft Agent Framework | `maximem-synap-microsoft-agent` | Python | Context + history providers |
| NVIDIA NeMo Agent Toolkit | `maximem-synap-nemo-agent-toolkit` | Python | `MemoryEditor` |
| LiveKit Agents | `maximem-synap-livekit-agents` | Python | Preload + recording + tools |
| Pipecat | `maximem-synap-pipecat` | Python | Frame processors |
| Claude Agent SDK | `maximem-synap-claude-agent` / `@maximem/synap-claude-agent` | Py + TS | Hooks + MCP server |
| Mastra | `@maximem/synap-mastra` | TypeScript | `SynapMemory` + tools |
| Vercel AI SDK | `@maximem/synap-vercel-adk` | TypeScript | Model middleware |

For any of these, jump to `reference/frameworks/<name>.md`. They share a contract:

- **Read failures degrade gracefully** — context fetch errors return empty results and log; the agent keeps running.
- **Write failures surface explicitly** — ingestion errors raise `SynapIntegrationError` (or framework equivalent).
- **Same scoping model**: every helper accepts `user_id`, `customer_id` (B2B only, and required there), optional `conversation_id`.

Five of them also drive the live stream from the framework's own hooks: OpenAI Agents, Google ADK, the Claude Agent SDK, the Vercel AI SDK and Strands Agents. Each is silent until `listen()` is running, and none of them covers every event on its own — see the table in `reference/streaming.md` for what is left for you in each case.

## Custom stack (no integration package)

If the user's framework isn't in the list (rare), they wire `sdk.memories.create()` and `sdk.conversation.context.fetch()` directly. See `reference/ingestion.md` and `reference/context-fetch.md`. The live stream is plain SDK calls and works with any framework or none — `reference/streaming.md` still applies.

## Defaults to use unless told otherwise

When generating code, default to:

- Read the environment variable `SYNAP_API_KEY`. Never hardcode. `SYNAP_INSTANCE_ID` is optional — the instance is resolved from the key.
- Ingestion `mode="long-range"`, `document_type="ai-chat-conversation"`.
- Retrieval `mode="fast"`, `max_results=10`.
- Always pass `user_id`. Add `customer_id` only on a B2B instance, where it is required. Confirm the mode with `GET /api/v1/auth/whoami` (`user_context_isolation`); on B2C a `customer_id` is rejected with HTTP 400.
- `conversation_id` must be a valid UUID — if the user passes a session string, wrap it: `str(uuid5(NAMESPACE_URL, session_str))`.
- Open the live stream. In a long-lived process this is a default, not a question to ask: `listen()` at startup, the five events per turn, `stop_listening()` at shutdown. Skip it only for per-request serverless, edge runtimes and batch jobs, and say why when you do.

## What this skill does NOT do

- Configure MACA (Memory Architecture Configuration). That's a YAML file in the dashboard. Mention it exists; point to `https://docs.maximem.ai/concepts/customized-memory-architectures` and let the user configure it themselves.
- Create instances or API keys. The user must do this from `https://synap.maximem.ai`. The skill should never attempt to provision.
- Migrate data from another memory vendor. Point to `https://docs.maximem.ai/migration/overview`.

## Authoritative source

Every claim in this skill is grounded in `https://docs.maximem.ai`. If something here conflicts with the live docs, the live docs win. When in doubt, fetch the relevant `https://docs.maximem.ai/<path>.md` URL — Mintlify serves a clean markdown version of every page.

`https://docs.maximem.ai/llms.txt` is the canonical machine-readable index of all pages.

---
*Accurate as of `maximem-synap` 0.5.1 (Python) · `@maximem/synap-js-sdk` 0.5.1 (JS) — verified 2026-09-25. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
