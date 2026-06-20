---
name: synap
description: Add persistent, structured long-term memory to AI agents using Maximem Synap. Use this skill whenever the user is building, debugging, or evaluating an AI agent and mentions any of: "memory", "long-term memory", "persistent memory", "agent memory", "remember across sessions", "context window", "agent forgets", "user preferences", "personalization", "RAG over conversations", "multi-tenant memory", "memory layer", "Mem0", "Zep", "Letta", "SuperMemory", "Cognee", or asks how to integrate memory into LangChain, LangGraph, LlamaIndex, OpenAI Agents SDK, Pydantic AI, CrewAI, AutoGen, Google ADK, Haystack, Agno, Semantic Kernel, Microsoft Agent Framework, NVIDIA NeMo, LiveKit, Pipecat, Claude Agent SDK, Mastra, Vercel AI SDK, or MCP (no-code). Also trigger on direct mentions of "Synap", "Maximem", "maximem-synap", or `synap-*` package names. Covers SDK setup, scoping (User/Customer/Client), ingestion, retrieval, and one drop-in package per framework.
---

# Maximem Synap — Agent Memory Skill (Codex)

Synap is a managed memory layer for AI agents: it ingests conversations/documents, extracts structured knowledge (facts, preferences, episodes, entities), and serves ranked, scope-aware context back at retrieval time. No vector DB to run, no retrieval pipeline to build.

This is the Codex edition. The integration knowledge — `reference/`, `scripts/`, `examples/` — is byte-for-byte identical to the Claude Code skill; only this manifest and `AGENTS.md` differ. Read only the reference files you actually need.

## Sandbox & approvals (read first)

This skill does real work in the user's repo, so the procedure needs capabilities Codex gates behind approval. State what you need before you start; don't assume silent execution:

- **Network access** — to `pip install maximem-synap` / `npm install @maximem/synap-js-sdk` and the framework package, and for the SDK to reach Synap Cloud at runtime.
- **File writes** — to add integration code and (if missing) a git-ignored `.env`.
- **A secret** — the `synap_...` API key. Ask the user to provide it; never print it back or commit it.

If running with restricted network/filesystem, tell the user which commands to run themselves.

## Procedure — the order to do this in

There is **no CLI**. Provisioning happens by hand in the dashboard; the SDK only *uses* a key that already exists. **Do not skip the PAUSE.**

1. **Detect the stack.** Identify the user's framework (or "custom"). This selects which `reference/frameworks/<name>.md` to follow — see `reference/frameworks/_index.md`.
2. **Provision in the dashboard (manual).** Walk the user through `reference/dashboard-setup.md`: sign up → create Client → create Instance (+ upload a use-case `.md`, see `reference/use-case-markdown.md`) → set B2C/B2B → generate an API key.
3. **⏸ PAUSE.** Ask the user to paste their `synap_...` key (or set it themselves), then `export SYNAP_API_KEY=synap_...`. Do not write integration code before the key is set.
4. **Install.** The SDK + the framework package (needs network + approval — see "Sandbox & approvals"). Details in `reference/sdk-setup.md`.
5. **Integrate.** Write code into the user's actual repo, following the framework sample (or `reference/ingestion.md` + `reference/context-fetch.md` for a custom stack).
6. **Verify.** Run `python scripts/verify_synap.py`. Never report done without a green run.

## Load-bearing mental model

- **Scope chain (narrowest → broadest):** `USER → CUSTOMER → CLIENT → WORLD`. `user_id` on every call; `customer_id` on B2B (on B2C, `record_message`/`addMemory` still require it — pass the same value as `user_id`); `conversation_id` must be a valid UUID.
- **Two write paths:** `sdk.conversation.record_message(...)` (turn-by-turn; the only call that *registers* a `conversation_id`) vs `sdk.memories.create(...)` (durable knowledge; heavier; `mode="long-range"` default). A production chat agent uses both.
- **Four fetch interfaces — match retrieval to the scope you ingested at:** `sdk.user.context.fetch(user_id=...)`, `sdk.customer.context.fetch(customer_id=...)`, `sdk.client.context.fetch()`, `sdk.conversation.context.fetch(conversation_id=...)`. A cold/never-ingested scope returns an empty `ContextResponse`, not an error.
- **Async-first.** Every Synap call is awaited. Forgetting `await` is the #1 mistake.
- **Graceful reads, explicit writes.** Failed fetch → empty + log (agent keeps running). Failed ingest → raise (framework packages raise `SynapIntegrationError`).

Full detail: `reference/core-concepts.md`. SDK setup/auth/errors: `reference/sdk-setup.md`.

## Languages

- **Python 3.11+** (primary): `pip install maximem-synap` → `from maximem_synap import MaximemSynapSDK`; `await sdk.initialize()` / `await sdk.shutdown()` (no async context manager).
- **TypeScript/Node 18+**: `npm install @maximem/synap-js-sdk` → `createClient({ apiKey })`, `await sdk.init()`, flat camelCase API (`sdk.addMemory`, `sdk.fetchUserContext`, `sdk.getContextForPrompt`). The JS SDK spawns Python as a subprocess, so it **also needs Python 3.11+ on the host** and does not run on Edge/Workers/Bun/Deno/Node-only-Lambda.

## The 19 integrations

One file per integration under `reference/frameworks/` (router: `reference/frameworks/_index.md`). 18 are drop-in packages; `mcp.md` is a no-code hosted MCP server (URL + bearer token).

## What this skill does NOT do

- Provision instances or API keys — the user does that in the dashboard at `https://synap.maximem.ai`. Never attempt it from code.
- Configure MACA (the memory-architecture file) — point to `https://docs.maximem.ai/concepts/customized-memory-architectures`.

## Authoritative source

Everything here is grounded in `https://docs.maximem.ai` (Mintlify serves a clean `.md` for any page; `https://docs.maximem.ai/llms.txt` is the index). If this skill ever conflicts with the live docs, the live docs win.

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.3.0 (JS) — verified 2026-06-20. Codex skill format confirmed against OpenAI Codex docs (developers.openai.com/codex/skills); re-verify if the format changes.*
