# Repo guidance — Maximem Synap integration

There is a Synap memory-integration skill in this directory (`SKILL.md` + `reference/` + `scripts/` + `examples/`). When the user is building, debugging, or evaluating an AI agent and memory comes up — "remember across sessions", "agent forgets", personalization, multi-tenant memory, or any framework integration (LangChain, LangGraph, LlamaIndex, OpenAI Agents, Pydantic AI, CrewAI, AutoGen, Google ADK, Haystack, Agno, Semantic Kernel, Microsoft Agent Framework, NeMo, LiveKit, Pipecat, Claude Agent SDK, Mastra, Vercel AI SDK, MCP), or any mention of Synap / `maximem-synap` / `synap_*` — **open `SKILL.md` and follow its procedure.**

## The procedure in one breath

1. Detect the framework → pick `reference/frameworks/<name>.md`.
2. Walk the user through manual dashboard provisioning (`reference/dashboard-setup.md`) — **there is no CLI.**
3. **PAUSE** for the `synap_...` API key; `export SYNAP_API_KEY=...`. Don't write code before it's set.
4. Install the SDK + framework package (`reference/sdk-setup.md`).
5. Integrate using the framework sample.
6. Verify with `python scripts/verify_synap.py` — never report done without a green run.

## What this needs from the sandbox

State it up front, don't assume silent execution: **network** (pip/npm install + reaching Synap Cloud), **file writes** (integration code, git-ignored `.env`), and a **secret** (the API key — never print or commit it).

## Non-negotiables

- Every SDK call is `async` — always `await`.
- `conversation_id` must be a valid UUID.
- `user_id` on every call; on B2B pass `customer_id` too. `record_message` / `addMemory` require `customer_id` even on B2C (use the same value as `user_id`).
- Match the retrieval interface to the scope you ingested at.
- Reads degrade gracefully; writes surface failures.
- Never provision instances/keys from code — the user does that at `https://synap.maximem.ai`.

Source of truth: `https://docs.maximem.ai`.

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.3.0 (JS) — verified 2026-06-20.*
