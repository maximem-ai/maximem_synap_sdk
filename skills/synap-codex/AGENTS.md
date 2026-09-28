# Repo guidance — Maximem Synap integration

There is a Synap memory-integration skill in this directory (`SKILL.md` + `reference/` + `scripts/` + `examples/`). When the user is building, debugging, or evaluating an AI agent and memory comes up — "remember across sessions", "agent forgets", personalization, multi-tenant memory, or any framework integration (LangChain, LangGraph, LlamaIndex, OpenAI Agents, Pydantic AI, CrewAI, AutoGen, Google ADK, Haystack, Agno, Semantic Kernel, Microsoft Agent Framework, NeMo, LiveKit, Pipecat, Claude Agent SDK, Mastra, Vercel AI SDK, MCP), or any mention of Synap / `maximem-synap` / `synap_*` — **open `SKILL.md` and follow its procedure.**

## The procedure in one breath

1. Detect the framework → pick `reference/frameworks/<name>.md`.
2. Walk the user through manual dashboard provisioning (`reference/dashboard-setup.md`) — **there is no CLI.**
3. **PAUSE** for the `synap_...` API key; `export SYNAP_API_KEY=...` and `export SYNAP_INSTANCE_ID=inst_...` (the dashboard shows both). Don't write code before they're set. Set the instance id as an env var, never as an `instance_id=` constructor argument.
4. Install the SDK + framework package (`reference/sdk-setup.md`).
5. Integrate using the framework sample.
6. **Wire the live stream** (`reference/streaming.md`). Part of the integration, not an extra — see below.
7. Verify with `python scripts/verify_synap.py` plus the checklist at the end of `reference/streaming.md` — never report done without a green run.

## What this needs from the sandbox

State it up front, don't assume silent execution: **network** (pip/npm install + reaching Synap Cloud), **file writes** (integration code, git-ignored `.env`), and a **secret** (the API key — never print or commit it).

## The live stream is not optional

Synap has three operations, not two: write (`memories.create`), read (`fetch`),
and the live stream (`sdk.instance.listen()` plus five events per turn). Most
integrations stop after two, and that is the most common way one ends up
working on paper and delivering nothing.

- **Anticipation runs on the `assistant_message` event.** Report the user's turn
  and never the reply, and you get zero prefetching while everything looks
  healthy from outside.
- **Reported turns become long-term memory on their own.** No `memories.create()`
  per conversation.
- **Without a stream every `fetch()` is a cold retrieval**, paying full network
  and ranking cost on every turn.

In a long-lived process (server, worker, voice session), wire it. Skip it only
for per-request serverless, edge runtimes and batch jobs, where nothing can hold
a connection, and say why. Read `reference/streaming.md` for the loop, the
silent-failure rules, and which framework packages report which events for you.

## Non-negotiables

- Every SDK call is `async` — always `await`.
- `conversation_id` must be a valid UUID.
- `user_id` on every call; on B2B pass `customer_id` too, where it is required. On B2C, `user_id` is the whole scope: a `customer_id` is rejected with HTTP 400 on every call, `record_message` / `addMemory` included, and passing the user id as one is the same bug. Read the mode from `GET /api/v1/auth/whoami` (`user_context_isolation`: `equals_customer` = B2C, `strict` = B2B); the Python SDK raises client-side from 0.4.7.
- `customer.context.fetch` / `POST /v1/context/customer/fetch` is B2B only and is rejected on B2C.
- Match the retrieval interface to the scope you ingested at.
- Open the stream in a long-lived process, and never hand-set `role` on a tool or reasoning event — use `record_tool_call` / `record_tool_result` / `record_thinking`. A wrong pair fails silently.
- Reads degrade gracefully; writes surface failures.
- Never provision instances/keys from code — the user does that at `https://synap.maximem.ai`.

Source of truth: `https://docs.maximem.ai`.

---
*Accurate as of `maximem-synap` 0.5.1 (Python) · `@maximem/synap-js-sdk` 0.5.1 (JS) — verified 2026-09-25.*
