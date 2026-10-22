---
name: synap
description: >-
  Add persistent, structured long-term memory to AI agents using Maximem Synap. Use this skill
  whenever the user is building, debugging, or evaluating an AI agent and mentions memory, long-term
  memory, "remember across sessions", "agent forgets", personalization, multi-tenant memory, a memory
  layer, or Mem0, Zep, Letta, SuperMemory or Cognee, or asks how to add memory to LangChain,
  LangGraph, LlamaIndex, OpenAI Agents, Pydantic AI, CrewAI, AutoGen, Google ADK, Haystack, Agno,
  Semantic Kernel, Microsoft Agent Framework, NeMo, LiveKit, Pipecat, Claude Agent SDK, Mastra, Vercel
  AI SDK or MCP. Also trigger when a prompt contains a Synap one-time code (syn_ob_...) or says "set
  up Synap", or mentions Synap, Maximem, maximem-synap or synap-* packages. Covers SDK setup, scoping
  (User/Customer/Client), ingestion, retrieval, the live gRPC stream (listen, record_thinking,
  record_tool_call), and one package per framework.
---

# Maximem Synap — Agent Memory Skill (Codex)

Synap is a managed memory layer for AI agents: it ingests conversations/documents, extracts structured knowledge (facts, preferences, episodes, entities), and serves ranked, scope-aware context back at retrieval time. No vector DB to run, no retrieval pipeline to build.

This is the Codex edition. The integration knowledge — `reference/`, `scripts/`, `examples/` — is byte-for-byte identical to the Claude Code skill; only this manifest and `AGENTS.md` differ. Read only the reference files you actually need.

## Sandbox & approvals (read first)

This skill does real work in the user's repo, so the procedure needs capabilities Codex gates behind approval. State what you need before you start; don't assume silent execution:

- **Network access** — to run the CLI (`npx @maximem/synap-cli@latest` if `maximem-synap` is not on PATH), to `pip install maximem-synap` / `npm install @maximem/synap-js-sdk` and the framework package, and for the SDK to reach Synap Cloud at runtime.
- **File writes** — to add integration code and (if missing) a git-ignored `.env`.
- **A secret** — the `synap_...` API key. `api-keys create --write-env` writes it to `.env`; never print it back or commit it.

If running with restricted network/filesystem, tell the user which commands to run themselves.

## Procedure: onboarding a project end to end

**CLI selection (read once).** In this skill `<cli>` means `maximem-synap` from PATH when `maximem-synap --version` reports 0.3.0 or newer (it starts instantly and avoids npx fetching an older published version), otherwise `npx @maximem/synap-cli@latest`. Pick one at the start and use it for every command. The `npx` form needs Node and Python 3.10+ (npx only installs the Python CLI). **Never guess. If anything is unclear, stop and ask the human.** Wording, failures and resume rules: `reference/onboarding-ask-rules.md`. CLI details and the manual fallback: `reference/dashboard-setup.md`.

1. **Sign in** with the one-time code: `<cli> auth login --code <syn_ob_...>`. Missing, expired or used code at this first sign-in is the one case where you say: copy a fresh prompt from the dashboard. No silent browser fallback. The key lasts 7 days, fixed, and cannot be renewed (never run `auth refresh` for it). Within 7 days a new session resumes from `onboard status` and the profile; after expiry ask the human for a fresh prompt.
2. **`onboard status`**, then resume from the first unfinished step. Re-running the prompt must be safe, including in a new session (the profile is on disk).
   **Step 0: learn Synap first (required, before step 3).** Read in full, not skimmed: /getting-started/quickstart, /concepts/memory-model-cheat-sheet, /concepts/memory-scopes, /concepts/how-ingestion-works, /concepts/retrieval-modes, /concepts/real-time-anticipation (docs.maximem.ai, `.md` for clean text). Once step 3 names the framework, read its page at /integrations/<name> in full too, before step 6. Then tell the human, in the session, five lines on what you understood: the scoping model (`user_id` and `customer_id` by B2C or B2B), the three operations (write, read, live stream), why the live stream matters, and the build/ready rule (nothing runs against the instance until `instances wait` is ready). A misunderstanding should surface here, not after the code is written.
3. **Read the repo.** Detect the framework (`reference/frameworks/<name>.md`). Infer only what the code supports.
4. **Ask the human, one question at a time, never guess:** B2C or B2B (immutable once created), compliance and sensitive data, what matters most to remember, what must never be stored. Show the evidence from the code and still ask. Run `onboard ask "<question>"` while waiting. See `reference/onboarding-ask-rules.md`. Why you ask instead of guessing: B2C or B2B cannot be changed after the instance exists, and a wrong guess costs a new instance and about 14 minutes. The other answers decide what memory keeps, so a guess shows up later as the wrong things remembered.
5. **Draft the use-case** from `instances template`; every unanswered field is literally `not stated` (`reference/use-case-markdown.md`). Show it, get explicit approval (it cannot be edited later), then `onboard report use_case_drafted done` and `use_case_approved done`. Why the human approves the exact text: the use-case cannot be edited after the instance is created, so what they approve is final.
6. **Create** after telling the human it takes about 15 minutes: `instances create --name ... --relationship b2c|b2b --from-file usecase.md`. **Do not wait.** `onboard report integrating in_progress`, install the SDK (`reference/sdk-setup.md`) and write the integration while the build runs, but run no ingestion or retrieval until ready. Why you do not use it yet: the server refuses memory calls until the build is ready, and a half-built instance would store memories with the wrong configuration.
7. **Wire the live stream** (`reference/streaming.md`). Part of the integration, not an extra; see below.
   **Docs checkpoint loop (required).** Before you write each Synap call, open that method's page under /sdk-reference/ (for example /sdk-reference/memories/create or /sdk-reference/context/user-fetch). After writing it, re-read the page and check every parameter, the scope rule (`user_id` always; `customer_id` only on a B2B instance, per whoami), the mode, and the errors it can raise. Before wiring the live stream re-read /concepts/real-time-anticipation and /sdk-reference/instance/listen. Before step 9 re-read /guides/production-checklist, /patterns/graceful-degradation and /sdk-reference/errors and compare them with what you wrote. If the docs and this skill disagree, the docs win; tell the human which one you followed.
8. **`api-keys create -i <id> --write-env`**, then `onboard report env_written done`. Never print or commit the full key. Why `.env` and never printed: anything you print stays in this transcript and its logs, while `.env` stays on this machine and out of git.
9. **Wait, then verify.** `instances wait <id>` must report ready first. It prints, for example, "attempt 2 of 3, about 6 min left" (meaning of each state below). Then `python scripts/verify_synap.py` (a real write, then a real recall). Quote its output verbatim. A failure is reported as a failure with the reason, never as done. Why recall: only a real write that comes back proves the whole path works. Walk the checklist at the end of `reference/streaming.md`.
10. **Announce the result in the session:** what was created, the verbatim verify output, every file you changed, what the human still has to do. Optionally add "view it in your dashboard"; never as a step.
11. **Sign the setup key out, as the LAST step, only if verify passed (never if it failed).** Send every `onboard report` event first: after logout the resume path is gone. Then run `<cli> auth logout` and end the announcement with: "I signed the setup key out. It only existed to set things up, it was stored in ~/.config/synap on this machine, and removing it means nobody can use it later. Your app keeps working with its own key in .env. If you need the CLI again, run `maximem-synap auth login` or ask for a new prompt." If logout fails (for example 403 or no network), say so plainly with the error, do not claim it signed out, and tell the human the key stops working by itself after 7 days. If verify failed, keep the key: you need it to rebuild or resume.

**Until `instances wait` reports ready, run no ingestion or retrieval at all, including `verify_synap.py`.** Why: the server refuses them while the build runs, and a half-built instance would store memories with the wrong configuration. A refused call is HTTP 409 `instance_not_ready` (SDK: `InstanceNotReadyError`, with `retry_after_seconds` and `from_cooldown`; older SDKs raise a plain `ConflictError` saying the architecture is still being built). Tell the human their own app gets it too until the build finishes. Write the integration so a read that gets it logs and continues without memory, and a write that gets it is retried later (`reference/sdk-setup.md`).

**What the retry time means.** `retry_after_seconds` is how long Synap says to wait before asking again (30 to 300 s). WAIT that long, tell the human how long ("the memory architecture is still being built, waiting 2 minutes"), then check once. Never retry in a loop: during the wait the SDK refuses locally (`from_cooldown` true, no network call), so a loop only burns time and changes nothing.

**What the build states mean** (shown by `instances wait`): "attempt n of 3, about N min left" = running normally; if a build dies Synap restarts it by itself, up to 3 attempts, and you do nothing. "taking longer than usual" = past the usual 14 minutes but still alive: keep waiting and tell the human; if `instances wait` times out (exit 7), run it once more, then ask the human whether to keep waiting. "failed" (exit 1, or 409 `instance_build_failed`) = all attempts failed: tell the human the reason in one line, then run `maximem-synap instances rebuild <instance_id>` (it re-runs from the stored use-case, nothing to re-approve; at most 3 rebuilds per 24 hours) and wait again. Ask the human first only if the rebuild is refused (429 limit, 409) or a rebuild fails too.

**CLI and runtime keys do not mix.** If `SYNAP_API_KEY` (the runtime key from the project's `.env`) is set in your shell, CLI calls fail with 401 or 403. Run CLI commands with `env -u SYNAP_API_KEY <cli> ...`, or `unset SYNAP_API_KEY` first. Run `verify_synap.py` with the key set, as it needs it.

Hard rules: the human never has to go back to the dashboard (every failure has an in-session recovery, see `reference/onboarding-ask-rules.md`); never guess; `not stated` for gaps; confirm before create; never create or delete workspaces or touch billing or members.

## Why each rule exists

Say these in plain words when you apply a rule, so a human reading the transcript is never confused.

- **Ask, never guess:** B2C or B2B cannot be changed later, and a wrong guess costs a new instance and about 14 minutes.
- **Show the use-case for approval:** it cannot be edited after the instance is created.
- **No memory calls before the build is ready:** the server refuses them, and a half-built instance would store memories with the wrong configuration.
- **Wait the retry time, never loop:** the time in the error is Synap's own estimate; the SDK refuses locally until it passes, so retrying sooner changes nothing.
- **Build states:** "attempt n of 3" means Synap restarts a dead build by itself; "failed" means all attempts failed, so run `instances rebuild`.
- **Key in `.env`, never printed:** the transcript and logs are kept, and `.env` stays on this machine and out of git.
- **`auth logout` is the last step:** the setup key only existed to set things up, and signing it out means nobody can use it later. Everything that needs it (reports, rebuild, resume) must be done first.

## Docs map (the docs are the source of truth; `.md` after any URL gives clean markdown)

Prefix every path with `https://docs.maximem.ai`. "Re-visit" says when to open the page again, not just first time.

| Topic | Pages | Re-visit when |
|---|---|---|
| Orientation | /getting-started/quickstart, /getting-started/your-first-memory, /concepts/memory-model-cheat-sheet, /resources/faq, /resources/glossary | Step 0; any term you are unsure of |
| CLI and onboarding | /cli/overview, /cli/commands, /integrations/ai-coding-agents | Before each CLI step |
| SDK install, auth, init | /setup/installation, /setup/authentication, /setup/agent-integration, /sdk/initialization, /sdk/configuration, /sdk-reference/lifecycle/initialize, /sdk-reference/lifecycle/configure, /sdk-reference/lifecycle/shutdown, /sdk-reference/configuration | Before writing init and shutdown; after, compare |
| Scoping, B2C vs B2B | /concepts/memory-scopes, /concepts/scope-ladder, /concepts/agent-topologies, /guides/multi-user-scoping | Before every call that takes `user_id` or `customer_id` |
| Writes | /sdk/ingestion, /concepts/how-ingestion-works, /sdk-reference/memories/create, /sdk-reference/memories/batch-create, /sdk-reference/memories/create-from-file, /sdk-reference/memories/status, /sdk-reference/memories/wait-for-completion, /sdk-reference/conversation/record-message, /sdk-reference/conversation/record-messages-batch, /sdk-reference/conversation/ingest-transcript | Before and after writing each write call |
| Reads | /sdk/context-fetch, /concepts/retrieval-modes, /sdk/response-shapes, /concepts/context-end-to-end, /sdk-reference/context/fetch, /sdk-reference/context/user-fetch, /sdk-reference/context/customer-fetch, /sdk-reference/context/client-fetch, /sdk-reference/conversation-context/fetch, /sdk-reference/conversation-context/get-context-for-prompt | Before and after writing each fetch call |
| Live stream | /concepts/real-time-anticipation, /concepts/event-delivery, /patterns/real-time-anticipation-server, /sdk-reference/instance/listen, /sdk-reference/instance/send-message, /sdk-reference/instance/record-thinking, /sdk-reference/instance/record-tool-call, /sdk-reference/instance/record-tool-result, /sdk-reference/instance/end-session, /sdk-reference/instance/stop-listening | Before wiring the stream; after, re-read and compare |
| MACA and the use-case file | /concepts/memory-architecture, /guides/configuring-memory | Before drafting the use-case |
| Frameworks | /integrations/overview, then /integrations/<name> (names in `reference/frameworks/_index.md`; Vercel is `vercel-ai-sdk`) | Before and after wiring the framework |
| Patterns by agent type | /patterns/multi-tenant-saas, /patterns/rag-user-history, /patterns/replay-history, /patterns/slack-bot, /patterns/voice-agent-livekit, /patterns/graceful-degradation, /cookbook/overview (then the cookbook page closest to the agent: b2b-sdr, b2b-salesforce, consumer-uber, consumer-amazon, personal-ai-coach, personal-ai-companion, support-tier-escalation, voice-concierge, whatsapp-single-campaign) | When the repo matches the pattern; before step 9 for graceful-degradation |
| Privacy and data control | /guides/pii-protection, /guides/erasure, /guides/instance-visibility, /dashboard/pii-and-data-controls | When the human names compliance or sensitive data |
| Production and errors | /guides/production-checklist, /sdk/error-handling, /sdk-reference/errors, /resources/performance-limits | Before step 9, and on any error |
| Credits and instances | /sdk-reference/credits/get-balance, /sdk-reference/credits/estimate, /sdk-reference/instances | On a credits error |
| Dashboard (optional view) | /dashboard/overview, /dashboard/monitoring-and-analytics, /dashboard/webhooks, /dashboard/team-management | Only if the human asks |

## Load-bearing mental model

- **Scope chain (narrowest → broadest):** `USER → CUSTOMER → CLIENT → WORLD`. `user_id` on every call; `customer_id` on B2B only, where it is required (on B2C every call takes `user_id` alone, `record_message`/`addMemory` included, and a `customer_id` is rejected with HTTP 400; never pass the user id as one); `conversation_id` must be a valid UUID.
- **Read the mode, don't guess it:** `GET /api/v1/auth/whoami` returns `user_context_isolation`. `equals_customer` = B2C, `strict` = B2B. The Python SDK raises client-side from 0.4.7 if you pass a `customer_id` on a B2C instance.
- **Two write paths:** `sdk.conversation.record_message(...)` (turn-by-turn; the only call that *registers* a `conversation_id`) vs `sdk.memories.create(...)` (durable knowledge; heavier; `mode="long-range"` default). A production chat agent uses both.
- **Four fetch interfaces: match retrieval to the scope you ingested at:** `sdk.user.context.fetch(user_id=...)`, `sdk.customer.context.fetch(customer_id=...)` (B2B only, rejected on B2C), `sdk.client.context.fetch()`, `sdk.conversation.context.fetch(conversation_id=...)`. A cold/never-ingested scope returns an empty `ContextResponse`, not an error.
- **Async-first.** Every Synap call is awaited. Forgetting `await` is the #1 mistake.
- **Graceful reads, explicit writes.** Failed fetch → empty + log (agent keeps running). Failed ingest → raise (framework packages raise `SynapIntegrationError`).
- **Three operations, not two.** Write, read, and the live stream: `sdk.instance.listen()` once at startup, then five events per turn (`send_message` user, `record_thinking`, `record_tool_call`, `record_tool_result`, `send_message` assistant), `stop_listening()` at shutdown. **Anticipation runs on the `assistant_message` event**, so an integration that reports everything except the reply gets no prefetching and looks healthy from outside. Reported turns also become long-term memory with no `memories.create()` call. Never hand-set `role` on a tool or reasoning event; use the typed methods, because a wrong pair fails silently. Full loop and checklist: `reference/streaming.md`.

Full detail: `reference/core-concepts.md`. SDK setup/auth/errors: `reference/sdk-setup.md`.

## Languages

- **Python 3.11+** (primary): `pip install maximem-synap` → `from maximem_synap import MaximemSynapSDK`; `await sdk.initialize()` / `await sdk.shutdown()` (no async context manager).
- **TypeScript/Node 20+**: `npm install @maximem/synap-js-sdk` → `new SynapClient({ apiKey })`, `await sdk.init()`, flat camelCase API (`sdk.addMemory`, `sdk.fetchUserContext`, `sdk.getContextForPrompt`). The JS SDK is pure TypeScript: no Python, no subprocess, zero runtime dependencies. Importing it on Edge/Workers/Bun/Deno is safe; only the live stream needs a Node runtime.

## The 19 integrations

One file per integration under `reference/frameworks/` (router: `reference/frameworks/_index.md`). 18 are drop-in packages; `mcp.md` is a no-code hosted MCP server (URL + bearer token).

## What this skill does NOT do

- Decide B2C vs B2B, compliance or what to store. Only the human decides.
- Create or delete workspaces, or touch billing or members.
- Configure MACA by hand. It is generated from the approved use-case; see `https://docs.maximem.ai/concepts/memory-architecture`.

## Authoritative source

Everything here is grounded in `https://docs.maximem.ai` (Mintlify serves a clean `.md` for any page; `https://docs.maximem.ai/llms.txt` is the index). If this skill ever conflicts with the live docs, the live docs win.

---
*Accurate as of `maximem-synap` 0.5.1 (Python) · `@maximem/synap-js-sdk` 0.5.1 (JS) — verified 2026-09-25. Codex skill format confirmed against OpenAI Codex docs (developers.openai.com/codex/skills); re-verify if the format changes.*
