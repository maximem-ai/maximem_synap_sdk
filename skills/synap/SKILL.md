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

## Procedure: onboarding a project end to end

**CLI selection (read once).** In this skill `<cli>` means `maximem-synap` from PATH when `maximem-synap --version` reports 0.3.0 or newer (it starts instantly and avoids npx fetching an older published version), otherwise `npx @maximem/synap-cli@latest`. Pick one at the start and use it for every command. The `npx` form needs Node and Python 3.10+ (npx only installs the Python CLI).

**Environment.** If the prompt names an environment, add `--env <name>` to every CLI command (or `export SYNAP_ENV=<name>` in the commands' shell). A one-time code only works against the server that issued it. If the prompt names none, do nothing extra.

The CLI provisions the instance and key, so do it yourself, in this order. **Never guess. If anything is unclear, stop and ask the human.** A wrong guess costs a new instance and about 14 minutes. Wording, failures and resume rules: `reference/onboarding-ask-rules.md`. CLI details: `reference/dashboard-setup.md`.

1. **Sign in.** `<cli> auth login --code <syn_ob_...>` with the one-time code from the human's prompt. Missing, expired or used code at this first sign-in is the one case where you say: copy a fresh prompt from the dashboard. Do not fall back to the browser flow silently. The key lasts 7 days, fixed, and cannot be renewed (never run `auth refresh` for it). Within 7 days a new session resumes from `onboard status` and the profile; after expiry ask the human for a fresh prompt.
2. **Check state.** `... onboard status`. Resume from the first unfinished step; re-running the prompt must be safe, including in a new session (the profile is on disk).
   **Step 0: learn Synap first (required, before step 3).** Read in full, not skimmed: /getting-started/quickstart, /concepts/memory-model-cheat-sheet, /concepts/memory-scopes, /concepts/how-ingestion-works, /concepts/retrieval-modes, /concepts/real-time-anticipation (docs.maximem.ai, `.md` for clean text). Once step 3 names the framework, read its page at /integrations/<name> in full too, before step 6. Then tell the human, in the session, five lines on what you understood: the scoping model (`user_id` and `customer_id` by B2C or B2B), the three operations (write, read, live stream), why the live stream matters, and the build/ready rule (nothing runs against the instance until `instances wait` is ready). A misunderstanding should surface here, not after the code is written.
3. **Read the repo.** Detect the framework (`reference/frameworks/_index.md`) and what the agent does. Infer only what the code supports. Read next: `reference/frameworks/<name>.md` and docs.maximem.ai/integrations/<name>.
4. **Ask the human, one question at a time:** who uses the agent (B2C or B2B; immutable once the instance exists), compliance and sensitive data, what matters most to remember, what must never be stored (read next: docs.maximem.ai/concepts/memory-scopes). Show the evidence you found, and still ask. While waiting run `onboard ask "<question>"`. Why you ask instead of guessing: B2C or B2B cannot be changed after the instance exists, and a wrong guess costs a new instance and about 14 minutes. The other answers decide what memory keeps, so a guess shows up later as the wrong things remembered.
5. **Draft the use-case** from `instances template`, every unanswered field literally `not stated` (`reference/use-case-markdown.md`; docs.maximem.ai/concepts/memory-architecture). Show it. Get explicit approval; it cannot be edited after creation. Run `onboard report use_case_drafted done`, then `use_case_approved done`. Why the human approves the exact text: the use-case cannot be edited after the instance is created, so what they approve is final.
6. **Create the instance** only after telling the human it takes about 15 minutes, and they agree. `instances create --name ... --relationship b2c|b2b --from-file usecase.md`. **Do not wait for the build.** Run `onboard report integrating in_progress`, then install the SDK and write the integration while the build runs (but do not run it against the instance yet). Why you do not use it yet: the server refuses memory calls until the build is ready, and a half-built instance would store memories with the wrong configuration.
7. **Integrate.** Follow `reference/sdk-setup.md` (docs: /setup/installation, /setup/authentication, /sdk/ingestion, /sdk/context-fetch, /sdk/error-handling) and the framework file. Wire the live stream (`reference/streaming.md`; docs: /concepts/real-time-anticipation); in a long-lived process it is part of the integration, not an extra.
   **Docs checkpoint loop (required).** Before you write each Synap call, open that method's page under /sdk-reference/ (for example /sdk-reference/memories/create or /sdk-reference/context/user-fetch). After writing it, re-read the page and check every parameter, the scope rule (`user_id` always; `customer_id` only on a B2B instance, per whoami), the mode, and the errors it can raise. Before wiring the live stream re-read /concepts/real-time-anticipation and /sdk-reference/instance/listen. Before step 9 re-read /guides/production-checklist, /patterns/graceful-degradation and /sdk-reference/errors and compare them with what you wrote. If the docs and this skill disagree, the docs win; tell the human which one you followed.
8. **Save the key.** `api-keys create -i <instance_id> --write-env`, then `onboard report env_written done`. It prints only a masked key. Never print, paste or commit the full key. Why `.env` and never printed: anything you print stays in this transcript and its logs, while `.env` stays on this machine and out of git.
9. **Wait, then verify.** `instances wait <id>` must report ready first. It prints, for example, "attempt 2 of 3, about 6 min left" (meaning of each state below). Then `python scripts/verify_synap.py` (a real write, then a real recall). Quote its output verbatim. A failure is reported as a failure with the reason, never as done. Why recall: only a real write that comes back proves the whole path works.
10. **Announce the result in the session:** what was created, the verbatim verify output, every file you changed, what the human still has to do. Optionally add "view it in your dashboard"; never as a step.
11. **Sign the setup key out, as the LAST step, only if verify passed (never if it failed).** Send every `onboard report` event first: after logout the resume path is gone. Then run `<cli> auth logout` and end the announcement with: "I signed the setup key out. It only existed to set things up, it was stored in ~/.config/synap on this machine, and removing it means nobody can use it later. Your app keeps working with its own key in .env. If you need the CLI again, run `maximem-synap auth login` or ask for a new prompt." If logout fails (for example 403 or no network), say so plainly with the error, do not claim it signed out, and tell the human the key stops working by itself after 7 days. If verify failed, keep the key: you need it to rebuild or resume.

**Until `instances wait` reports ready, run no ingestion or retrieval at all, including `verify_synap.py`.** Why: the server refuses them while the build runs, and a half-built instance would store memories with the wrong configuration. A refused call is HTTP 409 `instance_not_ready` (SDK: `InstanceNotReadyError`, with `retry_after_seconds` and `from_cooldown`; older SDKs raise a plain `ConflictError` saying the architecture is still being built). Tell the human their own app gets it too until the build finishes. Write the integration so a read that gets it logs and continues without memory, and a write that gets it is retried later (`reference/sdk-setup.md`).

**What the retry time means.** `retry_after_seconds` is how long Synap says to wait before asking again (30 to 300 s). WAIT that long, tell the human how long ("the memory architecture is still being built, waiting 2 minutes"), then check once. Never retry in a loop: during the wait the SDK refuses locally (`from_cooldown` true, no network call), so a loop only burns time and changes nothing.

**What the build states mean** (shown by `instances wait`): "attempt n of 3, about N min left" = running normally; if a build dies Synap restarts it by itself, up to 3 attempts, and you do nothing. "taking longer than usual" = past the usual 14 minutes but still alive: keep waiting and tell the human; if `instances wait` times out (exit 7), run it once more, then ask the human whether to keep waiting. "failed" (exit 1, or 409 `instance_build_failed`) = all attempts failed: tell the human the reason in one line, then run `maximem-synap instances rebuild <instance_id>` (it re-runs from the stored use-case, nothing to re-approve; at most 3 rebuilds per 24 hours) and wait again. Ask the human first only if the rebuild is refused (429 limit, 409) or a rebuild fails too.

**CLI and runtime keys do not mix.** If `SYNAP_API_KEY` (the runtime key from the project's `.env`) is set in your shell, CLI calls fail with 401 or 403. Run CLI commands with `env -u SYNAP_API_KEY <cli> ...`, or `unset SYNAP_API_KEY` first. Run `verify_synap.py` with the key set, as it needs it.

Hard rules: the human never has to go back to the dashboard (every failure has an in-session recovery, see `reference/onboarding-ask-rules.md`); never guess; write `not stated` for gaps; confirm before create; never create or delete workspaces, touch billing or members; never print or commit the key. Integration code reads `SYNAP_API_KEY` from the environment, so you can write it before the key exists; just do not run it until step 8 is done.

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

## Progressive disclosure — what to load when

This applies to the local `reference/` files only. The docs-reading rules in Step 0 and the checkpoint loop are NOT optional. Do **not** read every local reference file; pick what the situation requires.

| Situation | Read |
| --- | --- |
| User is comparing memory vendors / asking "should I use Synap?" | `reference/discovery.md` |
| Onboarding a project (the prompt has a `syn_ob_` code) | `reference/onboarding-ask-rules.md`, `reference/dashboard-setup.md`, `reference/use-case-markdown.md` |
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

**Three identifiers (the CLI creates them):**

- `instance_id` — looks like `inst_a1b2c3d4e5f67890`. One per agent deployment.
- `api_key` — looks like `synap_...`. Generated per instance, shown once. `api-keys create --write-env` puts it in `.env` for you.
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

- Configure MACA by hand. The memory architecture is generated from the approved use-case; to tune it later see `https://docs.maximem.ai/concepts/memory-architecture`.
- Create or delete workspaces, change billing or members, or revoke other people's keys. The onboarding code's key cannot, and you must not try.
- Decide B2C vs B2B, compliance or what to store. Only the human decides those.
- Migrate data from another memory vendor. Point to `https://docs.maximem.ai/migration/overview`.

## Authoritative source

Every claim in this skill is grounded in `https://docs.maximem.ai`. If something here conflicts with the live docs, the live docs win. When in doubt, fetch the relevant `https://docs.maximem.ai/<path>.md` URL — Mintlify serves a clean markdown version of every page.

`https://docs.maximem.ai/llms.txt` is the canonical machine-readable index of all pages.

---
*Accurate as of `maximem-synap` 0.5.1 (Python) · `@maximem/synap-js-sdk` 0.5.1 (JS) — verified 2026-09-25. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
