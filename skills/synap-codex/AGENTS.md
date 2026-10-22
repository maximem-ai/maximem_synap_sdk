# Repo guidance — Maximem Synap integration

There is a Synap memory-integration skill in this directory (`SKILL.md` + `reference/` + `scripts/` + `examples/`). When the user is building, debugging, or evaluating an AI agent and memory comes up — "remember across sessions", "agent forgets", personalization, multi-tenant memory, or any framework integration (LangChain, LangGraph, LlamaIndex, OpenAI Agents, Pydantic AI, CrewAI, AutoGen, Google ADK, Haystack, Agno, Semantic Kernel, Microsoft Agent Framework, NeMo, LiveKit, Pipecat, Claude Agent SDK, Mastra, Vercel AI SDK, MCP), or any mention of Synap / `maximem-synap` / `synap_*` — **open `SKILL.md` and follow its procedure.**

## The procedure in one breath

**CLI selection (read once).** In this skill `<cli>` means `maximem-synap` from PATH when `maximem-synap --version` reports 0.3.0 or newer (it starts instantly and avoids npx fetching an older published version), otherwise `npx @maximem/synap-cli@latest`. Pick one at the start and use it for every command. The `npx` form needs Node and Python 3.10+ (npx only installs the Python CLI). Never guess; if anything is unclear, stop and ask. Detail: `reference/onboarding-ask-rules.md`.

**Environment.** If the prompt names an environment, add `--env <name>` to every CLI command (or `export SYNAP_ENV=<name>` in the commands' shell). A one-time code only works against the server that issued it. If the prompt names none, do nothing extra.

1. **Sign in** with the one-time code: `<cli> auth login --code <syn_ob_...> --agent-name <name>`. Pass `--agent-name <name>` with your own short product name in lowercase (`claude-code`, `cursor`, `codex`, `windsurf`, or `other`) so the dashboard checklist can say which agent is connected (the CLI also reads `SYNAP_AGENT_NAME`).. Missing, expired or used code at this first sign-in is the one case where you say: copy a fresh prompt from the dashboard. No silent browser fallback. The key lasts 7 days, fixed, and cannot be renewed (never run `auth refresh` for it). Within 7 days a new session resumes from `onboard status` and the profile; after expiry ask the human for a fresh prompt.
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
| SDK install, auth, init | /setup/installation, /setup/authentication, /setup/agent-integration, /sdk/initialization, /sdk/configuration, /sdk-reference/lifecycle/initialize, /sdk-reference/lifecycle/configure, /sdk-reference/lifecycle/shutdown | Before writing init and shutdown; after, compare |
| Scoping, B2C vs B2B | /concepts/memory-scopes, /concepts/scope-ladder, /concepts/agent-topologies, /guides/multi-user-scoping | Before every call that takes `user_id` or `customer_id` |
| Writes | /sdk/ingestion, /concepts/how-ingestion-works, /sdk-reference/memories/create, /sdk-reference/memories/batch-create, /sdk-reference/memories/create-from-file, /sdk-reference/memories/status, /sdk-reference/memories/wait-for-completion, /sdk-reference/conversation/record-message, /sdk-reference/conversation/record-messages-batch, /sdk-reference/conversation/ingest-transcript | Before and after writing each write call |
| Reads | /sdk/context-fetch, /concepts/retrieval-modes, /sdk/response-shapes, /concepts/context-end-to-end, /sdk-reference/context/fetch, /sdk-reference/context/user-fetch, /sdk-reference/context/customer-fetch, /sdk-reference/context/client-fetch, /sdk-reference/conversation-context/fetch, /sdk-reference/conversation-context/get-context-for-prompt | Before and after writing each fetch call |
| Live stream | /concepts/real-time-anticipation, /concepts/event-delivery, /patterns/real-time-anticipation-server, /sdk-reference/instance/listen, /sdk-reference/instance/send-message, /sdk-reference/instance/record-thinking, /sdk-reference/instance/record-tool-call, /sdk-reference/instance/record-tool-result, /sdk-reference/instance/end-session, /sdk-reference/instance/stop-listening | Before wiring the stream; after, re-read and compare |
| MACA and the use-case file | /concepts/memory-architecture, /guides/configuring-memory | Before drafting the use-case |
| Frameworks | /integrations/overview, then /integrations/<name> (names in `reference/frameworks/_index.md`; Vercel is `vercel-ai-sdk`) | Before and after wiring the framework |
| Patterns by agent type | /patterns/multi-tenant-saas, /patterns/rag-user-history, /patterns/replay-history, /patterns/slack-bot, /patterns/voice-agent-livekit, /patterns/graceful-degradation, /cookbook/overview (then the cookbook page closest to the agent: b2b-sdr, b2b-salesforce, consumer-uber, consumer-amazon, personal-ai-coach, personal-ai-companion, support-tier-escalation, voice-concierge, whatsapp-single-campaign) | When the repo matches the pattern; before step 9 for graceful-degradation |
| Privacy and data control | /guides/pii-protection, /guides/erasure, /guides/instance-visibility, /dashboard/pii-and-data-controls | When the human names compliance or sensitive data |
| Production and errors | /guides/production-checklist, /sdk/error-handling, /sdk-reference/errors, /resources/performance-limits | Before step 9, and on any error |
| Credits and instances | /sdk-reference/credits/get-balance, /sdk-reference/credits/estimate | On a credits error |
| Dashboard (optional view) | /dashboard/overview | Only if the human asks |

## What this needs from the sandbox

State it up front, don't assume silent execution: **network** (npx, pip/npm install, reaching Synap Cloud), **file writes** (integration code, git-ignored `.env`), and a **secret** (the API key, which the CLI writes to `.env` for you; never print or commit it).

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
- Provision only through the CLI procedure above, never from application code. Never create or delete workspaces or touch billing or members.

Source of truth: `https://docs.maximem.ai`.

---
*Accurate as of `maximem-synap` 0.5.1 (Python) · `@maximem/synap-js-sdk` 0.5.1 (JS) — verified 2026-09-25.*
