# Codex edition notes (moved from the Codex SKILL.md and AGENTS.md)

The Codex wrapper files are short pointers. The longer text they used to carry is kept here.

## Load-bearing mental model

- **Scope chain (narrowest → broadest):** `USER → CUSTOMER → CLIENT`. `user_id` on every call; `customer_id` on B2B only, where it is required (on B2C every call takes `user_id` alone, `record_message`/`addMemory` included, and a `customer_id` is rejected with HTTP 400; never pass the user id as one); `conversation_id` must be a valid UUID.
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



---

<!-- Moved from skills/synap-codex/AGENTS.md -->
# Repo guidance — Maximem Synap integration

There is a Synap memory-integration skill in this directory (`SKILL.md` + `reference/` + `scripts/` + `examples/`). When the user is building, debugging, or evaluating an AI agent and memory comes up — "remember across sessions", "agent forgets", personalization, multi-tenant memory, or any framework integration (LangChain, LangGraph, LlamaIndex, OpenAI Agents, Pydantic AI, CrewAI, AutoGen, Google ADK, Haystack, Agno, Semantic Kernel, Microsoft Agent Framework, NeMo, LiveKit, Pipecat, Claude Agent SDK, Mastra, Vercel AI SDK, MCP), or any mention of Synap / `maximem-synap` / `synap_*` — **open `SKILL.md` and follow its procedure.**

## The procedure in one breath

**CLI selection (read once).** In this skill `<cli>` means `maximem-synap` from PATH when `maximem-synap --version` reports 0.3.3 or newer (it starts instantly and avoids npx fetching an older published version), otherwise `npx @maximem/synap-cli@latest`. Pick one at the start and use it for every command. The `npx` form needs Node and Python 3.10+ (npx only installs the Python CLI). Never guess; if anything is unclear, stop and ask. Detail: `reference/onboarding-ask-rules.md`.

**Environment (never say this word to the human).** If the prompt names an environment, add `--env <name>` to every CLI command (or `export SYNAP_ENV=<name>` in the commands' shell). A one-time code only works against the server that issued it. If the prompt names none, do nothing extra.

**Quick reference (CLI).** Put global flags before the subcommand: `<cli> --output json instances list` (`--json` is an alias only if your CLI accepts it). Ids can be shown truncated in a table: use the full `inst_...` id from `--output json`, never a shortened one. `api-keys create --write-env` writes `SYNAP_API_KEY` and `SYNAP_INSTANCE_ID` (or the `_<NAME>` forms with `--env-suffix NAME`), and when the CLI is pointed at another server also `SYNAP_BASE_URL`, `SYNAP_GRPC_HOST`, `SYNAP_GRPC_PORT` and `SYNAP_GRPC_USE_TLS`: do not add those by hand. Always pass `--env <name>` when the prompt names an environment, always pass `--agent-name <name>` at sign-in, and run CLI commands with `env -u SYNAP_API_KEY` when that variable is set in your shell. Full CLI reference: `reference/dashboard-setup.md`.

1. **Sign in** with the one-time code: `<cli> auth login --code <syn_ob_...> --agent-name <name>`. Pass `--agent-name <name>` with your own short product name in lowercase (`claude-code`, `cursor`, `codex`, `windsurf`, or `other`) so the dashboard checklist can say which agent is connected (the CLI also reads `SYNAP_AGENT_NAME`).. Missing, expired or used code at this first sign-in is the one case where you say: copy a fresh prompt from the dashboard. No silent browser fallback. The key lasts 7 days, fixed, and cannot be renewed (never run `auth refresh` for it). Within 7 days a new session keeps the sign-in (the profile is on disk) but still follows step 2: it trusts only what it reported itself; after expiry ask the human for a fresh prompt.
2. **`onboard status`** shows this session's own steps and a read-only list of OTHER sessions. Trust only the steps you reported yourself in this session; the other sessions are someone else's work: never act on them or answer their questions, and never skip drafting, approval or creation because they look done. Login printed your profile name: put `--profile <name>` before the subcommand of EVERY command, and never log out or touch a profile that is not yours (`onboarding-details.md`, "Sessions and profile pin"). After the which-agent answer and before creating the instance run `onboard target "<AgentName>"` (the client's agent, not an instance id). Re-running the prompt is safe.
2b. **What exists:** `<cli> instances list --json`; tell the human in plain words each instance (name, what it is for if known, ready, being built or failed: `reference/instances-explained.md`). If any exist, ask ONE question with your recommendation and tip: reuse one (only if it is for THIS agent; one instance serves one agent) or create a new one. Reuse skips drafting, approval, creation and the build wait (if ready); details in `onboarding-details.md` step 2b.

**Step 0: learn Synap first (required; do it after step 2 and before step 3).** Read in full, not skimmed: /getting-started/quickstart, /concepts/memory-model-cheat-sheet, /concepts/memory-scopes, /concepts/how-ingestion-works, /concepts/retrieval-modes, /concepts/real-time-anticipation (docs.maximem.ai, `.md` for clean text). Once step 3 names the framework, read its page at /integrations/<name> in full too, before step 6. Then tell the human, in the session, four lines on what you understood: (1) the scoping model (`user_id` and `customer_id` by B2C or B2B), (2) the three operations (write, read, live stream), (3) why the live stream matters, (4) the build/ready rule (nothing runs against the instance until `instances wait` is ready). A misunderstanding should surface here, not after the code is written.

3. **Read the repo.** Detect the framework (`reference/frameworks/<name>.md`). Infer only what the code supports.
4. **Ask the human, one question at a time, never guess:** B2C or B2B (chosen when the instance is created), what identifies a returning person in this app (a stable user id; if there is none, memory can only be per session), compliance and sensitive data, what matters most to remember, what must never be stored. Show the evidence from the code and still ask. Run `onboard ask "<question>"` while waiting. See `reference/onboarding-ask-rules.md`. Why you ask instead of guessing: B2C or B2B is chosen at creation (a later change goes through the email request below), and a wrong guess costs a new instance and 15 to 20 minutes. If the repo has more than one agent (several agent classes, services, personas, bots), ask first which agent gets memory now and say that further agents each need their own instance later. Every question carries your recommendation (what you suggest and why, from the repo) and a tip on how Synap uses the answer, written only after you read the docs page(s) named for that question in `reference/onboarding-ask-rules.md`. The other answers decide what memory keeps, so a guess shows up later as the wrong things remembered. If the answer means memory cannot serve the approved goal (for example a session id as the user id while the goal is remembering returning users), say so plainly before approval and ask again. If the human names health, financial, children's or other personal data, read /guides/pii-protection first (rules in `reference/onboarding-ask-rules.md`).
5. **Draft the use-case** from `instances template`; every unanswered field is literally `not stated` (`reference/use-case-markdown.md`). Show it, get explicit approval (once the file is sent, it can only be changed by making a request through email to gaurav@maximem.ai), then `onboard report use_case_drafted done` and `use_case_approved done`. Why the human approves the exact text: the human cannot edit it themselves after it is sent (it can only be changed by making a request through email to gaurav@maximem.ai), so say that when you ask. Show the EXACT file you will upload: print its full contents in chat (or give its path and show the full text), and upload exactly that file, never a longer or different one. At approval run `shasum -a 256 usecase.md` and quote the hash; if the CLI prints a sha256 on create it must match. If you change one character after approval, ask again.
6. **Create** after telling the human it usually takes 15 to 20 minutes: `instances create --name ... --relationship b2c|b2b --from-file usecase.md`. A 409 `instance_name_taken` means the name is in use: say so plainly and ask whether to reuse that instance or choose another name. A 409 `instance_limit_reached` (the workspace or this session hit its instance limit) or 429 `instance_create_rate_limited`: stop, tell the human the limit and count from the message, and ask whether to reuse an existing instance or contact support. Never retry, delete instances to make room, or use another key or workspace. **Do not wait.** `onboard report integrating in_progress`, install the SDK (`reference/sdk-setup.md`) and write the integration while the build runs, but run no ingestion or retrieval until ready. Why you do not use it yet: the server refuses memory calls until the build is ready, and a half-built instance would store memories with the wrong configuration.
7. **Wire the live stream** (`reference/streaming.md`). Part of the integration, not an extra; see below.
   **Docs checkpoint loop (required).** Before you write each Synap call, open that method's page under /sdk-reference/ (for example /sdk-reference/memories/create or /sdk-reference/context/user-fetch). After writing it, re-read the page and check every parameter, the scope rule (`user_id` always; `customer_id` only on a B2B instance, per whoami), the mode, and the errors it can raise. Before wiring the live stream re-read /concepts/real-time-anticipation and /sdk-reference/instance/listen. Before step 9 print in chat a **"Docs read" list**: every page you actually OPENED (read, not just downloaded or curled; a page you did not read does not count) with one line on what it changed in the code. Step 9 is gated on these being opened: /guides/production-checklist, /patterns/graceful-degradation, /sdk-reference/errors and each /sdk-reference method page you used. Compare them with what you wrote. If the docs and this skill disagree, the docs win; tell the human which one you followed.
8. **`api-keys create -i <id> --write-env`**, then `onboard report env_written done`. Never print or commit the full key. This setup key manages only instances this session created: a 403 `instance_not_created_by_this_session` (exit 5) on a reused instance means ask the human to create that key in the dashboard (wording in `onboarding-details.md`, "Setup-key limits"), never look for a way around it. One key per instance: if the project will have more than one Synap instance (ask when unsure), add `--env-suffix NAME` (CLI 0.3.3+; named after the instance or agent: letters, digits, underscore), which writes `SYNAP_API_KEY_<NAME>` and `SYNAP_INSTANCE_ID_<NAME>`; the CLI refuses with exit 6 to overwrite another instance's plain variables unless you add a suffix or `--force`. The app must read the matching variable and pass it explicitly; verify with `verify_synap.py --suffix NAME` (`reference/sdk-setup.md`, "Several instances"). Why `.env` and never printed: anything you print stays in this transcript and its logs, while `.env` stays on this machine and out of git. **Protect the build window:** the project's own tests or dev server may load `.env` and call Synap while the build is still running. Run the project's tests with `env -u SYNAP_API_KEY ...` (or the integration disabled) until verify has passed, and tell the human why: a memory call on a building instance is refused and the write may be lost.
9. **Wait, then verify.** `instances wait <id>` must report ready first. Run it in the BACKGROUND (redirect its output to a log file, then poll the log or `<cli> onboard status`) so the human can keep talking to you; the build usually takes 15 to 20 minutes (measured: 14m45s and about 20m). It prints, for example, "attempt 2 of 3, about 6 min left" (meaning of each state below). Then run `python scripts/verify_synap.py > verify.out 2>&1; echo "exit=$?"` (add `--suffix NAME` for a suffixed key; no pipe, so you see the real exit code) and paste the WHOLE of verify.out unedited, including every `[WARN]` line; a WARN is reported and explained, never dropped. It ends with `RESULT: PASS` or `RESULT: FAIL <reason>`, then `EXIT CODE: <n>`. A failure is reported as a failure with the reason, never as done. Why recall: only a real write that comes back proves the whole path works. Walk the checklist at the end of `reference/streaming.md`.
10. **Clean up test data, then announce the result in the session.** Every test or probe user id you create is `test_user_<purpose>_<8 random chars>`; write each down. For exactly those ids run `python scripts/cleanup_test_data.py --user-id <id> ... --dry-run`, then without `--dry-run`, and paste both outputs (it touches only ids starting `test_user_`; never delete any other, never the human's real users; full text in `onboarding-details.md` step 10). Announce: what was created, the test data removed and what could not be, the verbatim verify output, every file you changed, what the human still has to do, a line "Nothing was committed" and the list of files created and changed. Optionally add "view it in your dashboard"; never as a step.
11. **Sign the setup key out, as the LAST step, only if verify passed (never if it failed).** Send every `onboard report` event first: after logout the resume path is gone. Verify first (the dashboard shows finished only after the first memory and a useful recall; an earlier sign-out shows "ended before verification"). Then run `<cli> auth logout`. Only when it exits 0 and reports the key revoked, end the announcement with: "I signed the setup key out. It only existed to set things up and was stored in ~/.config/synap on this machine; it is revoked, so it can no longer be used. Your app keeps working with its own key in .env. If you need the CLI again, run `maximem-synap auth login` or ask for a new prompt." If it fails (for example 403 or no network), say so plainly with the error and do not claim it signed out. Exit code 9 (`revocation_pending`) means the CLI could not revoke the key after 3 tries and kept the local profile marked pending: the key is STILL ACTIVE until its expiry (7 days at most; it can read and create instances and keys). Rerun `auth logout` once or twice; if it is still pending, tell the human plainly that the key stays active until its expiry and that they can revoke it on the dashboard API keys page, and put it in the final report as an OPEN ITEM, not as done. `--json` output has `revoked` and `pending`. If verify failed, keep the key: you need it to rebuild or resume.

**Until `instances wait` reports ready, run no ingestion or retrieval at all, including `verify_synap.py`.** Why: the server refuses them while the build runs, and a half-built instance would store memories with the wrong configuration. A refused call is HTTP 409 `instance_not_ready`. Handle BOTH forms: newer SDKs raise `InstanceNotReadyError` (fields `retry_after_seconds`, `from_cooldown`, `instance_id`; the SDK tells you how long to wait); older SDKs, including the published 0.5.16, raise a `ConflictError` whose message says the memory architecture is still being built. Catch the typed class first, then check the `ConflictError` message. Never loop inside the call. Examples: `reference/sdk-setup.md`. Tell the human their own app gets it too until the build finishes. Write the integration so a read that gets it logs and continues without memory, and a write that gets it is retried later (`reference/sdk-setup.md`).

**What the retry time means.** `retry_after_seconds` is how long Synap says to wait before asking again (30 to 300 s). WAIT that long, tell the human how long ("the memory architecture is still being built, waiting 2 minutes"), then check once. Never retry in a loop: during the wait the SDK refuses locally (`from_cooldown` true, no network call), so a loop only burns time and changes nothing.

**What the build states mean** (shown by `instances wait`): "attempt n of 3, about N min left" = running normally; if a build dies Synap restarts it by itself, up to 3 attempts, and you do nothing. "taking longer than usual" = past the usual 15 to 20 minutes but still alive: keep waiting and tell the human; if `instances wait` times out (exit 7), run it once more, then ask the human whether to keep waiting. "failed" (exit 1, or 409 `instance_build_failed`) = all attempts failed: tell the human the reason in one line, then run `maximem-synap instances rebuild <instance_id>` (it re-runs from the stored use-case, nothing to re-approve; at most 3 rebuilds per 24 hours) and wait again. Ask the human first only if the rebuild is refused (429 limit, 409) or a rebuild fails too.

**CLI and runtime keys do not mix.** If `SYNAP_API_KEY` (the runtime key from the project's `.env`) is set in your shell, CLI calls fail with 401 or 403. Run CLI commands with `env -u SYNAP_API_KEY <cli> ...`, or `unset SYNAP_API_KEY` first. Run `verify_synap.py` with the key set, as it needs it.

Hard rules: the human never has to go back to the dashboard (every failure has an in-session recovery, see `reference/onboarding-ask-rules.md`); never guess; `not stated` for gaps; confirm before create; never create or delete workspaces or touch billing or members; Never run `git commit`, `git add -A`, `git push`, `git stash` or a commit `--amend`: leave every change uncommitted and every new file untracked, for the human to review and commit. Never say "production", "prod", "staging" or "environment" about Synap to the human: for them there is just Synap (if the prompt names an environment, use it silently with `--env`). Never show raw status words (`building`, `failed`, `attempt 2 of 3`, `b2c`, `no_instance`) to the human: translate them with `reference/instances-explained.md`; if a word is not there, read the docs first and say it simply.

## Why each rule exists

Say these in plain words when you apply a rule, so a human reading the transcript is never confused.

- **Ask, never guess:** B2C or B2B is chosen at creation and a change goes through the email request, and a wrong guess costs a new instance and 15 to 20 minutes.
- **Show the use-case for approval:** once the file is sent, it can only be changed by making a request through email to gaurav@maximem.ai.
- **No memory calls before the build is ready:** the server refuses them, and a half-built instance would store memories with the wrong configuration.
- **Wait the retry time, never loop:** the time in the error is Synap's own estimate; the SDK refuses locally until it passes, so retrying sooner changes nothing.
- **Build states:** "attempt n of 3" means Synap restarts a dead build by itself; "failed" means all attempts failed, so run `instances rebuild`.
- **Key in `.env`, never printed:** the transcript and logs are kept, and `.env` stays on this machine and out of git.
- **`auth logout` is the last step:** the setup key only existed to set things up, and once revoked it can no longer be used (say so only after `auth logout` exits 0). Everything that needs it (reports, rebuild, resume) must be done first.

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

In a long-lived process (server, worker, voice session) the stream is the default:
confirm it with the human (default yes) and wire it fully unless they explicitly say no.
Where nothing can hold a connection (per-request serverless, edge runtimes, batch jobs)
say in one sentence that it is not available and use the explicit SDK path
(`integration-depth.md`). Read `reference/streaming.md` for the loop, the
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
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here.*
