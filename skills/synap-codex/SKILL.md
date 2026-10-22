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

Synap is a managed memory layer for AI agents: it ingests conversations and documents, extracts structured knowledge, and serves scope-aware context at retrieval time. This is the Codex edition; `reference/`, `scripts/` and `examples/` are identical to the Claude Code skill. Read only the reference files you need.

## Sandbox & approvals (read first)

The procedure needs capabilities Codex gates behind approval. State what you need before you start:

- **Network access** — to run the CLI (`npx @maximem/synap-cli@latest` if `maximem-synap` is not on PATH), to `pip install maximem-synap` / `npm install @maximem/synap-js-sdk` and the framework package, and for the SDK to reach Synap Cloud at runtime.
- **File writes** — to add integration code and (if missing) a git-ignored `.env`.
- **A secret** — the `synap_...` API key. `api-keys create --write-env` writes it to `.env`; never print it back or commit it.

If running with restricted network/filesystem, tell the user which commands to run themselves.

## Procedure: onboarding a project end to end

**Never guess. If anything is unclear, stop and ask the human.** Exact wording, edge cases, retry rules, build states and the sign-out message: `reference/onboarding-details.md`. Questions and failure table: `reference/onboarding-ask-rules.md`. CLI reference: `reference/dashboard-setup.md`.

**CLI.** `<cli>` means `maximem-synap` from PATH if `maximem-synap --version` reports 0.3.0 or newer, otherwise `npx @maximem/synap-cli@latest` (needs Node and Python 3.10+). **Environment:** if the prompt names one, add `--env <name>` to every CLI command (or `export SYNAP_ENV=<name>`); a one-time code only works on the server that issued it. If it names none, do nothing extra.

**Quick reference.** Put `--json` or `--output json` before the subcommand (`<cli> --json instances list`); use the full `inst_...` id from JSON, never a truncated table id. `api-keys create --write-env` writes `SYNAP_API_KEY` and `SYNAP_INSTANCE_ID`, and off production also `SYNAP_BASE_URL`, `SYNAP_GRPC_HOST`, `SYNAP_GRPC_PORT`, `SYNAP_GRPC_USE_TLS` (never add those by hand). Always pass `--env` when the prompt names an environment and `--agent-name` at sign-in. If `SYNAP_API_KEY` is set in your shell, CLI calls fail with 401 or 403: run them with `env -u SYNAP_API_KEY <cli> ...`.

1. **Sign in.** `<cli> auth login --code <syn_ob_...> --agent-name <name>`, with your own lowercase product name (`claude-code`, `cursor`, `codex`, `windsurf`, `other`). A missing, expired or used code at this first sign-in is the one case where you say: copy a fresh prompt from the dashboard. No silent browser fallback. The key lasts 7 days; never `auth refresh` it.
2. **State.** `<cli> onboard status`; resume from the first unfinished step. Re-running the prompt is safe, also in a new session (the profile is on disk).

**Start variants (ask which applies; detail: `reference/no-code-path.md`).** Has a one-time code: step 1. No code: ask whether to get one (copy a fresh prompt from the dashboard) or sign in once with `<cli> auth login` (the human approves in the browser). Already has an instance and a runtime key: skip steps 1, 5, 6, 9's wait and the key step; read the key from the environment or `.env`, never ask the human to paste a key into chat (if it is missing, name the variable and have them add it to `.env` themselves); read the mode with whoami; still do Step 0, identity, the depth choice, the read path and verify.


**Step 0: learn Synap first (required, after step 2 and before step 3).** Read in full, not skimmed: /getting-started/quickstart, /concepts/memory-model-cheat-sheet, /concepts/memory-scopes, /concepts/how-ingestion-works, /concepts/retrieval-modes, /concepts/real-time-anticipation (docs.maximem.ai, `.md` for clean text), and once step 3 names the framework, /integrations/<name> before step 6. Then tell the human four lines on what you understood: (1) the scoping model (`user_id` and `customer_id` by B2C or B2B), (2) the three operations (write, read, live stream), (3) why the live stream matters, (4) the build/ready rule.

3. **Read the repo.** Detect the framework (`reference/frameworks/_index.md`). Infer only what the code supports.
4. **Ask the human, one question at a time, showing the evidence and still asking:** B2C or B2B (immutable); what identifies a returning person in this app (a stable user id; if there is none, memory can only be per session); compliance and sensitive data; what matters most to remember; what must never be stored. If the answers mean memory cannot serve the goal (a session id as the user id while the goal is remembering returning users), say so plainly before approval and ask again. For health, financial, children's or other personal data read /guides/pii-protection first. Run `onboard ask "<question>"` while waiting. Rules: `reference/onboarding-ask-rules.md`. Identity: never use a per-chat or session id as `user_id` unless the human confirms the app has no persistent identity; if so, say plainly that returning-person memory, profile and per-person erasure are lost, and warn that a shared-device id leaks between people.
**Choose the integration depth with the human (required, after step 4, before step 5).** The human decides; never decide alone or silently pick Basic. First form a recommendation from the repo (app kind, tool or reasoning loop, process lifetime, caller, deployment target, what they asked for) and an honest estimate of what the stream would add for THIS app (small for a short chat with no tools). Reuse the modality and turn speed already gathered for the use-case. Then say in plain words what Synap can do (memory ready before the user asks; conversations recorded as memory; better prediction from reasoning and tool use; fast or deeper recall; user profile; privacy controls) and ask ONE question, recommendation first, with 2 or 3 options: **Basic** (stores and recalls when your code asks; works anywhere), **Full** (the live stream; needs a long-running process; "I recommend X because Y"), and for serverless, edge or short-lived processes the **per-turn HTTP route** (no prepared context, no reasoning or tool traces; a worker next to the function is NOT a documented way to get the stream's benefits). Check dependency conflicts (e.g. protobuf) before offering or wiring the stream. Say what Basic or HTTP leaves out. Report `onboard report integrating in_progress --detail "depth=full|basic|http-turns"` and repeat the choice in the final report. Detail: `reference/integration-depth.md`.

For Full, send all five events every turn (calls: `reference/streaming.md`):

| Event | In plain words | Lost without it |
|---|---|---|
| `user_message` | the user spoke | the turn is not recorded |
| `agent_thinking` | why the agent chose it | only the edges of the turn are seen |
| `tool_call` | which tool, which arguments | what the agent is about to need |
| `tool_result` | what the tool returned | the strongest hint of the next question |
| `assistant_message` | the reply | anticipation never runs |

5. **Draft the use-case** from `instances template` (`reference/use-case-markdown.md`), every unanswered field literally `not stated`. Show the EXACT file you will upload, in full, and upload exactly that file, never a longer or different one. Quote its `shasum -a 256`; if you change one character after approval, ask again. Then `onboard report use_case_drafted done` and, after an explicit yes, `use_case_approved done`.
6. **Create** only after telling the human the build usually takes 15 to 20 minutes and getting a yes (ask before create): `instances create --name ... --relationship b2c|b2b --from-file usecase.md`. Do not wait: `onboard report integrating in_progress`, install the SDK and write the integration, but run no memory call yet.
7. **Integrate at the depth chosen.** Follow `reference/sdk-setup.md` and the framework file. **Done means** (`reference/integration-checklist.md`): the READ is wired into the model prompt (an integration that only writes is not done; say where) and memory informs while the app's own rules decide; writes init the SDK once, use a short timeout (about 5 s) and bounded retries, never block the user turn if avoidable, never swallow a failure silently, and you state what is lost on failure; if the app redacts, test it against the never-store list with realistic samples and tell the human which fields pass through; the integration is OFF-safe and logs one startup line saying ON or OFF; no staging key in production; pin the SDK and CLI versions; update `.env.example` and the README; add tests (Synap failing, the 409 path, redaction). For Full wire the whole stream with session end and clean shutdown (`reference/streaming.md`). **Docs checkpoint:** before each Synap call open its /sdk-reference page; after writing it re-read it and check every parameter, the scope rule (`user_id` always; `customer_id` only on B2B), the mode and the errors. Before step 9 print in chat a **"Docs read"** list: every page you actually OPENED (downloaded does not count), one line each on what it changed; step 9 is gated on /guides/production-checklist, /patterns/graceful-degradation, /sdk-reference/errors and each method page you used. If the docs and this skill disagree, the docs win; say so.
8. **Key.** `api-keys create -i <id> --write-env`, then `onboard report env_written done`. Never print, paste or commit the key. **Protect the build window:** run the project's tests with `env -u SYNAP_API_KEY ...` (or the integration disabled) until verify has passed, and tell the human why: a memory call on a building instance is refused and may be lost.
9. **Wait, then verify.** Run `instances wait <id>` in the BACKGROUND (output to a log file; poll the log or `onboard status`) so the human can keep talking to you; the build usually takes 15 to 20 minutes. Only when it reports ready, run `python scripts/verify_synap.py > verify.out 2>&1; echo "exit=$?"` and paste the WHOLE of verify.out unedited, WARN lines included (a WARN is explained, never dropped). It ends with `RESULT: PASS` or `RESULT: FAIL <reason>`, then `EXIT CODE: <n>`. A failure is reported as a failure. Then run `python scripts/check_integration_wiring.py <project dir>` and paste it (it fails when the project writes but never reads), drive the app's own endpoint with a sentinel and recall it through the app's own read code (or say that was not done), and report whether recalled text is used in a prompt.
10. **Announce in the session** with the filled `reference/final-report-template.md` (what is wired: write, read, stream, identity, sensitive data, tests; what was skipped and why), the verbatim verify output, every file changed. A known defect you caused is fixed first or asked about, never listed as a later decision. "Optional: view it in your dashboard" is never a step.
11. **Sign out last, only if verify passed:** send every `onboard report` first, then `<cli> auth logout`, and end with the required explanation (`reference/onboarding-details.md`). If verify failed, keep the key to rebuild or resume.

**Until `instances wait` reports ready, run no ingestion or retrieval at all, including `verify_synap.py`.** A refused call is HTTP 409 `instance_not_ready`. Handle BOTH forms: newer SDKs raise `InstanceNotReadyError` (`retry_after_seconds`, `from_cooldown`, `instance_id`), older SDKs (including 0.5.16) raise a `ConflictError` whose message says the memory architecture is still being built. Catch the typed class first, then the message; never loop inside the call. Tell the human their own app gets it too until the build finishes; reads degrade to no memory, writes retry later (examples: `reference/sdk-setup.md`).

Hard rules: the human never has to go back to the dashboard (every failure has an in-session recovery); never guess; `not stated` for gaps; confirm before create; never create or delete workspaces or touch billing or members; never print or commit the key.

## Why each rule exists

- **Ask, never guess / approve the exact file:** B2C or B2B and the use-case cannot change after create; a wrong guess costs a new instance and 15 to 20 minutes.
- **No memory calls before ready:** the server refuses them; wait the stated retry time, never loop.
- **Key in `.env`, never printed:** transcripts and logs are kept.
- **`auth logout` is last:** reports, rebuild and resume need the setup key first.

## Docs map (`docs.maximem.ai`, `.md` after any path gives clean markdown; full list: `reference/docs-map.md`)

| Topic | Pages | Re-visit when |
|---|---|---|
| Orientation | /getting-started/quickstart, /concepts/memory-model-cheat-sheet, /resources/faq | Step 0 |
| CLI | /cli/overview, /cli/commands | Each CLI step |
| SDK setup, auth | /setup/installation, /setup/authentication, /sdk/initialization, /sdk-reference/lifecycle/initialize | Before init code |
| Scoping, B2C vs B2B | /concepts/memory-scopes, /guides/multi-user-scoping | Every call with `user_id` or `customer_id` |
| Writes and reads | /sdk/ingestion, /sdk/context-fetch, /sdk-reference/memories/create, /sdk-reference/context/fetch | Before and after each call |
| Live stream | /concepts/real-time-anticipation, /sdk-reference/instance/listen | Before wiring it |
| Use-case and MACA | /concepts/memory-architecture, /guides/configuring-memory | Before drafting |
| Frameworks | /integrations/overview, /integrations/<name> | Before and after wiring |
| Privacy | /guides/pii-protection, /guides/erasure | Sensitive data named |
| Production, errors | /guides/production-checklist, /patterns/graceful-degradation, /sdk-reference/errors, /sdk/error-handling | Before step 9; on any error |

## What to load when

| Situation | Read |
| --- | --- |
| Onboarding | `reference/onboarding-details.md`, `onboarding-ask-rules.md`, `integration-depth.md`, `integration-checklist.md`, `final-report-template.md`, `no-code-path.md`, `use-case-markdown.md` |
| SDK, mental model, frameworks | `reference/mental-model.md`, `sdk-setup.md`, `frameworks/<name>.md`, `streaming.md` (long-lived processes) |
| Ingestion, reads, scoping, production, errors | `ingestion.md`, `context-fetch.md`, `core-concepts.md`, `production.md` |
| Comparing vendors, MCP (no code) | `discovery.md`, `frameworks/mcp.md` |

The docs-reading rules in Step 0 and the checkpoint are NOT optional; this table is only about local files.

## What this skill does NOT do

- Decide B2C vs B2B, compliance or what to store. Only the human decides.
- Create or delete workspaces, or touch billing or members.
- Configure MACA by hand: it is generated from the approved use-case (`https://docs.maximem.ai/concepts/memory-architecture`).
- Migrate data from another memory vendor: point to `https://docs.maximem.ai/migrations/overview`.

## Authoritative source

`https://docs.maximem.ai` is the source of truth; if this skill conflicts with it, the docs win. Index: `https://docs.maximem.ai/llms.txt`.

---
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. Source of truth: https://docs.maximem.ai (append `.md` to any page). Codex skill format confirmed against OpenAI Codex docs (developers.openai.com/codex/skills); re-verify if the format changes.*
