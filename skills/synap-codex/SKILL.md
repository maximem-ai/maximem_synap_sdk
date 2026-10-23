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

- **Network access** — to run the CLI, install the SDK and framework package, and for the SDK to reach Synap Cloud.
- **File writes** — to add integration code and (if missing) a git-ignored `.env`.
- **A secret** — the `synap_...` API key. `api-keys create --write-env` writes it to `.env`; never print it back or commit it.

If restricted, tell the user which commands to run.

## Procedure: onboarding a project end to end

**Never guess. If anything is unclear, stop and ask the human.** Exact wording, edge cases, retry rules, build states and the sign-out message: `reference/onboarding-details.md`. Questions and failure table: `reference/onboarding-ask-rules.md`. CLI reference: `reference/dashboard-setup.md`.

**CLI.** `<cli>` means `maximem-synap` from PATH if `maximem-synap --version` reports 0.3.0 or newer, otherwise `npx @maximem/synap-cli@latest` (needs Node and Python 3.10+). **Environment:** if the prompt names one, add `--env <name>` to every CLI command (or `export SYNAP_ENV=<name>`); a one-time code only works on the server that issued it. 

**Quick reference.** Put `--json` or `--output json` before the subcommand; use the full `inst_...` id, never a truncated table id. `api-keys create --write-env` writes `SYNAP_API_KEY` and `SYNAP_INSTANCE_ID`, and off production also `SYNAP_BASE_URL`, `SYNAP_GRPC_HOST`, `SYNAP_GRPC_PORT`, `SYNAP_GRPC_USE_TLS` (never add those by hand). Always pass `--env` when the prompt names an environment and `--agent-name` at sign-in. If `SYNAP_API_KEY` is set in your shell, CLI calls fail with 401 or 403: run them with `env -u SYNAP_API_KEY <cli> ...`.

1. **Sign in.** `<cli> auth login --code <syn_ob_...> --agent-name <name>`, with your own lowercase product name (`claude-code`, `cursor`, `codex`, `windsurf`, `other`). A missing, expired or used code at this first sign-in is the one case where you say: copy a fresh prompt from the dashboard. No silent browser fallback. The key lasts 7 days; never `auth refresh` it.
2. **State.** `<cli> onboard status`; resume from the first unfinished step. Re-running the prompt is safe, also in a new session (the profile is on disk).

**Start variants (ask which applies; detail: `reference/no-code-path.md`).** One-time code: step 1. No code: a fresh dashboard prompt, or `<cli> auth login` once (the human approves in the browser). Existing instance and runtime key: skip steps 1, 5, 6, the key step and the wait; read the key from the environment or `.env`, never ask the human to paste a key into chat (name the variable and have them add it to `.env`); read the mode with whoami; still do Step 0, identity, the path choice, the read path and verify.

**Step 0: learn Synap first (required, after step 2 and before step 3).** Read in full, not skimmed: /getting-started/quickstart, /concepts/memory-model-cheat-sheet, /concepts/memory-scopes, /concepts/how-ingestion-works, /concepts/retrieval-modes, /concepts/real-time-anticipation (docs.maximem.ai, `.md` for clean text), and once step 3 names the framework, /integrations/<name> before step 6. Then tell the human four lines on what you understood: (1) the scoping model (`user_id` and `customer_id` by B2C or B2B), (2) the three operations (write, read, live stream), (3) why the live stream matters, (4) the build/ready rule.

3. **Read the repo.** Detect the framework (`reference/frameworks/_index.md`). Infer only what the code supports.
4. **Ask the human, one question at a time, showing the evidence and still asking:** B2C or B2B (immutable); what identifies a returning person in this app (a stable user id; if there is none, memory can only be per session); compliance and sensitive data; what matters most to remember; what must never be stored. If the answers mean memory cannot serve the goal (a session id as the user id while the goal is remembering returning users), say so plainly before approval and ask again. For health, financial, children's or other personal data read /guides/pii-protection first. Run `onboard ask "<question>"` while waiting. Rules: `reference/onboarding-ask-rules.md`. Never use a per-chat or session id as `user_id` unless the human confirms there is no persistent identity, then say plainly that returning memory, profile and per-person erasure are lost; a shared-device id leaks between people.
**Choose the integration path (required, after step 4, before step 5). Always use the SDK, never raw HTTP calls.** There are two SDK paths, and they differ in who does the work. On the **implicit path** (the live stream: `listen`, `send_message`, `record_*`) the app reports events and Synap works as they flow. On the **explicit path** the app calls the SDK's store and fetch methods (`record_message`, `memories.create`, `fetch`) itself. For any app that CAN run a long-lived process (server, worker, container, desktop or CLI agent, voice agent) the implicit path is the DEFAULT and brings Synap's best results; never present it as optional. Before asking, say what it gives (docs-backed only): context prepared before the user asks (a read is often a local cache hit in about a millisecond when Synap prepared it); turns recorded and compacted into memory as they flow, with no write calls; reasoning and tool use (stream only) sharpening what is prepared; delivery that survives dropped connections. Then ask as a confirmation, default yes: "I will wire the live stream so you get X, Y, Z. Tell me if you do not want it." Implement it FULLY unless the human EXPLICITLY says no ("I do not know", silence, "whatever you think" mean implement): the five events below, session end, clean shutdown, the docs checkpoint, after checking dependency conflicts (e.g. protobuf). If they say no, record it with the one-sentence consequence, build the explicit path and mention it in the final report. **Serverless, Edge or per-request hosts (Vercel, Lambda, short-lived workers):** the implicit path needs a long-lived process (`/concepts/grpc-vs-rest`), so in one plain sentence say it is not available on this host, build the explicit path with the production rules, and add one neutral sentence that if the app ever runs a long-lived process the implicit path becomes available. Do not pitch re-architecting. Report `onboard report integrating in_progress --detail "path=implicit|explicit-no|explicit-host"`. Sources: `reference/integration-depth.md`.

The five events for the implicit path, every turn (calls and guides: `reference/streaming.md`):

| Event | In plain words | Lost without it |
|---|---|---|
| `user_message` | the user spoke | the turn is not recorded |
| `agent_thinking` | why the agent chose it | only the edges of the turn are seen |
| `tool_call` | which tool, which arguments | what the agent is about to need |
| `tool_result` | what the tool returned | the strongest hint of the next question |
| `assistant_message` | the reply | anticipation never runs |

5. **Draft the use-case** from `instances template` (`reference/use-case-markdown.md`), every unanswered field literally `not stated`. Show the EXACT file you will upload, in full, and upload exactly that file, never a longer or different one. Quote its `shasum -a 256`; if you change one character after approval, ask again. Then `onboard report use_case_drafted done` and, after an explicit yes, `use_case_approved done`.
6. **Create** only after telling the human the build usually takes 15 to 20 minutes and getting a yes (ask before create): `instances create --name ... --relationship b2c|b2b --from-file usecase.md`. Do not wait: `onboard report integrating in_progress`, install the SDK and write the integration, but run no memory call yet.
7. **Integrate on the chosen path.** Follow `reference/sdk-setup.md` and the framework file. **Done means** (`reference/integration-checklist.md`): the READ is wired into the model prompt (write-only is not done; say where) and memory informs while the app's own rules decide; writes init the SDK once, set `TimeoutConfig` at init (about 5 s; the SDK has no per-call timeout), enforce the total deadline in the app (`asyncio.wait_for` or a worker thread), use bounded retries, never block the user turn if avoidable, never swallow a failure silently, and state what is lost; redaction, if any, is tested against the never-store list with realistic samples; one startup log line says ON or OFF; no staging key in production; versions pinned; `.env.example` and README updated; tests cover Synap failing, the 409 path and redaction. On the implicit path wire the whole stream (`reference/streaming.md`). **Docs checkpoint:** open each call's /sdk-reference page before writing it, re-read it after, and check parameters, the scope rule (`user_id` always; `customer_id` only on B2B), mode and errors. Before step 9 print a **"Docs read"** list: every page you actually OPENED (downloaded does not count), one line each on what it changed; step 9 is gated on /guides/production-checklist, /patterns/graceful-degradation, /sdk-reference/errors and each method page used. If docs and skill disagree, the docs win; say so.
8. **Key.** `api-keys create -i <id> --write-env`, then `onboard report env_written done`. Never print, paste or commit the key. **Protect the build window:** run the project's tests with `env -u SYNAP_API_KEY ...` (or the integration disabled) until verify has passed, and tell the human why: a memory call on a building instance is refused and may be lost.
9. **Wait, then verify.** Run `instances wait <id>` in the BACKGROUND (output to a log file; poll the log or `onboard status`) so the human can keep talking to you. Only when it reports ready, run `python scripts/verify_synap.py > verify.out 2>&1; echo "exit=$?"` (prefix `SYNAP_VERIFY_REQUIRE_STREAM=1` on the implicit path) and paste the WHOLE of verify.out unedited, WARN lines included (explain every WARN). It ends with `RESULT: PASS` or `RESULT: FAIL <reason>`, then `EXIT CODE: <n>`. Then run `python scripts/check_integration_wiring.py <project dir>` and paste it, drive the app's own endpoint with a sentinel and recall it through the app's own read code (or say that was not done), and say whether recalled text reaches a prompt.
10. **Announce in the session** with the filled `reference/final-report-template.md` (what is wired: write, read, stream, identity, sensitive data, tests; what was skipped and why), the verbatim verify output, every file changed. A known defect you caused is fixed first or asked about, never listed as a later decision. "Optional: view it in your dashboard" is never a step.
11. **Sign out last, only if verify passed:** send every `onboard report` first, then `<cli> auth logout`, and say it is signed out ONLY if it exits 0 and reports revoked (exit 9 `revocation_pending`: retry, then an open item; see `reference/onboarding-details.md`). If verify failed, keep the key.

**Until `instances wait` reports ready, run no ingestion or retrieval at all, including `verify_synap.py`.** A refused call is HTTP 409 `instance_not_ready`. Handle BOTH forms: newer SDKs raise `InstanceNotReadyError` (`retry_after_seconds`, `from_cooldown`, `instance_id`), older SDKs (including 0.5.16) raise a `ConflictError` whose message says the memory architecture is still being built. Catch the typed class first, then the message; never loop inside the call. Tell the human their own app gets it too; reads degrade to no memory, writes retry later (`reference/sdk-setup.md`).

Hard rules: the human never has to return to the dashboard; never guess; `not stated` for gaps; confirm before create; never create or delete workspaces or touch billing or members; never print or commit the key.

## Why each rule exists

- **Ask, never guess; approve the exact file:** B2C or B2B and the use-case cannot change after create.
- **No memory calls before ready:** the server refuses them.
- **Key in `.env`, never printed; `auth logout` last:** transcripts are kept; reports need the key first.

## Docs map (`docs.maximem.ai`, `.md` after any path gives clean markdown; full list: `reference/docs-map.md`)

| Topic | Pages | Re-visit when |
|---|---|---|
| Orientation, CLI | /getting-started/quickstart, /concepts/memory-model-cheat-sheet, /cli/overview | Step 0; each CLI step |
| SDK setup, auth | /setup/installation, /setup/authentication, /sdk/initialization, /sdk-reference/lifecycle/initialize | Before init code |
| Scoping, B2C vs B2B | /concepts/memory-scopes, /guides/multi-user-scoping | Every call with `user_id` or `customer_id` |
| Writes and reads | /sdk/ingestion, /sdk/context-fetch, /sdk-reference/memories/create, /sdk-reference/context/fetch | Before and after each call |
| Live stream (implicit path) | /concepts/real-time-anticipation, /concepts/grpc-vs-rest, /concepts/event-delivery, /patterns/real-time-anticipation-server, /sdk-reference/instance/listen, send-message, record-thinking, record-tool-call, record-tool-result, end-session, stop-listening (all in sdk-reference/instance), /capture/traces-and-tool-calls | Before wiring it; again after |
| Use-case and MACA | /concepts/memory-architecture, /guides/configuring-memory | Before drafting |
| Frameworks | /integrations/overview, /integrations/<name> | Before and after wiring |
| Privacy | /guides/pii-protection, /guides/erasure | Sensitive data named |
| Production, errors | /guides/production-checklist, /patterns/graceful-degradation, /sdk-reference/errors, /sdk/error-handling | Before step 9; on any error |

## What to load when

Local references live in `reference/`: `onboarding-details.md` (full steps), `onboarding-ask-rules.md`, `integration-depth.md` (the path choice and its sources), `integration-checklist.md`, `final-report-template.md`, `no-code-path.md`, `use-case-markdown.md`, `sdk-setup.md`, `streaming.md` (the implicit path), `frameworks/<name>.md`, `mental-model.md`, `docs-map.md`; the README lists them all. `codex-edition-notes.md` holds the Codex-specific notes. The docs-reading rules in Step 0 and the checkpoint are NOT optional.

## What this skill does NOT do

- Decide B2C vs B2B, compliance or what to store. Only the human decides.
- Create or delete workspaces, or touch billing or members.
- Configure MACA by hand: it is generated from the approved use-case (`https://docs.maximem.ai/concepts/memory-architecture`).
- Migrate data from another memory vendor: point to `https://docs.maximem.ai/migrations/overview`.

## Authoritative source

`https://docs.maximem.ai` is the source of truth; if this skill conflicts with it, the docs win. Index: `https://docs.maximem.ai/llms.txt`.

---
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. Source of truth: https://docs.maximem.ai (append `.md` to any page). Codex skill format confirmed against OpenAI Codex docs (developers.openai.com/codex/skills); re-verify if the format changes.*
