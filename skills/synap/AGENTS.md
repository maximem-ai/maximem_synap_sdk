# Maximem Synap: agent rules (single-file edition)

This file is **self-sufficient**: copy it alone into `AGENTS.md`, `.cursor/rules/synap.mdc`, `CONVENTIONS.md` or `.clinerules` and it carries the whole flow in condensed form. For the exact wording of any step, fetch the live files named in "Where the detail is" (raw files on the public repo and the docs, both plain markdown). With a skills installer you can instead install the full skill (`npx skills add maximem-ai/maximem_synap_sdk --skill synap -y`), whose `SKILL.md` is then the source of truth.

**Never guess. If anything is unclear, stop and ask the human. Never ask the human to paste a key into chat.**

## Start (ask which applies)

- **One-time code** (`syn_ob_...` in the prompt): follow the steps below from step 1.
- **No code:** the human either copies a fresh prompt from the dashboard, or you run `<cli> auth login` once (they approve in the browser), then continue at step 2.
- **Already has an instance and a runtime key:** skip sign-in, use-case, create, key and wait. Read `SYNAP_API_KEY` from the environment or `.env` (if missing, name the variable and have them add it to `.env`), read B2C or B2B from whoami (`user_context_isolation`: `equals_customer` is B2C, `strict` is B2B), then do Step 0, identity, depth, integrate, verify.

`<cli>` means `maximem-synap` from PATH if `maximem-synap --version` is 0.3.0 or newer, else `npx @maximem/synap-cli@latest` (Node and Python 3.10+). If the prompt names an environment add `--env <name>` to every command. Put `--json` or `--output json` before the subcommand. If `SYNAP_API_KEY` is set in your shell run CLI commands with `env -u SYNAP_API_KEY`.

## The steps

1. Sign in: `<cli> auth login --code <syn_ob_...> --agent-name <claude-code|cursor|codex|windsurf|other>`. A missing, expired or used code is the one case to say "copy a fresh prompt from the dashboard".
2. `<cli> onboard status`; resume from the first unfinished step (safe to re-run).
3. **Step 0, learn Synap first:** read in full https://docs.maximem.ai/getting-started/quickstart.md, /concepts/memory-model-cheat-sheet.md, /concepts/memory-scopes.md, /concepts/how-ingestion-works.md, /concepts/retrieval-modes.md, /concepts/real-time-anticipation.md and your framework's /integrations/<name>.md. Tell the human four lines on what you understood (scoping, the three operations, the live stream, the build/ready rule).
4. Read the repo; infer only what the code supports.
5. Ask one question at a time, showing evidence and still asking: B2C or B2B (immutable); **what identifies a returning person (a stable user id)**; compliance and sensitive data (read /guides/pii-protection.md for health, financial or children's data); what to remember; what never to store. Never use a per-session id as `user_id` unless the human confirms there is no persistent identity, and then say plainly that returning memory, profile and per-person erasure are lost; a shared-device id leaks between people.
6. **Choose the integration depth with the human** (the human decides, you recommend first, with an honest estimate for this app): **Basic** (stores and recalls when your code asks), **Full** (the live stream; needs a long-running process), and for serverless, edge or short-lived processes the **per-turn HTTP route** (`conversation.record_message`; no prepared context, no reasoning or tool traces; a worker next to a function is not a documented way to get the stream's benefits). Check dependency conflicts (protobuf) before wiring the stream. Report `onboard report integrating in_progress --detail "depth=..."`.
7. Draft the use-case from `<cli> instances template`, every unanswered field literally `not stated`. Show the exact file in full, quote its `shasum -a 256`, upload exactly that file, ask again after any change, then `onboard report use_case_drafted done` and, after an explicit yes, `use_case_approved done`.
8. Say the build usually takes 15 to 20 minutes and get a yes (ask before create): `<cli> instances create --name ... --relationship b2c|b2b --from-file usecase.md`. Do not wait.
9. Integrate. **Done means:** the READ is wired into the model prompt (a write-only integration is not done; say where) and memory informs while the app's own rules decide; writes init the SDK once, use about a 5 s timeout and bounded retries, never block the user turn if avoidable, never swallow a failure silently, and you say what is lost on failure; redaction is tested against the never-store list with realistic samples; OFF-safe with one startup log line saying ON or OFF; no staging key in production; versions pinned; `.env.example` and README updated; tests for Synap failing, the 409 path and redaction. Before each Synap call open its /sdk-reference page and re-read it after; print a **"Docs read"** list of pages you actually opened (/guides/production-checklist, /patterns/graceful-degradation, /sdk-reference/errors and each method page used) before verifying.
10. `<cli> api-keys create -i <id> --write-env`, then `onboard report env_written done`. Never print or commit the key. Run the project's tests with `env -u SYNAP_API_KEY` until verify passes.
11. Run `<cli> instances wait <id>` in the BACKGROUND (log file, poll `onboard status`). No ingestion or retrieval, including `verify_synap.py`, before it reports ready. A refused call is HTTP 409 `instance_not_ready`: newer SDKs raise `InstanceNotReadyError`, older ones (including 0.5.16) a `ConflictError` saying the memory architecture is still being built; handle both, typed first, wait the stated time, never loop inside the call.
12. Run `python scripts/verify_synap.py > verify.out 2>&1; echo "exit=$?"` (from the skill, https://raw.githubusercontent.com/maximem-ai/maximem_synap_sdk/main/skills/synap/scripts/verify_synap.py) and paste verify.out unedited, WARN lines included; it ends with `RESULT:` and `EXIT CODE:`. Also run `scripts/check_integration_wiring.py` (it fails when the project writes but never reads) and drive the app's own endpoint with a sentinel. Announce with the filled final-report template (what is wired: write, read, stream, identity, sensitive data, tests; what was skipped and why; open decisions; bugs not fixed). Then, only if verify passed, `<cli> auth logout` last, after every `onboard report`.

## Full stream: send all five events every turn

| Event | Call | Lost without it |
|---|---|---|
| `user_message` | `send_message(event_type="user_message")` | the turn is not recorded |
| `agent_thinking` | `record_thinking(...)` | only the edges of the turn are seen |
| `tool_call` | `record_tool_call(...)` | what the agent is about to need |
| `tool_result` | `record_tool_result(...)` | the strongest hint of the next question |
| `assistant_message` | `send_message(event_type="assistant_message")` | anticipation never runs |

One stream per process: `listen()` at startup, `end_session` when a conversation ends, `stop_listening()` on shutdown. Never hand-set `role` on tool or reasoning events. Never stream a turn and also `memories.create()` the same text. Scope: `user_id` always; `customer_id` only on B2B; `conversation_id` must be a UUID.

## Hard rules

Never guess; `not stated` for gaps; confirm before create; never print, paste or commit the key; never create or delete workspaces or touch billing or members; a known defect you caused is fixed before you report or asked about, never left as a later decision; the human never has to go back to the dashboard once the flow has started. The live docs at https://docs.maximem.ai win over this file.

## Where the detail is

Raw files on the public repo (`https://raw.githubusercontent.com/maximem-ai/maximem_synap_sdk/main/skills/synap/reference/`): `onboarding-details.md`, `onboarding-ask-rules.md`, `integration-depth.md`, `integration-checklist.md`, `final-report-template.md`, `no-code-path.md`, `use-case-markdown.md`, `sdk-setup.md`, `streaming.md`, `frameworks/<name>.md`. Docs index: https://docs.maximem.ai/llms.txt (the index of every page). To see installed versions run `pip show maximem-synap` or `npm ls @maximem/synap-js-sdk`; none is pinned here.
