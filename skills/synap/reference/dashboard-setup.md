# Provisioning with the CLI (and the manual fallback)

> **Read this next:** [CLI overview](https://docs.maximem.ai/cli/overview.md); [CLI command reference](https://docs.maximem.ai/cli/commands.md); [AI coding agents](https://docs.maximem.ai/integrations/ai-coding-agents.md); [Clients and instances](https://docs.maximem.ai/concepts/memory-scopes.md). The live docs win over this file.

The Synap CLI creates the instance and the API key, so the coding agent does it. Run it as
`<cli> <command>`, where `<cli>` is `maximem-synap` from PATH if `maximem-synap --version` reports 0.3.2 or newer, otherwise `npx @maximem/synap-cli@latest` (npx is only the installer: it puts the Python CLI on the PATH, so that form needs Node and Python 3.10+). Put `--output json` before the subcommand when you need to parse the result,
for example `... --output json instances get <id>`. The procedure order is in `SKILL.md`; the questions to
ask are in `onboarding-ask-rules.md`.

If `SYNAP_API_KEY` (the runtime key from the project's `.env`) is set in the shell, CLI calls fail with 401/403: run them with `env -u SYNAP_API_KEY ...` or unset it first.

The CLI never prompts when there is no terminal. A command that would prompt exits 2 and names the flag that
answers it. Exit codes: 0 ok, 1 generic, 2 usage, 3 auth, 4 not found, 5 permission or scope, 6 conflict,
7 timeout, 8 server or network, 9 revocation pending (`auth logout` could not revoke the key). A 409 `instance_build_failed` or `instance_not_ready` is exit 6 with the server's body under `details` and the next step in `hint`.

## Commands the onboarding uses, in order

| Step | Command | Notes |
|---|---|---|
| Sign in | `auth login --code <syn_ob_...> --agent-name <name>` | Code also accepted as env `SYNAP_ONBOARDING_CODE`. Single use, expires in 10 minutes. The key it stores lasts exactly 7 days, cannot be renewed (after that, ask the human for a fresh prompt), and cannot touch billing, members or deletes. |
| State | `onboard status` | Shows the lifecycle state and what the agent has reported. Use it to resume. |
| Template | `instances template --output-file usecase.md` | The current use-case template. See `use-case-markdown.md`. |
| Ask | `onboard ask "<question>"` / `onboard ask --clear` | Lets an open dashboard show that your agent is waiting. Always ask the human in the session too. |
| Report | `onboard report <step> <status> [--detail TEXT]` | Steps: `use_case_drafted`, `use_case_approved`, `integrating`, `env_written`. Status: `in_progress`, `done`, `failed`. |
| Create | `instances create --name NAME --relationship b2c\|b2b --from-file usecase.md` | `--relationship` has no safe default: pass what the human confirmed. Also takes `--agent-type`. Starts the build (15 to 20 minutes). |
| Key | `api-keys create -i <instance_id> --write-env [--env-suffix NAME] [--force]` | Writes `SYNAP_API_KEY` and `SYNAP_INSTANCE_ID` into `./.env` (or the path you give), keeps the file git-ignored, and prints only a masked key. With `--env-suffix NAME` (CLI 0.3.2 and newer; letters, digits, underscore) it writes `SYNAP_API_KEY_<NAME>` and `SYNAP_INSTANCE_ID_<NAME>` instead and leaves other instances' lines alone: use it whenever the project will have more than one instance. Without a suffix it refuses (exit 6) to overwrite another instance's plain variables unless you pass `--force`. |
| List | `instances list --json` | What the workspace already has (name, status, who uses it). Run it right after sign-in and tell the human in plain words (`instances-explained.md`). |
| Wait | `instances wait <instance_id> [--timeout S] [--interval S]` | Run it in the background with output to a log file and poll the log or `onboard status`, so the human can keep talking to you. Usually 15 to 20 minutes. Prints "attempt 2 of 3, about 6 min left" or "taking longer than usual" (Synap restarts a dead build by itself, up to 3 attempts). Exit 0 ready, 7 timeout, 1 failed (reason and the rebuild command printed). Final JSON has `attempt`, `max_attempts`, `elapsed_seconds`, `estimated_remaining_seconds`, `restartable`. Until it reports ready, runtime writes and fetches (including `verify_synap.py`) are rejected with HTTP 409 `instance_not_ready`. |
| Rebuild | `instances rebuild <instance_id>` | Only after a failed build. Re-runs from the stored use-case (nothing to re-approve), starts a new series of 3 attempts, then `instances wait` again. At most 3 per instance per 24 hours: a 429 says so with the server's message; a 409 means the build is not failed. |
| Verify | `python scripts/verify_synap.py` | Not a CLI command. Real write then recall. Waits the retry time the server gives while the build runs (message "The memory architecture is still being built: waiting N s"); exit 2 = still building when time ran out, 3 = the build failed (run `instances rebuild`). |
| Sign out | `auth logout` | The LAST step, only if verify passed. Revokes the setup key on the server and removes it from `~/.config/synap`, so it can no longer be used (only once the command exits 0); the app keeps its own key in `.env`. Send all `onboard report` events before it: afterwards there is no profile to resume from. If the revoke fails it retries 3 times, then exits 9 (`revocation_pending`): the key stays active until it expires (7 days at most) and the local profile is kept marked pending. Rerun it; if still pending, tell the human and list it as an open item. `--json` shows `revoked` and `pending`. |

`skill install [--agent claude-code|cursor|codex|auto]` installs this skill if it is not there yet.

## Things to surface to the human

- **B2C vs B2B is chosen when the instance is created; once created, a change also goes through a request by email to gaurav@maximem.ai.** It changes how you call the SDK:
  - **B2C** (`user_context_isolation = "equals_customer"`): memory is per end-user. Send `user_id` and nothing
    else, on every call, including the turn-by-turn `record_message`. A `customer_id` is rejected with HTTP 400,
    and the Python SDK from 0.4.7 raises before the request leaves the process. Never pass the user id as a
    customer id to fill the field. See `core-concepts.md`.
  - **B2B** (`strict`): memory is per tenant. Each `user_id` lives under a `customer_id`, and `customer_id` is
    required.
  - **Which one am I on?** `GET /api/v1/auth/whoami` returns `user_context_isolation`. Read it, do not guess.
- **The build takes 15 to 20 minutes.** Say so before `instances create`.
- **Roles:** Owner and Admin can create instances and keys; Member is read-only.
- **The instance is resolved from the API key.** You do not pass an instance ID on each call. Set
  `SYNAP_INSTANCE_ID` as an env var if you want to pin it; never as an `instance_id=` constructor argument.
- **One key = one instance.** Use a separate instance (and key) for each agent and for each deployment of the customer's own (dev, test, live). With more than one instance, the variables are named per instance (`sdk-setup.md`, "Several instances").
- **Connection settings.** The SDK talks to Synap's default servers unless you also set `SYNAP_BASE_URL`,
  `SYNAP_GRPC_HOST`, `SYNAP_GRPC_PORT` and `SYNAP_GRPC_USE_TLS` (see `sdk-setup.md`). Set the key and the four
  host variables from the same deployment, always.

## The key

Why `.env` and never printed: anything you print stays in the transcript and its logs, while `.env` stays on this machine and out of git.

`api-keys create --write-env` is the only way you handle the key. Do not copy it into chat, code, logs or a
commit. If `.env` is not git-ignored the CLI adds it to `.gitignore` and says so; tell the human. Load it in the
app with your framework's env loader. Resolution order and auth errors: `sdk-setup.md`.

## Manual fallback (only for developers WITHOUT a coding agent)

A coding agent never sends the human here. This section is for a developer who is setting up by hand, with no
agent running. Everything above can be done at **https://synap.maximem.ai**:

| # | Action in the dashboard | Result |
|---|---|---|
| 1 | Sign up or accept an invite | An account and a workspace (`cli_<hex16>`) |
| 2 | **Create an Instance**: Name, Agent Type, **User Relationship (B2C / B2B)**, and a use-case file ("Download Template" in the form; `.md`, `.markdown` or `.txt`, up to 512 KB) | `inst_<hex16>` and a generated memory architecture |
| 3 | Open the instance, **API Keys**, **Generate API Key**, copy it now (shown once) | A `synap_...` key |
| 4 | `export SYNAP_API_KEY=synap_...` (and `SYNAP_INSTANCE_ID=inst_...`), or put them in a git-ignored `.env` | Ready for the SDK |

Then install the SDK, integrate, and run `python scripts/verify_synap.py`.

---
> Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. The onboarding commands follow `docs/internal/agent_onboarding_contract.md` and were not yet verified against a released CLI.
> Live, changing detail: https://docs.maximem.ai (Mintlify serves a clean `.md` for any page).
