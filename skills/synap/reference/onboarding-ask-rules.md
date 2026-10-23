# Onboarding: what to ask, how, and what to do when it goes wrong

> **Read this next:** [Memory scopes (B2C vs B2B)](https://docs.maximem.ai/concepts/memory-scopes.md); [Memory Architecture and the use-case file](https://docs.maximem.ai/concepts/memory-architecture.md); [CLI overview](https://docs.maximem.ai/cli/overview.md); [Error handling](https://docs.maximem.ai/sdk/error-handling.md). The live docs win over this file.

Read this before step 4 of the procedure in `SKILL.md`. Step 0 (learn Synap) comes first. Commands are in `dashboard-setup.md`.

## The rule

Ask the human whenever you cannot know the answer from the code. Never guess. A wrong guess costs a new
instance and 15 to 20 minutes. Once the use-case file is sent, **it can only be changed by making a request
through email to gaurav@maximem.ai**; B2C or B2B is chosen when the instance is created, and a change also goes
through that email. Say this when you ask for approval of the file.

## How to ask

- **One question at a time.** Wait for the answer before the next.
- **Show the evidence first, then ask anyway.** For example: "I found a `tenants` table and an `org_id` on every
  user row, which looks like B2B. Is that right?" Evidence is not an answer.
- **Every question carries two extra parts, written by you for this repo:**
  1. **My recommendation:** what you suggest and why, from what the repo shows ("I suggest B2B, because every
     user row has an `org_id`"). If the repo shows nothing, say that and recommend the safest choice for the
     goal; never leave the human with a bare question.
  2. **Tip:** one or two plain sentences on how Synap uses this answer and what it changes in the result. Write
     it ONLY after you have READ the docs page(s) listed for that question in the table below, in this
     session. Use what the page says; if the page does not say it, leave it out. Never write a tip from memory.
- Plain words, no Synap jargon (see `instances-explained.md`). Offer 2 to 3 concrete options when the answer is a
  choice.
- Every question is asked here, in this session. Also run `onboard ask "<the question>"` so an open dashboard
  can show it, but the human must never need the dashboard to answer or to find out what is happening.
  When answered, run `onboard ask --clear`.
- Repeat the answer back in the draft, in the human's words.

## The questions (all required, in this order, all BEFORE you draft the use-case)

Question 0 is asked only when the repo has more than one agent. The docs to read are `https://docs.maximem.ai`
pages (add `.md`); read them before writing the tip. What each tip must be based on is in the last column.

| # | Question | Recommend from | Docs to read first | The tip must say (from those pages) |
|---|---|---|---|---|
| 0 | **Which agent gets memory now?** Only if the repo has more than one agent (several agent classes, services, personas, bots). | the agent with the most returning users or the most repeat context; name the file and class. | /concepts/agent-topologies, /concepts/memory-scopes ("What is an Instance?") | One instance has one memory architecture, built from one use-case file, so it is built for one agent's job. Agents that share the same users and behave the same way can share one instance; agents with different jobs, extraction needs or compliance rules need their own instance each, and per-user memory does not cross instances. Say: further agents each need their own instance later, with their own key (`sdk-setup.md`, "Several instances"). |
| 1 | **Who uses the agent?** B2C (each end user has their own private memory) or B2B (users belong to a customer organization; memory is per organization and per person). If the repo shows both patterns, or neither, say so. | the user model, tenant or org tables, auth. | /concepts/memory-scopes ("B2C vs B2B"), /getting-started/use-case | It sets how memory is scoped: B2C uses `user_id` only (a `customer_id` is refused); B2B needs `customer_id` and `user_id` on every call. It is chosen when the instance is created and a change goes through the email in "The rule". |
| 2 | **What identifies a returning person in this app (a stable user id)?** | the login id or account id the code has; flag device ids and session ids as weak. | /concepts/memory-scopes, /guides/multi-user-scoping, /guides/erasure | Memories are tied to the `user_id` you send (and `customer_id` on B2B); the same id on a later visit gets the same memory. With only a session id, each visit is a new person: no returning memory, no profile, no per-person erasure. Erasure of one person is requested by email to privacy@maximem.ai with the `user_id`. |
| 3 | **Compliance and sensitive data.** Does the agent see health, financial, children's or other regulated data? Any rules (GDPR, HIPAA, SOC 2, data residency)? | the fields and domain the code handles. | /guides/pii-protection, /guides/erasure, /concepts/memory-architecture | What Synap does differently: it finds sensitive values, replaces them with placeholders at rest and gives your app the real value back; seven kinds (card numbers, security codes, PINs, passwords, secret keys, private keys, raw biometrics) are never stored; the policy is approved by the human in the dashboard and applies going forward only; names and street addresses are not detected. Plus the sensitive-data rule below. |
| 4 | **What matters most to remember?** Top two or three kinds of facts, preferences or events that make the agent better next time. | what the prompts and features of the agent already use. | /guides/configuring-memory, /concepts/memory-architecture | These words go into the use-case file, and Synap generates the memory design from it: which kinds of memory it extracts, how retrieval ranks, how long things live. The clearer the priorities, the better the design. |
| 5 | **What must never be stored?** Secrets, payment details, health details, anything else. | the fields the code already redacts or the data it passes through. | /guides/pii-protection, /guides/configuring-memory | It goes in the use-case file as the never-store list, and your answer shapes what the memory design avoids. Whether a value is actually removed depends on the approved sensitive-data policy and on the app's own redaction; the seven always-blocked kinds need no setting. |
| 6 | **Live stream (the confirmation below).** | long-lived process or not. | /concepts/real-time-anticipation, /concepts/grpc-vs-rest | What the stream gives and why it needs a long-lived process (see "The integration-path confirmation"). |
| 7 | **Approve the exact use-case file, then create.** | the file you drafted. | /getting-started/use-case, /concepts/memory-architecture | The file is stored on the instance and the memory design is built from it in about 15 to 20 minutes; once sent it can only be changed by the email in "The rule". |

**Worked example** (question 1, a support bot whose database has `organizations` and `users.org_id`):

> "Who uses this agent: individual people with private memory (B2C), or people who belong to a customer
> organization (B2B)? I found an `organizations` table and `users.org_id` on every user, which points to B2B.
> **My recommendation:** B2B, because each user row belongs to an organization, so memory can be kept per
> organization and per person. **Tip:** in Synap this decides how memory is scoped: on B2B every call carries
> both the organization id and the person id; on B2C only the person id, and an organization id is refused. It
> is chosen when the instance is created; changing it later goes through a request by email to
> gaurav@maximem.ai."

Both labels (`My recommendation`, `Tip`) appear in every question, including yes/no confirmations and the
approval of the file.

**Details for question 2.** The answer maps to the SDK scope:
   - **B2C:** that id is the `user_id`, and a `customer_id` is never sent.
   - **B2B:** the organization's id is the `customer_id` and the person's id is the `user_id`; both are required.
   - **Shared devices:** a device id is shared when two people use one phone or computer; the second person would be shown the first person's memory. Use a device id only with a visible opt-in such as "continue my last session?", or use a login.
   - **No stable id (a session id as the user id):** use a per-chat or per-session id only if the human confirms the app has no persistent identity. Say plainly what is lost: no returning-person memory, no profile, no per-person erasure, and Synap will return nothing the app does not already have.
   - **Goal vs identity:** with a per-session id each visit is a new person. If the approved goal is "remember returning users" (or anything across sessions), say so
     plainly BEFORE approval, offer the options (add a stable id, an anonymous device id, or accept per-session
     memory and change the goal text), and ask again. Do not draft a use-case the integration cannot serve.

### Sensitive-data rule (general)

When the human mentions health, financial, children's or other personal data, or answers "store everything" for
such a domain:

1. Read `https://docs.maximem.ai/guides/pii-protection.md` before approval (and `/dashboard/pii-and-data-controls`
   if you need the controls' names).
2. Tell the human the PII options in plain words: what can be hidden or removed before storage, what Synap can
   hold, and what is left to their own code. Use only what that page says.
3. Require an explicit yes to the "store everything" choice after you have said what it means ("passwords,
   card numbers and clinical details typed into the chat would be remembered").
4. Write the human's answer verbatim in the use-case Compliance and Data Sensitivity section, in their words.
5. If the app redacts before sending, test that redaction against the human's never-store list with realistic samples in the local formats (phone numbers with spaces, landlines, dates of birth, postal codes, ids), add unit tests, and tell the human which fields pass through (Synap does not detect names or street addresses). Mention custom field types for local ids. Do not leave the policy "as a decision" silently.
6. The instance's sensitive-data policy is approved by the human in the dashboard and applies going forward only. If the use-case has a never-store list and the policy is not approved, it is a named open item in the final report.
7. Ask whether the app's consent screen tells the person that a third party stores their text, and flag it to the human if not. When user ids are per-person, ask how a deletion request will be served; when they are per-session, say it cannot be.

## The integration-path confirmation

Asked after the questions above and before you draft the use-case. Like every question it carries your recommendation and a tip (the wording below already does). Background, verified benefits and sources: `integration-depth.md`. This is a **confirmation with default yes**, not a menu:

- **Always use the SDK, never raw HTTP calls.** Both paths are SDK paths.
- **Lead with the live stream (the implicit path) as the default** for an app that can run a long-lived process (server, worker, container, voice agent, desktop or CLI agent). Never call it optional and never list it as one of three equal options.
- **Explain first, in plain words with no class or method names, what it gives** (only what the docs back): context prepared before the agent asks, so a read is often answered locally in about a millisecond when Synap prepared it; conversations recorded as memory without write calls; the agent's reasoning and tool use sharpening what is prepared; delivery that survives dropped connections.
- **Be truthful about the gain for this app** (a short chat with no tools gains less than a tool-using agent), and form the recommendation from the repo before asking. Reuse the modality and turn speed already gathered for the use-case Interaction Pattern; never ask twice.
- **Then confirm:** "I will wire the live stream so you get X, Y, Z. Tell me if you do not want it."
- **Implement it fully unless the human EXPLICITLY says no.** "I do not know", silence and "whatever you think" all mean implement it.
- **If they say no:** record the choice and the one-sentence consequence (no prepared context, no reasoning or tool signals, their code makes each call), build the explicit path, and put it in the final report.

Wording:

> "I will wire Synap's live stream so you get context prepared before your agent asks, conversations recorded as memory without write calls, and your agent's reasoning and tool use feeding better predictions. I am recommending it because this is a long-running chat server that handles many turns and calls tools. Tell me if you do not want it."

Examples (adapt the evidence, keep the shape):

- **Chat backend (long-running server):** confirm the stream. "Your server stays up across many turns and calls tools, so Synap can prepare the next answer between turns."
- **Voice agent:** confirm the stream. "Turns land seconds apart and latency matters; prepared memory is the point, and the call ending is a clean place to close the session."
- **Desktop or CLI agent:** confirm the stream if the process stays open for the session; if each run is a one-shot command it cannot hold a stream, so use the serverless wording.
- **Serverless function, Edge or per-request worker (Vercel, Lambda, short-lived):** do not present the stream as the default and do not pitch re-architecting. Say in one plain sentence: "Synap's live stream needs a process that stays running, and this host does not run one, so it is not available here; I will use the SDK's explicit path (record each turn and fetch memory), with the production rules for a short-lived function. If you later run a long-lived process, the live stream becomes available." Cite `https://docs.maximem.ai/concepts/grpc-vs-rest.md` as the reason.
- **"I do not know":** implement it. Give one sentence on what it does and say you will wire it.

Report the choice with `onboard report integrating in_progress --detail "path=implicit|explicit-no|explicit-host"` and in the final report ("stream ON by default", "human said no because ...", or "host cannot run it").

Anything else you cannot resolve from code (agent purpose, who the end users are, which framework when the repo
has two) is also a question, with its own recommendation and tip. Do not ask what the code answers (language, framework, package manager).

## On ambiguity

- Two plausible answers: ask which one. Do not pick the safer-looking one.
- The human says "I don't know" or "skip": write `not stated` in that field, tell them what that costs
  ("memory will not know to avoid X"), and ask once more whether they want to answer before you create.
- The human changes an answer after approval but before create: redraft, show it, get approval again, and
  report `use_case_drafted` again.
- After the use-case file is sent, it can only be changed by making a request through email to
  gaurav@maximem.ai (B2C or B2B is chosen at creation, and a change also goes through that email). Do not create a
  second instance to get around it without the human's explicit yes (another 15 to 20 minute build).

## Reporting progress

`onboard report <step> <status> [--detail TEXT]`

| When | Command |
|---|---|
| Draft shown in chat (report AFTER showing it) | `onboard report use_case_drafted done` |
| Human said yes to the exact file you will upload (shown in full, hash quoted) | `onboard report use_case_approved done` |
| Create finished, you start coding | `onboard report integrating in_progress` |
| Integration written | `onboard report integrating done` |
| Key is in `.env` | `onboard report env_written done` |
| Something failed | `onboard report <step> failed --detail "<one line reason>"` |

Send every report before `auth logout`: after logout there is no signed-in profile, so nothing can be reported or resumed from this machine. Report only what is true. An open dashboard shows build status, key and first write and recall from the server,
so reporting cannot make those look done, and you must not try. Never put keys, codes or personal data in
`--detail`.

## Announce in the session

The human should never have to open the dashboard to know what happened. Say, in the chat, at each point: the
use-case you will upload, "instance created: <name> (<id>), the build takes 15 to 20 minutes, I am integrating
meanwhile", "key saved to `.env` (masked: synap_...abcd)", and at the end the result: what was created, the
**verbatim** `verify_synap.py` output (the write and the recalled text), and every file you changed. You may add
one line: "Optional: view it in your dashboard at https://synap.maximem.ai". Never make it a step.

The last thing you do, only if `verify_synap.py` passed, is `auth logout` (never if verify failed: you would need the key to rebuild or resume). Then, ONLY if `auth logout` exited 0 and reported the key revoked, end the announcement with these words: "I signed the setup key out. It only existed to set things up and was stored in ~/.config/synap on this machine; it is revoked, so it can no longer be used. Your app keeps working with its own key in .env. If you need the CLI again, run `maximem-synap auth login` or ask for a new prompt."

## Failure handling

Every failure has a recovery inside this session. Do not tell the human to open the dashboard or to copy
anything from it. The one exception is the first sign-in.

| Failure | What to do |
|---|---|
| CLI calls return 401 or 403 although you just signed in | `SYNAP_API_KEY` (a runtime key from `.env`) is set in your shell and overrides the CLI's sign-in. Re-run with `env -u SYNAP_API_KEY ...` or `unset SYNAP_API_KEY`. |
| Code rejected as invalid at the first sign-in, and the prompt names an environment | (Never say "environment" to the human.) The `--env <name>` flag (or `SYNAP_ENV`) was most likely left out, so the code went to the wrong server. Retry **once** with it before telling the human to copy a fresh prompt. |
| Code missing, expired or already used at the **first** sign-in (`auth login --code` exits 3) | Nothing has started yet, so this is the one place to say: "this code is missing, expired or used; please copy a fresh prompt from the dashboard." Give the single reason if the CLI names it. Do not use the browser login unless the human asks. |
| Key expired or rejected later (exit 3 on another command) | The onboarding key lasts exactly 7 days and cannot be renewed (`auth refresh` is refused for it: do not try it). Run `onboard status` and `auth whoami` to see the reason. If the profile on disk is gone or the key expired, state that one reason and ask the human for a fresh prompt (new one-time code; this is the same exception as the first sign-in). |
| Key in `.env` lost, rejected, or the human deleted it | Run `api-keys create -i <instance_id> --write-env` for a new key (with several instances in the project add `--env-suffix NAME`, see `sdk-setup.md`, "Several instances"). To retire the old one, `api-keys list -i <instance_id>` then `api-keys revoke <credential_id>` (ask first). `api-keys rotate <credential_id>` also works, but it has no `--write-env`, so use it only if you can store the key without printing it. |
| Not enough credits (a runtime memory call fails with a credits error carrying `balance_credits` and `minimum_required_credits`) | Quote the numbers from the error in the chat and do not invent figures. Say what each option costs: reduce what you ask for, wait, or the human adds credits by whatever route they prefer. Ask which. Do not send them away without saying why, do not retry in a loop, and resume from `onboard status` once they answer. |
| Neither `maximem-synap` (0.3.2+) on PATH nor `npx`, or the CLI will not start | The `npx` package is only an installer: it puts the Python CLI on the PATH, so the host needs **both Node (for `npx`) and Python 3.10+**. Say exactly which is missing and ask the human to install it. `pipx install maximem-synap-cli` is not available yet (not on PyPI), so do not suggest it. |
| Python missing for `verify_synap.py` | The SDK needs Python 3.11+. Say so. If the project is TypeScript only, write and run an equivalent write-then-recall in the project's language and quote its output. If you cannot run it, give the human the exact command and say plainly that verification has not happened. Never skip it silently. |
| HTTP 409 `instance_name_taken` from `instances create` | Say plainly: "an instance with that name already exists in your workspace". Ask ONE question, with your recommendation and tip: reuse that instance (only if it is for this agent; one instance serves one agent) or choose another name. Pick nothing yourself. Handled only if returned; the server is about to refuse duplicate names. |
| Exit 5 (scope) or 6 (conflict, such as a name already used) | Report the CLI's message. For a name conflict, pick nothing yourself: ask for another name. Do not retry with a different flag to get around it. |
| Exit 7 (`instances wait` timeout) | The build may still be running (it prints "attempt n of 3, about N min left" or "taking longer than usual"). Run `instances wait` again once. If still not ready, tell the human how long it has been, report `onboard report integrating in_progress --detail "build still running"`, and ask whether to keep waiting. Why: a slow build that is still alive is never restarted, so waiting is the right move. |
| Build failed (`instances wait` exit 1, or `maca_failed`) | Synap already retried it by itself (attempt 3 of 3), so it is not a blip. Quote the reason verbatim, tell the human in one line, then run `instances rebuild <instance_id>` (re-runs from the stored use-case, nothing to re-approve, at most 3 rebuilds per 24 hours) and `instances wait` again. Why you act without asking: nothing new is created and nothing the human approved changes. Ask first only if the rebuild is refused (429 = the daily limit, with the server's message; 409 = the build is not in a failed state, so just wait) or a rebuild fails too. Do not create a second instance for this. |
| HTTP 409 `instance_build_failed` from any call, or from `instances ...` (exit 6, details carry `attempts` and `hint`) | Same as a failed build: the build gave up, waiting will not help. Run the hint, `instances rebuild <instance_id>`, then `instances wait`. Do not retry the failed call in a loop. |
| HTTP 409 `instance_not_ready` from any runtime call (SDK: `InstanceNotReadyError`; an older SDK raises a plain `ConflictError` saying the memory architecture is still being built) | The memory architecture is still building. Not a bug. `retry_after_seconds` is the wait Synap asks for (30 to 300 s): WAIT exactly that long, tell the human how long ("still being built, waiting 2 minutes"), then check once with `instances wait <id>`. Never retry in a loop: the SDK refuses locally during the wait (`from_cooldown` true, no network call), so a loop changes nothing. Tell the human their own app will see the same error until the build finishes. |
| `verify_synap.py` exits non-zero | Report it as a failure and quote the whole output. Read the last `[ERROR]` line: key not resolving an instance (re-run `api-keys create -i <id> --write-env`), build not finished (`instances wait`), recall not yet visible (re-run once after a few minutes). Never say "set up" or "done". |
| `auth logout` fails (403, network) or exits 9 (`revocation_pending`) | Say it plainly with the error text. Never say the key is signed out unless `auth logout` exited 0 and reported it revoked. On exit 9 retry `auth logout` once or twice; if it is still pending, tell the human the key stays ACTIVE until its expiry (7 days at most; scoped to read and create instances and keys) and that they can revoke it on the dashboard API keys page, and list it in the final report as an OPEN ITEM, not as done. Do not loop and do not run other CLI commands. |
| Network or sandbox blocks the CLI, install, or gRPC | List exactly which commands the human must run themselves, and what you will do once they have. |

## Resume checklist (re-running the prompt is safe, in a new session too)

A **new agent session** needs no dashboard visit. The sign-in profile is stored on disk by the CLI, so start with
`onboard status` and `auth whoami`. `onboard status` describes the whole workspace: trust only the steps you reported yourself in this
session, treat another session's steps as someone else's work, and never skip drafting, approval or creation because they look done
(if they look done, ask the human whether that instance is theirs and for this agent: that is the reuse path in `no-code-path.md`). Within 7 days of the first sign-in a new session resumes from the profile. Only if the profile is
gone **or** the key has expired (7 days fixed, no renewal): tell the human the single reason ("your sign-in on this machine is gone or
the 7-day key has expired") and ask them for a fresh prompt. The code in that prompt starts a new sign-in;
everything already created (instance, `.env`, integration code) is kept and reused.

1. Signed in? (`auth whoami` works.) If not, see the paragraph above.
2. Use-case approved? Only your own report in this session, or the human telling you so for an instance they chose to reuse, counts. Otherwise redo steps
   3 to 5 of the procedure.
3. Instance exists? `instances list --json` (step 2b). **Never create a second instance** if one already matches the project
   name or the human says it is theirs. Ask which one to use; a duplicate name is refused with 409 `instance_name_taken`.
4. Integration code present in the repo? Check `git status` and grep for `maximem_synap` or
   `@maximem/synap-js-sdk`. Continue from what is there; do not duplicate it.
5. Key in `.env`? Check the file has `SYNAP_API_KEY=` (or this instance's `SYNAP_API_KEY_<NAME>=`, written by `--env-suffix NAME`; do not print the value). If not, run
   `api-keys create -i <id> --write-env`. If it is there, do not mint another.
6. Build ready? `instances wait <id>` (a failed build: `instances rebuild <id>`, then wait again). Run no ingestion or retrieval, and not `verify_synap.py`, before it reports ready.
7. Verified? Run `scripts/verify_synap.py` again; it writes one more throwaway memory, which is fine.

If `auth whoami` shows you are signed out although setup was unfinished, the earlier session ran `auth logout` or the 7 days passed: ask the human for a fresh prompt. If setup had finished and verified, there is nothing to resume.

Tell the human where you resumed from, in one line.
