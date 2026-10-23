# Start variants: no one-time code, or an existing instance and key

> **Read this next:** [CLI overview](https://docs.maximem.ai/cli/overview.md); [Authentication](https://docs.maximem.ai/setup/authentication.md); [SDK initialization](https://docs.maximem.ai/sdk/initialization.md); [Memory scopes](https://docs.maximem.ai/concepts/memory-scopes.md). The live docs win over this file.

The procedure in `SKILL.md` assumes the prompt came with a one-time code. Not everyone starts there. At the start, **ask which applies** (one question, plain words), then follow the matching branch. Never ask the human to paste a key into chat, in any branch.

## A. The prompt has a one-time code (`syn_ob_...`)

Follow `SKILL.md` from step 1.

## B. No code, new instance wanted

Two ways in; let the human choose:

1. **Get a code.** Ask them to copy a fresh prompt from the dashboard (this is the one place that is allowed to send them there), then follow branch A.
2. **Sign in once from the terminal.** Run `<cli> auth login` (add `--env <name>` when the prompt names one; never say "environment" to the human). It prints a short code and a link; the human approves in the browser, once. Then continue from step 2 of `SKILL.md`. Everything after sign-in is the same, so the human never needs the dashboard again. Say clearly that this opens their browser once.

## C. Already has an instance and a runtime key

Common when a developer created the instance by hand or is adding Synap to a second repo. Skip creation; keep everything else.

**Skip:** the one-time code and sign-in (step 1), drafting and approving a use-case (step 5), `instances create` (step 6), minting a key (step 8), `instances wait` and the sign-out (steps 9 and 11) unless a CLI session already exists and the instance may still be building.

**Do:**
1. **Find the key without asking for it.** Look for `SYNAP_API_KEY` (or this agent's `SYNAP_API_KEY_<NAME>`, the name `--env-suffix NAME` gave it) in the environment and in the project's `.env` (check it is set; never print it). If it is missing, name the variable and the file and ask the human to add it to `.env` themselves, then continue. Never ask them to paste the key into the chat, and never write a key you were shown in chat into a file.
2. **Check the instance.** Initialize the SDK and read the mode from whoami: `user_context_isolation` is `equals_customer` for B2C (send `user_id` only) and `strict` for B2B (send `customer_id` and `user_id`). If a CLI session exists, `<cli> instances list --json` shows the instance and its build state. If the instance is still building, `verify_synap.py` waits (bounded) and exits 2 with `[NOT READY]`; tell the human and retry later.
3. **Step 0** (read the docs), then **step 4's identity question** (what identifies a returning person): the existing instance already fixes B2C or B2B, so do not ask that one again, but do ask the others that shape the code (identity, sensitive data and the never-store list, what to remember).
4. **The integration path** exactly as in `SKILL.md` (the live stream by default where the host can run it, confirmed with the human), then **step 7**: wire the read path into the prompt, the write mechanics, redaction, the OFF-safe startup log, tests, and the docs checkpoint.
5. **Verify** (step 9's verify part): `python scripts/verify_synap.py > verify.out 2>&1; echo "exit=$?"` with the key loaded from `.env`, plus `scripts/check_integration_wiring.py` and the app's own path with a sentinel.
6. **Announce** with the filled `final-report-template.md`. Mark the skipped lines "existing instance and key, not created by me" and add: the use-case of the existing instance was not reviewed by you; ask the human whether it matches what the app does, because once a file is sent, it can only be changed by making a request through email to gaurav@maximem.ai. If the project has several Synap instances, each key lives under `SYNAP_API_KEY_<NAME>` (written by `api-keys create --env-suffix NAME`; `sdk-setup.md`, "Several instances"); read the one for the agent you are wiring and verify it with `verify_synap.py --suffix NAME`. Before the announcement run the test-data cleanup (`onboarding-details.md` step 10) for the ids you used.

## E. Reusing an instance the workspace already has (from the early instance list)

Right after sign-in `SKILL.md` step 2b lists the workspace's instances in plain words and asks one question: reuse or create new. If the human chooses to reuse, this is branch C with a CLI session, so you can mint the key yourself.

1. **Is it really for this agent?** One instance serves one agent (docs: /concepts/agent-topologies). Recommend reuse only when the human says this instance is for THIS agent. If it is for another agent, do not reuse it; offer a new instance.
2. **Skip:** drafting and approving a use-case (step 5), `instances create` (step 6) and, if the instance is ready, the build wait. If it is still being built, run `instances wait <id>` in the background before verify.
3. **Key:** if `.env` has no key for it, mint one with `api-keys create -i <id> --write-env` (add `--env-suffix NAME` when the project has or will have several instances). Never mint a second key if one is already in `.env`.
4. **Still do:** Step 0, the identity question and the other questions that shape the code (B2C or B2B is already fixed by the instance: read it from whoami), the depth and path choice, the read path, the docs checkpoint, verify, `check_integration_wiring.py`, the test-data cleanup and the final report.
5. **Say in the report:** the use-case of the reused instance was not drafted by you; ask the human whether it matches what the app does, because once a file is sent it can only be changed by making a request through email to gaurav@maximem.ai.
6. **Sessions are not shared.** An instance that `onboard status` shows as created or approved by another session is still someone else's work until the human tells you it is theirs and for this agent.

## D. Cannot tell

If the human is unsure whether they have a code or a key, look: a `syn_ob_...` code in the prompt means A; a `synap_...` key in the environment or `.env` and an instance in `whoami` means C; neither means B. Say which you found and confirm before acting.
