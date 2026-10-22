# Maximem Synap: agent rules

The single source of truth for this folder is `SKILL.md`. Read it and follow its procedure; this file is only a short index for agents that load `AGENTS.md` instead of a skill. Never guess: if anything is unclear, stop and ask the human. Full text of every step: `reference/onboarding-details.md`. The longer rules and cheat sheet that used to live in this file: `reference/agent-rules-cheatsheet.md`.

The steps (`<cli>` is `maximem-synap` 0.3.0+ from PATH, else `npx @maximem/synap-cli@latest`; add `--env <name>` when the prompt names an environment):

1. Sign in: `<cli> auth login --code <syn_ob_...> --agent-name <name>`.
2. `<cli> onboard status`; resume from the first unfinished step.
3. Step 0: read the core docs in full (quickstart, memory-model-cheat-sheet, memory-scopes, how-ingestion-works, retrieval-modes, real-time-anticipation) and tell the human four lines on what you understood.
4. Read the repo; infer only what the code supports.
5. Ask the human one question at a time: B2C or B2B, the stable user id, compliance and sensitive data (/guides/pii-protection), what to remember, what never to store.
6. Draft the use-case from `instances template`, gaps as `not stated`; show the exact file, quote its sha256, get an explicit yes.
7. Create the instance after saying it usually takes 15 to 20 minutes and getting a yes; do not wait.
8. Integrate with the docs checkpoint and print a "Docs read" list (production-checklist, graceful-degradation, /sdk-reference/errors, each method page).
9. `api-keys create -i <id> --write-env`; run the project's tests with `env -u SYNAP_API_KEY` until verify passes.
10. `instances wait` in the background; no memory call before ready (handle `InstanceNotReadyError` and the older `ConflictError`).
11. `python scripts/verify_synap.py > verify.out 2>&1`; paste it unedited; it ends with `RESULT:` and `EXIT CODE:`.
12. Announce the result, then `<cli> auth logout` last, only if verify passed.

Hard rules: never guess; `not stated` for gaps; confirm before create; never print or commit the key; never send the human to the dashboard. The live docs at https://docs.maximem.ai win over any file here.
