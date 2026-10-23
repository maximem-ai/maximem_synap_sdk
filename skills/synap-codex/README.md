# Synap skill — Codex edition

This is the **Codex wrapper** of the Maximem Synap integration skill. It exists alongside the
Claude Code skill at [`../synap/`](../synap/). The two share all the integration content and
differ only in the thin wrapper.

## Shared content vs. thin wrapper

| | Source | Notes |
|---|---|---|
| `reference/` | **byte-for-byte identical** to `../synap/reference/` | the load-bearing material |
| `scripts/` (`verify_synap.py`, `cleanup_test_data.py`, `check_integration_wiring.py`) | **byte-for-byte identical** to `../synap/scripts/` | the smoke test, the test-data cleanup, the wiring check |
| `examples/` | **byte-for-byte identical** to `../synap/examples/` | runnable samples |
| `SKILL.md` | **wrapper — differs** | Codex manifest: `name` + `description` only (no `allowed-tools`), plus a Sandbox & approvals section |
| `AGENTS.md` | **wrapper — differs** | self-sufficient single-file edition (the whole flow in condensed form); copying it alone works |

Do not fork the shared content, and do not edit it here. In the private monorepo
(`maximem-ai/maximem_synap`) this directory holds only the three wrapper files;
`reference/`, `scripts/` and `examples/` are copied in from `../synap/` by
`scripts/sync_to_public.sh` at publish time. Edit them in `../synap/`.

## Codex skill format used

Per OpenAI's Codex docs (https://developers.openai.com/codex/skills, /codex/guides/agents-md):

- A skill is a directory containing `SKILL.md` with YAML frontmatter `name` + `description`
  (no `allowed-tools` field — that is Claude-specific). Codex reads bundled files
  (`reference/`, `scripts/`, `examples/`) on demand.
- Skills are discovered from `~/.agents/skills/<name>/` (user-level, any repo) or
  `.agents/skills/<name>/` at the repo root (repo-level); `AGENTS.md` provides
  repo-level steering and is concatenated root→cwd with closer files winning.

> The Codex skill format is newer and still moving. If discovery doesn't work, confirm the
> current skills directory + manifest schema in the live Codex docs and adjust.

## What the agent does to your repo

- **It never commits.** The coding agent never runs `git commit`, `git add -A`, `git push`, `git stash` or an amend. Everything it changes stays uncommitted and every new file stays untracked, so you review the diff and commit it yourself. Its final report says "Nothing was committed" and lists the files it created and changed.
- **Every question comes with a recommendation and a tip.** When it asks you something (which agent gets memory first, B2C or B2B, what identifies a returning person, compliance, what matters, what must never be stored), it says what it suggests and why, and how Synap will use your answer.
- **One instance per agent, one key per instance.** If your repo has several agents it asks which one gets memory first; further agents each need their own instance later. With more than one instance, the agent mints each key with `api-keys create --write-env --env-suffix NAME`, which writes `SYNAP_API_KEY_<NAME>` and `SYNAP_INSTANCE_ID_<NAME>` into `.env` (and `verify_synap.py --suffix NAME` checks it).
- **It shows what already exists, and cleans up after itself.** Right after sign-in it lists the instances already in your workspace in plain words and asks whether to reuse one (only if it is for this agent) or create a new one. Every test user id it creates is `test_user_<purpose>_<8 random chars>`; before sign-out it removes those test memories with `scripts/cleanup_test_data.py` (a dry run first) and lists in its report what it could not remove. It never deletes an id that does not start with `test_user_`.
- **Plain words.** It does not show raw status words and never talks about Synap "environments"; `reference/instances-explained.md` is its glossary.
- **The use-case file.** Once it is sent, it can only be changed by making a request through email to gaurav@maximem.ai.

## Install

```bash
# Global (available in any repo)
mkdir -p ~/.agents/skills/synap
cp -R ./* ~/.agents/skills/synap/

# Or per-repo: drop AGENTS.md at the repo root so Codex picks it up every session
cp ./AGENTS.md /path/to/your/project/AGENTS.md
```

## Keeping in sync

Nothing to do by hand any more. The monorepo stores one copy of the shared
content, under `public_sdk/skills/synap/`, and `scripts/sync_to_public.sh`
copies it into both skill folders when it publishes them here. Two stored
copies is what let these drift for months, so there is now only one.

To check a published pair:

```bash
diff -rq skills/synap/reference skills/synap-codex/reference   # expect no output
```

---
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. Source of truth: https://docs.maximem.ai*
