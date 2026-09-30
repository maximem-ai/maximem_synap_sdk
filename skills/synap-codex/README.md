# Synap skill — Codex edition

This is the **Codex wrapper** of the Maximem Synap integration skill. It exists alongside the
Claude Code skill at [`../synap/`](../synap/). The two share all the integration content and
differ only in the thin wrapper.

## Shared content vs. thin wrapper

| | Source | Notes |
|---|---|---|
| `reference/` | **byte-for-byte identical** to `../synap/reference/` | the load-bearing material |
| `scripts/verify_synap.py` | **byte-for-byte identical** to `../synap/scripts/` | the smoke test |
| `examples/` | **byte-for-byte identical** to `../synap/examples/` | runnable samples |
| `SKILL.md` | **wrapper — differs** | Codex manifest: `name` + `description` only (no `allowed-tools`), plus a Sandbox & approvals section |
| `AGENTS.md` | **wrapper — differs** | short repo-level steering that points Codex at the skill |

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
*Accurate as of `maximem-synap` 0.5.1 (Python) · `@maximem/synap-js-sdk` 0.5.1 (JS) — verified 2026-09-25. Source of truth: https://docs.maximem.ai*
