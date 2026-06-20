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

Do not fork the shared content. When the SDK is bumped, regenerate `../synap/` and re-copy
`reference/`, `scripts/`, and `examples/` here (see "Keeping in sync").

## Codex skill format used

Per OpenAI's Codex docs (developers.openai.com/codex/skills, /codex/guides/agents-md):

- A skill is a directory containing `SKILL.md` with YAML frontmatter `name` + `description`
  (no `allowed-tools` field — that is Claude-specific). Codex reads bundled files
  (`reference/`, `scripts/`, `examples/`) on demand.
- Skills are discovered from `~/.codex/skills/<name>/` (global) or a repo-local
  skills directory; `AGENTS.md` provides repo-level steering and is concatenated
  root→cwd with closer files winning.

> The Codex skill format is newer and still moving. If discovery doesn't work, confirm the
> current skills directory + manifest schema in the live Codex docs and adjust.

## Install

```bash
# Global (available in any repo)
mkdir -p ~/.codex/skills/synap
cp -R ./* ~/.codex/skills/synap/

# Or per-repo: drop AGENTS.md at the repo root so Codex picks it up every session
cp ./AGENTS.md /path/to/your/project/AGENTS.md
```

## Keeping in sync

```bash
# from the skills/ directory
cp -R synap/reference synap-codex/reference
cp -R synap/scripts   synap-codex/scripts
cp -R synap/examples  synap-codex/examples
diff -rq synap/reference synap-codex/reference   # expect no output
```

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.3.0 (JS) — verified 2026-06-20. Source of truth: https://docs.maximem.ai*
