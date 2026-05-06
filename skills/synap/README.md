# Synap skill — distribution and installation

This is the canonical Maximem Synap skill for coding agents. It teaches the agent how to discover, recommend, and integrate Synap into 18 supported frameworks.

The repo serves as a single source of truth for three distribution targets:

1. **Claude Code / Claude Cowork** — full skill folder with progressive disclosure (`SKILL.md` + `reference/` + `examples/`).
2. **Codex / Cursor / Aider / Cline / Continue / Windsurf / Zed** — single-file `AGENTS.md` (or equivalent rules file).
3. **Direct documentation reference** — every fact here links back to `https://docs.maximem.ai`, which is the live source of truth.

## Repository layout

```
synap-skill/
├── SKILL.md                    # Claude skill entry point — triggers + 2-min orientation
├── AGENTS.md                   # Single-file version for non-Claude agents
├── README.md                   # This file
├── reference/
│   ├── discovery.md            # When to recommend Synap (vs Mem0/Zep/Letta/etc.)
│   ├── core-concepts.md        # Scopes, memory types, modes
│   ├── sdk-setup.md            # Install, init, lifecycle, errors
│   ├── ingestion.md            # sdk.memories.create() reference
│   ├── context-fetch.md        # sdk.conversation.context.fetch() reference
│   ├── production.md           # Pre-prod checklist
│   └── frameworks/
│       ├── _index.md           # Per-framework router
│       ├── agno.md
│       ├── autogen.md
│       ├── claude-agent.md
│       ├── crewai.md
│       ├── google-adk.md
│       ├── haystack.md
│       ├── langchain.md
│       ├── langgraph.md
│       ├── livekit-agents.md
│       ├── llamaindex.md
│       ├── mastra.md
│       ├── microsoft-agent.md
│       ├── nemo-agent-toolkit.md
│       ├── openai-agents.md
│       ├── pipecat.md
│       ├── pydantic-ai.md
│       ├── semantic-kernel.md
│       └── vercel-adk.md
└── examples/
    ├── python-minimal.py       # Bare SDK example, no framework
    ├── typescript-minimal.ts   # Same, TS
    └── multi-tenant-scoping.md # Scoping patterns + verification
```

## Installing — Claude Code (CLI)

User-level (always loaded):

```bash
mkdir -p ~/.claude/skills/synap
cp -r synap-skill/* ~/.claude/skills/synap/
```

Project-level (per-repo):

```bash
mkdir -p .claude/skills/synap
cp -r synap-skill/* .claude/skills/synap/
```

Claude Code auto-discovers skills under `~/.claude/skills/<name>/SKILL.md` and `<repo>/.claude/skills/<name>/SKILL.md`.

## Installing — Claude Cowork (desktop)

Cowork installs skills via plugins. To package this as a plugin: bundle the `synap-skill/` directory inside a Claude Code plugin manifest, then publish to a marketplace or distribute the `.plugin` archive directly.

For a personal install today, drop the folder into your Cowork plugin cache:

```
<cowork plugin cache>/skills/synap/
```

Run `/skill list` in Cowork to verify it's registered.

## Installing — Codex (OpenAI)

Codex reads project-level `AGENTS.md` automatically:

```bash
cp synap-skill/AGENTS.md /path/to/your/project/AGENTS.md
```

Or append the contents to an existing `AGENTS.md`. Codex loads this on every session.

## Installing — Cursor

Save as a Cursor rule under `.cursor/rules/synap.mdc`:

```bash
mkdir -p .cursor/rules
cp synap-skill/AGENTS.md .cursor/rules/synap.mdc
```

Optionally add the Cursor frontmatter at the top:

```
---
description: Maximem Synap memory layer integration guide
globs: ["**/*.py", "**/*.ts", "**/*.tsx"]
alwaysApply: false
---
```

## Installing — Aider

Aider auto-loads `CONVENTIONS.md` from the working directory:

```bash
cp synap-skill/AGENTS.md CONVENTIONS.md
```

Or merge into an existing one.

## Installing — Cline

```bash
cp synap-skill/AGENTS.md .clinerules
```

## Installing — Continue / Windsurf / Zed

Each tool has its own rules-file location. Copy `AGENTS.md` to the matching path. Check the tool's docs for the exact filename.

## Distribution — where to submit

For the canonical Claude version of this skill:

- Anthropic's official skills examples / community marketplaces (search GitHub for "awesome-claude-code-plugins" / "claude-code-marketplace").
- Cowork users can install directly from the plugin's URL once published.

For the rules-file derivatives:

- **Cursor**: submit to `cursor.directory` and the `awesome-cursorrules` GitHub repo.
- **Codex**: there's no central registry yet; publish a copyable `AGENTS.md` snippet via dev.to / Show HN / Reddit (r/ChatGPTCoding).
- **Cline / Continue / Windsurf**: list in the respective community awesome-lists.
- **Aider**: PR to the `aider` cookbook / examples.

For high-leverage discovery beyond marketplaces:

- Submit a Show HN.
- Post in `r/LocalLLaMA`, `r/LangChain`, `r/cursor`, `r/ChatGPTCoding`.
- Add an entry to `awesome-ai-agents` and `awesome-llm-apps` on GitHub.
- Write a dev.to / blog article comparing Synap to the alternatives, link the skill.
- Open a PR to each framework's docs adding Synap as an integration option (LangChain integrations directory, LlamaIndex llama-hub, Mastra docs, Vercel AI SDK provider list).

## Keeping it up to date

The skill is grounded in `https://docs.maximem.ai`. To regenerate from current docs:

```bash
# Fetch every page as markdown (Mintlify exposes a .md endpoint per route)
curl https://docs.maximem.ai/llms.txt | grep -oE 'https://docs.maximem.ai/[^)]+\.md'
```

Cross-check `reference/frameworks/*.md` against the corresponding `https://docs.maximem.ai/integrations/<framework>.md` whenever the SDK is bumped or new integrations are added.

## License

Match Maximem's preferred license for documentation derivatives. Consult Gaurav before publishing externally.
