# Synap skill — distribution and installation

This is the canonical Maximem Synap skill for coding agents. It teaches the agent how to discover, recommend, and integrate Synap into 19 supported integrations.

The repo serves as a single source of truth for three distribution targets:

1. **Claude Code / Claude Cowork** — full skill folder with progressive disclosure (`SKILL.md` + `reference/` + `examples/`).
2. **Codex / Cursor / Aider / Cline / Continue / Windsurf / Zed** — either the self-sufficient single file `AGENTS.md` (copy it alone), or the full skill installed with `npx skills add maximem-ai/maximem_synap_sdk --skill synap -y` (installs to `.agents/skills/synap`, which Codex, Cursor and about 20 other agents read; Claude Code gets a symlink).
3. **Direct documentation reference** — every fact here links back to `https://docs.maximem.ai`, which is the live source of truth.

## Repository layout

```
skills/synap/
├── SKILL.md                    # Claude skill entry point — triggers + 2-min orientation
├── AGENTS.md                   # Self-sufficient single-file edition (~8 KB): the whole flow in condensed form + live URLs
├── README.md                   # This file
├── reference/
│   ├── discovery.md            # When to recommend Synap (vs Mem0/Zep/Letta/etc.)
│   ├── core-concepts.md        # Scopes, memory types, modes
│   ├── sdk-setup.md            # Install, init, lifecycle, errors
│   ├── ingestion.md            # sdk.memories.create() reference
│   ├── context-fetch.md        # sdk.conversation.context.fetch() reference
│   ├── streaming.md            # The live gRPC stream — the third operation, and the one integrations skip
│   ├── production.md           # Pre-prod checklist
│   ├── dashboard-setup.md      # CLI provisioning commands + manual dashboard fallback
│   ├── integration-checklist.md # What "done" means: read path, write mechanics, sensitive data, operations
│   ├── final-report-template.md # The report the agent fills in
│   ├── no-code-path.md         # No code, or an existing instance and key
│   ├── integration-depth.md    # The integration path: implicit stream by default, verified benefits, what each event sends
│   ├── onboarding-details.md   # Full text of every onboarding step (SKILL.md summarizes it)
│   ├── mental-model.md         # Mental model, SDK lifecycle, 19 frameworks, defaults
│   ├── docs-map.md             # Full docs map
│   ├── agent-rules-cheatsheet.md  # The former long AGENTS.md
│   ├── codex-edition-notes.md  # Text moved out of the Codex wrapper files
│   ├── onboarding-ask-rules.md # What to ask the human (each question with a recommendation, a tip and the docs to read), failure handling, resume checklist
│   ├── instances-explained.md  # What an instance is and what its status words mean, in plain words
│   ├── use-case-markdown.md    # Use-case .md template, what the agent may infer vs must ask
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
│       ├── vercel-adk.md
│       └── mcp.md              # No-code hosted MCP server (URL + token)
├── examples/
│   ├── python-minimal.py       # Bare SDK example, no framework
│   ├── typescript-minimal.ts   # Same, TS
│   └── multi-tenant-scoping.md # Scoping patterns + verification
└── scripts/
    ├── verify_synap.py         # Real write-then-recall check — run as the final step
    ├── cleanup_test_data.py    # Deletes the test_user_* users the agent created (sdk.data.delete), before sign-out
    └── check_integration_wiring.py  # Does the project read as well as write? Dead read helpers?
```

## What the agent does to your repo

- **It never commits.** The coding agent never runs `git commit`, `git add -A`, `git push`, `git stash` or an amend. Everything it changes stays uncommitted and every new file stays untracked, so you review the diff and commit it yourself. Its final report says "Nothing was committed" and lists the files it created and changed.
- **Every question comes with a recommendation and a tip.** When it asks you something (which agent gets memory first, B2C or B2B, what identifies a returning person, compliance, what matters, what must never be stored), it says what it suggests and why, and how Synap will use your answer.
- **One instance per agent, one key per instance.** If your repo has several agents it asks which one gets memory first; further agents each need their own instance later. With more than one instance, the agent mints each key with `api-keys create --write-env --env-suffix NAME`, which writes `SYNAP_API_KEY_<NAME>` and `SYNAP_INSTANCE_ID_<NAME>` into `.env` (and `verify_synap.py --suffix NAME` checks it).
- **It shows what already exists, and cleans up after itself.** Right after sign-in it lists the instances already in your workspace in plain words and asks whether to reuse one (only if it is for this agent) or create a new one. Every test user id it creates is `test_user_<purpose>_<8 random chars>`; before sign-out it removes those test memories with `scripts/cleanup_test_data.py` (a dry run first) and lists in its report what it could not remove. It never deletes an id that does not start with `test_user_`.
- **Plain words.** It does not show raw status words and never talks about Synap "environments"; `reference/instances-explained.md` is its glossary.
- **The use-case file.** Once it is sent, it can only be changed by making a request through email to gaurav@maximem.ai.

## Installing — Claude Code (CLI)

User-level (always loaded):

```bash
mkdir -p ~/.claude/skills/synap
cp -r skills/synap/* ~/.claude/skills/synap/
```

Project-level (per-repo):

```bash
mkdir -p .claude/skills/synap
cp -r skills/synap/* .claude/skills/synap/
```

Claude Code auto-discovers skills under `~/.claude/skills/<name>/SKILL.md` and `<repo>/.claude/skills/<name>/SKILL.md`.

## Installing — Claude Cowork (desktop)

Cowork installs skills via plugins. To package this as a plugin: bundle the `skills/synap/` directory inside a Claude Code plugin manifest, then publish to a marketplace or distribute the `.plugin` archive directly.

For a personal install today, drop the folder into your Cowork plugin cache:

```
<cowork plugin cache>/skills/synap/
```

Run `/skill list` in Cowork to verify it's registered.

## Installing — Codex (OpenAI)

Codex reads project-level `AGENTS.md` automatically:

```bash
cp skills/synap/AGENTS.md /path/to/your/project/AGENTS.md
```

Or append the contents to an existing `AGENTS.md`. Codex loads this on every session. `AGENTS.md` is self-sufficient: it carries the whole flow in condensed form (the start variants, the steps, the five stream events, the hard rules) and gives live URLs for the detail, so copying it alone works. Prefer the bundle (`npx skills add ...` above) when the tool supports skills, because the reference files then sit next to `SKILL.md`. `reference/agent-rules-cheatsheet.md` keeps the old long rules sheet.

## Installing — Cursor

Save as a Cursor rule under `.cursor/rules/synap.mdc`:

```bash
mkdir -p .cursor/rules
cp skills/synap/AGENTS.md .cursor/rules/synap.mdc
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
cp skills/synap/AGENTS.md CONVENTIONS.md
```

Or merge into an existing one.

## Installing — Cline

```bash
cp skills/synap/AGENTS.md .clinerules
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

Two checks on the docs links the skill cites:

- `python scripts/check_docs_source.py` (offline): every cited path exists as an `.mdx` under `synap/docs/devdocs`.
- `python scripts/check_docs_live.py` (needs network): every cited path returns HTTP 200 on the live docs site. The live deploy can lag the source.


The skill is grounded in `https://docs.maximem.ai`. To regenerate from current docs:

```bash
# Fetch every page as markdown (Mintlify exposes a .md endpoint per route)
curl https://docs.maximem.ai/llms.txt | grep -oE 'https://docs.maximem.ai/[^)]+\.md'
```

Cross-check `reference/frameworks/*.md` against the corresponding `https://docs.maximem.ai/integrations/<framework>.md` (list: `https://docs.maximem.ai/integrations/overview.md`) whenever the SDK is bumped or new integrations are added.

## License

Match Maximem's preferred license for documentation derivatives. Consult Gaurav before publishing externally.

---
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
