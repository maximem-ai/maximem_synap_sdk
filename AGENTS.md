# AGENTS.md

## What this repo is
The published mirror of the Maximem Synap Python and JavaScript SDKs, 24 framework integrations, and the MCP server adapter. The Synap memory engine is a hosted service; these packages are clients for it.

## Important for contributors and agents
This repository is overwritten by an automated sync from Maximem's private monorepo. Pull requests that change files under `packages/` are closed, because the next sync would revert them. Open an issue instead: https://github.com/maximem-ai/maximem_synap_sdk/issues

## Build and test
Python SDK (Python 3.11+):

```bash
pip install -e "packages/sdks/maximem-synap[dev]" pytest-asyncio
pytest packages/sdks/maximem-synap/tests
```

JS parity suites need monorepo-only fixtures and are skipped in this mirror.

## Layout
- packages/sdks/          core Python and JavaScript SDKs
- packages/integrations/  one installable package per agent framework
- packages/mcps/          MCP server adapter (stateless, over the hosted API)
- skills/synap/           agent skill for Claude Code
- skills/synap-codex/     agent skill for Codex
- packages/sdks/maximem-synap/tests/  Python SDK tests

## Adding Synap to a user's project
Follow skills/synap/SKILL.md. It covers setup, scoping (client, customer, user, conversation), ingestion, retrieval, and one wiring guide per framework.
