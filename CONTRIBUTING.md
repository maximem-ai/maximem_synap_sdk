<!--
  This CONTRIBUTING guide is the SINGLE SOURCE OF TRUTH, maintained in the private
  monorepo at public_sdk/CONTRIBUTING.md and synced to the public repo root by
  scripts/sync_to_public.sh.
  DO NOT edit the copy in maximem-ai/maximem_synap_sdk directly: changes there are
  overwritten on the next sync. Edit here instead.
  Every packages/... path below is validated against the real layout by
  scripts/gen_readme_integrations.py --check, which the sync runs before copying.
-->

# Contributing to the Synap SDKs and integrations

Thanks for your interest. Please read the next section before writing any code, because how this repo accepts changes is probably not what you expect.

## How this repo works

This repository is a **published mirror**, not the development tree.

The Python SDK, the JavaScript SDK, and every framework integration are developed in Maximem's private monorepo and copied here by an automated sync. Everything under `packages/`, plus `README.md` and this file, is overwritten on each sync.

That has one practical consequence:

> **Pull requests against `packages/` will be closed.** Not because the change is unwelcome, but because merging it here would be silently reverted the next time the sync runs.

So instead:

| You want to | Do this |
|---|---|
| Report a bug, or a gap in the SDK/an integration | [Open an issue](https://github.com/maximem-ai/maximem_synap_sdk/issues) |
| Request support for a new framework | [Open an issue](https://github.com/maximem-ai/maximem_synap_sdk/issues) first, see [Proposing a new integration](#proposing-a-new-framework-integration) |
| Read, run, or debug the code against your own app | [Working with the code locally](#working-with-the-code-locally) below |
| Ask how something works | [docs.maximem.ai](https://docs.maximem.ai) |

Accepted issues are implemented in the monorepo. The fix appears here on the next sync, and on PyPI/npm at the next release.

**There is no CI in this repo.** No workflows run on pushes or PRs here. The test suites live in the monorepo and run there. If you are working locally, run the tests yourself as described below.

## Reporting an issue

A good issue includes:

- Which package is affected (for example `maximem-synap-langchain`)
- Installed versions: the SDK, the integration, the framework, and Python or Node
- What you expected to happen, and what actually happened
- A minimal reproduction, with the SDK call and arguments you used
- The full traceback if there is one

Please do not paste API keys, instance IDs, or customer data into an issue.

## Working with the code locally

### Repo layout

```
packages/
  sdks/
    maximem-synap/            # Python SDK       (PyPI: maximem-synap)
    maximem-synap-js/         # JavaScript SDK   (npm: @maximem/synap-js-sdk)
  integrations/               # one folder per framework
    synap-langchain/
      pyproject.toml
      synap_langchain/        # source
      tests/                  # tests for this package
    synap-langgraph/
    ...
  mcps/
    synap-mcp-server/         # Streamable-HTTP MCP server, source only, not published
  connectors/                 # reserved
```

The full published list, with install commands, is the table in [README.md](README.md#all-integrations).

### Python SDK

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

cd packages/sdks/maximem-synap
pip install -e ".[dev]"
```

### A Python integration

```bash
python -m venv .venv
source .venv/bin/activate

# Install the SDK from local source, not PyPI
pip install -e packages/sdks/maximem-synap

# Then the integration you are working on
cd packages/integrations/synap-langchain
pip install -e ".[dev]"
```

### JavaScript SDK

```bash
cd packages/sdks/maximem-synap-js
npm install
npm run check   # verifies the package loads
```

The JavaScript SDK bridges to the Python SDK in a subprocess, so it needs **both** Node 18+ and Python 3.11+ on the host. It is not usable in edge or browser-only runtimes.

### Running tests

```bash
cd packages/integrations/synap-langchain
python -m pytest tests/ -v
```

Tests mock the Synap SDK. Nothing in the suite makes a live API call, and no API key is required to run it.

### Code style

- [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Line length: 88
- Type hints on all public methods
- Google-style docstrings on all public classes and methods

```bash
ruff check packages/integrations/synap-langchain/
ruff format --check packages/integrations/synap-langchain/

# Auto-fix
ruff check --fix packages/integrations/synap-langchain/
ruff format packages/integrations/synap-langchain/
```

## Working against the SDK

If you are writing code on top of Synap, or proposing an integration, these are the rules the first-party integrations follow.

**Use only the public API.** Internal modules change without notice.

```python
# Good
from maximem_synap import MaximemSynapSDK
from maximem_synap.models.context import UnifiedContextResponse

# Bad: internal module
from maximem_synap.cache.anticipation_cache import AnticipationCache
```

**`fetch` is the entry point integrations should use.** It resolves every scope you give it in parallel, merges and deduplicates the results, and returns a `formatted_context` string ready for prompt injection:

```python
from maximem_synap import MaximemSynapSDK

class SynapMyFrameworkMemory:
    def __init__(self, sdk: MaximemSynapSDK, user_id: str, **kwargs):
        self.sdk = sdk
        self.user_id = user_id

    async def search(self, query: str):
        response = await self.sdk.fetch(
            user_id=self.user_id,
            search_query=[query],
        )
        return response.formatted_context
```

**Keep integrations thin.** An integration maps a framework's interfaces onto SDK calls. Business logic belongs in the SDK.

**Support both sync and async.** Most frameworks expose both. Provide async implementations with sync wrappers where the framework expects them.

**Never crash the host application.** Wrap SDK calls, log failures, and degrade to no-memory behaviour rather than raising into user code.

## Proposing a new framework integration

Open an issue naming the framework and the extension points you would map onto Synap. If it is a fit, we build it in the monorepo and it ships here.

We get a steady stream of proposed integrations that are an existing integration with the framework name substituted, and we cannot take those. A real integration has all of the following.

**1. Framework-native adapters.** The code imports the target framework and implements *that framework's* extension points. Each framework's shape is genuinely different: DSPy has no `Agent(instructions=...)` at all, Smolagents tools are `Tool` subclasses with a synchronous `forward()`, CAMEL wants a `camel.toolkits.FunctionTool` and its `ChatAgent` takes a string or `BaseMessage`, and the async `(context, agent)` instructions callable is specific to the OpenAI Agents SDK. If a package does not import the framework it claims to integrate, it is not an integration.

**2. Tests that exercise real framework types.** Mock the Synap SDK, not the framework. A suite copied from another integration with names replaced verifies nothing, because it never constructs the objects the framework would actually hand you.

**3. Honest dependencies.** Do not pin a heavy framework dependency the code never imports.

**4. Correct naming and metadata.** Folder `synap-<framework>`, distribution name `maximem-synap-<framework>`, Python 3.11+, and a dependency on both the SDK and the shared integration helpers:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "maximem-synap-myframework"
version = "0.1.0"
description = "Synap memory integration for MyFramework"
readme = "README.md"
requires-python = ">=3.11"
license = "Apache-2.0"
authors = [{name = "Synap Team"}]
keywords = ["synap", "memory", "myframework", "ai"]

dependencies = [
    "maximem-synap>=0.2.0",
    "maximem-synap-integrations-common>=0.1.0",
    "myframework>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-asyncio>=0.21"]

[tool.setuptools.packages.find]
where = ["."]
include = ["synap_myframework*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**5. Registration for publication.** A package is only released, and only listed in the README table, once it is registered in the monorepo's publish workflows. That step happens on our side; an integration that skips it never reaches PyPI or npm.

## Code of conduct

Be respectful and constructive. We are building something useful together.

## Questions

[Documentation](https://docs.maximem.ai) · [Dashboard](https://synap.maximem.ai) · [Issues](https://github.com/maximem-ai/maximem_synap_sdk/issues)
