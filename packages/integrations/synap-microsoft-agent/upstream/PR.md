# Upstream PR — add the `agent_framework.maximem_synap` namespace shim

Target: [`microsoft/agent-framework`](https://github.com/microsoft/agent-framework), MIT.
Status: **drafted, not filed.**

## What to change in their repo

Two new files plus two edits, mirroring `agent_framework/mem0/` exactly.

| Path | Action |
|---|---|
| `python/packages/core/agent_framework/maximem_synap/__init__.py` | new — copy from `upstream/agent_framework/maximem_synap/__init__.py` here |
| `python/packages/core/agent_framework/maximem_synap/__init__.pyi` | new — copy from `upstream/agent_framework/maximem_synap/__init__.pyi` here |
| `agent_framework/__init__.py` | add `maximem_synap` alongside `mem0`, `redis`, `azure`, in whatever list registers the sub-namespaces |
| Packaging manifest | include the new sub-package, the same way `mem0` is included |

Confirm the exact path layout against the repo at the time of filing — this
was written against the installed `agent-framework-core` 1.13.0 wheel, which
flattens `python/packages/core/`.

## PR title

> Add `agent_framework.maximem_synap` namespace shim for the Maximem Synap connector

## PR description

Adds a lazy namespace shim so `maximem-synap-microsoft-agent` is reachable at
`agent_framework.maximem_synap`, following the pattern already established for
`agent-framework-mem0`, `agent-framework-redis`, `agent-framework-azure-cosmos`,
and `agent-framework-azure-ai-search`.

The shim is the same 30-line lazy `__getattr__` as `agent_framework/mem0/__init__.py`,
with the same `IMPORT_PATH` / `PACKAGE_NAME` / `_IMPORTS` triple and the same
`ModuleNotFoundError` message. No runtime dependency is added: the import is
lazy and raises an install hint if the package is absent.

**One deliberate difference from the mem0 shape, called out so it is not a
surprise in review.** The mem0 shim points at a module named after its own
distribution (`agent-framework-mem0` → `agent_framework_mem0`). This one points
at `synap_microsoft_agent`, from the distribution `maximem-synap-microsoft-agent`.
That package already exists, already ships the `ContextProvider` and
`HistoryProvider` surfaces, and has now been extended with the harness storage
seams. Creating a second distribution just to match the naming convention would
split one integration across two packages and two install lines, so the shim
points at the existing module instead.

**What the package provides.** Both MAF surfaces:

- Agent SDK — `SynapContextProvider`, `SynapHistoryProvider`,
  `SynapShortTermContextProvider`.
- Agent Harness — `SynapMemoryStore` (a `MemoryStore`), `SynapAgentFileStore`
  (an `AgentFileStore`), and `create_synap_harness_memory`, which returns a
  correctly-wired `MemoryContextProvider` subclass.

As far as we can tell this is the first third-party `MemoryStore` implementation
for the harness. If that is wrong we would genuinely like to know, since it
would give us a reference to check our mapping against.

Everything imports from top-level `agent_framework` — nothing reaches into
`agent_framework._harness`, per the feature-stage warning that those members
may move.

## Secondary ask — include as a question, not a request

`MemoryStore.get_transcripts_directory` returns a `Path`, which is the one part
of the `MemoryStore` contract that assumes local disk. We work around it by
subclassing `MemoryContextProvider` and overriding `get_messages` and
`save_messages` — its only two consumers — so nothing ever reads or creates the
path. That works and needs nothing from you.

The question is whether a store-mediated transcript interface is on the roadmap,
which would let a cloud-backed store drop the subclass entirely. Framed as a
question because the workaround is fine and we are not blocked.

## Notes for whoever files this

- Sign the CLA first if the org has not already.
- The naming is `maximem_synap`, not `synap`. Company plus product, matching the
  `maximem-synap-*` distributions and the SDK's own `maximem_synap` module.
  Do not shorten it in review.
- Tier A and Tier B ship regardless of whether this lands. The shim buys
  discoverability, not function, so there is no reason to hold anything on it.
