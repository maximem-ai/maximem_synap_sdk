# synap-microsoft-agent

Synap memory integration for [Microsoft Agent Framework (MAF)](https://learn.microsoft.com/en-us/agent-framework/).

## Install

```bash
pip install synap-microsoft-agent
```

Requires `agent-framework>=1.0`, `maximem-synap>=0.2.0`.

## Quickstart

```python
from agent_framework import InMemoryHistoryProvider
from agent_framework.openai import OpenAIChatClient
from maximem_synap import MaximemSynapSDK
from synap_microsoft_agent import SynapContextProvider, SynapHistoryProvider

sdk = MaximemSynapSDK(api_key="sk-...")
client = OpenAIChatClient(model="gpt-4o-mini")

agent = client.as_agent(
    name="MemoryAgent",
    instructions="You are a helpful assistant.",
    context_providers=[
        SynapContextProvider(
            sdk=sdk,
            user_id="alice",
            customer_id="acme",
        ),
        SynapHistoryProvider(
            sdk=sdk,
            user_id="alice",
            customer_id="acme",
        ),
    ],
)

session = agent.create_session()
result = await agent.run("What's my trial expiring?", session=session)
```

## What each provider does

- **`SynapContextProvider`** — on every turn, fetches Synap context (facts, preferences, episodes, emotions, temporal events) and appends it as instructions. After the turn, records the user + assistant messages back to Synap.

- **`SynapHistoryProvider`** — persists the conversation message log. Loads prior turns on session resume. Subclass of MAF's `HistoryProvider`, so all its flags (`load_messages`, `store_inputs`, `store_outputs`, `store_context_messages`) work as documented.

- **`SynapShortTermContextProvider`** — injects a compacted summary of the current conversation, refreshed each turn.

You can use either or both; they coexist.

## The Agent Harness

MAF's harness (`create_harness_agent`) is a separate surface with its own memory
subsystem, and this package backs both of its storage seams. Needs
`agent-framework>=1.13`; the classes import lazily so the floor above stays at 1.0.

```python
from agent_framework import create_harness_agent
from synap_microsoft_agent import SynapAgentFileStore, create_synap_harness_memory

agent = create_harness_agent(
    client,
    history_provider=create_synap_harness_memory(sdk, user_id="alice", customer_id="acme"),
    file_memory_store=SynapAgentFileStore(sdk, user_id="alice", customer_id="acme"),
)
```

- **`SynapMemoryStore`** — backs the topic notebook (`MEMORY.md`, topic records,
  extraction, consolidation). Extraction and consolidation stay MAF's; storage and
  retrieval become Synap's. `MEMORY.md` gains a durable recall block that survives
  restarts and is shared across agents on the same scope.
- **`SynapAgentFileStore`** — backs the seven `file_memory_*` tools. `grep` searches
  by regex *and* by meaning; `delete` is real.
- **`create_synap_harness_memory`** — builds the provider wired correctly. Prefer it.

Two things to know:

1. **`create_harness_agent` takes exactly one `history_provider`,** and both
   `SynapHistoryProvider` and the harness memory provider are `HistoryProvider`s.
   Passing both silently drops one. Use the factory and pass only that.
2. **Topic records are held exactly, and by default only for the life of the
   process.** Synap's ingestion rewrites what you submit, and the harness does
   read-modify-write on records, so reading them back from Synap would corrupt
   them a little more each turn. Pass a `record_store` to survive restarts. The
   content is durable in Synap either way.

Every harness API is `@experimental` upstream. Pin a tested version and re-run the
suite on each MAF minor.

## Error policy

- **Read-side failures** (`fetch`, `get_context_for_prompt`) degrade gracefully — logged at `ERROR`, empty result returned. An outage never crashes an agent turn.
- **Write-side failures** — `SynapContextProvider.after_run` logs and swallows (context providers must not raise per MAF's hook contract). `SynapHistoryProvider.save_messages` surfaces errors as `SynapIntegrationError` so explicit persistence failures are observable.
- **Harness stores** — same split. `get_index_text` degrades to pointer lines with no recall block, because it feeds the system prompt on every turn. `write_topic` and `file_memory_write` raise. A topic that is not held raises `FileNotFoundError`, which is MAF's own not-found contract.

## Tests

```bash
pytest integrations/synap-microsoft-agent/tests -q
```

The harness tests skip cleanly when `agent-framework<1.13` is installed, so the SDK
surfaces stay testable on the floor version. `bench/smoke.py` drives MAF's real
`MemoryContextProvider` and `FileMemoryProvider` over both stores — run it with
`--mock` for no network, or with credentials and `--scope` against a seeded scope.
