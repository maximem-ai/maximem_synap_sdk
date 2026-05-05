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

You can use either or both; they coexist.

## Error policy

- **Read-side failures** (`fetch`, `get_context_for_prompt`) degrade gracefully — logged at `ERROR`, empty result returned. An outage never crashes an agent turn.
- **Write-side failures** — `SynapContextProvider.after_run` logs and swallows (context providers must not raise per MAF's hook contract). `SynapHistoryProvider.save_messages` surfaces errors as `SynapIntegrationError` so explicit persistence failures are observable.
