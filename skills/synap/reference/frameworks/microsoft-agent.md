# Microsoft Agent Framework (MAF)

`pip install synap-microsoft-agent`

For Microsoft Agent Framework / Azure AI Agents.

| Class | Purpose |
| --- | --- |
| `SynapContextProvider` | Injects memory into context before each turn; records turns after |
| `SynapHistoryProvider` | Persists/loads full conversation history |

## Quick start

```python
import os
from azure.ai.agents import AgentsClient
from synap_microsoft_agent import SynapContextProvider, SynapHistoryProvider

client = AgentsClient(endpoint=os.environ["AZURE_AI_ENDPOINT"])

agent = client.as_agent(
    context_providers=[
        SynapContextProvider(
            sdk=sdk,
            user_id="alice",
            customer_id="acme",   # optional
            max_context_results=6,
        ),
        SynapHistoryProvider(
            sdk=sdk,
            user_id="alice",
            conversation_id="thread-001",
        ),
    ],
)

response = await agent.run("What were the outcomes from my last meeting?")
```

## SynapContextProvider — semantic memory injection

Called by the MAF runtime at the start of each turn:

1. Fetches relevant memories for the incoming message
2. Appends them as a `system` context message
3. After the agent responds, ingests the full turn back into Synap

Step 1 failures degrade gracefully (empty context, error logged). Step 3 failures raise so callers know persistence failed.

## SynapHistoryProvider — verbatim history

Persists/reloads the full conversation message list:

- `load(thread_id)` — fetches prior messages and restores them to the MAF thread
- `save(thread_id, messages)` — writes new messages after each turn

Use `SynapHistoryProvider` when the LLM needs the full transcript, not just semantic context.

## When to use which

- **Semantic recall ("what does the agent know about me")** → `SynapContextProvider`.
- **Full transcript replay ("what did we say last session, verbatim")** → `SynapHistoryProvider`.
- **Both** → most production agents want both. They compose cleanly.

## Live doc

`https://docs.maximem.ai/integrations/microsoft-agent`
