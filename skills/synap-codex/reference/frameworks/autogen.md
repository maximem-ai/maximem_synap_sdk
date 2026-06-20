# AutoGen

`pip install synap-autogen`

For Microsoft AutoGen — `autogen-agentchat` / `autogen-core`.

| Class | Purpose |
| --- | --- |
| `SynapSearchTool` | `BaseTool` that searches Synap memory |
| `SynapStoreTool` | `BaseTool` that stores a memory |

Both support AutoGen's `CancellationToken`.

## Quick start

```python
from autogen_agentchat.agents import AssistantAgent
from synap_autogen import SynapSearchTool, SynapStoreTool

tools = [
    SynapSearchTool(sdk=sdk, user_id="alice", customer_id="acme"),
    SynapStoreTool(sdk=sdk, user_id="alice", customer_id="acme"),
]

agent = AssistantAgent(
    name="MemoryAgent",
    model_client=your_model_client,
    tools=tools,
    system_message=(
        "Use synap_search to recall user context. "
        "Use synap_store to remember new information."
    ),
)

await agent.run(task="What are my top priorities this week?")
```

## Tool schemas

`SynapSearchTool`:

```json
{
  "query": "string",
  "max_results": "int (default 5)",
  "mode": "\"fast\" | \"accurate\" (default \"fast\")"
}
```

Returns: list of `{content, type, confidence}`.

`SynapStoreTool`:

```json
{
  "content": "string",
  "memory_type": "string (default \"fact\")"
}
```

Returns: `{"status": "stored", "id": "..."}`.

## Cancellation

```python
from autogen_core import CancellationToken

token = CancellationToken()
result = await tool.run({"query": "project deadlines"}, cancellation_token=token)

# elsewhere in the same task group:
token.cancel()
```

## Live doc

`https://docs.maximem.ai/integrations/autogen`

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.3.0 (JS) — verified 2026-06-20. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
