# AutoGen

> **Read this next:** [This integration's docs page](https://docs.maximem.ai/integrations/autogen.md); [memories.create](https://docs.maximem.ai/sdk-reference/memories/create.md); [context fetch](https://docs.maximem.ai/sdk-reference/context/fetch.md); [listen](https://docs.maximem.ai/sdk-reference/instance/listen.md); [Errors](https://docs.maximem.ai/sdk-reference/errors.md); [Graceful degradation](https://docs.maximem.ai/patterns/graceful-degradation.md). **Checkpoint:** open the method's page before writing each call, re-read it after and compare every parameter, scope rule and error; if the docs and this file disagree, the docs win and tell the human.

`pip install maximem-synap-autogen`

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

**Scoping:** `customer_id` is B2B only, and required there. On a B2C instance (`user_context_isolation = "equals_customer"`) pass `user_id` alone: a `customer_id` comes back as HTTP 400. `GET /api/v1/auth/whoami` tells you which mode the instance is in.

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
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
