# OpenAI Agents SDK

`pip install synap-openai-agents`

For OpenAI's `agents` package (the official Agents SDK with `Agent`, `Runner`, `FunctionTool`).

| Function | Purpose |
| --- | --- |
| `create_search_tool` | Async function tool that searches Synap memory |
| `create_store_tool` | Async function tool that stores a memory in Synap |

## Quick start

```python
from agents import Agent, FunctionTool, Runner
from synap_openai_agents import create_search_tool, create_store_tool

search_fn = create_search_tool(sdk=sdk, user_id="alice", customer_id="acme")
store_fn = create_store_tool(sdk=sdk, user_id="alice", customer_id="acme")

agent = Agent(
    name="Memory Agent",
    instructions=(
        "Use synap_search to recall facts about the user. "
        "Use synap_store to remember new information."
    ),
    tools=[
        FunctionTool(search_fn, name_override="synap_search"),
        FunctionTool(store_fn, name_override="synap_store"),
    ],
)

result = await Runner.run(agent, "What do you know about my project deadlines?")
print(result.final_output)
```

## Tool signatures

```
synap_search(query: str, max_results: int = 5) -> list[dict]
# returns [{"content": "...", "type": "fact", "confidence": 0.91}, ...]

synap_store(content: str, memory_type: str = "fact") -> dict
# returns {"status": "stored", "id": "..."}
```

## Per-user agents

The tools close over `user_id` / `customer_id` at construction. For multi-tenant apps, build a fresh tool set per request:

```python
def build_agent_for(user_id: str, customer_id: str | None = None) -> Agent:
    return Agent(
        name="MemAgent",
        tools=[
            FunctionTool(create_search_tool(sdk=sdk, user_id=user_id, customer_id=customer_id), name_override="synap_search"),
            FunctionTool(create_store_tool(sdk=sdk, user_id=user_id, customer_id=customer_id), name_override="synap_store"),
        ],
    )
```

## Live doc

`https://docs.maximem.ai/integrations/openai-agents`

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.3.0 (JS) — verified 2026-06-20. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
