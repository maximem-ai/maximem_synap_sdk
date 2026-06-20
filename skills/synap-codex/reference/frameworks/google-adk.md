# Google ADK

`pip install synap-google-adk`

For Google's Agent Development Kit (`google.adk` package).

| Function | Purpose |
| --- | --- |
| `create_synap_tools` | Returns a list of two ADK `FunctionTool` objects: search + store |

## Quick start

```python
from google.adk.agents import Agent
from synap_google_adk import create_synap_tools

tools = create_synap_tools(
    sdk=sdk,
    user_id="alice",
    customer_id="acme",   # optional
)

agent = Agent(
    name="MemoryAgent",
    model="gemini-2.0-flash",
    instruction=(
        "Use the synap_search tool to recall context about the user. "
        "Use synap_store to remember new facts."
    ),
    tools=tools,
)
```

`create_synap_tools` returns `[search_memory, store_memory]` — pass directly to `Agent(tools=...)`.

## Tool signatures

```
search_memory(query: str, max_results: int = 5) -> list[dict]
# returns [{"content": "...", "type": "...", "confidence": float}, ...]

store_memory(content: str, memory_type: str = "fact") -> dict
# returns {"status": "stored", "id": "..."}
```

## Multi-user setup

The tools close over `user_id` / `customer_id` at construction. For multi-tenant, build per-user:

```python
def build_agent_for_user(user_id: str) -> Agent:
    tools = create_synap_tools(sdk=sdk, user_id=user_id)
    return Agent(name="MemAgent", model="gemini-2.0-flash", tools=tools)
```

## Live doc

`https://docs.maximem.ai/integrations/google-adk`

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.3.0 (JS) — verified 2026-06-20. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
