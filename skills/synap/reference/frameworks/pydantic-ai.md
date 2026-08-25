# Pydantic AI

`pip install synap-pydantic-ai`

| Export | Purpose |
| --- | --- |
| `SynapDeps` | Dataclass holding the SDK and user scope |
| `register_synap_tools(agent)` | Adds `synap_search` + `synap_store` tools and an instruction fragment to the agent |

## Quick start

```python
from pydantic_ai import Agent
from synap_pydantic_ai import SynapDeps, register_synap_tools

agent: Agent[SynapDeps, str] = Agent(
    "openai:gpt-4o",
    deps_type=SynapDeps,
    system_prompt="You are a helpful assistant with long-term memory.",
)

register_synap_tools(agent)

deps = SynapDeps(sdk=sdk, user_id="alice", customer_id="acme")   # customer_id: B2B only
result = await agent.run("What do you remember about my project?", deps=deps)
print(result.data)
```

`register_synap_tools` does three things in one call:

1. Adds `synap_search` (model recalls memories)
2. Adds `synap_store` (model persists new memories)
3. Appends a system-prompt fragment instructing the agent to use both

**Scoping:** `customer_id` is B2B only, and required there. On a B2C instance (`user_context_isolation = "equals_customer"`) pass `user_id` alone: a `customer_id` comes back as HTTP 400. `GET /api/v1/auth/whoami` tells you which mode the instance is in.

## SynapDeps shape

```python
@dataclass
class SynapDeps:
    sdk: MaximemSynapSDK
    user_id: str
    customer_id: str | None = None
    conversation_id: str | None = None
```

This is how you serve multiple users from a single agent instance: different `SynapDeps` per request. The example below is the B2C shape; on B2B carry the tenant's `customer_id` in the deps too:

```python
async def handle_request(user_id: str, message: str) -> str:
    deps = SynapDeps(sdk=sdk, user_id=user_id)
    result = await agent.run(message, deps=deps)
    return result.data
```

The agent itself is stateless w.r.t. user identity; `deps` carries it.

## Live doc

`https://docs.maximem.ai/integrations/pydantic-ai`

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.3.0 (JS) — verified 2026-06-20. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
