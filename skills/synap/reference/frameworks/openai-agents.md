# OpenAI Agents SDK

`pip install maximem-synap-openai-agents`

For OpenAI's `agents` package (the official Agents SDK with `Agent`, `Runner`, `FunctionTool`).

| Function | Purpose |
| --- | --- |
| `create_search_tool` | Async function tool that searches Synap memory |
| `create_store_tool` | Async function tool that stores a memory in Synap |

## Quick start

```python
from agents import Agent, Runner, function_tool
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
        function_tool(search_fn, name_override="synap_search"),
        function_tool(store_fn, name_override="synap_store"),
    ],
)

result = await Runner.run(agent, "What do you know about my project deadlines?")
print(result.final_output)
```

`create_search_tool` / `create_store_tool` return plain async callables. Wrap them with
`function_tool(...)` (the helper), **not** `FunctionTool(...)` (the dataclass). The
dataclass takes `name`, `description`, `params_json_schema` and `on_invoke_tool`, has no
`name_override`, and raises `TypeError` if you pass it a bare function.

**Scoping:** `customer_id` is B2B only, and required there. On a B2C instance (`user_context_isolation = "equals_customer"`) pass `user_id` alone: a `customer_id` comes back as HTTP 400. `GET /api/v1/auth/whoami` tells you which mode the instance is in.

## Tool signatures

```
synap_search(query: str) -> str
# returns the formatted context block, or "No relevant memories found."

synap_store(content: str) -> str
# returns "Memory stored (ingestion_id: ...)"
```

## Per-user agents

The tools close over `user_id` / `customer_id` at construction. For multi-tenant (B2B) apps, build a fresh tool set per request and pass both ids. On B2C, leave `customer_id` at `None`:

```python
def build_agent_for(user_id: str, customer_id: str | None = None) -> Agent:
    return Agent(
        name="MemAgent",
        tools=[
            function_tool(create_search_tool(sdk=sdk, user_id=user_id, customer_id=customer_id), name_override="synap_search"),
            function_tool(create_store_tool(sdk=sdk, user_id=user_id, customer_id=customer_id), name_override="synap_store"),
        ],
    )
```

## The live stream — wire this too

Tools are only half the integration. `SynapRunHooks` reports the run itself, so
Synap can predict the next turn instead of retrieving cold every time.

```python
from synap_openai_agents import SynapRunHooks, report_user_turn

await sdk.instance.listen()                       # once, at startup

hooks = SynapRunHooks(sdk, user_id="alice", customer_id="acme",
                      conversation_id=conv_id)

# no hook carries the user's input, so report it yourself before the run
await report_user_turn(sdk, user_text, conversation_id=conv_id,
                       user_id="alice", customer_id="acme")

result = await Runner.run(agent, user_text, hooks=hooks)

await sdk.instance.end_session(conv_id)           # when the conversation ends
await sdk.instance.stop_listening()               # once, at shutdown
```

`SynapRunHooks` reports tool calls, tool results, model reasoning and the reply.
The user's turn is the one thing left to you: `on_agent_start` fires with the
agent, not with what the user said.

`report_tool_results=False` turns off tool-result reporting if the caller does
not want their customers' tool output leaving the process. Everything is a
no-op until `listen()` is running, so adding the hooks changes nothing on its
own. Full picture: `reference/streaming.md`.

## Live doc

`https://docs.maximem.ai/integrations/openai-agents`

---
*Accurate as of `maximem-synap` 0.5.1 (Python) · `@maximem/synap-js-sdk` 0.5.1 (JS) — verified 2026-09-25. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
