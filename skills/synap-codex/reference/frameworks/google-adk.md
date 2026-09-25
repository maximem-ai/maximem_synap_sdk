# Google ADK

`pip install maximem-synap-google-adk`

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
    customer_id="acme",   # B2B only; omit on B2C
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

**Scoping:** `customer_id` is B2B only, and required there. On a B2C instance (`user_context_isolation = "equals_customer"`) pass `user_id` alone: a `customer_id` comes back as HTTP 400. `GET /api/v1/auth/whoami` tells you which mode the instance is in.

## Tool signatures

```
search_memory(query: str, max_results: int = 5) -> list[dict]
# returns [{"content": "...", "type": "...", "confidence": float}, ...]

store_memory(content: str, memory_type: str = "fact") -> dict
# returns {"status": "stored", "id": "..."}
```

## Multi-user setup

The tools close over `user_id` / `customer_id` at construction. For multi-tenant (B2B), build per-user and pass both ids. The example below is the B2C shape, `user_id` on its own:

```python
def build_agent_for_user(user_id: str) -> Agent:
    tools = create_synap_tools(sdk=sdk, user_id=user_id)
    return Agent(name="MemAgent", model="gemini-2.0-flash", tools=tools)
```

## The live stream — wire this too

Tools are only half the integration. `create_synap_callbacks` reports the run
itself, so Synap can predict the next turn instead of retrieving cold.

```python
from synap_google_adk import create_synap_tools, create_synap_callbacks, report_user_turn

await sdk.instance.listen()                       # once, at startup

agent = LlmAgent(
    model="gemini-2.0-flash",
    tools=create_synap_tools(sdk, user_id="alice"),
    **create_synap_callbacks(sdk, user_id="alice", customer_id="acme",
                             conversation_id=conv_id),
)

# ADK's before_agent_callback gets a context, not the user's text
await report_user_turn(sdk, user_text, conversation_id=conv_id,
                       user_id="alice", customer_id="acme")

await sdk.instance.end_session(conv_id)           # when the conversation ends
await sdk.instance.stop_listening()               # once, at shutdown
```

It returns `before_tool_callback`, `after_tool_callback` and
`after_model_callback`, splatted into `LlmAgent(...)`. They report tool calls,
tool results and the reply; the user's turn is yours.

Every callback returns `None` on purpose. ADK reads a returned value as an
override of the model or tool result, so telemetry that returned anything would
change what the agent does. Everything is a no-op until `listen()` is running.
Full picture: `reference/streaming.md`.

## Live doc

`https://docs.maximem.ai/integrations/google-adk`

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.3.0 (JS) — verified 2026-06-20. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
