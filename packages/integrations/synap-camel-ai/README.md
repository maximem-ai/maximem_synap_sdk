# maximem-synap-camel-ai

Synap memory integration for [CAMEL-AI](https://github.com/camel-ai/camel) — plug
Synap in as a native `AgentMemory` so your `ChatAgent` gains persistent long-term
memory, plus explicit tools and short-term context.

```bash
pip install maximem-synap-camel-ai camel-ai
```

## Surfaces

| Surface | CAMEL extension point | Purpose |
|---------|-----------------------|---------|
| `SynapAgentMemory` | `AgentMemory` → `ChatAgent(memory=...)` | Recall + persistence layered over CAMEL's own conversation history |
| `create_synap_tools` | `FunctionTool` | Explicit `search_memory` / `store_memory` the model can call |
| `synap_st_system_message` | `ChatAgent(system_message=...)` | Fold Synap short-term context into the system prompt |

## Native memory

`SynapAgentMemory` subclasses CAMEL's `ChatHistoryMemory` and **augments** it rather
than replacing it — CAMEL's `get_context()` is the sole source of the model's input,
so the memory must return the real conversation (system + user + assistant). On top
of that live history it:

- **recalls** Synap's long-term memories and prepends them as a system context block
  (using the latest user turn as the query);
- **persists** each completed turn's accumulated transcript to Synap for server-side
  extraction, under a stable document id.

```python
from camel.agents import ChatAgent
from camel.models import ModelFactory
from maximem_synap import MaximemSynapSDK
from synap_camel_ai import SynapAgentMemory

sdk = MaximemSynapSDK(api_key="sk-...")
memory = SynapAgentMemory(sdk, user_id="alice", customer_id="acme")

agent = ChatAgent(
    system_message="You are a concise, friendly support agent.",
    model=ModelFactory.create(model_platform="openai", model_type="gpt-4o"),
    memory=memory,            # constructor only — not agent.memory = ...
)
print(agent.step("Remind me what plan I'm on and my open ticket.").msgs[0].content)
```

Attach the memory via the **constructor** (`ChatAgent(memory=...)`), not
`agent.memory = ...` afterward.

## Explicit tools

```python
from synap_camel_ai import create_synap_tools

agent = ChatAgent(
    system_message="You are a helpful assistant.",
    tools=create_synap_tools(sdk, user_id="alice", customer_id="acme"),
)
```

## Short-term context

```python
from synap_camel_ai import SynapAgentMemory, synap_st_system_message

agent = ChatAgent(
    system_message=synap_st_system_message(
        sdk, conversation_id="conv_abc",
        system_message="You are a support agent.",
    ),
    memory=SynapAgentMemory(sdk, user_id="alice"),
)
```

## Error policy

- **Reads** (`search_memory`, memory recall) degrade — a Synap blip returns no recall
  and the turn continues.
- **The persistence write** inside the agent loop is best-effort by default
  (`on_error="fallback"`) so a transient outage never discards a model response; set
  `on_error="raise"` for strict environments.
- **The explicit `store_memory` tool** raises `SynapIntegrationError` on failure.

Requires Python 3.11+ and `camel-ai>=0.2.90`.
