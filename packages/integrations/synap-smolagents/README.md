# maximem-synap-smolagents

Synap memory integration for [Smolagents](https://github.com/huggingface/smolagents)
— give a `CodeAgent` / `ToolCallingAgent` persistent memory through Synap.

```bash
pip install maximem-synap-smolagents smolagents
```

## Surfaces

Smolagents keeps its own fixed step log (no pluggable memory backend), so Synap plugs
into the two extension points it *does* expose, plus static instructions:

| Surface | Smolagents extension point | Purpose |
|---------|----------------------------|---------|
| `create_synap_tools` | `@tool` | Explicit `search_memory` / `store_memory` the model can call |
| `create_synap_recorder` | `step_callbacks={ActionStep: ...}` | Record each completed action into Synap |
| `synap_st_instructions` | `CodeAgent(instructions=...)` | Fold Synap short-term context into the agent's instructions |

## Example

```python
from smolagents import CodeAgent, InferenceClientModel
from smolagents.memory import ActionStep
from maximem_synap import MaximemSynapSDK
from synap_smolagents import create_synap_tools, create_synap_recorder, synap_st_instructions

sdk = MaximemSynapSDK(api_key="sk-...")

agent = CodeAgent(
    model=InferenceClientModel(),
    tools=create_synap_tools(sdk, user_id="alice", customer_id="acme"),
    instructions=synap_st_instructions(
        sdk, conversation_id="conv_abc",
        instructions="You are a concise, friendly assistant.",
    ),
    step_callbacks={
        ActionStep: create_synap_recorder(
            sdk, user_id="alice", conversation_id="conv_abc", customer_id="acme"
        )
    },
)
agent.run("Remind me what plan I'm on and my open ticket.")
```

Because `CodeAgent` calls tools as plain Python, `search_memory` / `store_memory`
compose naturally into agent-written code.

## Notes

- **Synchronous framework.** Smolagents runs synchronously and its tools do not
  support `async`, so every surface bridges to the async Synap SDK internally. Run
  the agent on its own thread if your app is otherwise async.
- **Error policy.** The `search_memory` tool degrades (returns a not-found message);
  `store_memory` raises `SynapIntegrationError`; the recorder **logs and never
  raises** — it runs inside the agent loop, where a raising callback would abort the
  run.
- **Per-step documents.** The recorder writes each `ActionStep` to its own Synap
  document, so multi-step runs are captured in full.

Requires Python 3.11+ and `smolagents>=1.26`.
