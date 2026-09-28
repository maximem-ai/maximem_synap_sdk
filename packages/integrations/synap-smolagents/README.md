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
| `synap_step_callbacks` | `step_callbacks={MemoryStep: ...}` | Report the whole turn on Synap's live stream and record each completed action into Synap |
| `synap_st_instructions` | `CodeAgent(instructions=...)` | Fold Synap short-term context into the agent's instructions |

## Example

```python
from smolagents import CodeAgent, InferenceClientModel
from maximem_synap import MaximemSynapSDK
from synap_smolagents import create_synap_tools, synap_step_callbacks, synap_st_instructions

sdk = MaximemSynapSDK(api_key="sk-...")

agent = CodeAgent(
    model=InferenceClientModel(),
    tools=create_synap_tools(sdk, user_id="alice", customer_id="acme"),
    instructions=synap_st_instructions(
        sdk, conversation_id="conv_abc",
        instructions="You are a concise, friendly assistant.",
    ),
    step_callbacks=synap_step_callbacks(
        sdk, user_id="alice", conversation_id="conv_abc", customer_id="acme"
    ),
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
- **Live stream.** With `sdk.instance.listen()` running, the recorder also reports
  the turn as it happens: the user's task, the assistant's final answer, every tool
  call and its result, and the model's plan and step output as reasoning. Without a
  stream every one of those is a no-op, so nothing changes for callers who have not
  opted in.
- **Register against `MemoryStep`.** `synap_step_callbacks` does this for you.
  Registering the recorder against `ActionStep` alone (what earlier versions of this
  README showed) never delivers the final answer, and the assistant turn is the one
  event anticipation acts on.
- **Two gaps, on purpose.** Smolagents never passes its `TaskStep` to a callback, so
  the user turn is read from `agent.task` and reported at the end of the first step
  rather than before it (`report_user_task=False` turns it off). And a step with
  several parallel tool calls merges their outputs into one `observations` string,
  so that result is reported with no `tool_call_id` rather than being attributed to
  a guessed call.

Requires Python 3.11+ and `smolagents>=1.26`.
