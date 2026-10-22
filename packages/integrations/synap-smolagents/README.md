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
| `synap_step_callbacks` | `step_callbacks={MemoryStep: ...}` | Record the user and assistant turns as a Synap conversation, and report tool calls and reasoning on the live stream |
| `attach_synap_history` | `agent.write_memory_to_messages` | Send Synap's compaction instead of the agent's earlier steps (applied by `synap_step_callbacks`) |
| `synap_st_instructions` | `CodeAgent(instructions=...)` | Fold Synap's compaction block into the agent's instructions (for an agent with no recorder) |

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
- **Turns are conversation messages.** The user's task and the assistant's final
  answer are written to the conversation `conversation_id` names: on the gRPC stream
  (the first turn opens it; you do not call `listen()`), otherwise with
  `sdk.conversation.record_message`, never both. `SYNAP_AUTO_LISTEN=false` (or
  `set_auto_listen(sdk, False)` from `synap_integrations_common`) keeps turns on REST. The server compacts that
  conversation and promotes it to long-term memory, and the SDK's local history
  search can find old turns after compaction. Reuse one `conversation_id` across
  `agent.run()` calls that belong together.
- **No per-step documents by default.** Earlier versions ingested every step's
  model output with `memories.create`, which left no conversation on the server.
  That is now off (`ingest_steps=True` turns it back on): with the turns recorded,
  it would extract the final answer twice.
- **Live stream.** On the stream the recorder also reports every tool call and its
  result, and the model's plan and step output as reasoning. Without a stream those
  are dropped.
- **Synap's compaction replaces the agent's earlier steps.** Smolagents builds the
  model's input in `write_memory_to_messages` and has no compaction of its own. The
  recorder wraps that method on the agent it is handed: once Synap has compacted the
  conversation, the model sees the compaction block, Synap's turns after it and the
  current task, instead of the earlier steps in `agent.memory` (which is never
  edited). Until then, and if Synap cannot answer, the agent's messages are
  untouched. The conversation is read once per run. Smolagents has no hook before a
  model call, so the recorder attaches after the first step; call
  `attach_synap_history(agent, sdk, conversation_id)` right after building the agent
  to cover the first step of a resumed conversation. `compact_history=False` turns
  it off.
- **Register against `MemoryStep`.** `synap_step_callbacks` does this for you.
  Registering the recorder against `ActionStep` alone (what earlier versions of this
  README showed) never delivers the final answer, and the assistant turn is the one
  event anticipation acts on.
- **Two gaps, on purpose.** Smolagents never passes its `TaskStep` to a callback, so
  the user turn is read from `agent.task` and written at the end of the first step
  rather than before it (`report_user_task=False` turns it off). And a step with
  several parallel tool calls merges their outputs into one `observations` string,
  so that result is reported with no `tool_call_id` rather than being attributed to
  a guessed call.

Requires Python 3.11+ and `smolagents>=1.26`.
