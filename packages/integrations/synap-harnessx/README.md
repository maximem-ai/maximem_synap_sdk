# synap-harnessx

Synap integration for [HarnessX](https://github.com/datagol/agent-harness-x), a provider-agnostic Python toolkit for building LLM agents.

HarnessX ships local memory (`AgentMemory`, `LongTermMemory`) that lives on the agent's disk. This package adds Synap's hosted memory through HarnessX's own extension system: one `SynapExtension` and no changes to your agent code.

## Install

```bash
pip install maximem-synap-harnessx
```

Requires `harnessx>=0.9.1`, `maximem-synap>=0.5.3`.

## Quickstart

```python
from harnessx import Agent, AgentConfig
from maximem_synap import MaximemSynapSDK
from synap_harnessx import SynapExtension

sdk = MaximemSynapSDK()   # reads SYNAP_API_KEY from the environment
await sdk.initialize()    # required. An uninitialised SDK fails every call, and the
                          # extension logs that and carries on with no memory.

agent = Agent(
    config=AgentConfig(model="claude-sonnet-4-6"),
    extensions=[SynapExtension(sdk=sdk, user_id="alice")],
)

async with agent:
    print((await agent.run("What do I prefer for hotels?")).output)
```

The extension takes an initialised SDK and never creates one. Get an API key from the [dashboard](https://synap.maximem.ai).

## What one extension does

| Job | HarnessX surface | What happens |
|---|---|---|
| Recall | `on_turn_start` + a prompt provider | Fetches memory using the user's message as the search query and injects a `<synap_memory>` block into the system prompt |
| Tools | `ctx.register_tool` | Registers `search_memory` (allowed) and `store_memory` (asks for permission) |
| Recording | `on_turn_start`, `on_turn_end` | Reports the user's turn and the assistant's reply |
| Stream | `TOOL_CALL_START` / `TOOL_CALL_END` hooks, `teardown` | Reports tool calls and results, and ends the session when the agent closes |
| Session state | `on_save_session` / `on_load_session` | Keeps the Synap conversation id across `save_session` / `load_session` |

HarnessX prompt providers are synchronous and take no arguments, so they can neither await the SDK nor see the question. The fetch therefore happens in the async `on_turn_start`, which receives the user message, and the provider returns what it cached. The block is refreshed every turn.

The conversation id defaults to the agent's session id, which is a UUID, as Synap requires. Pass `conversation_id=` to pin your own.

## Reporting on Synap's live stream

Tool calls and tool results are reported only while `sdk.instance.listen()` is running. With no stream they are no-ops and the extension costs nothing.

```python
await sdk.instance.listen()
...
await sdk.instance.stop_listening()
```

Turn recording is **stream first, REST only as a fallback, never both**. The server persists `user_message` and `assistant_message` from the stream itself, so sending them over REST as well stores and extracts the turn twice. Unlike the Claude Agent SDK, HarnessX hands the extension the final reply in `on_turn_end`, so the assistant turn needs no manual call.

## Configuration

```python
SynapExtension(
    sdk=sdk,
    user_id="alice",             # required. On B2B add customer_id; on B2C it is rejected.
    customer_id="acme",
    conversation_id=None,        # default: the agent's session id
    max_results=20,
    mode="fast",                 # recall runs every turn
    precision_level="high",
    memory_prompt=SYNAP_MEMORY_PROMPT,  # None fetches but injects nothing
    fetch_without_query=True,
    register_tools=True,
    store_permission=PermissionLevel.ASK,
    record_turns=True,
    report_tools=True,
)
```

`make_synap_tools(sdk, user_id=...)` returns the two tool handlers if you would rather register them yourself.

## Error policy

| Operation | Behaviour |
|---|---|
| Recall | Degrades to an empty block and logs at `ERROR`. `on_turn_start` is fail-fast in HarnessX, so an exception there would end the run |
| Recording and stream reports | Never raise |
| `search_memory`, `store_memory` | Raise `SynapIntegrationError`, which HarnessX turns into an error result the model can read |

## Notes

- **Store with context.** `store_memory` returns "accepted" for any text, but Synap's extraction creates no memories from a bare one-sentence statement. The tool description asks the model for a couple of sentences with context. If you write your own prompts around it, keep that.
- **One extension per agent.** The cached memory block and conversation id are per instance.
- **Choose Synap or local memory.** The built-in `register_memory_tools` registers `save_memory` and `recall_memories` over local disk. These tools have different names, so they do not collide, but the model will see two memory systems.
- **Tool results leave your process.** A tool result is usually your customer's data. It is an anticipation hint and never becomes a long-term memory, but it is sent. Set `report_tools=False` to stop that.

## Tests

```bash
pytest packages/integrations/synap-harnessx/tests -q
```
