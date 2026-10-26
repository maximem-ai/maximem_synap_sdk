# HarnessX

Python only.

```bash
pip install maximem-synap-harnessx
```

For [HarnessX](https://github.com/datagol/agent-harness-x) (`from harnessx import Agent`).

| Export | Purpose |
| --- | --- |
| `SynapExtension` | One extension: recall, `search_memory` / `store_memory` tools, turn recording, stream events |
| `make_synap_tools` | The two tool handlers, to register yourself |
| `SYNAP_MEMORY_PROMPT` | The default block template |

## Quick start

```python
from harnessx import Agent, AgentConfig
from maximem_synap import MaximemSynapSDK
from synap_harnessx import SynapExtension

sdk = MaximemSynapSDK()      # reads SYNAP_API_KEY from the environment
await sdk.initialize()       # required, once, before the first turn

agent = Agent(
    config=AgentConfig(model="claude-sonnet-4-6"),
    extensions=[SynapExtension(
        sdk=sdk,
        user_id="alice",
        customer_id="acme",   # B2B only; omit on B2C
    )],
)

async with agent:
    result = await agent.run("What did I tell you about my trial account?")
```

## How it works

- Recall runs in `Extension.on_turn_start`, because HarnessX prompt providers are sync and take no arguments. The block is cached and the provider returns it. It is refreshed every turn.
- The conversation id is the agent's session id (a UUID). Pass `conversation_id=` to pin one. It is saved and restored with `save_session` / `load_session`.
- `search_memory` is `ALLOW`; `store_memory` is `ASK` by default (`store_permission=`).
- User and assistant turns are recorded stream first, REST as a fallback, never both.
- Tool calls and results are reported on the stream through the `TOOL_CALL_START` / `TOOL_CALL_END` hooks. The app still calls `sdk.instance.listen()`.

## Scoping notes

`customer_id` is B2B only. An empty one is treated as absent. Everything else is as in `core-concepts.md`.

Canonical package doc: `packages/integrations/synap-harnessx/README.md`.
