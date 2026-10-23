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

`SynapAgentMemory` also reports the turn on Synap's live stream — see
[Live stream reporting](#live-stream-reporting).

## Native memory

`SynapAgentMemory` subclasses CAMEL's `ChatHistoryMemory` and **augments** it rather
than replacing it — CAMEL's `get_context()` is the sole source of the model's input,
so the memory must return the real conversation (system + user + assistant). On top
of that live history it:

- **recalls** Synap's long-term memories and prepends them as a system context block
  (using the latest user turn as the query);
- **records** every user and assistant turn as a Synap conversation message, under
  `conversation_id` (default `camel-{agent_id}`; pass your own to continue a
  conversation across agents or processes). The server compacts that
  conversation and promotes it to long-term memory, and the SDK's local history
  search can find old turns after compaction;
- **replaces CAMEL's earlier turns with Synap's compaction** once the server has
  compacted the conversation: the model sees CAMEL's system message, the
  compaction block, the turns after it, and the current turn. CAMEL's own
  summarizer stays installed as the fallback (it does not fire on the shorter
  history), and a conversation with no compaction yet is left as CAMEL holds it.

```python
from camel.agents import ChatAgent
from camel.models import ModelFactory
from maximem_synap import MaximemSynapSDK
from synap_camel_ai import SynapAgentMemory

sdk = MaximemSynapSDK(api_key="sk-...")
memory = SynapAgentMemory(
    sdk, user_id="alice", customer_id="acme", conversation_id="conv_abc",
)

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

`synap_st_system_message` folds the conversation's compaction block (not its
recent turns, which `ChatAgent` already sends) into the system message once, at
construction. With `SynapAgentMemory` the live compaction replaces that snapshot
on every step.

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

## Live stream reporting

CAMEL has no callback protocol, so `SynapAgentMemory.write_records` is the only
place an integration can see what an agent is doing — and it sees all of it,
because `ChatAgent.update_memory` is how CAMEL records every part of a turn.
The first write opens the stream (`sdk.instance.listen()`); you do not call it.
The memory reports five events on it so Synap's anticipation agent can watch the
turn unfold:

| What CAMEL writes | What goes on the stream |
|-------------------|-------------------------|
| a `USER` record | the user turn |
| an `ASSISTANT` record carrying `meta_dict["tool_calls"]` | one `tool_call` per entry, with the parsed arguments and the provider's `tool_call_id` |
| a `FUNCTION` record | the `tool_result`, under that same id |
| an `ASSISTANT` record with `reasoning_content` | the reasoning step, then the assistant turn |

**Stream first, REST otherwise, never both.** The user and assistant turns go
out as stream events and the server persists them from the event. If the stream
cannot open they are written with `sdk.conversation.record_message`. Tool calls,
tool results and reasoning are stream-only. Set `SYNAP_AUTO_LISTEN=false` (or
call `set_auto_listen(sdk, False)` from `synap_integrations_common`) to keep
turns on REST.

CAMEL's own summary (`[CONTEXT_SUMMARY] ...`, and the user message it rewrites)
is never written as a turn: the real turns are already on Synap.

> Previously, turns were ingested as a transcript document
> (`memories.create`). That left no conversation on the server, so compaction and
> history search had nothing to work on. It is replaced, not kept alongside: the
> server promotes recorded messages to long-term memory, so both would extract
> the same text twice.

```python
sdk = MaximemSynapSDK(api_key="sk-...")

memory = SynapAgentMemory(
    sdk, user_id="alice", customer_id="acme",
    conversation_id="conv_abc",   # defaults to camel-{agent_id}
)
```

Reporting never raises into `ChatAgent.step` and never changes what the agent
answers.

## Error policy

- **Reads** (`search_memory`, memory recall) degrade — a Synap blip returns no recall
  and the turn continues.
- **The turn write** inside the agent loop is best-effort by default
  (`on_error="fallback"`) so a transient outage never discards a model response; set
  `on_error="raise"` to raise `SynapIntegrationError` when a turn was not written.
- **The explicit `store_memory` tool** raises `SynapIntegrationError` on failure.

Requires Python 3.11+ and `camel-ai>=0.2.90`.
