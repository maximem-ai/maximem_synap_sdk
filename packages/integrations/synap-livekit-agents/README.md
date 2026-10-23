# synap-livekit-agents

Synap integration for [LiveKit Agents](https://docs.livekit.io/agents) — preload long-term memory before a realtime call, record every committed turn back to Synap, and expose Synap search/store as LLM-callable function tools.

## Install

```bash
pip install synap-livekit-agents
```

Requires `livekit-agents>=1.0`, `maximem-synap>=0.2.0`.

## Quickstart

```python
from livekit.agents import Agent, AgentSession, ChatContext
from maximem_synap import MaximemSynapSDK
from synap_livekit_agents import (
    preload_synap_context,
    attach_synap_recording,
    synap_search_tool,
    synap_store_tool,
)

sdk = MaximemSynapSDK(api_key="sk-...")

async def entrypoint(ctx):
    chat_ctx = ChatContext()
    await preload_synap_context(
        chat_ctx, sdk, user_id="alice", customer_id="acme",
    )

    agent = Agent(
        instructions="You are a helpful assistant.",
        chat_ctx=chat_ctx,
        tools=[
            synap_search_tool(sdk, user_id="alice", customer_id="acme"),
            synap_store_tool(sdk, user_id="alice", customer_id="acme"),
        ],
    )
    session = AgentSession(...)
    conversation_id = attach_synap_recording(
        session, sdk, user_id="alice", customer_id="acme",
    )
    await session.start(agent=agent, room=ctx.room)
```

## Scope

- **`preload_synap_context(chat_ctx, sdk, *, user_id, ...)`** — async helper. Fetches user-scoped context from Synap and prepends a single `system`-role `ChatMessage` to the `ChatContext` so the agent starts with long-term memory in scope. Read failures degrade silently — a Synap blip must never prevent a call from starting.
- **`attach_synap_recording(session, sdk, *, user_id, ...)`** — reports the whole turn, not just its two ends. Wires four listeners onto one `AgentSession` and returns the `conversation_id` used for this call (auto-generated when absent). Callbacks never raise — write failures are logged and swallowed, consistent with LiveKit's sync-event contract.

  | LiveKit event | Reported as |
  | --- | --- |
  | `conversation_item_added` | the user or assistant turn |
  | `tool_execution_updated` (`ToolCallStarted`) | the tool call, at dispatch |
  | `function_tools_executed` | the tool result, and the call if the dispatch event never arrived |
  | `close` | end of the Synap session |

  The same call also installs Synap's compaction: once Synap has compacted the conversation, the agent's LLM is sent the compaction plus the turns after it instead of every earlier turn, by wrapping `Agent.llm_node` (the framework hands it a copy, so the agent's own `chat_ctx` is never rewritten). Before the first compaction, or if Synap does not answer within 1.5s, the history goes through untouched. A `RealtimeModel` session is not covered: the provider holds that history and `llm_node` is never called. `install_synap_history(agent, sdk, conversation_id=...)` does the same for an agent without a session attached.

  A tool call and its result share LiveKit's own `call_id`, so the anticipation agent reads them as one exchange. On a voice call the tool call is reported the moment the tool is *dispatched*, not when the batch finishes: the gap between a caller's question and the answer is dead air, and that gap is the whole window anticipation has to work in.
- **`synap_search_tool(sdk, *, user_id, ...)`** / **`synap_store_tool(sdk, *, user_id, ...)`** — factories that return `FunctionTool` instances registerable via `Agent(tools=[...])`. The LLM can invoke `synap_search(query)` to retrieve formatted context, or `synap_store(content, category)` to write a new memory.

## The gRPC stream is the default; REST only as a fallback, never both

The first turn, tool call or tool result opens the stream (`sdk.instance.listen()`) if you did not, and everything after rides it: turns go out as `user_message` / `assistant_message` events and `sdk.conversation.record_message` is **not** called, because the server persists those events itself and doing both would store the turn twice. When the stream cannot be opened or lives on another event loop, turns go over REST. Set `SYNAP_AUTO_LISTEN=false` (or `set_auto_listen(sdk, False)` from `synap_integrations_common`) to stay on REST.

Tool calls, tool results and reasoning are anticipation hints. The server does not persist them, so they have no REST fallback and are simply silent when no stream is open.

`preload_synap_st` / `refresh_synap_st` never send a turn twice: once Synap has compacted they splice the compaction into the context instead of stacking a block beside the turns, and before that the block carries only the turns the context does not already hold.

## Error policy

- **Reads** (`preload_synap_context`, `synap_search_tool`) degrade gracefully. SDK failures log at `ERROR`, return an empty / "no relevant memory" placeholder, and the call proceeds.
- **Writes** (`attach_synap_recording`, `synap_store_tool`) surface `SynapIntegrationError` via `wrap_sdk_errors_async`. The recording callback catches+logs (can't bubble an exception out of a sync LiveKit event emission without tearing down the session); the store tool lets the `SynapIntegrationError` propagate to the LLM runtime so the model sees a tool failure.

## Constructor args

All helpers accept:

- `sdk`: a configured `MaximemSynapSDK`
- `user_id`: required — Synap memory is user-scoped
- `customer_id`: B2B instances only, where it is required. Not accepted on a B2C instance, which rejects it with HTTP 400.
- `conversation_id` *(recording only)*: optional explicit id; auto-generated when absent
- `mode` *(read helpers only)*: `"accurate"` or `"fast"`, passed through to `sdk.fetch`
- `max_results` *(read helpers only)*: cap per fetch
