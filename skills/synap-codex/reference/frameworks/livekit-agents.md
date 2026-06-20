# LiveKit Agents (voice)

`pip install synap-livekit-agents`

For LiveKit's voice agent framework. Memory in voice contexts has special needs — preload at session start (no time for tool calls mid-utterance) and record turns as they commit.

| Export | Purpose |
| --- | --- |
| `preload_synap_context` | Injects long-term memory into the `ChatContext` before the session starts |
| `attach_synap_recording` | Records every committed turn back to Synap during the session |
| `synap_search_tool` | LLM-callable function tool for on-demand search |
| `synap_store_tool` | LLM-callable function tool for on-demand store |

## Quick start

```python
from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.agents.llm import ChatContext
from synap_livekit_agents import (
    preload_synap_context,
    attach_synap_recording,
    synap_search_tool,
    synap_store_tool,
)

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # 1. Preload long-term memory into the chat context
    chat_ctx = ChatContext()
    await preload_synap_context(
        chat_ctx=chat_ctx,
        sdk=sdk,
        user_id="alice",
        customer_id="acme",
        max_results=8,
    )

    agent = Agent(
        instructions="You are a voice assistant with long-term memory.",
        chat_ctx=chat_ctx,
        tools=[
            synap_search_tool(sdk=sdk, user_id="alice"),
            synap_store_tool(sdk=sdk, user_id="alice"),
        ],
    )

    session = AgentSession(...)

    # 2. Attach recording — every committed turn ingested
    conversation_id = attach_synap_recording(
        session=session,
        sdk=sdk,
        user_id="alice",
        customer_id="acme",
    )

    await session.start(agent=agent, room=ctx.room)
```

## preload_synap_context

Loads the user's long-term memories as system messages **before** the session starts. The LLM sees the user's history from turn one — no tool call latency.

```python
await preload_synap_context(
    chat_ctx=chat_ctx,
    sdk=sdk,
    user_id="alice",
    customer_id="acme",    # optional
    max_results=8,
    mode="fast",           # "fast" or "accurate"
)
```

Failures degrade gracefully — the session starts with empty context rather than raising.

## attach_synap_recording

Subscribes to the `AgentSession`'s turn-commit events and ingests asynchronously. Returns the `conversation_id` for the session.

```python
conversation_id = attach_synap_recording(
    session=session,
    sdk=sdk,
    user_id="alice",
    customer_id="acme",
    conversation_id="call-001",   # optional; auto-generated if omitted
)
```

## Function tools (mid-conversation lookup)

```python
tools = [
    synap_search_tool(sdk=sdk, user_id="alice", max_results=5),
    synap_store_tool(sdk=sdk, user_id="alice"),
]
```

These are `@llm.ai_callable`-decorated functions exposed to the model as function calls.

## Pattern

For voice agents the right pattern is: **preload + record + (optionally) tools**. Preload handles the "what does the agent already know" recall; record handles persistence; tools are for explicit "let me check / let me save that" moments.

## Live doc

`https://docs.maximem.ai/integrations/livekit-agents`

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.2.4 (JS) — verified 2026-06-17. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
