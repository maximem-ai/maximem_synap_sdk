# LiveKit Agents (voice)

> **Read this next:** [This integration's docs page](https://docs.maximem.ai/integrations/livekit-agents.md); [memories.create](https://docs.maximem.ai/sdk-reference/memories/create.md); [context fetch](https://docs.maximem.ai/sdk-reference/context/fetch.md); [listen](https://docs.maximem.ai/sdk-reference/instance/listen.md); [Errors](https://docs.maximem.ai/sdk-reference/errors.md); [Graceful degradation](https://docs.maximem.ai/patterns/graceful-degradation.md). **Checkpoint:** open the method's page before writing each call, re-read it after and compare every parameter, scope rule and error; if the docs and this file disagree, the docs win and tell the human.

`pip install maximem-synap-livekit-agents`

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
        customer_id="acme",   # B2B only; omit on B2C
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
        customer_id="acme",   # B2B only; omit on B2C
    )

    await session.start(agent=agent, room=ctx.room)
```

**Scoping:** `customer_id` is B2B only, and required there. On a B2C instance (`user_context_isolation = "equals_customer"`) pass `user_id` alone: a `customer_id` comes back as HTTP 400. `GET /api/v1/auth/whoami` tells you which mode the instance is in.

## preload_synap_context

Loads the user's long-term memories as system messages **before** the session starts. The LLM sees the user's history from turn one — no tool call latency.

```python
await preload_synap_context(
    chat_ctx=chat_ctx,
    sdk=sdk,
    user_id="alice",
    customer_id="acme",    # B2B only; omit on B2C
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
    customer_id="acme",           # B2B only; omit on B2C
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
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
