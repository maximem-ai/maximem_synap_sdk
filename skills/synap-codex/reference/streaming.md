# The live stream — the third operation

> **Read this next:** [gRPC vs REST](https://docs.maximem.ai/concepts/grpc-vs-rest.md); [Traces and tool calls](https://docs.maximem.ai/capture/traces-and-tool-calls.md); [Real-time anticipation](https://docs.maximem.ai/concepts/real-time-anticipation.md); [Event delivery](https://docs.maximem.ai/concepts/event-delivery.md); [Server pattern](https://docs.maximem.ai/patterns/real-time-anticipation-server.md); [listen](https://docs.maximem.ai/sdk-reference/instance/listen.md); [send-message](https://docs.maximem.ai/sdk-reference/instance/send-message.md); [record-thinking](https://docs.maximem.ai/sdk-reference/instance/record-thinking.md); [record-tool-call](https://docs.maximem.ai/sdk-reference/instance/record-tool-call.md); [record-tool-result](https://docs.maximem.ai/sdk-reference/instance/record-tool-result.md); [end-session](https://docs.maximem.ai/sdk-reference/instance/end-session.md); [stop-listening](https://docs.maximem.ai/sdk-reference/instance/stop-listening.md). **Checkpoint:** open the method's page before writing each call, re-read it after and compare every parameter, scope rule and error; if the docs and this file disagree, the docs win and tell the human.

**Read this before you finish any agent integration.** Ingestion and retrieval
are the two operations every guide in this skill starts with. There is a third,
and skipping it is the single most common way a Synap integration ends up
working on paper and doing nothing useful in production.

## What it is

`sdk.instance.listen()` opens a long-lived gRPC stream to Synap. Your agent
reports what it is doing as it happens: the user spoke, the agent thought, it
called a tool, the tool answered, the agent replied.

Synap uses that to **anticipate**. It predicts what the next turn will need and
pushes the context before the user asks, so `fetch()` resolves from a local
cache in about a millisecond instead of a network round trip. Reported turns
also become long-term memory on their own, with no `memories.create()` call.

## Why the implicit path is the default

The stream is a pipe your code writes into. It is not a tap on the agent. Synap
sees exactly what you report and nothing else, so an unreported event does not
exist.

Three specific consequences, in the order they bite:

1. **No `assistant_message` means no anticipation at all.** That is the event
   the prediction runs on. An integration reporting user turns and tool calls
   but never the reply gets zero prefetching, and looks healthy from outside.
2. **No stream means every retrieval is cold.** `fetch()` still works. It just
   pays the full network and ranking cost on every turn, forever.
3. **No stream means your code records each turn itself.** Turns still become
   memory if you send them with `conversation.record_message` (one request per
   turn) or `memories.create()`; the stream is what removes that work.

Where the app can run it, the stream is the default (`integration-depth.md`) and wiring
the whole of it is part of the integration, not an enhancement to suggest later, unless the
human explicitly says no.

## Which hosts can run it

The live stream (the implicit path) is the default for any app that can keep a process
running for the length of a conversation: a server, worker, container, voice agent, or a
desktop or CLI agent that stays open. The path choice and the wording are in
`integration-depth.md`. The technical facts, all from the docs:

- **The stream needs a long-lived process** that owns it (`listen()` at startup,
  `stop_listening()` at shutdown). A serverless function or request handler that exits
  after one response (Lambda, Vercel functions, Cloud Run scaled to zero), a short-lived
  worker or job, a script or a backfill is opened and torn down on every request with no
  time for a prepared bundle to arrive (grpc-vs-rest, "Which to choose").
- **On such a host the implicit path is not available.** Say that in one plain sentence,
  build the explicit SDK path (`conversation.record_message` per turn, which stores and
  compacts turns into memory the same way, plus a normal `fetch`) with the production
  rules, and add one neutral sentence that if the app ever runs a long-lived process the
  implicit path becomes available. Do not pitch re-architecting and do not invent a
  worker or proxy design the docs do not describe: prepared bundles go only to the streams
  that sent events for that conversation and live in the SDK's own process.
- **Edge runtimes and browsers cannot open the stream.** Importing the JS SDK is safe
  there, but `listen()` throws: grpc-js needs `node:http2`. Stream from a Node runtime.
- **Batch or offline jobs.** Nothing is waiting for a next turn. Ingest directly
  (`ingest_transcript`, `memories.create`).

If the human explicitly says no on a host that could run it, record the choice and the
one-sentence consequence, build the explicit path, and say so in the final report.

## The implementation guides (live docs, plain markdown)

Python and JavaScript/TypeScript share the same pages (each shows both languages in tabs).
Open the page before writing the call and re-read it after.

| Guide | URL |
|---|---|
| What the stream does and does not do | https://docs.maximem.ai/concepts/real-time-anticipation.md |
| gRPC vs REST: which methods use which, costs, hosts | https://docs.maximem.ai/concepts/grpc-vs-rest.md |
| Event delivery: ids, acknowledgements, buffering, replay | https://docs.maximem.ai/concepts/event-delivery.md |
| One shared stream for a multi-tenant server, health check | https://docs.maximem.ai/patterns/real-time-anticipation-server.md |
| Traces: reasoning, tool calls, tool results | https://docs.maximem.ai/capture/traces-and-tool-calls.md |
| `instance.listen` | https://docs.maximem.ai/sdk-reference/instance/listen.md |
| `instance.send_message` (user and assistant turns) | https://docs.maximem.ai/sdk-reference/instance/send-message.md |
| `instance.record_thinking` | https://docs.maximem.ai/sdk-reference/instance/record-thinking.md |
| `instance.record_tool_call` | https://docs.maximem.ai/sdk-reference/instance/record-tool-call.md |
| `instance.record_tool_result` | https://docs.maximem.ai/sdk-reference/instance/record-tool-result.md |
| `instance.end_session` | https://docs.maximem.ai/sdk-reference/instance/end-session.md |
| `instance.stop_listening` | https://docs.maximem.ai/sdk-reference/instance/stop-listening.md |
| Agent integration walkthrough (startup, loop, shutdown) | https://docs.maximem.ai/setup/first-integration.md |
| Production checklist and graceful degradation | https://docs.maximem.ai/guides/production-checklist.md and https://docs.maximem.ai/patterns/graceful-degradation.md |
| Errors | https://docs.maximem.ai/sdk-reference/errors.md |
| TypeScript framework pages that drive the stream for you | https://docs.maximem.ai/integrations/vercel-ai-sdk.md, https://docs.maximem.ai/integrations/mastra.md, https://docs.maximem.ai/integrations/eve.md |

There are no separate JavaScript-only stream pages. The Python-only detail (for example
`ListeningNotActiveError`) and the JS spellings (`options` objects, snake_case on
`sdk.instance`) are both on the pages above.

## The five events

| What happened | Call | Cost of skipping it |
|---|---|---|
| The user spoke | `send_message(event_type="user_message")` | Nothing from this conversation is recorded at all. |
| The agent reasoned | `record_thinking(...)` | Prediction sees the edges of the turn and not the intent inside it. |
| It called a tool | `record_tool_call(...)` | Synap cannot see which tool and arguments the agent is about to rely on. |
| The tool answered | `record_tool_result(...)` | The result is the strongest clue to the next question, and it is missing. |
| The agent replied | `send_message(event_type="assistant_message")` | **Anticipation never runs.** |

If you can only wire one event beyond the user's turn, wire
`assistant_message`.

## Where each call goes (Python)

```python
# once, at process start — after sdk.initialize()
await sdk.instance.listen()

# ── per turn, inside your request handler ──────────────────────────────
# 1. the user's turn, BEFORE you retrieve
await sdk.instance.send_message(
    content=user_text, role="user", event_type="user_message",
    conversation_id=conv_id, user_id=user_id, customer_id=customer_id,
)

context = await sdk.fetch(...)          # served from cache if anticipated

# 2. each reasoning step, as the agent produces it
await sdk.instance.record_thinking(
    content=thought,
    conversation_id=conv_id, user_id=user_id, customer_id=customer_id,
)

# 3. and 4. around every tool call
await sdk.instance.record_tool_call(
    tool_name, tool_args, tool_call_id=call_id,
    conversation_id=conv_id, user_id=user_id, customer_id=customer_id,
)
result = await run_tool(...)
await sdk.instance.record_tool_result(
    result, tool_name=tool_name, tool_call_id=call_id,
    conversation_id=conv_id, user_id=user_id, customer_id=customer_id,
)

# 5. the reply, AFTER the agent produces it
await sdk.instance.send_message(
    content=reply, role="assistant", event_type="assistant_message",
    conversation_id=conv_id, user_id=user_id, customer_id=customer_id,
)

# when one conversation ends and the process keeps running
await sdk.instance.end_session(conv_id)

# once, at process shutdown — before sdk.shutdown()
await sdk.instance.stop_listening()
```

No session call appears in the per-turn loop. The SDK opens a session on the
first event of a conversation and closes it on `stop_listening()`.
`end_session` exists for a conversation that ends while the process lives on: a
voice call hangs up, a chat window closes.

## TypeScript

Same method names, snake_case, options object:

```typescript
await sdk.instance.listen();

await sdk.instance.send_message({
  content: userText, role: 'user', event_type: 'user_message',
  conversation_id: convId, user_id: userId, customer_id: customerId,
});
await sdk.instance.record_thinking({ content: thought, conversation_id: convId, user_id: userId });
await sdk.instance.record_tool_call({ tool_name: name, tool_call_id: callId, conversation_id: convId, user_id: userId });
await sdk.instance.record_tool_result({ result, tool_name: name, tool_call_id: callId, conversation_id: convId, user_id: userId });
await sdk.instance.send_message({
  content: reply, role: 'assistant', event_type: 'assistant_message',
  conversation_id: convId, user_id: userId, customer_id: customerId,
});

await sdk.instance.end_session(convId);
await sdk.instance.stop_listening();
```

## Rules that fail silently if you break them

These produce no error. The data is simply wrong or gone.

- **Never set `role` by hand for a tool or reasoning event.** Use
  `record_tool_call` / `record_tool_result` / `record_thinking`. The server
  reads `event_type` first, but a mismatched pair used to be filed as the
  assistant's reply, and hand-rolled `send_message(event_type="tool_call",
  role="assistant")` is how that happens.
- **Every event needs `conversation_id` and `user_id`.** An event missing
  either is discarded server-side with no client-visible error. On B2B add
  `customer_id`; on B2C a `customer_id` is rejected.
- **`conversation_id` must be a UUID**, reused across the whole conversation.
- **One stream per process, not one per user session.** Stream quotas are per
  instance; one per session exhausts them under real concurrency.
- **Never stream a turn and also `memories.create()` the same text.** It is
  extracted twice and costs twice.

## Check your own work

An integration is not finished until every one of these is true. Check them
before reporting back to the user.

- [ ] `listen()` is called once at startup, `stop_listening()` on shutdown.
- [ ] Every turn reports a `user_message` **and** an `assistant_message`.
- [ ] Tool calls and results are reported, sharing a `tool_call_id`.
- [ ] Reasoning is reported wherever the framework exposes it.
- [ ] Every event carries `conversation_id` and `user_id` (plus `customer_id` on B2B).
- [ ] No hand-set `role` on a tool or reasoning event.
- [ ] `end_session` is called when a conversation ends and the process lives on.
- [ ] `python scripts/verify_synap.py` passes.

## Frameworks that do it for you

Five packages drive the stream from the framework's own hooks. Use them instead
of writing the loop by hand. Each one is silent unless `listen()` is running.

| Framework | Reported for you | Still yours |
|---|---|---|
| OpenAI Agents | Tool calls, tool results, model reasoning, the reply | The user's turn, before the run |
| Google ADK | Tool calls, tool results, the reply | The user's turn, before the run |
| Claude Agent SDK | The user's turn, tool calls, tool results, session end | The reply: no hook carries the final assistant text, so call `report_assistant_turn` when your run returns |
| Vercel AI SDK | Turns, tool calls, reasoning, from the model middleware | Nothing |
| Strands Agents | Turns and tool intent | Tool results and reasoning |

Every other framework, and a custom stack, uses the loop above. It is plain SDK
calls and composes with anything.

## Full reference

`https://docs.maximem.ai/setup/agent-integration` and
`https://docs.maximem.ai/concepts/real-time-anticipation`.
