# Choose the integration path (the implicit path is the default)

> **Read this next:** [gRPC vs REST](https://docs.maximem.ai/concepts/grpc-vs-rest.md); [Real-time anticipation](https://docs.maximem.ai/concepts/real-time-anticipation.md); [Event delivery](https://docs.maximem.ai/concepts/event-delivery.md); [Real-time anticipation in a server](https://docs.maximem.ai/patterns/real-time-anticipation-server.md); [Traces and tool calls](https://docs.maximem.ai/capture/traces-and-tool-calls.md); [instance.listen](https://docs.maximem.ai/sdk-reference/instance/listen.md); [instance.send_message](https://docs.maximem.ai/sdk-reference/instance/send-message.md); [conversation.record_message](https://docs.maximem.ai/sdk-reference/conversation/record-message.md); [Retrieval modes](https://docs.maximem.ai/concepts/retrieval-modes.md); [PII protection](https://docs.maximem.ai/guides/pii-protection.md). The live docs win over this file.

This is the detail behind the "integration path" step in `SKILL.md`. Rule one: **always use the SDK, never raw HTTP calls.** Both paths below are SDK paths. They differ in who does the work:

- **Implicit path (the live stream):** the app opens the SDK's stream (`listen`) and reports what its agent does (`send_message`, `record_thinking`, `record_tool_call`, `record_tool_result`, `end_session`). Synap does the work as the events flow.
- **Explicit path:** the app calls the SDK's store and fetch methods itself on each call (`conversation.record_message`, `memories.create`, `fetch`). The app does the work; each call returns an answer, and it runs in any runtime.

Neither "does not work". For an app that CAN run a long-lived process, the implicit path is the default because it is where Synap's best results come from. Do not present it as optional or as one of equal choices.

## 1. What the implicit path gives (every line backed by the docs or the code)

| Benefit | Evidence |
|---|---|
| **Context prepared before the user asks.** The reply you report ends a turn; Synap predicts the next need and pushes a bundle into the app's own memory, so the next read is often answered locally in about a millisecond. A borderline match waits up to 250 ms for Synap; no match is an ordinary request. A read that sets `query_date` or `max_results_per_query` always goes to Synap. | real-time-anticipation ("What comes back"), grpc-vs-rest ("How a context fetch uses the cache") |
| **Conversation turns are recorded and become long-term memory with no write calls.** Compaction promotes them (3,000 tokens, 10 messages, or 5 minutes idle). | real-time-anticipation ("What the stream does to memory") |
| **Reasoning, tool calls and tool results sharpen what Synap prepares.** They travel only on the stream; there is no HTTP route for them. They are short-lived hints (the 80 most recent events, gone 30 minutes after the last), never stored as memory. Matters for agents that use tools or plan across steps; an agent that answers in one model call is served well by its turns alone. | traces-and-tool-calls |
| **Delivery that survives a dropped connection.** Events are buffered (100 events or 5 minutes), replayed with their original ids and journalled to disk. | event-delivery |
| **Sessions are managed for you.** The SDK opens a session on the first event for a conversation, which warms the first turn; `end_session` finalises the last turn sooner than waiting for quiet. | send-message, end-session |
| **A learning loop.** After each fetch the SDK reports what was asked and whether the cache answered, which is how Synap scores its predictions. | real-time-anticipation ("What the SDK sends on its own") |

### Claims the docs do NOT support (do not make them)

- **"Without the stream there is no automatic memory."** False. Turns sent with `conversation.record_message` are stored, compacted and turned into long-term memory the same way as streamed turns (grpc-vs-rest). On both paths the server records the turn through the same tracker, and compaction triggers fire from it (server code).
- **"Compaction depends on the stream."** No. Token, message-count and idle triggers apply to any recorded conversation. The only stream-specific thing in the public docs is the session end finalising the last turn sooner.
- **"Short-term context management needs the stream."** No. `get_context_for_prompt`, `compact` and the compaction status calls are REST, and the SDK's local turn history is fed by `record_message`, `record_messages_batch` and the stream alike (SDK source).
- **"The explicit path is broken or second-class."** No. It is complete on its own; it gives no prepared context and no traces.
- **"Run a small worker next to the function to get the benefits."** Not documented. Prepared bundles go only to the streams that sent events for that conversation and live in the SDK's process, so a separate worker does not feed a separate function.
- **The ingestion `mode` (`fast` or `long-range`)** is a no-op today per the docs; never offer it.

## 2. Which hosts can run it

From grpc-vs-rest ("Which to choose"): a long-running agent (chat server, voice agent, worker living across many turns, container, desktop or CLI agent that stays open) can hold the stream. A serverless function or request handler that exits after one response, a short-lived worker or job, a script or backfill, an edge runtime or a browser cannot: the stream would be opened and torn down on every request with no time for a bundle to arrive, and `listen()` throws on Edge and in browsers.

**On such a host:** say it in one plain sentence ("the live stream needs a long-lived process, and this host does not run one, so it is not available here"), cite grpc-vs-rest, build the explicit path with the production rules (`integration-checklist.md`), and add one neutral sentence: if the app ever runs a long-lived process, the implicit path becomes available. Do NOT pitch re-architecting the deployment, do not present the stream as the default for this host, and do not invent a worker or proxy design the docs do not describe.

**Before offering or wiring the stream:** check installed packages for conflicts with the SDK (for example the protobuf version range against other Google or gRPC libraries); `verify_synap.py` must open the stream, and a stream warning is a blocking dependency issue on the implicit path. Prepared context lives in the process: with several replicas a conversation's next turn may land on another instance. `POST /v1/events/batch` exists only in unreleased docs and in no released SDK; do not offer it.

## 3. The recommendation and the confirmation (a default, not a menu)

Form the recommendation from the repo first (app kind, tool or reasoning loop, process lifetime, deployment target, what the human asked for), and be truthful about the gain for THIS app: a short chat with one model call and no tools gains less than a tool-using agent. Reuse the modality and turn speed already gathered for the use-case Interaction Pattern; never ask twice. Then, for an app that can run it, say what it gives (section 1, in plain words, no class names) and ask as a confirmation with default yes:

> "I will wire the live stream so you get context prepared before your agent asks, conversations recorded as memory without write calls, and your agent's reasoning and tool use feeding better predictions. It needs your process to stay running, which it does here. Tell me if you do not want it."

- **Implement it fully** unless the human EXPLICITLY says no. "I do not know", silence and "whatever you think" mean implement.
- **If the human says no:** record the choice and the one-sentence consequence (no prepared context, no reasoning or tool signals, your code makes each write and read call itself), build the explicit path, and put it in the final report.

## 4. Implementing the implicit path fully, and saying so

Send ALL of these for every turn, with the same `conversation_id`, `user_id` (and `customer_id` on B2B only):

| Event | Call | Plain words | Lost without it |
|---|---|---|---|
| `user_message` | `send_message(role="user", event_type="user_message")` | the user spoke | the turn is not recorded |
| `agent_thinking` | `record_thinking(...)` | why the agent chose what it did | Synap sees only the edges of a turn |
| `tool_call` | `record_tool_call(...)` | which tool, with which arguments | Synap cannot see what the agent is about to need |
| `tool_result` | `record_tool_result(...)` | what the tool returned | the strongest hint of the next question is missing |
| `assistant_message` | `send_message(role="assistant", event_type="assistant_message")` | the reply | anticipation never runs for the next turn |

Also: `listen()` once at startup (one stream per process, never one per user), `end_session(conversation_id)` when a conversation ends while the process lives on, `stop_listening()` on shutdown, and the docs checkpoint loop for every call. Send reasoning only when the model really returns it; never invent it. Send only what is needed from a tool result. Exact calls and rules: `streaming.md`.

### What each event sends (from the SDK source)

Every event is one `ConversationEvent` with these fields: `event_type`, `content`, `role`, `conversation_id`, `user_id`, `customer_id`, `session_id`, `metadata` (a string map), `timestamp_ms`, `sent_at_ms`, `event_id` (minted by the SDK), `tool_name`, `tool_args_json`, `tool_result_json`, `tool_call_id`, `search_queries`, `context_types`. What fills them:

| Event | `role` | `content` | Other fields filled |
|---|---|---|---|
| `user_message` | `user` | the user's text | ids, `session_id`, `metadata` |
| `agent_thinking` | empty | one reasoning step | ids, `session_id`, `metadata` with `step_index` and `thought_type` if given |
| `tool_call` | `assistant` | empty | `tool_name`, `tool_args_json`, `tool_call_id`, ids |
| `tool_result` | `tool` | empty | `tool_result_json`, `tool_name`, `tool_call_id`, ids |
| `assistant_message` | `assistant` | the reply | ids, `session_id`, `metadata` |

Added by the SDK: `session_id` (the session it opens on the first event of a conversation, only when both `conversation_id` and `user_id` are given), the agent's model and context window in `metadata` when configured, a `session_start`/`session_end` control message per conversation, and `context_fetch`, `context_used`, `context_assembled` events after each fetch (ids and counts, no prompt content). `record_tool_result` also keeps a copy of the result in a local archive on the machine.

One hint is NOT in the five: a `context_request` event (via `send_message`) with `search_queries` and `context_types` tells Synap which retrieval the agent plans next. Send it when the agent knows its next lookup; `record_tool_call` has no parameter for those two fields.

Persistence: only `user_message` and `assistant_message` become conversation history and, at compaction, long-term memory. Reasoning, tool calls and tool results are short-lived hints (the 80 most recent events, gone 30 minutes after the last one).

## 5. Report the choice

`onboard report integrating in_progress --detail "path=implicit"` (or `path=explicit-no` when the human said no, `path=explicit-host` when the host cannot run the stream); keep it short and free of personal data. Repeat it in the final report: "stream ON by default", or "human said no because ...", or "host cannot run it".
