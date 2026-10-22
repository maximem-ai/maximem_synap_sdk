# Choose the integration depth with the human

> **Read this next:** [gRPC vs REST](https://docs.maximem.ai/concepts/grpc-vs-rest.md); [Real-time anticipation](https://docs.maximem.ai/concepts/real-time-anticipation.md); [Event delivery](https://docs.maximem.ai/concepts/event-delivery.md); [Real-time anticipation in a server](https://docs.maximem.ai/patterns/real-time-anticipation-server.md); [Traces and tool calls](https://docs.maximem.ai/capture/traces-and-tool-calls.md); [instance.send_message](https://docs.maximem.ai/sdk-reference/instance/send-message.md); [conversation.record_message](https://docs.maximem.ai/sdk-reference/conversation/record-message.md); [Retrieval modes](https://docs.maximem.ai/concepts/retrieval-modes.md); [PII protection](https://docs.maximem.ai/guides/pii-protection.md). The live docs win over this file.

The human decides how deep the integration goes. You never decide alone and you never silently pick the basic one. This file is the detail behind the "integration depth" step in `SKILL.md`: what to present, how to form a recommendation, what each choice really gives, and what to implement.

## 0. Value comes first, depth is the last choice

The depth choice is about transport only. It comes after the value chain in `integration-checklist.md`: a stable identity, then a read wired into the model prompt, then write mechanics. Full with a per-session id still gives no cross-visit memory, and a stream that never feeds a prompt delivers nothing.

## 1. The capability menu (plain words, benefits only)

Present this short list, in the human's language, with no class or method names. Every line is backed by the docs named in brackets.

- **Memory that is ready before the user asks.** While the agent runs, Synap predicts what the next turn will need and prepares it, so the next read is often answered from the app's own memory in about a millisecond instead of a network call. If nothing prepared matches, the read just takes the normal path, so nothing breaks. Needs the live stream. [real-time-anticipation, grpc-vs-rest]
- **Conversations become memory without write calls.** Turns the agent reports are kept and turned into long-term memory automatically (a turn is recallable once its conversation compacts, up to about five minutes later; use an explicit write for anything that must be recallable at once). Works on the live stream, and also by sending one HTTP request per turn. [real-time-anticipation, record-message]
- **Better prediction from the agent's reasoning, tool calls and tool results.** Synap then sees why the agent did what it did, not only the words, and can prepare for the likely next question. Needs the live stream: there is no HTTP route for these. Reasoning and tool data are used to anticipate; they are not stored as memory. [traces-and-tool-calls]
- **Faster or deeper recall.** Each read can use a fast mode (lower latency, direct matches) or an accurate mode (follows relationships and re-ranks, higher latency). Start fast; use accurate for the valuable questions. [retrieval-modes]
- **When a write becomes readable.** Explicit memory writes are processed straight away in the background (extraction is asynchronous, so allow a short delay); conversation turns follow compaction. (The ingestion `mode` option does nothing today per the docs, so do not offer it as a choice.) [real-time-anticipation, retrieval-modes]
- **Session handling.** Synap tracks a session per conversation; telling it when a conversation ends lets it finalize the last turn sooner. [end-session]
- **User profile and start-of-conversation read.** Attributes the human defines (for example preferred language) plus summaries of recent conversations, read in one fast call at the start of a conversation. Needs a stable id. [user-profile, context-fetch]
- **Outcome at completion.** One `ingest_transcript` at the natural end of a conversation carries the app's outcome and real start and end times. [ingest-transcript]
- **Privacy controls.** Personal data can be protected by the instance's PII policy, and anything sent as a tool result is subject to it. [pii-protection, record-tool-result]

## 2. Form a recommendation BEFORE you ask

From the repo, work out and write down (for yourself, and to show as reasons):

| Signal | Why it matters |
|---|---|
| What kind of app: chat backend, voice agent, background worker, desktop or CLI agent, serverless function | The stream needs a process that stays up for the length of a conversation |
| Does it run a loop with tools or reasoning | Only then do the tool and reasoning events carry anything |
| How long the process lives, and who calls it | A long-lived server or worker can own one stream; a process that exits after one response cannot |
| The deployment target (a Vercel or Lambda function, Cloud Run scaling to zero, an edge runtime, a long-running container, a desktop app) | Edge runtimes and browsers cannot open the stream at all |
| What the human asked for | An explicit wish for "anticipation", "voice latency" or "just store facts" changes the recommendation |

The modality and turn speed you learn here are the same facts the use-case's Interaction Pattern section needs. Ask them once, reuse the answer in both places, and never ask the human twice.

Recommend **Full** for a long-running agent that holds a connection across many turns (chat server, voice agent, worker, desktop agent); recommend **Basic** or the HTTP route for a function that exits after one response, a short-lived worker, a script or a backfill. These follow the docs' "Which to choose" table in gRPC vs REST.

**Be truthful about the gain for THIS app.** The stream's reasoning and tool events only matter for agents that use tools or plan across steps; the docs say an agent that answers in one model call with no tools is served well by its turns alone. For a short chat with one model call and no tools the extra gain is small (a read that was a request becomes a local hit when a prepared bundle matches), and it needs a long-running host. Say that in the recommendation instead of recommending Full by reflex. Always explain the choice to the human.

**Pre-flight for Full:** check installed packages for conflicts with the SDK before offering or wiring the stream (for example the protobuf version range against other Google or gRPC libraries); `verify_synap.py` must open the stream, and a stream warning is a blocking dependency issue when the human chose Full. The prepared context lives in the process: with several replicas, a conversation's next turn may land on another instance. `POST /v1/events/batch` exists only in unreleased docs and in no released SDK; do not offer it.

## 3. Ask (one question, recommendation first, two or three options)

Wording and examples: `onboarding-ask-rules.md`. Offer exactly these:

1. **Basic.** Synap stores and recalls memories when your code asks. Works anywhere, including serverless. What you do not get: memory prepared before the user asks, and Synap seeing the agent's reasoning and tool use.
2. **Full.** The live stream: Synap also prepares what your agent will need next and records conversations automatically. It needs a process that stays running. Say "I recommend X because Y" and name the repo evidence.
3. **Only for a serverless, edge or short-lived target: the per-turn HTTP route.** Each turn is recorded with one HTTP request (`conversation.record_message`) and your code reads memory with a normal fetch. What it gives: turns are stored and become long-term memory, and it works in any runtime. What it does NOT give: prepared context (every read is a request) and the reasoning/tool traces (no HTTP route exists for them).

The human decides. If they pick Basic or the HTTP route, say in one sentence what they will not get, then record the choice (below).

### What is NOT confirmed for serverless apps

The docs do not describe any way for a function to get the stream's benefits. In particular: they say a conversation's prepared bundles are delivered only to the streams that sent events for that conversation, and the SDK keeps the prepared context in its own process memory. So "run a small always-on worker next to the function" is NOT a documented route: a worker that holds the stream would not hand prepared context to a separate function. The one documented way to get Full is to run the agent loop itself (the part that sends events and reads) in a long-running process. Offer that as a larger change (a small always-on service), say it is a change to how they deploy, and do not promise more.

## 4. If the human picks Full: implement it fully, and say so

Send ALL of these for every turn, with the same `conversation_id`, `user_id` (and `customer_id` on B2B only):

| Event | Call | Plain words | Lost without it |
|---|---|---|---|
| `user_message` | `send_message(role="user", event_type="user_message")` | the user spoke | the turn is not recorded |
| `agent_thinking` | `record_thinking(...)` | why the agent chose what it did | Synap sees only the edges of a turn, not the intent |
| `tool_call` | `record_tool_call(...)` | which tool, with which arguments | Synap cannot see what the agent is about to need |
| `tool_result` | `record_tool_result(...)` | what the tool returned | the strongest hint of the next question is missing |
| `assistant_message` | `send_message(role="assistant", event_type="assistant_message")` | the reply | anticipation never runs for the next turn |

Also: `listen()` once at startup (one stream per process, never one per user), a session closed with `end_session(conversation_id)` when a conversation ends while the process lives on, `stop_listening()` on shutdown, and the docs checkpoint loop for every call. Send reasoning only when the model really returns it; never invent it. Send only what is needed from a tool result, not the whole record. Exact calls and rules: `streaming.md`.

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

If a step exists use it; none is dedicated, so report with the existing integration step and keep the detail short and free of personal data: `onboard report integrating in_progress --detail "depth=full (stream), reason: long-running chat server"` (or `depth=basic`, `depth=http-turns`). Always repeat the choice and the reasons in the final announcement, including what a Basic choice leaves out.
