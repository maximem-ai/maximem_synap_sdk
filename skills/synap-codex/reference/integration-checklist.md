# Integration checklist: what "done" means

> **Read this next:** [Context fetch](https://docs.maximem.ai/sdk/context-fetch.md); [User profile](https://docs.maximem.ai/sdk/user-profile.md); [Conversation ingest_transcript](https://docs.maximem.ai/sdk-reference/conversation/ingest-transcript.md); [record_messages_batch](https://docs.maximem.ai/sdk-reference/conversation/record-messages-batch.md); [PII protection](https://docs.maximem.ai/guides/pii-protection.md); [Custom field types](https://docs.maximem.ai/guides/custom-field-types.md); [Erasure](https://docs.maximem.ai/guides/erasure.md); [Production checklist](https://docs.maximem.ai/guides/production-checklist.md); [Graceful degradation](https://docs.maximem.ai/patterns/graceful-degradation.md); [Error handling](https://docs.maximem.ai/sdk/error-handling.md). The live docs win over this file.

Step 7 of `SKILL.md` is finished only when every line below is true or is named as skipped, with the reason, in the final report (`final-report-template.md`). The order matters: **value comes before transport.** Deciding how Synap is reached (Basic, the live stream, per-turn HTTP: `integration-depth.md`) is the last choice, not the first. An app that sends every message to Synap and never reads anything back pays for storage and gets nothing.

## 1. Value before transport

The chain, in order: a stable identity, then a read wired into the model prompt, then write mechanics, then the depth.

1. **Identity.** Use what the human said identifies a returning person (the answer from step 4, reused from the use-case; do not ask again). Never use a per-chat or per-session id as `user_id` unless the human confirmed the app has no persistent identity. If it is per-session, say plainly: Synap will return nothing the app does not already have, there is no returning-person memory, no profile, and no per-person erasure (a deletion request cannot be served). Even the Full stream gives no cross-visit memory with a per-session id. Warn that a device id is shared on shared devices (a second person on the same phone or computer would be shown the first person's memory), so a device id needs a visible opt-in such as "continue my last session?" or a login.
2. **Read path.** Wire recall into the model prompt, and say where (file and function). An integration that only writes is not done. If the human wants write-only, record that and why in the final report; do not leave a read helper defined and never called. Fetch tuning, per docs: `mode` fast on the hot path and accurate for the valuable one-off question, `types`, `max_results`, and `precision_level` ("medium" is faster with less precisely filtered results). Format recalled items into a short block and keep it out of the way of the app's own instructions.
3. **Memory informs, your rules decide.** Recalled memory may shape what the agent asks or says. It must never override the app's own safety, business or policy rules, and the deterministic checks the app already has must not depend on memory. Say this in the code comment where the memory block is added.

## 2. Pick the right write path (one owner per piece of content)

| Situation | Use | Trade-off |
|---|---|---|
| Each turn as it happens, any runtime | `conversation.record_message`, or `record_messages_batch` to send several turns in one request | Turns are stored and compacted into long-term memory; abandoned chats are still stored |
| A short chat with a natural end (a completed form, a closed ticket, a resolved case) | one `conversation.ingest_transcript` at completion, with `analysis` (the app's outcome, for example the decision reached), `started_at` and `ended_at` | One call per chat, extracted once, carries the outcome; chats abandoned before completion are not stored, and later turns of a pushed transcript are not extracted (start a new conversation) |
| Documents and facts that are not conversation turns | `memories.create` | Immediate; never for text you also sent as a turn (extracted twice) |
| A long-running process with the stream | the stream (`integration-depth.md`) | Turns are promoted at compaction, up to about five minutes |

Wire the app's OUTCOME or summary where the app has one (a decision, a result, a resolved status), not just the chat text: the outcome is usually the most valuable fact.

## 3. Write mechanics (serverless and inline apps)

- **Init the SDK once** (module scope per process or container), never inside a request. In a synchronous web framework, hold one SDK on one background event loop; `asyncio.run` per call re-initializes per call, costs seconds, and raises if the route later becomes async.
- **Short timeout** (about 5 seconds for reads and writes; the SDK default is 30 seconds with 3 retries) and **bounded retries**.
- **Never block the user turn if avoidable.** Do the write after the reply is built, in a background task where the platform allows it (check the platform: a serverless function may freeze after it responds).
- **Never swallow a failure silently.** Log it with the correlation id, count it, and keep the app answering.
- **Say what is lost on failure.** A refused write (409 during the build, 429, an outage) is retried later or dropped. State which, per write path, in the final report. A dropped write is a loss the human must know about.
- Safety-relevant endpoints get the strictest version of all of these: memory never adds to their latency or failure surface.

## 4. Use the features that fit (offer as menu items, plain benefits)

- **User profile**: the 3 to 5 attributes the human says matter (for example preferred language), read in one call at the start of a conversation with `context_mode="conversation-summary"`, which also returns summaries of the last few conversations. Needs a stable id.
- **Outcome at completion** via `ingest_transcript` (above).
- **`record_messages_batch`** to halve round trips.
- **Fetch tuning** (above). Compaction only matters when a conversation outgrows the model's window.
- **Erasure by person** exists; it needs a per-person id. When ids are per-person, ask how a deletion request will be served.

## 5. Sensitive data

- If the app redacts before sending, check its redaction against the human's stated never-store list. Test it with realistic samples in the local formats (phone numbers with spaces, landlines, dates of birth, postal codes, ids), and add unit tests. Tell the human which fields pass through (names and street addresses are not detected by Synap, per the PII guide).
- Local identifiers can be covered with custom field types; mention them. Do not leave the policy "as a decision for later" silently.
- The instance's sensitive-data policy is approved by the human in the dashboard and applies going forward only. When the use-case has a never-store list and the policy is not approved, it is a named open item in the final report.
- For health, financial or children's data, ask whether the app's consent screen tells the person that a third party stores their text, and flag it to the human if not.

## 6. Operational

- **OFF-safe and visibly ON.** The integration does nothing and never raises when its env vars are unset, and logs one line at startup saying whether memory is ON or OFF and which instance. Without that line a missing env var is a silent no-op in production.
- **No staging key in production.** List the variables the deployed environment needs (key, instance id, and the base-URL and gRPC variables on a non-production environment) and say that the integration is silently OFF without them. Never copy staging values to production.
- **Pin the versions**: the SDK package and the CLI version you used.
- **Dependency pre-flight before offering or wiring the stream:** check the installed packages for conflicts with the SDK (for example the protobuf version range against other Google or gRPC libraries). A conflict that stops the stream is reported to the human as a dependency issue with options; it is not left behind as a warning.
- **Update `.env.example` and the README run section** with the new variables.
- **Tests**: at least one that mocks Synap failing and shows the app still answers, one for the 409 not-ready path, and one for redaction if the app redacts.
- **Skill files**: the installer adds `.agents/`, `.claude/` and `skills-lock.json`; say so and ask whether to commit or ignore them.

## 7. Verify the real path

`verify_synap.py` proves Synap works for this key; it is not the app. Also:

1. Run `python scripts/check_integration_wiring.py <project dir>` and paste its output. It lists write sites, read sites, dead read helpers and whether the stream is wired, and fails when the project writes but never reads.
2. Drive the app's own endpoint with a sentinel phrase, then recall the sentinel through the app's own read code (after compaction, which can take about five minutes for recorded turns), or state that this was not done.
3. Report whether the recalled text is actually used in a model prompt, and where.

## 8. A known defect is fixed before you report

A regression you caused (added latency, lost writes, an unwired read) is fixed before you report, or you ask the human first. It is not listed as a "decision for later".
