# Core concepts

The minimum mental model needed to write correct Synap code. If you've already read SKILL.md, skim this for the parts you skipped.

## Hierarchy

```
Client (your org)              cli_<hex16>   one per Synap account
  └─ Instance (one agent)      inst_<hex16>  one per agent deployment, has its own MACA + storage
       └─ memories             scoped to one of: user / customer / client / world
```

The SDK talks to one **instance** at a time. Multiple agents = multiple instances = multiple SDK constructions (each with its own `instance_id`).

## B2C or B2B: check this before you write a call

Every instance is one or the other, fixed when it was created (the "User Relationship"
setting in the dashboard). It decides which ids the API accepts, so establish it first:

```bash
curl -H "Authorization: Bearer $SYNAP_API_KEY" \
  https://synap-cloud-prod.maximem.ai/api/v1/auth/whoami
```

The response carries `user_context_isolation`:

| `user_context_isolation` | Mode | What the caller sends |
| --- | --- | --- |
| `equals_customer` | B2C | `user_id` only. A `customer_id` is rejected with HTTP 400. |
| `strict` | B2B | `user_id` **and** `customer_id`. A `user_id` on its own is an error. |

On B2C that holds for every call: ingestion, retrieval, and `record_message` alike. Do not
pass the user id as a customer id to fill the field. `sdk.customer.context.fetch`
(`POST /v1/context/customer/fetch`) is B2B only and is rejected on a B2C instance. The
Python SDK from 0.4.7 raises client-side if you pass a `customer_id` on a B2C instance.

When you don't know the mode, call whoami. Never send both ids to see which one sticks.

## The four scope levels

Memories live at one scope. Retrieval searches narrower-to-broader and merges with narrower-takes-priority on conflicts.

| Scope | Identified by | Visible to | Use for |
| --- | --- | --- | --- |
| **User** | `user_id` alone on B2C; `user_id` + `customer_id` on B2B | only that user | personal prefs, individual history |
| **Customer** | `customer_id` only, B2B instances only | all users in that org | company policies, shared org context |
| **Client** | implicit (no ids) | all users across all customers in your app | product docs, feature announcements |
| **World** | global | every Synap instance | rarely used by app developers; managed by Synap |

**The single most common mistake:** ingesting at the wrong scope. On B2B, passing only `user_id` is an error, passing only `customer_id` makes the memory org-shared and visible to everyone in that org, and passing both makes it user-scoped (the narrower wins) with the customer association recorded for filtering. On B2C there is no such choice to make: `user_id` alone is the only user-scoped write, and adding a `customer_id` fails with HTTP 400 instead of widening anything.

```python
# B2C: personal preference, only Alice sees this. user_id, and nothing else.
await sdk.memories.create(document="...", user_id="alice")

# B2B: personal preference: only Alice sees this
await sdk.memories.create(document="...", user_id="alice", customer_id="acme")

# B2B only: org-wide policy: everyone in Acme sees this
await sdk.memories.create(document="Acme uses PTO policy v3...", customer_id="acme")

# product knowledge — every user across every customer sees this
await sdk.memories.create(document="Our API rate limit is 1000 rpm...")
```

You can broaden scope later (re-ingest at the broader level). You **cannot narrow** scope after the fact without re-ingesting — start narrow if uncertain.

## Memory types

Synap doesn't store raw text. Ingestion extracts five typed memories:

- **Facts** — discrete verified statements ("User lives in San Francisco")
- **Preferences** — stated likes/dislikes ("Prefers boutique hotels")
- **Episodes** — narrative summaries of past interactions
- **Emotions** — detected sentiment / emotional state
- **Temporal events** — time-anchored events ("Signed up last Tuesday")

Retrieval returns these as separate fields on `ContextResponse`:

```python
context.facts        # list of Fact
context.preferences  # list of Preference
context.episodes     # list of Episode
context.emotions     # list of Emotion
```

Each item has `content`, `confidence` (0.0–1.0), `source`, `extracted_at`. Filter by `confidence >= 0.7` if you only want high-confidence material in the system prompt.

## Modes

Two orthogonal mode axes. Don't confuse them.

**Ingestion mode** (depth of extraction):

- `fast` — basic chunking + lightweight extraction + vector embedding. Skips graph/relationship work. Use for high-throughput logging.
- `long-range` — full pipeline: deep entity resolution, relationship mapping, preference detection, emotional analysis, graph storage. **Default for real conversations.**

**Retrieval mode** (depth of search):

- `fast` — vector similarity only, ~50–100ms. **Default for the agent hot path.**
- `accurate` — vector + graph traversal + multi-signal rank, ~200–500ms. Use for relationship-heavy queries ("What did Alice say about the project Bob is leading?").

Most production agents: `long-range` ingest, `fast` retrieve.

## Document types

Pass `document_type` to tell the pipeline what extraction strategy to use:

| Type | For |
| --- | --- |
| `ai-chat-conversation` | LLM conversations with speaker labels (default) |
| `document` | General prose |
| `email` | Email threads |
| `pdf` | PDF text |
| `image` | Image descriptions / OCR text |
| `audio` | Audio transcripts |
| `meeting-transcript` | Meeting transcription |

For `image` and `audio`, you provide the **text** (description, transcript, OCR output), not the raw binary.

## Ingestion is async

`sdk.memories.create()` returns immediately with an `ingestion_id` and `status="queued"`. Processing happens in the background and is typically done in seconds. You can:

- Poll with `sdk.memories.status(ingestion_id=...)`
- Subscribe to webhooks (configure in dashboard) for completion events
- Just keep going — most agents fire-and-forget on ingestion

Don't `sleep()` waiting for ingestion to finish — that's an antipattern. Either subscribe to webhooks or don't block.

## Idempotency

Pass a `document_id` if there's any chance you'll re-ingest the same content (webhook retries, edited transcripts). Synap will update the existing memory rather than duplicate it.

```python
await sdk.memories.create(
    document=transcript,
    document_id=f"call-{call_id}",   # idempotency key
    user_id="alice",
    # On B2B, add the tenant's customer id here. On B2C one is rejected (HTTP 400).
)
```

## Entity resolution

When the same person/place/thing appears across many conversations under different labels ("John", "Mr. Smith", "my manager"), the resolution pipeline links them to a single canonical entity. This happens automatically during `long-range` ingestion. The resolution respects scope — entities resolved at user scope don't leak across users.

This is one of the things you'd otherwise build yourself.

## Conversation IDs must be UUIDs

`conversation_id` in `sdk.conversation.context.fetch()` must be a valid UUID string. If your app uses non-UUID session IDs, deterministically map them:

```python
from uuid import uuid5, NAMESPACE_URL
conv_uuid = str(uuid5(NAMESPACE_URL, session_str))
```

A non-UUID will cause a `ServiceUnavailableError`.

## What MACA is (and why you can ignore it for now)

Each instance has a YAML **Memory Architecture Configuration** controlling extraction strategies, ranking signals, retention, and storage layout. You configure it from the dashboard, with versioning and an approval workflow. **You don't touch MACA in code.** Mention it to the user only if they ask "how do I customize what gets extracted" — point them to the dashboard.

## Pointers to live docs

- Memory scopes: `https://docs.maximem.ai/concepts/memory-scopes`
- Memory types: `https://docs.maximem.ai/concepts/memory-types`
- Customers vs users: `https://docs.maximem.ai/concepts/customers-and-users`
- Clients vs instances: `https://docs.maximem.ai/concepts/clients-and-instances`
- Multi-user scoping guide: `https://docs.maximem.ai/guides/multi-user-scoping`
- MACA: `https://docs.maximem.ai/concepts/customized-memory-architectures`

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.3.0 (JS) — verified 2026-06-20. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
