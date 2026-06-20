# Ingestion — `sdk.memories.create()` and friends

The write side. Use this when no integration package fits, or when you need a primitive the integration doesn't expose.

## The single most common call

```python
response = await sdk.memories.create(
    document="User: I'm vegetarian.\nAssistant: Got it.",
    document_type="ai-chat-conversation",
    user_id="alice",
    customer_id="acme",          # optional, scopes to org
    mode="long-range",           # "fast" or "long-range"
    metadata={"session_id": "..."},  # opaque, not indexed
)
print(response.ingestion_id)     # uuid
print(response.status)           # always "queued" initially
```

Returns immediately. Processing happens in the background.

## Parameters

| Param | Type | Notes |
| --- | --- | --- |
| `document` | `str` | Required. Raw content. For chat, include speaker labels. |
| `document_type` | `str` | Default `"ai-chat-conversation"`. Drives extraction strategy. See list below. |
| `user_id` | `str` | Required for user-scoped memories. |
| `customer_id` | `str` | Required for customer-scoped or shared user-customer memories. |
| `mode` | `str` | `"long-range"` (default) or `"fast"`. |
| `document_id` | `str` | Idempotency key. Same id → updates, doesn't duplicate. |
| `document_created_at` | `datetime` | Original creation time. Improves temporal reasoning during retrieval. |
| `metadata` | `dict` | Stored but not indexed. Use for your bookkeeping. |

The official docs mark `customer_id` as required, but in practice many integrations support user-only scoping when org-shared memory isn't needed. When in doubt, pass both.

## Document types

| Type | Optimized for |
| --- | --- |
| `ai-chat-conversation` | Multi-turn LLM convos, speaker turns, intent + preference detection |
| `document` | Generic prose, paragraph chunking |
| `email` | Sender/recipient extraction, action items, thread context |
| `pdf` | Section-aware chunking; pass extracted text, not bytes |
| `image` | Pass description or OCR text, not raw image |
| `audio` | Pass transcript, not raw audio |
| `meeting-transcript` | Multi-speaker extraction, action items, decisions |

For images and audio, the user does media processing upstream and passes text to Synap.

## Speaker labels matter

For `ai-chat-conversation`, **always** prefix turns with speaker labels (`User:`, `Assistant:`, or actual names). The pipeline uses these to attribute preferences to the right participant. Without them, "I prefer X" gets ambiguously attributed.

```python
# good
document = """User: I prefer concise answers.
Assistant: Understood, I'll keep responses tight."""

# bad
document = """I prefer concise answers.
Understood, I'll keep responses tight."""
```

## Modes

- **`long-range`** (default): full extraction pipeline including entity resolution, relationship mapping, preference detection, emotional analysis, and graph storage. Use for real conversations.
- **`fast`**: basic chunking + lightweight extraction + vector embedding. Skips graph work. Use for high-volume logging where deep extraction isn't worth the latency.

You can re-ingest later in `long-range` by resubmitting with the same `document_id` — extractions will be redone.

## Batch ingestion

For backfills, migrations, or bulk imports:

```python
from synap.types import CreateMemoryRequest

documents = [
    CreateMemoryRequest(
        document="...",
        document_type="ai-chat-conversation",
        user_id="alice",
        mode="long-range",
    ),
    CreateMemoryRequest(
        document="Sprint planning notes...",
        document_type="meeting-transcript",
        customer_id="acme",
        mode="fast",
    ),
]

batch = await sdk.memories.batch_create(documents=documents, fail_fast=False)
print(f"submitted={batch.total} ok={batch.succeeded} failed={batch.failed}")
```

`fail_fast=False` (default): bad documents are rejected individually; valid ones go through. `fail_fast=True`: any validation failure aborts the entire batch.

## Status polling

```python
status = await sdk.memories.status(ingestion_id=response.ingestion_id)
# status.status: queued | processing | completed | failed | partial_success
# status.memory_ids on completed
# status.error_message on failed
```

Don't busy-poll. Either subscribe to webhooks (`https://docs.maximem.ai/dashboard/webhooks`) or move on without waiting.

## Updating memories

```python
await sdk.memories.update(
    memory_id=memory_id,
    document="updated transcript...",
    merge_strategy="smart-merge",   # "replace" | "append" | "smart-merge"
    metadata={"reason": "new turns"},
)
```

- `replace` — drop and re-extract from new content.
- `append` — add to end; extractions only from appended part.
- `smart-merge` — dedup overlapping content, prefer newer for conflicts.

## Deletion

```python
await sdk.memories.delete(memory_id=memory_id)
```

**Permanent. Irreversible.** All extractions, entity associations, and graph edges derived from the memory are removed. There is no undelete.

## Best practices

- **Stable user/customer IDs.** Inconsistent IDs fragment a user's memory. Pick a deterministic identifier and stick to it.
- **Idempotency for retried sources.** Webhook-driven ingestion? Pass `document_id`.
- **Backfilled data needs `document_created_at`.** Otherwise temporal reasoning thinks everything happened today.
- **Choose mode by workload.** `long-range` for conversations, `fast` for high-volume logs.
- **Don't `sleep()` for ingestion to complete.** Use webhooks or fire-and-forget.

## Live doc

`https://docs.maximem.ai/sdk/ingestion`

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.3.0 (JS) — verified 2026-06-20. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
