# Ingestion — `sdk.memories.create()` and friends

> **Read this next:** [Ingestion (SDK)](https://docs.maximem.ai/sdk/ingestion.md); [How ingestion works](https://docs.maximem.ai/concepts/how-ingestion-works.md); [memories.create](https://docs.maximem.ai/sdk-reference/memories/create.md); [batch-create](https://docs.maximem.ai/sdk-reference/memories/batch-create.md); [create-from-file](https://docs.maximem.ai/sdk-reference/memories/create-from-file.md); [status](https://docs.maximem.ai/sdk-reference/memories/status.md); [wait-for-completion](https://docs.maximem.ai/sdk-reference/memories/wait-for-completion.md); [record-message](https://docs.maximem.ai/sdk-reference/conversation/record-message.md); [record-messages-batch](https://docs.maximem.ai/sdk-reference/conversation/record-messages-batch.md); [ingest-transcript](https://docs.maximem.ai/sdk-reference/conversation/ingest-transcript.md); [Errors](https://docs.maximem.ai/sdk-reference/errors.md). **Checkpoint:** open the method's page before writing each call, re-read it after and compare every parameter, scope rule and error; if the docs and this file disagree, the docs win and tell the human.

The write side. Use this when no integration package fits, or when you need a primitive the integration doesn't expose.

## The single most common call

```python
response = await sdk.memories.create(
    document="User: I'm vegetarian.\nAssistant: Got it.",
    document_type="ai-chat-conversation",
    user_id="alice",
    customer_id="acme",          # B2B only: required there, rejected on B2C
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
| `document` | `str`, or a list of strings and parts | Required. Raw content. For chat, include speaker labels. A list may hold images; see below. |
| `document_type` | `str` | Default `"ai-chat-conversation"`. Drives extraction strategy. See list below. |
| `user_id` | `str` | Required for user-scoped memories, in both modes. |
| `customer_id` | `str` | **B2B only.** Required on B2B for user-scoped and customer-scoped memories. On B2C it is rejected with HTTP 400. |
| `mode` | `str` | `"long-range"` (default) or `"fast"`. |
| `document_id` | `str` | Idempotency key. Same id → updates, doesn't duplicate. |
| `document_created_at` | `datetime` | Original creation time. Improves temporal reasoning during retrieval. |
| `metadata` | `dict` | Stored but not indexed. Use for your bookkeeping. |

Which ids you send is not a judgement call, the instance settles it. B2C (`user_context_isolation = "equals_customer"`) takes `user_id` alone and rejects a `customer_id` with HTTP 400; B2B (`strict`) requires `customer_id` and errors on a `user_id` sent by itself. When in doubt, call `GET /api/v1/auth/whoami` and read `user_context_isolation`. Do not pass both to cover yourself, and never reuse the user id as the customer id.

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

## Images in conversations

An image shared in a conversation can go in as it is. Put a `FilePart` where it sits; Synap reads it and stores a description in its place.

```python
from maximem_synap import FilePart

response = await sdk.memories.create(
    document=["User: here's the receipt", FilePart(path="receipt.png")],
    user_id="alice",
)
response.attachments   # one per file: filename, media_type, status ("read" | "failed"), error

# The same list works as `content` here:
await sdk.conversation.record_message(conversation_id=cid, role="user",
    content=["here's the receipt", FilePart(path="receipt.png")], user_id="alice")
await sdk.instance.send_message(content=["here's the receipt", FilePart(data=png_bytes,
    filename="receipt.png")], conversation_id=cid, user_id="alice")
```

JS: `filePart({ path | data | url })` and `textPart(text)` from `@maximem/synap-js-sdk`.

- `FilePart` takes exactly one of `data`, `path` or `url`. The SDK fetches a `url`; the server never does.
- png, jpeg and webp only, up to 10 per call, 10 MB each. Other files get HTTP 415: use `memories.create_from_file`.
- A list of text only is joined with `"\n"` and sent as that string, byte for byte.
- `send_message` caps the images in one event at 2.5 MB (the stream message limit is 4 MB). Use `record_message` or `memories.create` for larger ones.
- `objective=` (optional) on `memories.create`: `"conversation"`, `"bootstrap"`, `"knowledge-base"`, `"record"`, `"profile"` or free text. Stored only; nothing acts on it yet.
- Needs an SDK and server with file parts. An older server makes `memories.create` raise `InvalidInputError` asking to upgrade the server.

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

The server accepts `long-range` (default) or `fast`, but per the docs both run the same pipeline today and nothing downstream reads the value, so the choice does not change what is extracted or how fast. Do not choose by workload and do not offer it as a decision (`https://docs.maximem.ai/concepts/retrieval-modes.md`). The retrieval `mode` (`fast` or `accurate`) does change results. Resubmitting with the same `document_id` redoes the extraction.

## Batch ingestion

For backfills, migrations, or bulk imports:

```python
from maximem_synap import CreateMemoryRequest

documents = [
    CreateMemoryRequest(
        document="...",
        document_type="ai-chat-conversation",
        user_id="alice",          # B2C shape; on B2B add customer_id="acme"
        mode="long-range",
    ),
    CreateMemoryRequest(
        document="Sprint planning notes...",
        document_type="meeting-transcript",
        customer_id="acme",       # customer scope: B2B instances only
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

Don't busy-poll. Either subscribe to webhooks or move on without waiting.

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
- **The ingestion `mode` makes no difference today**; leave the default.
- **Don't `sleep()` for ingestion to complete.** Use webhooks or fire-and-forget.

## Live doc

`https://docs.maximem.ai/sdk/ingestion`

---
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. Source of truth: https://docs.maximem.ai (append `.md` to any page).*

> While a new instance is still building, writes get HTTP 409 `instance_not_ready`: retry later, see `sdk-setup.md`.
