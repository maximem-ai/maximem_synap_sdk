# Changelog: maximem-synap-langgraph

## 0.6.0 (unreleased)

Behaviour change: methods that used to answer wrongly now raise.

### Changed
- `SynapStore.get` / `aget`, `delete` / `adelete` and `list_namespaces` /
  `alist_namespaces` raise `SynapStoreUnsupportedError` (new, a subclass of
  `NotImplementedError`, exported from `synap_langgraph`). Synap is a
  search-based memory with no exact key lookup and no namespace listing.
  Before, `get` returned `None` for everything, `delete` raised "no memory
  found" and `list_namespaces` returned `[]`, because the server does not
  return the key tag `put` writes. A `batch` / `abatch` holding a `GetOp`, a
  delete `PutOp` or a `ListNamespacesOp` raises before running any of its ops.
  Migration: read with `search`; delete with `adelete_memory(namespace,
  hit.value["_synap"]["id"])` or `aforget(namespace, description)`.

### Added
- `put` / `aput` take `memories.create` options as optional keyword-only
  arguments: `document_type`, `document_id`, `document_created_at`, `mode`,
  `objective`, `metadata` (merged with the store's tag), and `wait=True`
  (+ `timeout_seconds`, `poll_interval_seconds`), which returns the SDK's
  `MemoryStatusResponse`. Without them `put` is unchanged.
- `search` / `asearch` take `sdk.fetch` options for one call as extra keyword
  arguments (`mode`, `precision_level`, `types`, `query_date`,
  `max_results_per_query`, `conversation_id`, `scopes` ...).
- `arecord_message` / `record_message` take the SDK's `model` and
  `context_window`, and `content` may be a list of strings, `TextPart` and
  `FilePart`; such a turn goes over REST (`sdk.conversation.record_message`).
- New store methods, identity from the namespace, each with a sync twin:
  `acreate_memory` (any document, file parts included),
  `acreate_memory_from_file`, `aget_ingestion_status`, `await_ingestion`
  (`wait_for_ingestion`), `aget_memory`, `aupdate_memory`, `adelete_memory`,
  `aforget`, `afetch_raw` (the SDK's `UnifiedContextResponse` with the SDK's
  own defaults), `aget_profile`, `aingest_transcript`,
  `aget_context_for_prompt`, `acompact` (with `compaction_level`), `aget_compacted`,
  `aget_compaction_status`, `aget_scope_ladder`, `adelete_data`
  (`sdk.data.delete`, user and customer from the namespace only),
  `arestore` (`sdk.data.restore`), `adeletion_status` (`sdk.data.status`).
- `adelete_memory` / `delete_memory` take the SDK's `permanent=False`.

### Known limits
- Calls by id (`aget_memory`, `aupdate_memory`, `adelete_memory`,
  `aget_ingestion_status`, `await_ingestion`, `arestore`,
  `adeletion_status`, and every method taking a
  `conversation_id`) are NOT checked against the namespace's identity. The
  server does not verify ownership by id, so any caller holding an id can act
  on it, another user's memory included.
- Not exposed, by owner decision: `memories.batch_create` and
  `conversation.record_messages_batch` (each item carries its own user ids
  and would bypass the namespace identity). They stay reachable on
  `store.sdk`.

### Unchanged
- `put` still writes the `lg_store*` tag: `search` reads it for sub-namespace
  filtering on a server that returns caller metadata.
- `search` keeps its `mode="accurate"` default, by owner decision: an explicit
  exception to "same defaults as the SDK" (`sdk.fetch` defaults to `"fast"`).
  `include_conversation_context=False` and the ranked shape are unchanged too.
  `afetch_raw` uses the SDK's own defaults.
