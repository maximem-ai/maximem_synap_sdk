# Production checklist

Work through this before the user's first production deployment, and again before any release that touches the memory layer.

## Credentials

- [ ] `SYNAP_API_KEY` is in a secret manager (AWS Secrets Manager, GCP Secret Manager, Vault, Doppler, etc.) — never in code, never in `.env` files committed to git.
- [ ] Separate API keys per environment (dev / staging / prod). Revoke and rotate quarterly or on staff changes.
- [ ] `SYNAP_INSTANCE_ID` for production points at a **separate instance** from staging/dev. Do not share an instance across environments — memories will mix.

## Scoping

- [ ] Every ingestion call passes `user_id` (or explicit `customer_id` for org-shared content). No accidental client-scoped writes.
- [ ] User IDs and customer IDs are stable, deterministic identifiers — not display names, not anything that can change.
- [ ] If multi-tenant, verify scope isolation: ingest a memory under `customer_id=A`, fetch from a `customer_id=B` context, confirm it's not visible.
- [ ] If using `user_id` only (no `customer_id`), confirm that's intended and not an oversight.

## SDK lifecycle

- [ ] `await sdk.initialize()` runs once at process start, not per-request.
- [ ] `await sdk.shutdown()` runs on graceful shutdown (SIGTERM handler, FastAPI lifespan, Lambda extension, etc.).
- [ ] Single SDK instance per `instance_id` per process — singleton behavior is intentional, don't fight it.
- [ ] In serverless, the SDK is module-level and initialized lazily on first invocation; cold-start latency is documented.

## Read path

- [ ] Read failures degrade gracefully — agent continues with empty context when fetch fails. Wrap in `try/except SynapError`.
- [ ] `mode="fast"` in the conversation hot path. `accurate` only where latency budget allows.
- [ ] `conversation_id` is always a valid UUID — wrap non-UUID session ids with `uuid5(NAMESPACE_URL, ...)`.
- [ ] `max_results` set to what the prompt actually consumes. Don't fetch 50 if the prompt template only renders 5.
- [ ] `types` filter set if only specific memory types are used downstream.

## Write path

- [ ] Write failures **surface explicitly** — log them, alert on rate. The agent shouldn't silently lose memory.
- [ ] Ingestion is fire-and-forget (returns `queued`). No `sleep()` waiting for completion.
- [ ] `document_id` is set for any source that can be retried (webhooks, message queues, scheduled re-ingests).
- [ ] `document_created_at` is set when ingesting historical / backfilled data.
- [ ] Speaker labels (`User:`, `Assistant:`) are present in conversation documents.
- [ ] Bulk loads use `batch_create()` with `fail_fast=False`, not a loop of `create()` calls.

## Retries and timeouts

- [ ] `RetryPolicy` is configured (default 3 attempts is usually fine).
- [ ] `TimeoutConfig.read` is set to fit the agent's latency budget. Default 30s is too long for an interactive agent — drop to 5s for the read path if you can.
- [ ] On retry exhaustion, the read path falls back to no-memory rather than blocking.

## Monitoring and observability

- [ ] `correlation_id` from every error is logged. Synap support can trace requests from this.
- [ ] Metrics: ingestion volume, ingestion failures, retrieval latency p50/p95/p99, retrieval failures, cache hit rate.
- [ ] Alerts: spike in write failures (data loss), spike in read latency (UX impact), zero-memory responses for known-active users (likely scope misconfiguration).
- [ ] Dashboard analytics enabled at `https://synap.maximem.ai` — ingestion throughput, context retrieval usage, system health.

## Webhooks (optional, recommended)

- [ ] If you need to know when ingestion completes, configure webhooks instead of polling. See `https://docs.maximem.ai/dashboard/webhooks`.
- [ ] Webhook handler is idempotent — same event may arrive twice.
- [ ] Webhook signature is verified before processing.

## Data hygiene

- [ ] PII / sensitive data flow understood — what's being ingested, who can read it via scope retrieval, retention policy.
- [ ] User deletion path implemented — `sdk.memories.delete()` is irreversible; have a tested procedure.
- [ ] If GDPR / right-to-be-forgotten matters, document how a user's memories are purged: enumerate via `sdk.user.context.fetch()` (or custom listing) and `delete()` each.

## Deployment

- [ ] Initial backfill (if any) runs in `long-range` mode with proper `document_created_at` and idempotent `document_id`s.
- [ ] Staging instance has been smoke-tested with a representative conversation flow (ingest → wait → fetch → assert facts/preferences match).
- [ ] Rollback plan: how to disable Synap reads (return empty context) and writes (skip the call) without breaking the agent.

## MACA configuration

- [ ] Use-case markdown was uploaded when the instance was created, OR a custom MACA was reviewed and applied.
- [ ] If the agent's domain shifts (new product, new user segment), the use-case markdown is updated and MACA is regenerated. See `https://docs.maximem.ai/concepts/customized-memory-architectures`.

## Upgrade path

- [ ] SDK version is pinned in `requirements.txt` / `package.json`. Don't auto-update on production.
- [ ] Read the changelog before bumping: `https://docs.maximem.ai/resources/changelog`.
- [ ] Migration guide consulted on major version changes: `https://docs.maximem.ai/migration/overview`.

## Live doc

The canonical version of this checklist with rationale per item: `https://docs.maximem.ai/guides/production-checklist`
