# Multi-tenant scoping — patterns

The hardest thing to get right with any memory system is scoping for B2B SaaS. This file walks through the four common patterns. Use these as templates.

## 1. Single-user personal assistant

One user, no orgs, full personalization.

```python
await sdk.memories.create(
    document=transcript,
    user_id=user_id,         # only ID needed
    mode="long-range",
)

context = await sdk.conversation.context.fetch(
    conversation_id=conv_id,
    search_query=[user_msg],
)
```

Result: Each user has fully isolated memory. Nothing leaks. No customer scope.

## 2. Multi-tenant B2B SaaS — the canonical pattern

Multiple customer organizations, each with multiple users. Per-user personalization plus org-shared knowledge.

```python
# Personal — only Alice sees this
await sdk.memories.create(
    document=user_chat_transcript,
    user_id="alice",
    customer_id="acme",
    mode="long-range",
)

# Org-shared policy — everyone in Acme sees this, no one outside
await sdk.memories.create(
    document="Acme PTO policy v3: ...",
    document_type="document",
    customer_id="acme",          # NO user_id → customer scope
    mode="long-range",
)

# Product-wide knowledge — everyone everywhere sees this
await sdk.memories.create(
    document="API rate limit is 1000 rpm.",
    document_type="document",
    # no user_id, no customer_id → client scope
    mode="long-range",
)

# Retrieval — full chain, narrower wins
context = await sdk.conversation.context.fetch(
    conversation_id=conv_id,
    search_query=[query],
    mode="fast",
)
# context.facts may include items from user, customer, AND client scope.
# Narrower-scope memories take priority on conflicts.
```

## 3. Org-only (shared knowledge base)

No per-user personalization — everyone in the same org sees the same memories.

```python
await sdk.memories.create(
    document=transcript,
    customer_id="acme",          # customer-scoped only
    mode="long-range",
)

context = await sdk.customer.context.fetch(
    conversation_id=conv_id,
    search_query=[query],
)
```

Use this for shared workspaces / team chatbots where individual personalization isn't a feature.

## 4. Customer-first with optional personalization

Most memories are org-shared; a few user-specific overrides.

```python
# Default — org-shared
await sdk.memories.create(
    document=meeting_transcript,
    customer_id="acme",
    mode="long-range",
)

# Personal override — only Alice
await sdk.memories.create(
    document="User: I prefer dark mode.",
    user_id="alice",
    customer_id="acme",
)

# Retrieval — narrower priority means Alice's prefs surface first
context = await sdk.conversation.context.fetch(
    conversation_id=conv_id,
    search_query=[query],
)
```

## Verification — never skip this

After wiring scoping, **test it**. Ingest one customer's data, fetch from another customer's context, confirm zero leakage:

```python
# Setup
await sdk.memories.create(
    document="Acme secret: project Atlas launches Q3",
    customer_id="acme_corp",
)
await sdk.memories.create(
    document="Globex secret: project Beacon launches Q4",
    customer_id="globex",
)

await asyncio.sleep(5)  # let ingestion settle

# Acme user fetching → should see Atlas, not Beacon
ctx_acme = await sdk.conversation.context.fetch(
    conversation_id=acme_conv,
    search_query=["secret project launches"],
)
assert any("Atlas" in f.content for f in ctx_acme.facts)
assert not any("Beacon" in f.content for f in ctx_acme.facts)

# Globex user fetching → should see Beacon, not Atlas
ctx_globex = await sdk.conversation.context.fetch(
    conversation_id=globex_conv,
    search_query=["secret project launches"],
)
assert any("Beacon" in f.content for f in ctx_globex.facts)
assert not any("Atlas" in f.content for f in ctx_globex.facts)
```

If this assertion ever fails, you have a scope-leakage bug. Fix it before shipping.

## Anti-patterns

- **Reusing the same `user_id` across customers.** Synap will treat them as the same user — preferences will leak across orgs. Use `f"{customer_id}:{user_id}"` if your user IDs aren't globally unique, or fix at the source.
- **Display names as `user_id`.** Names change. IDs shouldn't. Use immutable identifiers.
- **Ingesting at client scope by accident.** Forgetting to pass `customer_id` makes every memory visible to every customer. Audit your ingestion call sites in code review.
- **Skipping `customer_id` on retrieval but providing it on ingest.** The retrieval doesn't know what org context to honor. Pass it consistently on both sides.

## Live doc

The full guide: `https://docs.maximem.ai/guides/multi-user-scoping`

---
*Accurate as of `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.2.4 (JS) — verified 2026-06-17. Source of truth: https://docs.maximem.ai (append `.md` to any page).*
