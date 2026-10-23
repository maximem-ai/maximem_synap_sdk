# Context fetch — `sdk.conversation.context.fetch()` and scoped variants

> **Read this next:** [Context fetch (SDK)](https://docs.maximem.ai/sdk/context-fetch.md); [Retrieval modes](https://docs.maximem.ai/concepts/retrieval-modes.md); [Response shapes](https://docs.maximem.ai/sdk/response-shapes.md); [fetch](https://docs.maximem.ai/sdk-reference/context/fetch.md); [user-fetch](https://docs.maximem.ai/sdk-reference/context/user-fetch.md); [customer-fetch](https://docs.maximem.ai/sdk-reference/context/customer-fetch.md); [client-fetch](https://docs.maximem.ai/sdk-reference/context/client-fetch.md); [conversation-context fetch](https://docs.maximem.ai/sdk-reference/conversation-context/fetch.md); [get-context-for-prompt](https://docs.maximem.ai/sdk-reference/conversation-context/get-context-for-prompt.md); [Errors](https://docs.maximem.ai/sdk-reference/errors.md). **Checkpoint:** open the method's page before writing each call, re-read it after and compare every parameter, scope rule and error; if the docs and this file disagree, the docs win and tell the human.

The read side. This sits in the agent's hot path; latency matters.

## The single most common call

```python
context = await sdk.conversation.context.fetch(
    conversation_id="3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c",   # UUID required
    search_query=["project deadlines", "Q2 planning"],
    max_results=10,
    types=["facts", "preferences"],   # or omit for all
    mode="fast",                       # or "accurate"
)
```

Returns a `ContextResponse` with separate fields per memory type.

## `conversation_id` must be a UUID

Non-UUID strings raise `InvalidConversationIdError` ("Invalid conversation ID: ..."),
client-side, before any request leaves. If your app's session/thread id isn't a
UUID, deterministically map it once:

```python
from uuid import uuid5, NAMESPACE_URL
conv_id = str(uuid5(NAMESPACE_URL, your_session_string))
```

Cache the mapping if you need to do this often.

## Modes

| | `fast` | `accurate` |
| --- | --- | --- |
| Latency | ~50–100 ms | ~200–500 ms |
| Search | vector similarity only | vector + graph traversal + re-rank |
| Ranking | cosine similarity | similarity + recency + graph centrality |
| Best for | every-turn retrieval | relationship-heavy queries |

**Default to `fast`.** Switch to `accurate` only when the query crosses entities ("What did Alice say about the project Bob is leading?") or when latency budget allows.

## Search queries

- **One query** — straightforward semantic match.
- **Multiple queries** — independent searches merged + deduped + re-ranked. Broaden recall when one phrasing might miss things.
- **No query** — returns the most recent + relevant memories for the conversation, no semantic filter.

```python
# Multi-query example — cast a wider net
context = await sdk.conversation.context.fetch(
    conversation_id=conv_id,
    search_query=[
        "dietary restrictions",
        "favorite cuisines",
        "food allergies",
    ],
    max_results=15,
)
```

## `types` filter

Cut response size by asking only for what you'll use:

```python
types=["facts", "preferences"]   # only these two
types=["episodes", "temporal"]   # narrative + time-anchored
types=["all"]                    # explicit equivalent of omitting
```

Valid: `"facts"`, `"preferences"`, `"episodes"`, `"emotions"`, `"temporal"`, `"all"`.

## Ranked results

Pass `response_shape="ranked"` and read `context.items`: one list, most relevant
memory first across every type, ready to hand to the model in that order.

```python
context = await sdk.user.context.fetch(
    user_id=user_id,
    search_query=["refund for a cancelled ride"],
    response_shape="ranked",
)
for item in context.items:
    item.rank         # 1 is the most relevant
    item.category     # "facts" | "preferences" | "episodes" | "emotions" | "temporal_events" | "excerpts"
    item.content
    item.rank_score   # higher is better; the order is already sorted on it
```

`"excerpts"` are quotes from the source conversations and documents, placed by
their own score. The typed lists below are still filled, as views of the same
items, except that excerpts are not in `facts`. When the token budget is exceeded the
cut keeps whole memory types in a fixed order, so the list is in rank order but may
not keep the best items overall. Without `response_shape` you get the typed lists only, which lose the
order across types. Needs an SDK release that has `response_shape`; older SDKs
reject the argument.

## Response structure

```python
context.facts            # list[Fact]
context.preferences      # list[Preference]
context.episodes         # list[Episode]
context.emotions         # list[Emotion]
context.metadata         # ResponseMetadata
context.raw              # dict — raw API response, forward-compat escape hatch
```

Each item:

```python
fact.id            # UUID
fact.content       # str — natural-language statement
fact.confidence    # float 0.0–1.0
fact.source        # str — source memory id
fact.extracted_at  # datetime
fact.metadata      # dict
```

Filter by `confidence >= 0.7` if you only want high-trust facts in the system prompt.

## Response metadata

```python
m = context.metadata
m.correlation_id          # log this for support
m.source                  # "cache" or "cloud"
m.ttl_seconds             # cache validity
m.compaction_applied      # True if context was compressed to fit token budget
```

## Scoped retrieval

Beyond conversation-level, three scope-specific endpoints. `sdk.customer.context.fetch`
(`POST /v1/context/customer/fetch`) is **B2B only** and is rejected on a B2C instance.
`sdk.user.context.fetch` and `sdk.client.context.fetch` work in both modes; on B2B a user
fetch takes a `customer_id` alongside the `user_id`. `GET /api/v1/auth/whoami` returns
`user_context_isolation`, which tells you which mode you are in.

```python
# All user-scoped + broader memories for the user behind a conversation
user_ctx = await sdk.user.context.fetch(
    user_id=user_id,          # REQUIRED. Omitting it is a TypeError, not an empty result.
    conversation_id=conv_id,
    search_query=["travel preferences"],
    max_results=20,
    mode="accurate",
)

# Customer-shared memories (org-wide knowledge). B2B instances only.
cust_ctx = await sdk.customer.context.fetch(
    customer_id=customer_id,  # REQUIRED. Omitting it is a TypeError.
    conversation_id=conv_id,
    search_query=["engineering OKRs"],
    max_results=15,
)

# Client-wide product knowledge
client_ctx = await sdk.client.context.fetch(
    conversation_id=conv_id,
    search_query=["product roadmap"],
    max_results=10,
)
```

Higher scopes don't leak narrower-scope memories. User context includes broader scopes by default; customer context excludes user-scoped data.

## Pattern: system-prompt injection

The 80% use case: fetch context, format as a system message, send to the LLM, record the turns afterward with `conversation.record_message` (the first call registers the `conversation_id`; record the user turn before the first fetch). Never also send the same text with `memories.create()`: it is extracted twice.

```python
async def turn(conv_id: str, user_id: str, user_message: str) -> str:
    # 1. fetch
    context = await sdk.conversation.context.fetch(
        conversation_id=conv_id,
        search_query=[user_message],
        max_results=10,
        mode="fast",
    )

    # 2. format
    lines = []
    if context.facts:
        lines.append("## Facts")
        lines += [f"- {f.content}" for f in context.facts if f.confidence >= 0.7]
    if context.preferences:
        lines.append("\n## Preferences")
        lines += [f"- {p.content}" for p in context.preferences]
    memory_block = "\n".join(lines) if lines else "No prior context."

    system_prompt = f"""You are a helpful assistant with persistent memory.
Use this context, but do not mention you're reading from a memory system.

{memory_block}"""

    # 3. LLM call
    answer = await your_llm.complete(system=system_prompt, user=user_message)

    # 4. record the turns (one owner for conversation turns: record_message, not memories.create)
    await sdk.conversation.record_message(
        conversation_id=conv_id, role="user", content=user_message,
        user_id=user_id,          # B2C shape; on B2B add customer_id=...
    )
    await sdk.conversation.record_message(
        conversation_id=conv_id, role="assistant", content=answer,
        user_id=user_id,
    )

    return answer
```

This is what most integrations do under the hood.

## Performance tips

- **`mode="fast"` in the agent loop.** Use `accurate` for reports / summaries / batch.
- **Smaller `max_results` is faster.** Default 10 is fine; drop to 5 for tight budgets.
- **Filter `types`.** Don't fetch episodes if you only show facts.
- **Cache aggressively.** The SDK cache already helps; reuse the same `search_query` if the conversation hasn't shifted topic.
- **The live stream** can make a read a local cache hit when Synap prepared the context. Whether to wire it is the human's choice: see `integration-depth.md`.

## Live doc

`https://docs.maximem.ai/sdk/context-fetch`

---
*Versions: run `pip show maximem-synap` (Python) or `npm ls @maximem/synap-js-sdk` (JS) for what is installed; no version is pinned here. Source of truth: https://docs.maximem.ai (append `.md` to any page).*

> While a new instance is still building, fetches get HTTP 409 `instance_not_ready`: degrade to no memory and continue, see `sdk-setup.md`.
