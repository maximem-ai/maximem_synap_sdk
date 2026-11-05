# maximem-synap-langgraph

Synap memory integration for [LangGraph](https://langchain-ai.github.io/langgraph/).

## Install

```bash
pip install maximem-synap-langgraph
```

Requires `langgraph>=1.0` and `maximem-synap>=0.7.1` (see `pyproject.toml`). The shared store (identity taken from the namespace), `scope_path` and the deeper-ladder features also need the `maximem-synap` release that has `get_scope_ladder`; a store built with `user_id` / `customer_id` does not.

## Before / After

**Before**: LangGraph's in-memory store, memory lost on restart and not shared:

```python
from langgraph.store.memory import InMemoryStore

app = graph.compile(store=InMemoryStore())
```

**After**: one line changes, and the store is Synap:

```python
from maximem_synap import MaximemSynapSDK
from synap_langgraph import SynapStore

sdk = MaximemSynapSDK(api_key="sk-...")
app = graph.compile(store=SynapStore(sdk))      # one shared store for all your users
```

**Who the memory belongs to is in the namespace.** `SynapStore(sdk)` is built once and serves every user. On each call the first entries of the namespace are the identity; everything after them is yours:

```python
async def recall(state, config, *, store):
    who = config["configurable"]
    # B2B account (customer, user):
    hits = await store.asearch((who["customer_id"], who["user_id"], "prefs"),
                               query=state["question"], limit=5)
```

| Your account | Namespace starts with | Example |
|---|---|---|
| B2C (users only) | `(user_id, ...)` | `("alice", "prefs")` |
| B2B (customer, user), the default | `(customer_id, user_id, ...)` | `("acme", "alice", "prefs")` |
| Deeper ladder (customer, team, user) | `(customer_id, team_id, user_id, ...)` | `("acme", "payments", "dana", "prefs")` |

How many entries are identity comes from your account's scope ladder (`await sdk.get_scope_ladder()`, cached by the SDK). A namespace that is too short, or empty, **raises**: the store never falls back to another user. On a deeper ladder the store sends the scope path for you.

**A pool at a higher level** (for example shared by every user of a customer) is a second shared store, built once:

```python
company = SynapStore(sdk, level="customer")            # identity = ("acme",)
await company.aput(("acme", "policies"), "refunds", {"days": 30})
await company.asearch(("acme", "policies"), query="refund window")
```

`level="client"` (the whole account, no identity) needs `allow_client_level=True`.

**Several queries, one server call.** `await store.asearch(ns, query=["tea", "airport cab"], limit=8)` sends one `sdk.fetch(search_query=[...])`, not one call per query. Blank and repeated entries are dropped. Cap what each query may contribute with `fetch_options={"max_results_per_query": 3}`.

**Fetch options per call.** On a shared store the constructor's `fetch_options` apply to everyone. For one search use `await store.asearch_with(ns, query=..., fetch_options={"mode": "fast", "precision_level": "medium", "types": ["facts"], "max_results_per_query": 3, "query_date": "2023-03-11", "conversation_id": thread_id})` (sync: `search_with`), or wrap calls in `with store.with_fetch_options(mode="fast"): ...`, or put them in the run config: `config={"configurable": {"synap_fetch_options": {...}}}` (read when the store is called inside a node). They are context-local, so concurrent calls for other users do not see them. `user_id`, `customer_id`, `scope_path`, `search_query`, `response_shape` and `max_results` cannot be set (`ValueError`).

**One store, several SDKs.** `SynapStore(sdk_provider=lambda: current_sdk.get())` resolves the SDK on every call (sync or async callable); `SynapStore(sdk)` is unchanged. `sdk_provider()` returning `None` raises `SynapIntegrationError`. Sync methods call the provider in the caller's thread.

**Conversation ids.** Any string works (`pgv2_...`, `chat-42`). The stream gets your id; REST-backed calls (`record_message` fallback, `get_context_for_prompt`, the `conversation_id` of a fetch) get the UUID the server derives from it (`uuid5(NAMESPACE_URL, id)`; a UUID is unchanged). `synap_integrations_common.coerce_conversation_id` is that function.

**How many places `search` asks for.** `store.asearch(ns, query=..., limit=8)` asks the server for 8 places, not more: the server fills a top-N differently from a top-M (it reserves places for chunks), so over-asking changes which 8 you get. The store asks for at least 10 only when it drops items itself afterwards (a sub-namespace in the prefix, a `filter`, or `semantic_fallback=False`).

**Text comes back as written.** A memory whose text is a JSON object (what `put` writes) comes back as that dict, with the exact text in `value["_synap"]["text"]`. Any other text comes back untouched as `value["_raw"]`: `1.50` stays `"1.50"`, `true` stays `"true"`, `"quoted"` keeps its quotes. (Before 0.5.1 a bare number, boolean, string or list was parsed and came back as `value["_value"]`.)

**Search follows the SDK.** `store.asearch` returns Synap's own ranked list, best first, with `score` set to the rank score. Every item's `value["_synap"]` carries what Synap knows about it: `scope` (`user` / `customer` / `client`), `assertion_source` (who said it), `said_at`, `event_date`, `valid_until`, `type`, `rank`, `rank_score`, `confidence`, `source`, `dates_known`, and any other metadata. `created_at` / `updated_at` are the day the user said it (else the event date); with neither they are 1970-01-01 UTC and `_synap["dates_known"]` is `False`. Memories that come back carrying the store's tag are filtered by the sub-namespace you pass; memories without it are always returned. Today's server does not return the tag, so against it every memory is returned and the sub-namespace does not filter (see the warning below).

**Read failures.** By default a failed read returns `[]` and logs at `ERROR`. To tell an outage from an empty memory: `SynapStore(sdk, on_error="raise")` raises `SynapIntegrationError`. This also covers a scope that `sdk.fetch` swallowed ("Scope 'user' fetch failed ... non-fatal"): the SDK leaves a failed scope out of `scopes_queried` and returns the rest, and the store compares that list with the scopes it asked for. With `"fallback"` it logs a WARNING ("read is incomplete") and uses the scopes that answered. An empty `scopes_queried` from the real `sdk.fetch` means every scope it ran failed (a healthy empty memory still lists its scopes), so it is reported; an answer that has items but lists no scope is taken as a stand-in SDK and treated as complete. The conversation scope is expected only when a `conversation_id` and a query are both sent.

**Pass anything else `sdk.fetch` takes** with `fetch_options`:

```python
store = SynapStore(sdk, fetch_options={
    "mode": "accurate", "precision_level": "high", "types": ["facts", "preferences"],
    "max_results_per_query": 5, "scopes": ["user", "customer", "client"],
})
```

The options apply to every call of the store, so on a shared store do not put per-user values in them (a `conversation_id`, for example).

**`SynapStore(sdk, user_id="alice", customer_id="acme")` still works** (one user per store). ⚠ It reads and writes that one user's memory for everyone who runs the graph, whatever namespace is passed: do not compile it into a graph that serves several users. This includes `delete_memory` and `forget`: `aforget(("acme", "bob"), "old address")` on a store built for alice acts on **alice's** memory. A shared store has none of this.

Because the SDK reads every level above the one you ask for, a store built with both `user_id` and `customer_id` now also returns customer-level memories from `search` (it used to ask for the user scope only).

**Know the namespace convention.** The first `N` entries are the identity; a namespace that is *longer* is accepted and the extra entries become your sub-namespace. A wrong-length convention is therefore not detected: on a B2C account, `("acme", "alice", "prefs")` files everything under the user `acme`. Use the table above; the store cannot know which you meant.

## Synap is search-based: no lookup by key

Agents ask Synap by search. Synap has **no exact `(namespace, key)` lookup and no namespace listing**, so these `BaseStore` methods raise `SynapStoreUnsupportedError` (a subclass of `NotImplementedError`) before anything is read or written:

| `BaseStore` method | Use instead |
|---|---|
| `get` / `aget(namespace, key)` | `search` / `asearch(namespace, query="...")`. Each hit carries its Synap id in `value["_synap"]["id"]`. |
| `delete` / `adelete(namespace, key)` | `adelete_memory(namespace, memory_id)` with the id of a search hit, or `aforget(namespace, "a description of it")`. |
| `list_namespaces` / `alist_namespaces()` | Keep the namespaces you use in your own app. |

A `batch` / `abatch` holding a `GetOp`, a delete `PutOp` (value `None`) or a `ListNamespacesOp` raises the same error and runs none of its ops.

**Migrating from 0.5.x.** Before 0.6.0 `get` returned `None` for everything, `delete(namespace, key)` raised `SynapIntegrationError` ("no memory found") and `list_namespaces` returned `[]`, because today's Synap server does not return the metadata a caller attaches to a memory, so the key tag `put` writes never came back. Code that calls `store.get(...)` and treats `None` as "not stored yet" now gets `SynapStoreUnsupportedError`: replace it with `search`. Code that called `store.delete(ns, key)` should search for the memory and call `adelete_memory(ns, hit.value["_synap"]["id"])`, or `aforget(ns, description)`.

`put` still writes the tag (sub-namespace, key, identity): `search` reads it to filter by sub-namespace and to give each hit its key and true namespace, on a server that returns it. Against today's server every memory looks untagged, so `search` returns every memory of the identity (yours from `put` and the ones Synap extracted from conversations), without sub-namespace filtering, and `SynapStore(..., semantic_fallback=False)` returns nothing.

## The Python SDK, on the store

The Python SDK's capabilities are reachable from the store, except the two batch calls listed below, so most code never has to drop to the SDK. Standard LangGraph code (`put`, `search` with `BaseStore`'s arguments) is unchanged; the additions are optional keyword-only arguments and extra methods. The identity always comes from the namespace (shared store) or the constructor (single-user store), never from an argument. Raw results are the SDK's own typed objects. Every async method has a sync twin without the leading `a` (`await_ingestion` / `wait_for_ingestion`).

| You want (SDK call) | On the store |
|---|---|
| Store a value with create options (`memories.create`) | `await store.aput(ns, key, value, document_type=, document_id=, document_created_at=, mode=, objective=, metadata=)` |
| Store and wait until it is searchable (`wait_for_completion`) | `await store.aput(ns, key, value, wait=True, timeout_seconds=300)` returns the `MemoryStatusResponse` |
| Store any document as is: text, dict, `TextPart`, `FilePart`, a list of parts (`memories.create`) | `await store.acreate_memory(ns, document, ...same options, wait=False)` returns `CreateMemoryResponse` |
| Ingest a file (`memories.create_from_file`) | `await store.acreate_memory_from_file(ns, file_path=... / file=..., filename=... / text=..., document_type=, mode=, metadata=)` |
| Ingestion status / wait (`memories.status`, `wait_for_completion`) | `await store.aget_ingestion_status(ingestion_id)`, `await store.await_ingestion(ingestion_id, timeout_seconds=, poll_interval_seconds=)` |
| One memory by id (`memories.get`, `update`, `delete`) | `aget_memory(ns, memory_id)`, `aupdate_memory(ns, memory_id, document, merge_strategy=, document_type=, metadata=)`, `adelete_memory(ns, memory_id, permanent=False)` (soft delete by default, restorable) |
| Delete a user's, conversation's or customer's data (`data.delete`) | `await store.adelete_data(ns, "user" / "conversation" / "customer", conversation_id=, dry_run=False, confirm=, wait=True, instances=, permanent=False)` returns `DeleteDataResult`. The user and customer come only from the namespace; `"customer"` needs a customer-level namespace |
| Restore soft-deleted memories (`data.restore`), read a delete's record (`data.status`) | `await store.arestore(ns, memory_ids=None, deletion_id=None)` returns `RestoreResult`; `await store.adeletion_status(ns, deletion_id)` returns `DeleteDataResult` |
| Forget by description (`memories.forget`) | `await store.aforget(ns, "description", mode="preview" or "direct", conversation_id=)` |
| Search with per-call fetch options (`fetch`) | `await store.asearch(ns, query=..., mode=, precision_level=, types=, query_date=, max_results_per_query=, conversation_id=, scopes=, ...)` |
| The raw fetch answer, with the SDK's own defaults (`fetch`; `user/customer/client/conversation.context.fetch` are `fetch` with `scopes=`) | `await store.afetch_raw(ns, query=None, **any_fetch_option)` returns `UnifiedContextResponse` |
| User profile (`user.get_profile`) | `await store.aget_profile(ns)` |
| Scope ladder (`get_scope_ladder`) | `await store.aget_scope_ladder(refresh=False)` |
| Record a turn (`conversation.record_message`) | `await store.arecord_message(conversation_id, role, content, namespace=ns, model=, context_window=)`; `content` may be a list of strings, `TextPart` and `FilePart`. With `model`, `context_window` or a list it goes over REST (`sdk.conversation.record_message`) instead of the stream |
| Push a whole transcript (`conversation.ingest_transcript`) | `await store.aingest_transcript(ns, conversation_id, transcript, conversation_type=, analysis=, metadata=, started_at=, ended_at=)` |
| Compacted context for a prompt (`get_context_for_prompt`) | `await store.aget_context_for_prompt(conversation_id, style=, search_history=, query=)`, or `synap_st_prompt` / `create_synap_st_node` |
| Compaction (`compact`, `get_compacted`, `get_compaction_status`) | `acompact(conversation_id, strategy=, compaction_level=, target_tokens=, force=)`, `aget_compacted(conversation_id, version=, format=)`, `aget_compaction_status(conversation_id)` |
| Live stream reporting (`instance.*`) | `SynapLangGraphCallbackHandler`, `create_synap_turn_node` and friends |

> **⚠ Calls by id are not checked against the namespace.** `aget_memory`, `aupdate_memory`, `adelete_memory`, `aget_ingestion_status`, `await_ingestion`, `arestore`, `adeletion_status` and the methods that take a `conversation_id` (`arecord_message`, `aingest_transcript`, `aget_context_for_prompt`, `acompact`, `aget_compacted`, `aget_compaction_status`) pass the id to the server as given. The namespace is only checked for shape. The server does not verify who owns a memory id, ingestion id or conversation id, so **any caller holding an id can read, rewrite or delete what it points to, another user's included**. Do not pass ids from untrusted input (for example a model's tool call) without your own check.

**Not on the store, by owner decision:** `memories.batch_create` and `conversation.record_messages_batch`. Each request in them carries its own user ids, which would bypass the namespace identity. Call `acreate_memory` / `arecord_message` once per item, or call them on `store.sdk` directly (then you pass the ids yourself). Also not on the store: SDK administration (`configure`, `cache`, `credits`, `subscribe_to_compaction_updates`, `archive_tool_result`), which stays on `store.sdk`.

**Defaults that differ from `sdk.fetch`, by owner decision.** `search` keeps `mode="accurate"` as its default (the SDK's `fetch` defaults to `"fast"`): an explicit exception to "same defaults as the SDK". It also reads with `include_conversation_context=False` (both set in the constructor; pass `mode="fast"` per call or in the constructor to change), always `response_shape="ranked"` and `max_results=limit+offset`. `afetch_raw` sends only what you pass, so it uses the SDK's own defaults (`mode="fast"`, `max_results=20`, by-type).

## What's in the box

- **`SynapStore`** — implements LangGraph's `BaseStore` for cross-thread long-term memory. Semantic search via `store.search(namespace, query=...)` routes to `sdk.fetch(...)`, so your graph nodes get Synap-powered recall out of the box.
  - **Shared store, identity in the namespace** (see Before / After), or a single-user store with `user_id` / `customer_id`. With only a `customer_id` the single-user store operates on the customer-wide pool.
  - **All memory types, in Synap's ranked order**, each labelled in `value["_synap"]`.
  - **Anticipation (optional).** Construct with `include_conversation_context=True` and feed turns via `store.record_message(conversation_id, role, content)` so just-stated context is in play on the next read. The turn goes out on the live stream when one is open and over REST only when there is none, never both. (This lives alongside the `BaseStore` API — anticipation has no key/value analogue.)

- **`SynapCheckpointSaver`**: **deprecated; it cannot restore what it saves.** Checkpoints go through `sdk.memories.create`, which extracts facts from them; `fetch` returns those facts, never the checkpoint, so a thread always resumes from scratch (and the base64 blob lands in the user's memory extraction). Constructing it warns. Use `SqliteSaver` / `PostgresSaver` for checkpoints and `SynapStore` for long-term memory.

- **`create_synap_node`** — re-exported from `synap-langchain` for users who discovered our LangGraph support through the LangChain package. This is the canonical home.

- **`synap_st_prompt`**: Synap's compaction as the `prompt` callable of `create_react_agent`. Once Synap has compacted the conversation, the model sees the compaction above your system prompt, then the turns after it, then the current turn; the turns it replaces are the ones your graph's state holds. Until then your messages go through as they are.

- **`create_synap_st_node`**: the same compaction, exposed as a `StateGraph` node that writes the string into state for your LLM node to consume (empty until the conversation has been compacted).

- **`SynapLangGraphCallbackHandler`**: reports the whole turn on Synap's live gRPC stream so the anticipation agent can see it: the user turn, the reasoning, every tool call and result, and the assistant turn. Pass it like any LangChain callback. It hears LangChain model and tool callbacks only.

- **`create_synap_turn_node`, `create_synap_tool_call_node`, `create_synap_tool_result_node`, `create_synap_reasoning_node`**: the same reports as graph nodes, for a graph whose LLM is **not** a LangChain model (an in-house client). See "Reporting without LangChain".

## Reporting a run on the live stream

LangGraph runs on LangChain's callback machinery, so a Synap handler in the run config reaches the chat model and the tool nodes inside your compiled graph.

```python
from langchain_core.messages import HumanMessage
from synap_langgraph import SynapLangGraphCallbackHandler

handler = SynapLangGraphCallbackHandler(
    sdk=sdk,
    conversation_id="conv_abc123",
    user_id="user_456",
    customer_id="cust_789",
)

await app.ainvoke(
    {"messages": [HumanMessage("where is my order QX41RT")]},
    config={"callbacks": [handler]},
)
```

Build one handler per conversation. What it reports:

| Event | Where it comes from |
|---|---|
| user turn | `on_chat_model_start`, once per turn (a ReAct loop calls the model again after every tool result with the same human message still in the list) |
| reasoning | the AIMessage's `reasoning` content blocks in `on_llm_end` — `on_agent_action` is an `AgentExecutor` callback and never fires in a graph |
| tool call | `on_tool_start`, with the parsed argument dict |
| tool result | `on_tool_end`, tied to the call by the shared `run_id` |
| assistant turn | `on_llm_end` |

**The stream opens itself.** The first turn calls `sdk.instance.listen()` for you; you do not. The user turn, the assistant turn, tool calls, results and reasoning then go on it. A turn goes over REST only when the stream cannot carry it (never both: the server persists a streamed turn itself). `SYNAP_AUTO_LISTEN=false`, or `synap_integrations_common.set_auto_listen(sdk, False)`, turns the auto-open off. The same rule applies to `SynapStore.record_message`, which works for B2C stores built with `user_id` alone. None of these calls can raise into your graph.

A run is one user turn and one assistant turn: the question the run was given and the answer it ended with, not every model call in the tool loop. Tool callbacks of a *sync* tool run on a different event loop from the stream's and are not reported; write the tool as `async def` to get tool events.

## Reporting without LangChain

> **Use the callback handler OR these nodes, not both.** On a LangChain model, `SynapLangGraphCallbackHandler` and the turn nodes each report the user and assistant turns, so together every turn is sent **twice** (and extracted twice). They are alternatives; nothing dedupes them.

The callback handler fires only for LangChain models and tools. If your nodes call your own LLM client, add nodes that report from the graph itself. Identity and conversation come from `config["configurable"]` on each run (`user_id`, `conversation_id`, and `customer_id` / `scope_path` when you have them), so one compiled graph serves many users:

```python
from synap_langgraph import create_synap_turn_node

g.add_node("emit_user", create_synap_turn_node(sdk, role="user", content="question"))
g.add_node("emit_assistant", create_synap_turn_node(
    sdk, role="assistant", content=lambda state: state["answer"]))
# START -> emit_user -> ... your nodes ... -> emit_assistant -> END

await app.ainvoke(
    {"question": "where is my order"},
    config={"configurable": {"user_id": "alice", "customer_id": "acme", "conversation_id": "conv-1"}},
)
```

**Reporting options (all four node factories).** `transport_key="synap_transport"` writes how the node's report travelled into that state key: for a turn `"stream"`, `"rest"`, `"failed"` (with `on_error="fallback"`) or `"skipped"` (nothing said); for tool and reasoning nodes `"stream"` or `"none"` (they exist only on the stream). `auto_listen=False` keeps REST fallback without ever opening a stream for that SDK (default `None` = the SDK's behaviour: open on first report unless `SYNAP_AUTO_LISTEN=false`); it is set on the SDK object each time the node runs. `sdk_provider=callable` resolves the SDK at run time instead of binding one (also on `create_synap_st_node` and `synap_st_prompt`). The reasoning node takes `step_index` and `thought_type` as constants or as `lambda state: ...`. Conversation ids may be any string; REST calls get the derived UUID.

**Deliberate gap: the caller profile.** The package does not expose the caller profile (`context_mode="conversation-summary"`). It is not a key/value memory: it is one block per user, outside `BaseStore`'s shapes, and `await sdk.user.context.fetch(user_id=..., context_mode="conversation-summary")` is already one call on the SDK. Wrapping it would add an API with no behaviour of its own.

`content` is a state key or a callable. Tool calls, tool results and reasoning have matching factories. A missing `user_id` or `conversation_id` is logged and nothing is sent, or raises with `on_error="raise"`. `scope_path` is sent with the user and assistant turns only (the SDK's tool and reasoning calls take no path).

## Short-term context (compacted conversation, on every LLM step)

LangGraph's built-in memory truncates recent turns to a token budget. Synap's short-term context is the **compaction** the Synap server maintains per conversation: a richer, more token-efficient view of "what happened so far" than the turns it replaces. Drop it into a prebuilt agent or a custom graph; both helpers serve from the SDK's local cache when warm and fall back to the Synap server when cold. By default neither adds the recent turns: a graph that keeps its own message list already sends them, and adding them again showed each one twice. A graph that keeps **no** message list (it sends every turn to Synap and reads them back) passes `include_recent_turns=True` (see below).

The SDK helper they both wrap is `sdk.conversation.context.get_context_for_prompt(conv_id, style=...)`, which is cache-first whenever the `SYNAP_SDK_ST_AUTHORITATIVE` flag is on.

### A) Prebuilt agent — one-line drop-in

```python
from langgraph.prebuilt import create_react_agent  # or langchain.agents.create_agent
from maximem_synap import MaximemSynapSDK
from synap_langgraph import synap_st_prompt

sdk = MaximemSynapSDK(api_key="sk-...")

agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-20241022",
    tools=[...],
    prompt=synap_st_prompt(
        sdk,                                 # no conversation_id: read from the run's config
        system="You are a helpful agent.",   # your own instructions
    ),
)

await agent.ainvoke(
    {"messages": [("user", "hi")]},
    config={"configurable": {"conversation_id": "conv_abc123"}},
)
```

The agent is built once and serves every conversation: the conversation id is read from `config["configurable"]["conversation_id"]` when the prompt runs. Pass `conversation_id="..."` to pin one conversation instead (an explicit argument wins). With neither, the helper logs and degrades, or raises with `on_error="raise"`. `style` is deprecated: it is validated and sent as before but has no effect on what the helper returns (it reads the compaction block, which has one rendering).

What the model sees at every step once the conversation has been compacted (system message content, then the turns Synap holds after the compaction, then the current turn):

```
<synap_short_term_context>
... Synap's compaction ...
</synap_short_term_context>

You are a helpful agent.
```

Before the first compaction, and whenever Synap cannot answer, the model sees only `You are a helpful agent.` and your messages. The graph's own state is never changed. For `langchain.agents.create_agent` (which has no `prompt=`), use `synap_langchain.SynapHistoryMiddleware`.

### B) Custom graph — write ST into state, consume in your own LLM node

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from synap_langgraph import create_synap_st_node

class State(TypedDict):
    messages: Annotated[list, add_messages]
    synap_st: str   # populated by the node

async def my_llm_node(state: State):
    system_text = f"You are a helpful agent.\n\n{state['synap_st']}".strip()
    ...   # build your prompt template with system_text and invoke the model

graph = StateGraph(State)
graph.add_node("st", create_synap_st_node(sdk))   # conversation id from the run's config
graph.add_node("llm", my_llm_node)
graph.add_edge(START, "st")
graph.add_edge("st", "llm")
graph.add_edge("llm", END)
```

### Graphs that keep no message list: `include_recent_turns=True`

`create_synap_st_node(sdk, include_recent_turns=True)` also writes Synap's recent turns (oldest first, `{"role", "content"}` dicts) to `state["synap_recent_turns"]` (`recent_key=` renames it; declare it in your state). `synap_st_prompt(sdk, include_recent_turns=True)` instead adds them to the system message in a `<synap_recent_turns>` block, passes your messages through unchanged, and leaves out a trailing user turn that is the same as your last user message. Use it when the user's latest message ("actually, I moved to California") must reach the model through Synap.

### Which to use

| You're building... | Use |
| --- | --- |
| A prebuilt `create_react_agent` | **`synap_st_prompt`**: drop into `prompt=` |
| A `langchain.agents.create_agent` agent | `synap_langchain.SynapHistoryMiddleware` |
| A custom `StateGraph` with multi-LLM / conditional routing / per-step prompt composition | **`create_synap_st_node`** — read `state["synap_st"]` in your nodes |
| Both | They compose — adapter Option A is sugar over Option B + a SystemMessage prepend |

### Error policy

- SDK failures **never crash the graph** by default (`on_error="fallback"`): logged at `ERROR` via `SynapIntegrationError`'s log path, then the helper degrades to your bare system prompt (or empty state slot).
- Pass `on_error="raise"` for strict environments that want the failure surfaced as `SynapIntegrationError`.
- No compaction yet is a legitimate empty case, not a failure; the user's system prompt and your messages are passed through as-is.

### Conversation ID

Yours, never inferred from LangGraph's `thread_id` (the two namespaces can diverge: a thread may span several Synap conversations, or the reverse). Put `conversation_id` in `config["configurable"]` for each run, or pass it to the helper to pin one conversation.

## Quickstart

```python
from langgraph.checkpoint.memory import InMemorySaver  # or SqliteSaver / PostgresSaver
from langgraph.graph import StateGraph, START, END
from maximem_synap import MaximemSynapSDK
from synap_langgraph import SynapStore

sdk = MaximemSynapSDK(api_key="sk-...")

store = SynapStore(sdk)          # shared: the user comes from the namespace

graph = StateGraph(MyState)
# ... add nodes / edges ...
app = graph.compile(checkpointer=InMemorySaver(), store=store)

# Store usage inside a node (B2B: customer, user, then your own namespace):
async def remember(state, runtime):
    await runtime.store.aput(
        ("acme", "alice", "preferences"),
        "language",
        {"value": "English"},
    )
```

## Error policy

- **Writes** (`SynapStore.put`) and every SDK method of the store (`acreate_memory`, `adelete_memory`, `aforget`, `afetch_raw` ...) surface SDK failures as `SynapIntegrationError` (the SDK's exception is its `__cause__`). Silent drops would hide ingestion outages.
- **Reads** (`search`) degrade gracefully by default (`on_error="fallback"`): they log at `ERROR` and return `[]` rather than crashing the graph. `on_error="raise"` raises `SynapIntegrationError` instead.
- **`get`, `delete`, `list_namespaces`** raise `SynapStoreUnsupportedError` (see "Synap is search-based").
- **Identity** errors (namespace too short or empty, unknown `level`, no ladder) always raise.

> Job/document-level attribution (mapping fragments back to a source id) is not done in the store; build it in app code.

## When to use which checkpointer

| Goal | Saver |
| ---- | ----- |
| Durable thread checkpoints, exact restore | LangGraph's `SqliteSaver` or `PostgresSaver` |
| Thread state in Synap | not possible: `SynapCheckpointSaver` is deprecated because Synap returns extracted facts, not the checkpoint |

Cross-thread long-term memory (`BaseStore`) maps cleanly to Synap — use `SynapStore` as your default.
