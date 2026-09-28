# synap-deepagents

Synap integration for [deepagents](https://docs.langchain.com/oss/python/deepagents/overview) — LangChain's agent harness.

deepagents stores memory as `AGENTS.md` files and pastes them whole into the system prompt. This package plugs Synap in underneath, and adds a retrieval path the stock middleware cannot express.

## Install

```bash
pip install maximem-synap-deepagents
```

Requires `deepagents>=0.7.4`, `maximem-synap>=0.2.0`.

## Quickstart

Mount Synap on a route with `CompositeBackend`:

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from maximem_synap import MaximemSynapSDK
from synap_deepagents import SynapBackend

sdk = MaximemSynapSDK(api_key="sk-...")

backend = CompositeBackend(
    default=FilesystemBackend(root_dir="/path/to/repo"),
    routes={"/memories/": SynapBackend(sdk, user_id="alice")},
)

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    backend=backend,
    memory=["/memories/AGENTS.md"],
)

agent.invoke({"messages": [{"role": "user", "content": "What do I prefer?"}]})
```

> **Mount it on a route, never as `backend=` on its own.** `SynapBackend` as the
> default backend would route the agent's source-code reads and writes through a
> memory API, and the agent would lose its working tree. The constructor cannot
> stop you.

## The four surfaces

| Surface | Class | Use when |
|---|---|---|
| Backend | `SynapBackend` | You want memory to reach the agent through its own `read_file` / `grep` / `write_file` tools |
| Middleware | `SynapMemoryMiddleware` | You want recall scoped to the user's actual question |
| Tools | `SynapSearchTool`, `SynapStoreTool` | You want the model to reach for memory deliberately |
| Stream | `SynapStreamMiddleware` | You want Synap to see the turn as it happens, not just answer questions about it |

They compose. A common setup is the backend for automatic recall plus the tools for deliberate lookups, with the stream middleware alongside.

## Reporting the run on Synap's live stream

The first three surfaces read. `SynapStreamMiddleware` writes: it reports the five events Synap's anticipation agent watches for, on the gRPC stream `sdk.instance.listen()` opened.

```python
from deepagents import create_deep_agent
from maximem_synap import MaximemSynapSDK
from synap_deepagents import SynapMemoryMiddleware, SynapStreamMiddleware

sdk = MaximemSynapSDK(api_key="sk-...")
await sdk.instance.listen()

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    middleware=[
        SynapMemoryMiddleware(sdk=sdk, user_id="alice"),
        SynapStreamMiddleware(sdk=sdk, conversation_id="conv-123", user_id="alice"),
    ],
)
```

| Event | Hook | Why that hook |
|---|---|---|
| user turn | `before_agent` | The entry node runs once per invocation. `before_model` runs once per lap of a tool loop, with the same question still in the list, so it would report and extract the question several times per turn |
| reasoning | `after_model` | Each lap has its own reasoning; steps are numbered within the turn |
| tool call | `wrap_tool_call` | Reported before the tool runs |
| tool result | `wrap_tool_call` | Same hook, so the call and the result share `tool_call["id"]` by construction rather than by agreement |
| assistant turn | `after_agent` | The exit node. `assistant_message` means a turn just ended, and the text the model produces alongside a tool call is not the answer to anything |

Turn recording is **stream first, REST only as a fallback, never both** — the server persists `user_message` and `assistant_message` from the stream itself, so recording them again over REST writes and extracts the turn twice. Tool calls, results and reasoning have no REST equivalent: with no stream open they are simply not reported, and the middleware costs nothing.

Nothing in it can raise into the graph, and one middleware covers one conversation (the de-duplication is per instance).

## `grep` is a semantic search

This is the part worth knowing. On a Synap route, the agent's `grep` tool is **not** a regex match over file bytes — the pattern is passed to Synap as a natural-language query:

```python
# The agent runs this:
grep("what deployment process does the user follow", path="/memories/")
# It becomes this:
sdk.fetch(search_query=["what deployment process does the user follow"], mode="accurate")
```

Stock deepagents `grep` can only find what is literally written in a file. This finds what the user said, however they said it. Matches are attributed to `/memories/AGENTS.md` so the agent can read the file for more.

Tell your agent about this in its system prompt if it tends to write regexes. A model assuming literal matching will write `^prefer.*` and misread the misses.

## Query-conditioned recall

`create_deep_agent(memory=[...])` installs deepagents' own `MemoryMiddleware`, which calls `backend.download_files(paths)` — paths, no query. Even with `SynapBackend` underneath, that is one unqueried digest per run.

`SynapMemoryMiddleware` reads the pending user message first and passes it as the search query:

```python
from synap_deepagents import SynapMemoryMiddleware

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    middleware=[SynapMemoryMiddleware(sdk=sdk, user_id="alice")],
)
```

Use `memory=[...]` **or** `SynapMemoryMiddleware`, not both — they write to the same part of the system prompt, and you would pay for two retrievals to say the same thing twice.

## Short-term context

Long-term memory and short-term context are different things. Short-term is the compacted history of the *current* conversation:

```python
from synap_deepagents import synap_st_instructions

system_prompt = await synap_st_instructions(
    sdk, "conv_abc", system="You are a helpful coding agent."
)
agent = create_deep_agent(model="...", system_prompt=system_prompt)
```

That is a snapshot taken once. For a long-running agent, use `SynapShortTermMiddleware` instead, which refreshes each turn.

## Error policy

| Operation | Behaviour |
|---|---|
| Reads (`read`, `grep`, `ls`, `glob`, `download_files`) | Degrade — log at `ERROR`, return empty or `file_not_found` |
| Writes (`write`, `edit`, `upload_files`) | Raise `SynapIntegrationError` |
| Recall in middleware | Degrades to an empty block — a Synap outage must not end the run |
| `delete` | **Not implemented** — see below |

Read failures are always reported as `file_not_found`, never any other code. That is deliberate: deepagents' `MemoryMiddleware` raises `ValueError` on any download error code *except* `file_not_found`, so returning anything else during a Synap outage would end the agent run instead of degrading to an empty memory block.

## Why there is no `delete`

`sdk.memories.create()` returns an `ingestion_id`; `sdk.memories.delete()` needs a `memory_id`. They are different identifiers, and the memory does not exist yet at write time — ingestion is queued. So a path written through this backend cannot be resolved back to a durable memory, and a path-addressed delete cannot be honoured.

`delete` is optional in `BackendProtocol`, so this backend inherits the default that raises `NotImplementedError`, and `CompositeBackend` reports it cleanly. Remove memories through the Synap API or dashboard with a memory id.

## Queued writes and read-after-write

Synap ingestion is asynchronous: `create` returns `{ingestion_id, document_id, status: QUEUED}`. A memory written a moment ago is not yet retrievable, so an agent that writes a file and reads it back in the same turn would get nothing — which reads as data loss.

`SynapBackend` keeps a short-TTL, per-process write-through cache so your own writes are always readable back. It is a read-after-write guarantee, **not** semantic dedup, and it does not survive a restart. Set `cache_ttl_seconds=0` to disable it only if you can tolerate that.

The package never polls `wait_for_completion` on the agent's path. Waiting would cost turn latency without improving the answer — Synap may split one submitted document into several memories or merge it with existing ones, so there is no count to wait for.

## Configuration

```python
SynapBackend(
    sdk,
    user_id="alice",              # required. On B2B add customer_id; on B2C it is rejected.
    customer_id="acme",
    conversation_id=None,
    recall_filename="AGENTS.md",  # basename that maps to the synthesized recall doc
    max_results=20,
    mode="fast",                  # retrieval mode for reads (on the startup path)
    grep_mode="accurate",         # retrieval mode for grep (a deliberate question)
    precision_level="high",
    document_type="document",     # writes are explicit intent, not chat transcript
    ingest_mode="fast",
    include_conversation_context=False,
    cache_ttl_seconds=300,
)
```

## Tests

```bash
pytest integrations/synap-deepagents/tests -q
```
