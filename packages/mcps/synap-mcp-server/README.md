# synap-mcp-server

[![Maximem Synap MCP connector – tool definition quality and endpoint health on Glama](https://glama.ai/mcp/connectors/ai.maximem/synap/badges/score.svg)](https://glama.ai/mcp/connectors/ai.maximem/synap)

Hosted remote **MCP server** (Streamable HTTP) that re-fronts the existing Synap public
REST operations as MCP tools, so no-code platforms (Gumloop, n8n) can give their agents
persistent memory with nothing but a pasted **MCP URL** and a **Bearer token**.

It is a **stateless adapter** — no new backend, storage, or pipeline. Each tool call maps
to one existing REST operation, and the incoming `Bearer synap_<key>` token is forwarded
verbatim to `synap-cloud`, which owns auth.

- Public endpoint (prod): `https://synap-mcp.maximem.ai/mcp`
- Health: `https://synap-mcp.maximem.ai/health`

## Tools

| Tool | REST operation | Notes |
|---|---|---|
| `log_exchange` | `POST /api/v1/memories/create` (`mode=long-range`) | Forward every turn; extraction decides what persists. `user_id` scopes per end-user, `customer_id` is B2B only (see Scoping); optional `conversation_id`. |
| `recall_context` | `POST /v1/context/{client\|user\|customer}/fetch` (`mode=fast`) | Hot path; no IDs ⇒ client scope. `customer` scope is B2B only. |
| `list_recent_memories` | broad `/v1/context/.../fetch` (no query) | Debug / "test my memory". Same scoping rules. |
| `check_memory_status` | `GET /api/v1/memories/status/{id}` | Takes no scope ids: the `ingestion_id` already identifies the write. |

## Scoping: one contract, two modes

The REST API enforces this. It is not advice, and it has been live on production
since 2026-08-26 07:07 UTC.

| Instance mode (`user_context_isolation`) | What to send | What happens otherwise |
|---|---|---|
| `equals_customer` (**B2C**) | `user_id` and nothing else. The user id is the whole identity. | Any `customer_id` is rejected with **HTTP 400**. `/v1/context/customer/fetch` is not available at all. |
| `strict` (**B2B**) | `customer_id` is **required**, alongside `user_id`. | A `user_id` on its own is an error. Unchanged by this contract. |

`GET /api/v1/auth/whoami` reports the mode as `user_context_isolation`. This server
reads it once per token and refuses a `customer_id` locally on a B2C instance, so a
model gets a sentence telling it what to send instead of a status code it cannot act
on. If the mode cannot be read, nothing is refused and the API stays authoritative.

Pass no IDs at all ⇒ **client scope** (shared per credential), which is valid in
both modes. Note: client-scope writes are not surfaced on the dashboard Memories page
(relational), so per-user scoping is recommended when dashboard visibility matters.
Fill `user_id` from an n8n expression / Gumloop input, and pass the same value on
`log_exchange` and `recall_context` so writes and reads address the same person.

**Host header.** The server disables the MCP transport's DNS-rebinding check (it sits behind a proxy + Bearer auth), so any reverse proxy can forward the real `Host` — no Host-rewrite hack needed.

## Run locally

```bash
pip install -e ".[dev]"
SYNAP_API_URL=http://localhost:8000 uvicorn synap_mcp_server.server:app --port 8090
curl http://localhost:8090/health        # {"status":"ok",...}
```

## Reporting the turn to the anticipation agent

An MCP server is not an agent. It exposes tools to somebody else's agent and
is handed one stateless HTTP call at a time, so most of what an agent does is
invisible here: the caller's tool calls, its tool results and its reasoning
never reach this process, and nothing tells a stateless server that a
conversation has ended. Those are not reported, because there is nothing to
report.

The turn does exist. `log_exchange` is handed the user's message and the
assistant's reply as arguments. Set `MCP_STREAM_EVENTS=true` and each logged
exchange is also reported as a conversation turn on `POST /v1/events/batch`,
the HTTP door onto the same listening path the gRPC stream feeds, so the
anticipation agent sees the turn and not only the memory extracted from it.
There is no gRPC stream here to use: this server holds no SDK, keeps no state
between requests, and every request can carry a different tenant's key.

It is **off by default**, on purpose. Turning it on adds a second backend call
per `log_exchange`, writes the turn to conversation history in addition to the
long-range document, and runs the anticipation agent, which bills model calls.

What does not change either way: `log_exchange` still queues the long-range
document and still returns the `ingestion_id` that `check_memory_status` and
`wait_for_processing` are built on. A turn report is best effort and can never
alter what a tool answers. A turn with no `user_id` or no `conversation_id` is
not reported at all, because the events route requires both.

## Test

```bash
pip install -e ".[dev]"
pytest -q
```

## Config (env)

| Var | Default | Meaning |
|---|---|---|
| `SYNAP_API_URL` | `http://synap-cloud:8000` | Backing REST API base URL (internal docker hostname in prod). |
| `MCP_PORT` | `8090` | Listen port. |
| `MCP_RECALL_TIMEOUT_S` | `10` | Recall (read) timeout. |
| `MCP_INGEST_TIMEOUT_S` | `8` | Log/ingest (write) timeout. |
| `MCP_DEFAULT_MAX_RESULTS` | `10` | Default recall result count. |
| `MCP_STREAM_EVENTS` | `false` | Also report a logged exchange as a conversation turn for anticipation. See above before enabling. |
| `MCP_EVENTS_TIMEOUT_S` | `4` | Turn-report timeout. Best effort; a miss never changes a tool's answer. |
| `LOG_LEVEL` | `INFO` | Log level. |
| `ENVIRONMENT` | `production` | Reported in `/health`. |

No Synap API key is configured on the server — the end user's key arrives per-request as a
Bearer token.

See the [Synap MCP documentation](https://docs.maximem.ai/integrations/mcp) for platform setup guides
(Gumloop, n8n) and deployment details.
