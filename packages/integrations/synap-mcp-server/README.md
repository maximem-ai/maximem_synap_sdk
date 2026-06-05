# synap-mcp-server

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
| `log_exchange` | `POST /api/v1/memories/create` (`mode=long-range`) | Forward every turn; extraction decides what persists. |
| `recall_context` | `POST /v1/context/{client\|user\|customer}/fetch` (`mode=fast`) | Hot path; no IDs ⇒ client scope. |
| `list_recent_memories` | broad `/v1/context/.../fetch` (no query) | Debug / "test my memory". |

## Run locally

```bash
pip install -e ".[dev]"
SYNAP_API_URL=http://localhost:8000 uvicorn synap_mcp_server.server:app --port 8090
curl http://localhost:8090/health        # {"status":"ok",...}
```

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
| `LOG_LEVEL` | `INFO` | Log level. |
| `ENVIRONMENT` | `production` | Reported in `/health`. |

No Synap API key is configured on the server — the end user's key arrives per-request as a
Bearer token.

See [`../../synap/docs/mcp-server/`](../../synap/docs/mcp-server/) for the deployment plan,
implementation plan, test cases, and the Gumloop/n8n testing runbook.
