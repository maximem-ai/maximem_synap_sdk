# MCP server (no-code)

No package, no SDK, no code. Synap runs a hosted **Model Context Protocol** server; any
MCP-capable client (Claude Desktop, Cursor, a custom MCP client, another agent) connects to
it over HTTP and gets memory as a set of tools. Use this when the developer wants memory in
a tool that speaks MCP but doesn't want to write Python/TS.

## Connection

| Field | Value |
| --- | --- |
| Endpoint | `https://synap-mcp.maximem.ai/mcp` |
| Transport | Streamable HTTP (not stdio, not SSE) |
| Auth | `Authorization: Bearer synap_...` (your instance API key) |
| Health check | `https://synap-mcp.maximem.ai/health` |

The instance is resolved from the key, exactly like the SDK. The server forwards the token
to Synap Cloud verbatim — it never stores it.

## Client config

For a generic MCP client that supports remote HTTP servers with a bearer token:

```json
{
  "mcpServers": {
    "synap": {
      "url": "https://synap-mcp.maximem.ai/mcp",
      "headers": { "Authorization": "Bearer synap_..." }
    }
  }
}
```

Clients differ in how they take a remote URL + header (some use `"transport": "http"`, some
a connector UI). The two things that never change: the `/mcp` URL and the
`Authorization: Bearer synap_...` header. If a client only supports stdio, bridge it with a
generic "remote MCP over HTTP" proxy — but prefer a client with native HTTP MCP support.

## Tools the server exposes

| Tool | Does | Maps to |
| --- | --- | --- |
| `log_exchange` | Record a conversation turn (write) | `memories.create`, `mode=long-range` |
| `recall_context` | Fetch ranked context for the current turn (read) | client/user/customer context fetch, `mode=fast` |
| `list_recent_memories` | Broad recent-memory dump (debugging) | broad context fetch |
| `check_memory_status` | Poll an ingestion's status | `memories.status` |

## Scoping (same model as the SDK)

You control scope by which IDs you pass to the tools:

- **No `user_id` / `customer_id`** → **client scope** — shared across everyone on this key.
- **`user_id`** → **user scope** — private to that end-user.
- **`customer_id`** → **customer scope** — shared within that tenant (B2B).

Pass stable, immutable IDs — never display names.

## Failure semantics

The MCP layer mirrors the SDK contract: `recall_context` degrades gracefully (returns empty
context rather than failing the turn), while `log_exchange` surfaces write failures to the
client so a lost memory is visible, not silent. Tune timeouts server-side via
`MCP_RECALL_TIMEOUT_S` / `MCP_INGEST_TIMEOUT_S` if you self-host the MCP server.

## Live doc

`https://docs.maximem.ai/integrations/mcp`

---
> **Accurate as of** `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.2.4 (JS) — verified 2026-06-17.
> Hosted MCP server: `https://synap-mcp.maximem.ai/mcp`.
