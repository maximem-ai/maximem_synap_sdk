"""Hosted remote MCP server — a stateless Streamable HTTP adapter over the Synap public REST API.

Re-fronts the existing ingest (write) and context-fetch (read) operations as MCP tools so
no-code platforms (Gumloop, n8n) can give their agents persistent memory with nothing but a
pasted MCP URL and a Bearer token. No new backend, storage, or pipeline.
"""

__version__ = "0.1.0"
