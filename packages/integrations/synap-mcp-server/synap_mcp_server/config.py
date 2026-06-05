"""Environment-driven settings. No secrets live here — the end user's Synap key
arrives per-request as a Bearer token (see context.py)."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    synap_api_url: str = os.getenv("SYNAP_API_URL", "http://synap-cloud:8000")
    port: int = int(os.getenv("MCP_PORT", "8090"))
    recall_timeout_s: float = float(os.getenv("MCP_RECALL_TIMEOUT_S", "10"))
    ingest_timeout_s: float = float(os.getenv("MCP_INGEST_TIMEOUT_S", "8"))
    default_max_results: int = int(os.getenv("MCP_DEFAULT_MAX_RESULTS", "10"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("ENVIRONMENT", "production")
    # Browser origins allowed to call the MCP endpoint directly (the dashboard's
    # in-app "Test my memory"). Server-to-server callers (Gumloop/n8n) are unaffected.
    cors_allow_origins: tuple = tuple(
        o.strip()
        for o in os.getenv(
            "MCP_CORS_ALLOW_ORIGINS",
            "https://synap.maximem.ai,https://synap-admin.maximem.ai,"
            "http://localhost:3000,http://localhost:5173",
        ).split(",")
        if o.strip()
    )
    # DNS-rebinding protection. Enabled by default with an allowlist of the public
    # Host(s) this server answers on. The endpoint sits behind Cloudflare -> nginx ->
    # container, so allowed_hosts MUST include whatever Host the proxy forwards (the
    # public domain). If the proxy can't be made to forward a matching Host, set
    # MCP_DNS_REBINDING_PROTECTION=false to fall back to the previous behavior rather
    # than 400-ing every /mcp request. allowed_origins reuses the CORS allowlist.
    dns_rebinding_protection: bool = (
        os.getenv("MCP_DNS_REBINDING_PROTECTION", "true").lower()
        not in ("0", "false", "no")
    )
    allowed_hosts: tuple = tuple(
        h.strip()
        for h in os.getenv(
            "MCP_ALLOWED_HOSTS",
            "synap-mcp.maximem.ai,localhost,localhost:8090,127.0.0.1,127.0.0.1:8090",
        ).split(",")
        if h.strip()
    )


settings = Settings()
