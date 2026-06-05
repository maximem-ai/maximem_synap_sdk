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


settings = Settings()
