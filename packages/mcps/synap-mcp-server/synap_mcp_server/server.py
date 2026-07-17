"""FastMCP application — the uvicorn entrypoint (`synap_mcp_server.server:app`).

Builds a stateless Streamable HTTP MCP server mounted at /mcp, wraps it with middleware
that lifts the Bearer token off each request into a ContextVar, and exposes /health.
Stateless mode keeps the server safe for many concurrent no-code clients.
"""

import logging

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import tools
from .config import settings
from .context import set_token

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger("synap-mcp")

mcp = FastMCP(
    "synap",
    instructions=(
        "Synap memory tools. Forward every user message to log_exchange so it can be "
        "remembered, and call recall_context before replying to use what is already known."
    ),
    stateless_http=True,
    # DNS-rebinding protection restricts which Host/Origin headers the transport will
    # answer, blocking a malicious page from rebinding a victim's browser onto this
    # public endpoint. It's enabled with an explicit allowlist (settings.allowed_hosts
    # = the public proxied Host; allowed_origins = the dashboard CORS list). Because the
    # endpoint sits behind Cloudflare -> nginx, the proxy must forward a Host that is on
    # the allowlist; if that can't be arranged, MCP_DNS_REBINDING_PROTECTION=false
    # restores the prior permissive behavior without a code change.
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=settings.dns_rebinding_protection,
        allowed_hosts=list(settings.allowed_hosts),
        allowed_origins=list(settings.cors_allow_origins),
    ),
)
tools.register(mcp)

# Streamable HTTP Starlette app; the MCP endpoint is mounted at /mcp.
app = mcp.streamable_http_app()


class BearerCaptureMiddleware(BaseHTTPMiddleware):
    """Capture `Authorization: Bearer <token>` into the per-request ContextVar.

    The token is never stored beyond the request scope; downstream the REST client
    forwards it verbatim to synap-cloud, which owns validation.
    """

    async def dispatch(self, request: Request, call_next):
        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth[:7].lower() == "bearer " else None
        set_token(token)
        return await call_next(request)


app.add_middleware(BearerCaptureMiddleware)

# Allow the dashboard's in-app "Test my memory" (a browser → MCP call) cross-origin.
# Added last so it's the outermost layer and answers CORS preflight first.
# Server-to-server callers (Gumloop/n8n) don't send Origin and are unaffected.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)


async def _health(_request: Request) -> JSONResponse:
    return JSONResponse(
        {"status": "ok", "service": "synap-mcp", "environment": settings.environment}
    )


app.add_route("/health", _health, methods=["GET"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port, timeout_keep_alive=75)
