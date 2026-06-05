"""FastMCP application — the uvicorn entrypoint (`synap_mcp_server.server:app`).

Builds a stateless Streamable HTTP MCP server mounted at /mcp, wraps it with middleware
that lifts the Bearer token off each request into a ContextVar, and exposes /health.
Stateless mode keeps the server safe for many concurrent no-code clients.
"""

import logging

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
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
    # This server sits behind a reverse proxy / Cloudflare and authenticates every
    # request with a Bearer token, so the transport's DNS-rebinding Host/Origin check
    # adds no security — it only rejects the proxied public Host ("Invalid Host header"
    # on /mcp). Disable it so any proxy works without a Host-rewrite hack.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
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


async def _health(_request: Request) -> JSONResponse:
    return JSONResponse(
        {"status": "ok", "service": "synap-mcp", "environment": settings.environment}
    )


app.add_route("/health", _health, methods=["GET"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port, timeout_keep_alive=75)
