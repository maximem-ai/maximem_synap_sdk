FROM python:3.11-slim

WORKDIR /app

# Dependencies only (no torch / no heavy ML deps — this is a thin adapter).
#
# NOTE: this list duplicates [project.dependencies] in pyproject.toml — the
# COPY below is only there for layer invalidation, pip never reads it. Editing
# pyproject alone changes nothing about the built image, so keep both in sync.
#
# The <2 bound is load-bearing: mcp 2.0 removed `mcp.server.fastmcp` (now
# `mcp.server.mcpserver`), so server.py fails to import and this container
# crash-loops while the rest of the stack reports healthy. Lift it only
# together with migrating to the 2.0 API.
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "mcp[cli]>=1.26.0,<2" \
    "httpx>=0.27" \
    "uvicorn>=0.30" \
    "starlette>=0.37"

COPY synap_mcp_server ./synap_mcp_server

# Run as non-root.
RUN useradd -m -u 10001 mcp && chown -R mcp:mcp /app
USER mcp

EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8090/health')"

CMD ["uvicorn", "synap_mcp_server.server:app", \
     "--host", "0.0.0.0", "--port", "8090", \
     "--timeout-keep-alive", "75"]
