FROM python:3.11-slim

WORKDIR /app

# Dependencies only (no torch / no heavy ML deps — this is a thin adapter).
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "mcp[cli]>=1.26.0" \
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
