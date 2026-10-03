FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv pip install --system --no-cache .

# Holds no credentials: each request brings the caller's EverOS API key.
ENV EVEROS_MCP_TRANSPORT=http \
    EVEROS_MCP_HOST=0.0.0.0 \
    EVEROS_MCP_PORT=8765
EXPOSE 8765
USER nobody
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz')"
ENTRYPOINT ["everos-mcp"]
