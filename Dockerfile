# Pinned by digest; bump deliberately.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
# Install exactly the locked dependency versions, then the package itself.
RUN uv export --frozen --no-dev --no-emit-project -o /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt . \
    && rm /tmp/requirements.txt

# Holds no credentials: each request brings the caller's EverOS API key.
ENV EVEROS_MCP_TRANSPORT=http \
    EVEROS_MCP_HOST=0.0.0.0 \
    EVEROS_MCP_PORT=8765
EXPOSE 8765
USER nobody
# Same port variable the server reads, so overriding it keeps the check right.
HEALTHCHECK CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('EVEROS_MCP_PORT', '8765'))"
ENTRYPOINT ["everos-mcp"]
