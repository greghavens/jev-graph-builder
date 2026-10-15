# The MCP server over HTTPS (claude.ai, ChatGPT). Build from the repo root:
#   fly deploy . -c deploy/fly/mcp/fly.toml --dockerfile deploy/fly/mcp/Dockerfile
# with the build directory's registry copied to deploy/fly/mcp/registry first.
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 HF_HOME=/models PIP_NO_CACHE_DIR=1
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY jev_graph_builder ./jev_graph_builder
RUN pip install .
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
COPY deploy/fly/mcp/registry /registry
ENV JGB_REGISTRY_PATH=/registry JGB_DEVICE=cpu HF_HUB_OFFLINE=1
EXPOSE 8080
CMD ["jev-graph-builder", "mcp", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8080"]
