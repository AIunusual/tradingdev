FROM python:3.14.10-slim-bookworm AS base

# Required by the Linux LightGBM/XGBoost wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-cache
COPY src/ ./src/
RUN uv sync --locked --no-dev --no-editable --no-cache

FROM base AS runtime
RUN groupadd --gid 1000 tradingdev \
    && useradd --uid 1000 --gid tradingdev --create-home tradingdev \
    && mkdir /workspace \
    && chown tradingdev:tradingdev /workspace
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRADINGDEV_WORKSPACE=/workspace \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    RUFF_CACHE_DIR=/tmp/ruff \
    MYPY_CACHE_DIR=/tmp/mypy \
    XDG_CACHE_HOME=/tmp/cache
USER tradingdev
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import socket; socket.create_connection(('127.0.0.1', 8000), timeout=3).close()"]
ENTRYPOINT ["tradingdev-mcp"]
CMD ["--web", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]
