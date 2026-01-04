FROM ghcr.io/astral-sh/uv:python3.13-trixie AS builder
WORKDIR /app
COPY pyproject.toml uv.lock* README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev

FROM python:3.13-slim
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src/ ./src/
USER app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "yfinance_mcp.server"]
