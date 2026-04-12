FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Install the project itself
RUN uv sync --frozen

# Add venv to PATH so uvicorn can be called directly
ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "baize.main:app", "--host", "0.0.0.0", "--port", "8000"]
