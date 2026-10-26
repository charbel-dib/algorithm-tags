# Image to run the command line in production (prediction) or for batch training.
# Build:   docker build -t algo-tags .
# Predict: docker run --rm -v "$PWD/models:/app/models" -v "$PWD/examples:/app/examples" algo-tags predict --input examples/problem.json
# Logs are written to /app/logs (mount it with -v "$PWD/logs:/app/logs" to keep them).
#
# The base image is pinned by digest and uv by version, so that a rebuild gives the same image.
# Update the digest deliberately to get the security fixes of newer Python images.
FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/uv

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never

# 1. Dependencies only, at the exact versions of uv.lock. This layer is rebuilt only when the
#    lock changes, not when the code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-editable

# 2. The package itself, then a non-root user with the folders it writes to.
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable \
    && useradd --create-home app \
    && mkdir -p /app/logs /app/results /app/runs \
    && chown app /app/logs /app/results /app/runs

ENV PATH="/app/.venv/bin:$PATH"
USER app

ENTRYPOINT ["python", "-m", "algo_tags"]
CMD ["--help"]
