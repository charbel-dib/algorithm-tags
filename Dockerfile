# Image to run the command line in production (prediction) or for batch training.
# Build:   docker build -t algo-tags .
# Predict: docker run --rm -v "$PWD/models:/app/models" -v "$PWD/examples:/app/examples" algo-tags predict --input examples/problem.json
# Logs are written to /app/logs (mount it with -v "$PWD/logs:/app/logs" to keep them).
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt . \
    && useradd --create-home app \
    && mkdir -p /app/logs /app/results /app/runs \
    && chown app /app/logs /app/results /app/runs
USER app

ENTRYPOINT ["python", "-m", "algo_tags"]
CMD ["--help"]
