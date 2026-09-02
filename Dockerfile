# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for torch/faiss/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app

# Copy source code and pre-built artifacts
COPY src/       ./src/
COPY artifacts/ ./artifacts/

# Default environment — overridden by docker-compose
ENV MLFLOW_TRACKING_URI=http://mlflow:5000 \
    REDIS_HOST=redis \
    REDIS_PORT=6379 \
    EMBEDDING_CACHE_TTL_SECONDS=3600 \
    ARTIFACTS_DIR=/app/artifacts \
    DRIFT_KL_THRESHOLD=0.5 \
    DRIFT_WINDOW=100

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
