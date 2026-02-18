# ── SuperAI Platform - Production Dockerfile ─────────────────────
# Multi-stage build for minimal production image

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Production
FROM python:3.11-slim AS production

LABEL maintainer="SuperAI Engineering <eng@superai.platform>"
LABEL description="SuperAI Platform - Enterprise AI Inference Backend"
LABEL version="1.0.0"

# Security: non-root user
RUN groupadd -r superai && useradd -r -g superai -d /app -s /sbin/nologin superai

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY pyproject.toml .
COPY requirements.txt .

# Create necessary directories
RUN mkdir -p /models /data /logs && \
    chown -R superai:superai /app /models /data /logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()" || exit 1

USER superai

EXPOSE 8000

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    SUPERAI_HOST=0.0.0.0 \
    SUPERAI_PORT=8000

CMD ["uvicorn", "src.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--access-log", \
     "--log-level", "info"]