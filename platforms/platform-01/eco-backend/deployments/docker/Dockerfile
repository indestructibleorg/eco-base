# =============================================================================
# Eco-Backend Dockerfile
# =============================================================================

FROM python:3.11-slim as builder

# 設置工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 安裝Python依賴
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# =============================================================================
# 生產階段
# =============================================================================
FROM python:3.11-slim

# 設置環境變數
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/root/.local/bin:$PATH

# 安裝運行時依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 創建非root用戶
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# 設置工作目錄
WORKDIR /app

# 複製依賴
COPY --from=builder /root/.local /root/.local

# 複製應用代碼
COPY app/ ./app/
COPY migrations/ ./migrations/
COPY alembic.ini .

# 設置權限
RUN chown -R appuser:appgroup /app

# 切換到非root用戶
USER appuser

# 暴露端口
EXPOSE 8000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 啟動命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
