# =============================================================================
# Eco-Backend Main Application
# =============================================================================
# FastAPI 應用入口
# =============================================================================

import uuid
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.core.exceptions import register_exception_handlers
from app.core.security import get_cors_config
from app.core.metrics import (
    get_metrics, record_http_request, http_request_duration_seconds
)
from app.db.base import init_db, close_db
from app.api.v1.router import api_router
from app.services.platform_integration_service import platform_integration_service

# 配置日誌
configure_logging()
logger = get_logger("main")

# 創建限流器
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    # 啟動時
    logger.info(
        "application_starting",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )
    
    # 初始化數據庫
    await init_db()
    
    # 硬約束: 初始化平台集成服務，失敗則啟動失敗
    logger.info("initializing_platform_integration_service")
    try:
        await platform_integration_service.initialize(config={
            "supabase": {
                "api_key": settings.SUPABASE_API_KEY,
                "url": settings.SUPABASE_URL,
            },
            "openai": {
                "api_key": settings.OPENAI_API_KEY,
                "model": settings.OPENAI_MODEL or "gpt-4",
            },
            "pinecone": {
                "api_key": settings.PINECONE_API_KEY,
                "environment": settings.PINECONE_ENVIRONMENT,
            },
            "github": {
                "api_key": settings.GITHUB_API_KEY,
                "owner": settings.GITHUB_OWNER,
                "repo": settings.GITHUB_REPO,
            },
            "slack": {
                "api_key": settings.SLACK_API_KEY,
                "channel": settings.SLACK_CHANNEL,
            },
            "vercel": {
                "api_key": settings.VERCEL_API_KEY,
                "team_id": settings.VERCEL_TEAM_ID,
            },
        })
        logger.info("platform_integration_service_initialized")
    except Exception as e:
        logger.error("platform_integration_service_init_failed", error=str(e))
        # 硬約束: 初始化失敗導致啟動失敗
        raise RuntimeError(f"Failed to initialize platform integration service: {e}") from e
    
    yield
    
    # 關閉時
    logger.info("application_shutting_down")
    await close_db()


# 創建FastAPI應用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Eco-Platform Integration Backend API",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    openapi_url="/openapi.json" if not settings.is_production else None,
    lifespan=lifespan,
)

# 註冊限流器
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 註冊異常處理器
register_exception_handlers(app)

# 添加中間件 (注意：中間件執行順序與添加順序相反)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS中間件
cors_config = get_cors_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config["allow_origins"],
    allow_credentials=cors_config["allow_credentials"],
    allow_methods=cors_config["allow_methods"],
    allow_headers=cors_config["allow_headers"],
)


# Prometheus 指標追蹤中間件
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """指標收集中間件"""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # 記錄指標
    endpoint = request.url.path
    method = request.method
    status_code = response.status_code
    
    http_request_duration_seconds.labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)
    
    record_http_request(
        method=method,
        endpoint=endpoint,
        status_code=status_code
    )
    
    return response


# 註冊路由
app.include_router(api_router, prefix="/api")


# Prometheus 指標端點
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus 指標端點"""
    metrics_data, content_type = get_metrics()
    return Response(
        content=metrics_data,
        media_type=content_type
    )


# 健康檢查端點
@app.get("/health", tags=["Health"])
async def health_check():
    """健康檢查端點"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """就緒檢查端點"""
    from app.db.base import engine
    from sqlalchemy import text
    
    checks = {}
    all_healthy = True
    
    # 检查数据库连接
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = {"status": "ok"}
    except Exception as e:
        checks["database"] = {"status": "error", "error": str(e)}
        all_healthy = False
    
    # 检查平台集成服务
    try:
        from app.services.platform_integration_service import platform_integration_service
        platform_health = await platform_integration_service.health_check()
        checks["platform_integration"] = platform_health
        if platform_health.get("status") != "healthy":
            all_healthy = False
    except Exception as e:
        checks["platform_integration"] = {"status": "error", "error": str(e)}
        all_healthy = False
    
    # 检查 Redis 连接（如果配置了）
    if settings.REDIS_URL:
        try:
            import redis.asyncio as redis
            r = redis.from_url(settings.REDIS_URL)
            await r.ping()
            await r.close()
            checks["redis"] = {"status": "ok"}
        except Exception as e:
            checks["redis"] = {"status": "error", "error": str(e)}
            all_healthy = False
    
    status_code = 200 if all_healthy else 503
    
    return {
        "status": "ready" if all_healthy else "not_ready",
        "checks": checks,
    }


# 根端點
@app.get("/", tags=["Root"])
async def root():
    """根端點"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "documentation": "/docs",
        "health": "/health",
    }


# 啟動命令 (用於開發)
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.is_development,
        workers=1 if settings.is_development else settings.WORKERS,
    )
