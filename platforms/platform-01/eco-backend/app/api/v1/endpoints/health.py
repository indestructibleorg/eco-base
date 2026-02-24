# =============================================================================
# System Health Endpoints
# =============================================================================

from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.config import settings
from app.core.logging import get_logger
from app.db.base import get_db, engine

router = APIRouter()
logger = get_logger("health")


@router.get("/system")
async def system_health():
    """系統健康檢查"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/database")
async def database_health(db: AsyncSession = Depends(get_db)):
    """數據庫健康檢查"""
    try:
        # 執行簡單查詢測試連接
        result = await db.execute(text("SELECT 1"))
        await result.scalar()
        
        return {
            "status": "healthy",
            "component": "database",
            "message": "Database connection is working",
        }
    except Exception as e:
        logger.error("database_health_check_failed", error=str(e))
        return {
            "status": "unhealthy",
            "component": "database",
            "message": str(e),
        }


@router.get("/providers")
async def providers_health():
    """提供者健康檢查"""
    configs = settings.get_provider_configs()
    
    providers_status = {}
    for provider_id, config in configs.items():
        has_config = any(v for v in config.values() if v is not None)
        providers_status[provider_id] = {
            "configured": has_config,
            "status": "healthy" if has_config else "not_configured",
        }
    
    return {
        "status": "healthy",
        "component": "providers",
        "providers": providers_status,
    }


@router.get("/metrics")
async def system_metrics():
    """系統指標"""
    import psutil
    
    # 獲取系統資源使用情況
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "cpu": {
            "percent": cpu_percent,
            "count": psutil.cpu_count(),
        },
        "memory": {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used,
        },
        "disk": {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": (disk.used / disk.total) * 100,
        },
    }
