# =============================================================================
# Eco-Backend Logging Configuration
# =============================================================================
# 結構化日誌配置
# =============================================================================

import logging
import sys
from typing import Any, Dict
import structlog
from pythonjsonlogger import jsonlogger

from app.core.config import settings


def configure_logging() -> None:
    """配置結構化日誌"""
    
    # 配置標準庫日誌
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper()),
    )
    
    # 配置 structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.LOG_FORMAT == "json" 
            else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # 配置第三方庫日誌級別
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.DEBUG if settings.DATABASE_ECHO else logging.WARNING
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """獲取結構化日誌記錄器"""
    return structlog.get_logger(name)


class JSONFormatter(jsonlogger.JsonFormatter):
    """自定義JSON格式器"""
    
    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        
        # 添加自定義字段
        log_record["service"] = settings.APP_NAME
        log_record["environment"] = settings.ENVIRONMENT
        log_record["version"] = settings.APP_VERSION
        
        # 重命名字段
        if "levelname" in log_record:
            log_record["level"] = log_record.pop("levelname")
        if "asctime" in log_record:
            log_record["timestamp"] = log_record.pop("asctime")


# 請求上下文日誌中間件
class RequestContextLogMiddleware:
    """請求上下文日誌中間件"""
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.logger = get_logger("request")
    
    async def __call__(self, request):
        # 綁定請求上下文
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=getattr(request.state, "request_id", "unknown"),
            user_id=getattr(request.state, "user_id", "anonymous"),
            path=request.url.path,
            method=request.method,
        )
        
        self.logger.info(
            "request_started",
            path=request.url.path,
            method=request.method,
            client=request.client.host if request.client else None,
        )
        
        response = await self.get_response(request)
        
        self.logger.info(
            "request_completed",
            status_code=response.status_code,
            path=request.url.path,
            method=request.method,
        )
        
        return response
