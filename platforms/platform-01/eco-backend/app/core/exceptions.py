# =============================================================================
# Eco-Backend Exceptions
# =============================================================================
# 自定義異常類與錯誤處理
# =============================================================================

from typing import Any, Dict, Optional, List
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import traceback

from app.core.logging import get_logger
from app.utils.security import sanitize_payload

logger = get_logger("exceptions")


# =============================================================================
# 自定義異常類
# =============================================================================

class EcoBaseException(Exception):
    """基礎異常類"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class AuthenticationError(EcoBaseException):
    """認證錯誤"""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTHENTICATION_ERROR",
            details=details
        )


class AuthorizationError(EcoBaseException):
    """授權錯誤"""
    
    def __init__(self, message: str = "Permission denied", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="AUTHORIZATION_ERROR",
            details=details
        )


class ResourceNotFoundError(EcoBaseException):
    """資源未找到錯誤"""
    
    def __init__(self, resource: str, identifier: str, details: Optional[Dict] = None):
        super().__init__(
            message=f"{resource} with identifier '{identifier}' not found",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="RESOURCE_NOT_FOUND",
            details=details
        )


class ValidationError(EcoBaseException):
    """驗證錯誤"""
    
    def __init__(self, message: str = "Validation failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details=details
        )


class RateLimitError(EcoBaseException):
    """限流錯誤"""
    
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details
        )


class ProviderError(EcoBaseException):
    """第三方提供者錯誤"""
    
    def __init__(
        self,
        provider: str,
        message: str = "Provider error",
        details: Optional[Dict] = None
    ):
        super().__init__(
            message=f"Provider '{provider}' error: {message}",
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code="PROVIDER_ERROR",
            details={"provider": provider, **(details or {})}
        )


class CircuitBreakerError(EcoBaseException):
    """熔斷器錯誤"""
    
    def __init__(self, provider: str, details: Optional[Dict] = None):
        super().__init__(
            message=f"Circuit breaker open for provider '{provider}'",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="CIRCUIT_BREAKER_OPEN",
            details={"provider": provider, **(details or {})}
        )


# =============================================================================
# 提供者相關詳細錯誤
# =============================================================================

class ProviderTimeoutError(ProviderError):
    """提供者超時錯誤"""
    
    def __init__(self, provider: str, timeout: float, details: Optional[Dict] = None):
        super().__init__(
            provider=provider,
            message=f"Request timed out after {timeout}s",
            details={"timeout": timeout, **(details or {})}
        )
        self.error_code = "PROVIDER_TIMEOUT"
        self.status_code = status.HTTP_504_GATEWAY_TIMEOUT


class ProviderRateLimitError(ProviderError):
    """提供者限流錯誤"""
    
    def __init__(self, provider: str, retry_after: Optional[int] = None, details: Optional[Dict] = None):
        super().__init__(
            provider=provider,
            message="Provider rate limit exceeded",
            details={"retry_after": retry_after, **(details or {})}
        )
        self.error_code = "PROVIDER_RATE_LIMIT"
        self.status_code = status.HTTP_429_TOO_MANY_REQUESTS


class ProviderAuthError(ProviderError):
    """提供者認證錯誤"""
    
    def __init__(self, provider: str, details: Optional[Dict] = None):
        super().__init__(
            provider=provider,
            message="Provider authentication failed",
            details=details
        )
        self.error_code = "PROVIDER_AUTH_ERROR"
        self.status_code = status.HTTP_502_BAD_GATEWAY


class ProviderQuotaError(ProviderError):
    """提供者配額耗盡錯誤"""
    
    def __init__(self, provider: str, details: Optional[Dict] = None):
        super().__init__(
            provider=provider,
            message="Provider quota exceeded",
            details=details
        )
        self.error_code = "PROVIDER_QUOTA_EXCEEDED"
        self.status_code = status.HTTP_402_PAYMENT_REQUIRED


class ProviderUnavailableError(ProviderError):
    """提供者不可用錯誤"""
    
    def __init__(self, provider: str, details: Optional[Dict] = None):
        super().__init__(
            provider=provider,
            message="Provider service unavailable",
            details=details
        )
        self.error_code = "PROVIDER_UNAVAILABLE"
        self.status_code = status.HTTP_503_SERVICE_UNAVAILABLE


# =============================================================================
# 數據庫相關錯誤
# =============================================================================

class DatabaseError(EcoBaseException):
    """數據庫錯誤"""
    
    def __init__(self, message: str = "Database error", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="DATABASE_ERROR",
            details=details
        )


class DatabaseConnectionError(DatabaseError):
    """數據庫連接錯誤"""
    
    def __init__(self, message: str = "Database connection failed", details: Optional[Dict] = None):
        super().__init__(message=message, details=details)
        self.error_code = "DATABASE_CONNECTION_ERROR"


class DatabaseTimeoutError(DatabaseError):
    """數據庫超時錯誤"""
    
    def __init__(self, message: str = "Database query timeout", details: Optional[Dict] = None):
        super().__init__(message=message, details=details)
        self.error_code = "DATABASE_TIMEOUT"


# =============================================================================
# 緩存相關錯誤
# =============================================================================

class CacheError(EcoBaseException):
    """緩存錯誤"""
    
    def __init__(self, message: str = "Cache error", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="CACHE_ERROR",
            details=details
        )


# =============================================================================
# 安全相關錯誤
# =============================================================================

class SecurityError(EcoBaseException):
    """安全錯誤"""
    
    def __init__(self, message: str = "Security error", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="SECURITY_ERROR",
            details=details
        )


class SQLInjectionError(SecurityError):
    """SQL 注入錯誤"""
    
    def __init__(self, message: str = "SQL injection detected", details: Optional[Dict] = None):
        super().__init__(message=message, details=details)
        self.error_code = "SQL_INJECTION_DETECTED"
        self.status_code = status.HTTP_400_BAD_REQUEST


class XSSAttackError(SecurityError):
    """XSS 攻擊錯誤"""
    
    def __init__(self, message: str = "XSS attack detected", details: Optional[Dict] = None):
        super().__init__(message=message, details=details)
        self.error_code = "XSS_ATTACK_DETECTED"
        self.status_code = status.HTTP_400_BAD_REQUEST


# =============================================================================
# 配置相關錯誤
# =============================================================================

class ConfigurationError(EcoBaseException):
    """配置錯誤"""
    
    def __init__(self, message: str = "Configuration error", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="CONFIGURATION_ERROR",
            details=details
        )


# =============================================================================
# 異常處理器
# =============================================================================

async def eco_exception_handler(request: Request, exc: EcoBaseException) -> JSONResponse:
    """處理自定義異常"""
    logger.error(
        "exception_occurred",
        error_code=exc.error_code,
        message=exc.message,
        status_code=exc.status_code,
        path=request.url.path,
        method=request.method,
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            },
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """處理請求驗證錯誤"""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })
    
    logger.warning(
        "validation_error",
        path=request.url.path,
        errors=errors,
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {"errors": errors},
            },
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """處理HTTP異常"""
    logger.warning(
        "http_exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "details": {},
            },
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """處理通用異常 (帶敏感信息過濾)"""
    error_trace = traceback.format_exc()
    
    # 清理可能的敏感信息
    safe_message = str(exc)
    if len(safe_message) > 500:
        safe_message = safe_message[:500] + "..."
    
    logger.error(
        "unhandled_exception",
        exception_type=type(exc).__name__,
        message=safe_message,
        path=request.url.path,
        method=request.method,
    )
    
    # 生產環境不返回詳細錯誤信息
    from app.core.config import settings
    
    if settings.is_production:
        message = "An internal error occurred"
        details = {}
    else:
        message = safe_message
        # 清理堆疊跟踪中的敏感信息
        safe_traceback = error_trace.split("\n")[:20]  # 限制行數
        details = {"traceback": safe_traceback}
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": message,
                "details": details,
            },
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# =============================================================================
# 註冊異常處理器
# =============================================================================

def register_exception_handlers(app):
    """註冊所有異常處理器"""
    app.add_exception_handler(EcoBaseException, eco_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
