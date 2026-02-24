# =============================================================================
# Middleware
# =============================================================================

import time
import uuid
from typing import Optional, Dict, Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.logging import get_logger
from app.core.security import hash_api_key, verify_api_key
from app.core.exceptions import AuthenticationError, RateLimitError
from app.utils.security import sanitize_payload, sanitize_headers

logger = get_logger("middleware")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """請求ID中間件"""
    
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """日誌中間件 (帶敏感信息過濾)"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # 清理請求頭中的敏感信息
        safe_headers = sanitize_headers(dict(request.headers))
        
        # 記錄請求開始
        logger.info(
            "request_started",
            request_id=getattr(request.state, "request_id", None),
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None,
            user_agent=safe_headers.get("User-Agent"),
        )
        
        try:
            response = await call_next(request)
        except Exception as exc:
            # 記錄異常
            logger.error(
                "request_failed",
                request_id=getattr(request.state, "request_id", None),
                method=request.method,
                path=request.url.path,
                error=str(exc),
            )
            raise
        
        duration = time.time() - start_time
        
        # 記錄請求完成
        logger.info(
            "request_completed",
            request_id=getattr(request.state, "request_id", None),
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
        )
        
        return response


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """API密鑰認證中間件"""
    
    def __init__(
        self,
        app: ASGIApp,
        exempt_paths: Optional[list] = None,
    ):
        super().__init__(app)
        self.exempt_paths = exempt_paths or [
            "/health",
            "/ready",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/auth",
        ]
    
    async def dispatch(self, request: Request, call_next):
        # 檢查是否豁免
        path = request.url.path
        for exempt in self.exempt_paths:
            if path.startswith(exempt):
                return await call_next(request)
        
        # 檢查Authorization頭
        auth_header = request.headers.get("Authorization", "")
        
        if auth_header.startswith("Bearer "):
            # JWT認證 - 由端點處理
            return await call_next(request)
        
        if auth_header.startswith("ApiKey "):
            # API密鑰認證
            api_key = auth_header[7:]  # 去掉 "ApiKey " 前綴
            
            # 驗證API密鑰
            user_id = await self._validate_api_key(api_key)
            
            if user_id:
                request.state.user_id = user_id
                request.state.auth_type = "api_key"
                return await call_next(request)
        
        # 無認證信息，繼續處理（端點會檢查）
        return await call_next(request)
    
    async def _validate_api_key(self, api_key: str) -> Optional[str]:
        """驗證API密鑰"""
        from app.db.base import AsyncSessionLocal
        from app.models.user import ApiKey
        from sqlalchemy import select
        from datetime import datetime
        
        async with AsyncSessionLocal() as session:
            hashed_key = hash_api_key(api_key)
            
            result = await session.execute(
                select(ApiKey).where(
                    ApiKey.hashed_key == hashed_key,
                    ApiKey.is_active == True,
                )
            )
            key_record = result.scalar_one_or_none()
            
            if not key_record:
                return None
            
            # 檢查是否過期
            if key_record.expires_at and key_record.expires_at < datetime.utcnow():
                return None
            
            # 檢查月度配額
            if key_record.monthly_used >= key_record.monthly_quota:
                raise RateLimitError("Monthly quota exceeded")
            
            # 更新使用統計
            key_record.monthly_used += 1
            key_record.last_used_at = datetime.utcnow()
            await session.commit()
            
            return key_record.user_id


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """請求日誌記錄中間件 (帶敏感信息過濾)"""
    
    async def dispatch(self, request: Request, call_next):
        from app.db.base import AsyncSessionLocal
        from app.models.request_log import RequestLog
        from datetime import datetime
        
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = (time.time() - start_time) * 1000
        
        # 清理敏感信息
        safe_query_params = str(request.query_params)
        safe_user_agent = request.headers.get("User-Agent")
        
        # 異步記錄請求日誌
        async def log_request():
            try:
                async with AsyncSessionLocal() as session:
                    log_entry = RequestLog(
                        request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
                        user_id=getattr(request.state, "user_id", None),
                        method=request.method,
                        path=request.url.path,
                        query_params=safe_query_params[:500] if safe_query_params else None,  # 限制長度
                        client_ip=request.client.host if request.client else None,
                        user_agent=safe_user_agent[:200] if safe_user_agent else None,  # 限制長度
                        status_code=response.status_code,
                        duration_ms=duration,
                        created_at=datetime.utcnow(),
                    )
                    session.add(log_entry)
                    await session.commit()
            except Exception as e:
                logger.error("failed_to_log_request", error=str(e))
        
        # 不等待日誌記錄完成
        import asyncio
        asyncio.create_task(log_request())
        
        return response


class CORSMiddlewareCustom(BaseHTTPMiddleware):
    """自定義CORS中間件"""
    
    def __init__(
        self,
        app: ASGIApp,
        allow_origins: list = None,
        allow_methods: list = None,
        allow_headers: list = None,
        allow_credentials: bool = True,
    ):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["*"]
        self.allow_headers = allow_headers or ["*"]
        self.allow_credentials = allow_credentials
    
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("Origin", "")
        
        response = await call_next(request)
        
        # 設置CORS頭
        if "*" in self.allow_origins or origin in self.allow_origins:
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
        
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
        
        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response
