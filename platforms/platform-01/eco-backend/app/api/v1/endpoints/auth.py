# =============================================================================
# Authentication Endpoints
# =============================================================================

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.security import verify_token
from app.core.logging import get_logger
from app.core.exceptions import AuthenticationError
from app.db.base import get_db
from app.services.user_service import UserService, ApiKeyService
from app.schemas.user import (
    LoginRequest, TokenResponse, RefreshTokenRequest,
    UserCreate, UserResponse, ApiKeyCreate
)

router = APIRouter()
logger = get_logger("auth")
security = HTTPBearer()
limiter = Limiter(key_func=get_remote_address)


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: Request,
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """用戶註冊"""
    user_service = UserService(db)
    user = await user_service.create_user(user_data)
    return user


@router.post("/login", response_model=TokenResponse)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def login(
    request: Request,
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """用戶登錄"""
    user_service = UserService(db)
    user, access_token, refresh_token = await user_service.authenticate(
        username=login_data.username,
        password=login_data.password
    )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: Request,
    refresh_data: RefreshTokenRequest
):
    """刷新訪問令牌"""
    from app.core.security import (
        verify_token, create_access_token, create_refresh_token,
        add_token_to_blacklist
    )
    
    payload = verify_token(refresh_data.refresh_token, token_type="refresh")
    
    if not payload:
        raise AuthenticationError("Invalid or expired refresh token")
    
    user_id = payload.get("sub")
    email = payload.get("email")
    username = payload.get("username")
    
    # 將舊刷新令牌加入黑名單（防止重複使用）
    add_token_to_blacklist(refresh_data.refresh_token)
    
    # 生成新的令牌
    token_data = {"sub": user_id, "email": email, "username": username}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    logger.info("token_refreshed", user_id=user_id)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/api-keys")
async def create_api_key(
    request: Request,
    key_data: ApiKeyCreate,
    db: AsyncSession = Depends(get_db)
):
    """創建API密鑰"""
    from app.core.security import get_current_user_id
    
    # 獲取當前用戶
    credentials = await security(request)
    user_id = await get_current_user_id(credentials)
    
    api_key_service = ApiKeyService(db)
    api_key, raw_key = await api_key_service.create_api_key(user_id, key_data)
    
    return {
        "id": api_key.id,
        "name": api_key.name,
        "api_key": raw_key,  # 僅返回一次
        "rate_limit": api_key.rate_limit,
        "monthly_quota": api_key.monthly_quota,
        "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
        "created_at": api_key.created_at.isoformat() if api_key.created_at else None,
    }


@router.post("/logout")
async def logout(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """用戶登出（使令牌失效）"""
    from app.core.security import (
        get_current_user_id, add_token_to_blacklist, security_scheme
    )
    
    credentials = await security(request)
    user_id = await get_current_user_id(credentials)
    
    # 將當前令牌加入黑名單
    add_token_to_blacklist(credentials.credentials)
    
    logger.info("user_logged_out", user_id=user_id)
    
    return {"success": True, "message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """獲取當前用戶信息"""
    from app.core.security import get_current_user_id
    
    credentials = await security(request)
    user_id = await get_current_user_id(credentials)
    
    user_service = UserService(db)
    user = await user_service.get_user(user_id)
    return user
