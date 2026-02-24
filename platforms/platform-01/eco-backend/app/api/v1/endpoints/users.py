# =============================================================================
# User Endpoints
# =============================================================================

from typing import List
from fastapi import APIRouter, Depends, Request, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user_id
from app.core.logging import get_logger
from app.db.base import get_db
from app.services.user_service import UserService, ApiKeyService
from app.schemas.base import ResponseSchema
from app.schemas.user import (
    UserResponse, UserUpdate,
    ApiKeyResponse, ApiKeyCreate, ApiKeyUpdate, ApiKeyListResponse
)

router = APIRouter()
logger = get_logger("users")


@router.get("/me", response_model=ResponseSchema[UserResponse])
async def get_me(
    request: Request,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """獲取當前用戶信息"""
    user_service = UserService(db)
    user = await user_service.get_user(user_id)
    return ResponseSchema(data=user)


@router.patch("/me", response_model=ResponseSchema[UserResponse])
async def update_me(
    request: Request,
    update_data: UserUpdate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """更新當前用戶信息"""
    user_service = UserService(db)
    user = await user_service.update_user(user_id, update_data)
    return ResponseSchema(data=user)


# =============================================================================
# API Key Management
# =============================================================================

@router.get("/me/api-keys", response_model=ResponseSchema[ApiKeyListResponse])
async def list_api_keys(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """列出當前用戶的API密鑰"""
    api_key_service = ApiKeyService(db)
    api_keys, total = await api_key_service.list_api_keys(user_id, page, page_size)
    
    return ResponseSchema(
        data=ApiKeyListResponse(
            items=list(api_keys),
            total=total,
        )
    )


@router.post("/me/api-keys", response_model=ResponseSchema[dict])
async def create_api_key(
    request: Request,
    key_data: ApiKeyCreate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """創建API密鑰"""
    api_key_service = ApiKeyService(db)
    api_key, raw_key = await api_key_service.create_api_key(user_id, key_data)
    
    return ResponseSchema(
        data={
            "id": api_key.id,
            "name": api_key.name,
            "api_key": raw_key,
            "rate_limit": api_key.rate_limit,
            "monthly_quota": api_key.monthly_quota,
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
            "created_at": api_key.created_at.isoformat() if api_key.created_at else None,
        }
    )


@router.patch("/me/api-keys/{key_id}", response_model=ResponseSchema[ApiKeyResponse])
async def update_api_key(
    request: Request,
    key_id: str,
    update_data: ApiKeyUpdate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """更新API密鑰"""
    api_key_service = ApiKeyService(db)
    api_key = await api_key_service.update_api_key(user_id, key_id, update_data)
    return ResponseSchema(data=api_key)


@router.delete("/me/api-keys/{key_id}")
async def delete_api_key(
    request: Request,
    key_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """刪除API密鑰"""
    api_key_service = ApiKeyService(db)
    await api_key_service.delete_api_key(user_id, key_id)
    return ResponseSchema(data={"deleted": True})
