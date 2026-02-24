# =============================================================================
# User Schemas
# =============================================================================

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr, ConfigDict

from app.schemas.base import BaseSchema, TimestampSchema, IDSchema


# =============================================================================
# User Base Schemas
# =============================================================================

class UserBase(BaseSchema):
    """用戶基礎Schema"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    full_name: Optional[str] = Field(None, max_length=255)


class UserCreate(UserBase):
    """用戶創建Schema"""
    password: str = Field(..., min_length=8, max_length=100)


class UserUpdate(BaseSchema):
    """用戶更新Schema"""
    full_name: Optional[str] = Field(None, max_length=255)
    avatar_url: Optional[str] = Field(None, max_length=500)


class UserResponse(UserBase, IDSchema, TimestampSchema):
    """用戶響應Schema"""
    avatar_url: Optional[str] = None
    is_active: bool
    is_verified: bool
    last_login_at: Optional[datetime] = None


class UserInDB(UserBase):
    """數據庫用戶Schema"""
    id: str
    hashed_password: str
    is_active: bool
    is_verified: bool
    is_superuser: bool


# =============================================================================
# Authentication Schemas
# =============================================================================

class TokenResponse(BaseSchema):
    """令牌響應Schema"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class LoginRequest(BaseSchema):
    """登錄請求Schema"""
    username: str
    password: str


class RefreshTokenRequest(BaseSchema):
    """刷新令牌請求Schema"""
    refresh_token: str


class PasswordChangeRequest(BaseSchema):
    """密碼修改請求Schema"""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)


# =============================================================================
# API Key Schemas
# =============================================================================

class ApiKeyBase(BaseSchema):
    """API密鑰基礎Schema"""
    name: str = Field(..., min_length=1, max_length=100)


class ApiKeyCreate(ApiKeyBase):
    """API密鑰創建Schema"""
    permissions: Optional[List[str]] = None
    rate_limit: Optional[int] = Field(None, ge=1, le=10000)
    monthly_quota: Optional[int] = Field(None, ge=1)
    expires_at: Optional[datetime] = None


class ApiKeyUpdate(BaseSchema):
    """API密鑰更新Schema"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    is_active: Optional[bool] = None


class ApiKeyResponse(ApiKeyBase, IDSchema, TimestampSchema):
    """API密鑰響應Schema"""
    permissions: Optional[List[str]] = None
    rate_limit: int
    monthly_quota: int
    monthly_used: int
    is_active: bool
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None


class ApiKeyWithSecretResponse(ApiKeyResponse):
    """包含密鑰的API密鑰響應Schema"""
    api_key: str  # 僅在創建時返回


class ApiKeyListResponse(BaseSchema):
    """API密鑰列表響應Schema"""
    items: List[ApiKeyResponse]
    total: int


# =============================================================================
# Provider Config Schemas
# =============================================================================

class ProviderConfigBase(BaseSchema):
    """提供者配置基礎Schema"""
    provider_id: str = Field(..., min_length=1, max_length=50)
    provider_name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class ProviderConfigCreate(ProviderConfigBase):
    """提供者配置創建Schema"""
    config: dict  # 明文配置，將被加密存儲


class ProviderConfigUpdate(BaseSchema):
    """提供者配置更新Schema"""
    config: Optional[dict] = None
    is_active: Optional[bool] = None
    provider_name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class ProviderConfigResponse(ProviderConfigBase, IDSchema, TimestampSchema):
    """提供者配置響應Schema"""
    is_active: bool
    # 注意：config 字段不返回，需要單獨接口獲取


class ProviderConfigListResponse(BaseSchema):
    """提供者配置列表響應Schema"""
    items: List[ProviderConfigResponse]
    total: int
