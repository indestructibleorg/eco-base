"""
Authentication & authorization schemas.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


class UserRole(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    DEVELOPER = "developer"
    VIEWER = "viewer"


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class APIKeyCreate(BaseModel):
    name: str
    description: Optional[str] = None
    role: UserRole = UserRole.DEVELOPER
    rate_limit_per_minute: int = 600
    allowed_models: List[str] = []
    expires_at: Optional[datetime] = None


class APIKeyInfo(BaseModel):
    key_id: str
    name: str
    prefix: str  # first 8 chars for identification
    role: UserRole
    rate_limit_per_minute: int
    allowed_models: List[str]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    total_requests: int = 0
    is_active: bool = True


class APIKeyResponse(BaseModel):
    """Returned only on creation - full key shown once."""
    key: str
    info: APIKeyInfo


class UserInfo(BaseModel):
    user_id: str
    username: str
    email: Optional[str] = None
    role: UserRole
    is_active: bool = True
    created_at: datetime
    api_keys: List[APIKeyInfo] = []