# =============================================================================
# User Service
# =============================================================================
# 用戶業務邏輯服務層
# =============================================================================

from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.config import settings
from app.core.security import (
    verify_password, get_password_hash,
    create_access_token, create_refresh_token,
    generate_api_key, hash_api_key
)
from app.core.logging import get_logger
from app.core.exceptions import (
    AuthenticationError, ValidationError, ResourceNotFoundError
)
from app.models.user import User, ApiKey
from app.schemas.user import (
    UserCreate, UserUpdate, UserResponse,
    ApiKeyCreate, ApiKeyUpdate
)

logger = get_logger("user_service")


class UserService:
    """用戶服務"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_user(self, user_data: UserCreate) -> User:
        """創建用戶"""
        # 檢查郵箱是否已存在
        existing = await self._get_user_by_email(user_data.email)
        if existing:
            raise ValidationError("Email already registered")
        
        # 檢查用戶名是否已存在
        existing = await self._get_user_by_username(user_data.username)
        if existing:
            raise ValidationError("Username already taken")
        
        # 創建用戶
        user = User(
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            hashed_password=get_password_hash(user_data.password),
        )
        
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        
        logger.info("user_created", user_id=user.id, email=user.email)
        
        return user
    
    async def authenticate(
        self,
        username: str,
        password: str
    ) -> tuple[User, str, str]:
        """
        認證用戶
        
        Returns:
            (user, access_token, refresh_token)
        """
        # 查找用戶
        result = await self.db.execute(
            select(User).where(
                (User.email == username) | (User.username == username)
            )
        )
        user = result.scalar_one_or_none()
        
        # 驗證用戶和密碼
        if not user or not verify_password(password, user.hashed_password):
            raise AuthenticationError("Invalid credentials")
        
        if not user.is_active:
            raise AuthenticationError("Account is disabled")
        
        # 更新最後登錄時間
        user.last_login_at = datetime.utcnow()
        await self.db.commit()
        
        # 生成令牌
        token_data = {
            "sub": user.id,
            "email": user.email,
            "username": user.username
        }
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)
        
        logger.info("user_authenticated", user_id=user.id)
        
        return user, access_token, refresh_token
    
    async def get_user(self, user_id: str) -> User:
        """獲取用戶"""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise ResourceNotFoundError("User", user_id)
        
        return user
    
    async def update_user(
        self,
        user_id: str,
        update_data: UserUpdate
    ) -> User:
        """更新用戶"""
        user = await self.get_user(user_id)
        
        if update_data.full_name is not None:
            user.full_name = update_data.full_name
        
        if update_data.avatar_url is not None:
            user.avatar_url = update_data.avatar_url
        
        await self.db.commit()
        await self.db.refresh(user)
        
        logger.info("user_updated", user_id=user_id)
        
        return user
    
    async def _get_user_by_email(self, email: str) -> Optional[User]:
        """通過郵箱獲取用戶"""
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
    
    async def _get_user_by_username(self, username: str) -> Optional[User]:
        """通過用戶名獲取用戶"""
        result = await self.db.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()


class ApiKeyService:
    """API 密鑰服務"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_api_key(
        self,
        user_id: str,
        key_data: ApiKeyCreate
    ) -> tuple[ApiKey, str]:
        """
        創建 API 密鑰
        
        Returns:
            (api_key_record, raw_api_key) - 注意：raw_api_key 只返回一次
        """
        # 生成密鑰
        raw_key = generate_api_key()
        hashed_key = hash_api_key(raw_key)
        
        # 創建記錄
        api_key = ApiKey(
            user_id=user_id,
            name=key_data.name,
            hashed_key=hashed_key,
            permissions=",".join(key_data.permissions) if key_data.permissions else None,
            rate_limit=key_data.rate_limit or 100,
            monthly_quota=key_data.monthly_quota or 10000,
            expires_at=key_data.expires_at,
        )
        
        self.db.add(api_key)
        await self.db.commit()
        await self.db.refresh(api_key)
        
        logger.info("api_key_created", user_id=user_id, key_id=api_key.id)
        
        return api_key, raw_key
    
    async def list_api_keys(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20
    ) -> tuple[List[ApiKey], int]:
        """
        列出用戶的 API 密鑰
        
        Returns:
            (api_keys, total_count)
        """
        # 計算總數
        count_result = await self.db.execute(
            select(func.count()).where(ApiKey.user_id == user_id)
        )
        total = count_result.scalar()
        
        # 查詢數據
        result = await self.db.execute(
            select(ApiKey)
            .where(ApiKey.user_id == user_id)
            .offset((page - 1) * page_size)
            .limit(page_size)
            .order_by(ApiKey.created_at.desc())
        )
        api_keys = result.scalars().all()
        
        return list(api_keys), total
    
    async def update_api_key(
        self,
        user_id: str,
        key_id: str,
        update_data: ApiKeyUpdate
    ) -> ApiKey:
        """更新 API 密鑰"""
        result = await self.db.execute(
            select(ApiKey).where(
                ApiKey.id == key_id,
                ApiKey.user_id == user_id
            )
        )
        api_key = result.scalar_one_or_none()
        
        if not api_key:
            raise ResourceNotFoundError("API Key", key_id)
        
        if update_data.name is not None:
            api_key.name = update_data.name
        
        if update_data.is_active is not None:
            api_key.is_active = update_data.is_active
        
        await self.db.commit()
        await self.db.refresh(api_key)
        
        logger.info("api_key_updated", user_id=user_id, key_id=key_id)
        
        return api_key
    
    async def delete_api_key(self, user_id: str, key_id: str) -> None:
        """刪除 API 密鑰"""
        result = await self.db.execute(
            select(ApiKey).where(
                ApiKey.id == key_id,
                ApiKey.user_id == user_id
            )
        )
        api_key = result.scalar_one_or_none()
        
        if not api_key:
            raise ResourceNotFoundError("API Key", key_id)
        
        await self.db.delete(api_key)
        await self.db.commit()
        
        logger.info("api_key_deleted", user_id=user_id, key_id=key_id)
    
    async def validate_api_key(self, api_key: str) -> Optional[ApiKey]:
        """驗證 API 密鑰"""
        hashed_key = hash_api_key(api_key)
        
        result = await self.db.execute(
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
            from app.core.exceptions import RateLimitError
            raise RateLimitError("Monthly quota exceeded")
        
        # 更新使用統計
        key_record.monthly_used += 1
        key_record.last_used_at = datetime.utcnow()
        await self.db.commit()
        
        return key_record
