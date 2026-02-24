# =============================================================================
# Eco-Backend Security Module
# =============================================================================
# 認證授權與安全相關功能
# =============================================================================

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union, Set
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import hashlib

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("security")

# 密碼哈希上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer 認證
security_scheme = HTTPBearer(auto_error=False)

# 令牌黑名單（用於登出/刷新時使舊令牌失效）
# 生產環境應使用 Redis
_token_blacklist: Set[str] = set()


def add_token_to_blacklist(token: str) -> None:
    """將令牌加入黑名單"""
    _token_blacklist.add(token)
    logger.info("token_added_to_blacklist", token_preview=token[:20])


def is_token_blacklisted(token: str) -> bool:
    """檢查令牌是否在黑名單中"""
    return token in _token_blacklist


def clear_expired_tokens() -> None:
    """清理過期的黑名單令牌（應定期調用）"""
    # 實際實現中應檢查令牌過期時間
    # 這裡簡化處理，生產環境應使用 Redis TTL
    pass


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """驗證密碼"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """獲取密碼哈希"""
    return pwd_context.hash(password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """創建訪問令牌"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """創建刷新令牌"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    
    return encoded_jwt


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """解碼令牌"""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError as e:
        logger.warning("token_decode_failed", error=str(e))
        return None


def verify_token(token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
    """驗證令牌"""
    # 檢查令牌是否在黑名單中
    if is_token_blacklisted(token):
        logger.warning("token_blacklisted")
        return None
    
    payload = decode_token(token)
    
    if payload is None:
        return None
    
    # 檢查令牌類型
    if payload.get("type") != token_type:
        logger.warning("token_type_mismatch", expected=token_type, got=payload.get("type"))
        return None
    
    # 檢查過期時間
    exp = payload.get("exp")
    if exp is None or datetime.utcnow() > datetime.fromtimestamp(exp):
        logger.warning("token_expired")
        return None
    
    return payload


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme)
) -> str:
    """獲取當前用戶ID (依賴函數)"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    payload = verify_token(credentials.credentials)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_id


async def get_optional_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme)
) -> Optional[str]:
    """可選用戶ID獲取 (依賴函數)"""
    if credentials is None:
        return None
    
    payload = verify_token(credentials.credentials)
    if payload is None:
        return None
    
    return payload.get("sub")


def generate_api_key() -> str:
    """生成API密鑰"""
    return f"eco_{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    """哈希API密鑰"""
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """驗證API密鑰"""
    return hash_api_key(api_key) == hashed_key


class PermissionChecker:
    """權限檢查器"""
    
    def __init__(self, required_permissions: list):
        self.required_permissions = required_permissions
    
    async def __call__(self, user_id: str = Depends(get_current_user_id)) -> str:
        # 从数据库获取用户权限并检查
        user_permissions = await self._get_user_permissions(user_id)
        
        # 检查是否有所需权限
        for permission in self.required_permissions:
            if permission not in user_permissions:
                logger.warning(
                    "permission_denied",
                    user_id=user_id,
                    required_permission=permission,
                    user_permissions=list(user_permissions),
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {permission}",
                )
        
        return user_id
    
    async def _get_user_permissions(self, user_id: str) -> Set[str]:
        """从数据库获取用户权限"""
        try:
            from app.db.base import AsyncSessionLocal
            from app.models.user import User
            from sqlalchemy import select
            
            async with AsyncSessionLocal() as session:
                # 查询用户及其角色权限
                result = await session.execute(
                    select(User).where(User.id == user_id)
                )
                user = result.scalar_one_or_none()
                
                if not user:
                    return set()
                
                # 收集用户权限
                permissions = set()
                
                # 添加用户直接权限
                if hasattr(user, 'permissions') and user.permissions:
                    permissions.update(user.permissions)
                
                # 添加角色权限
                if hasattr(user, 'roles') and user.roles:
                    for role in user.roles:
                        if hasattr(role, 'permissions') and role.permissions:
                            permissions.update(role.permissions)
                
                return permissions
                
        except Exception as e:
            logger.error("failed_to_get_user_permissions", user_id=user_id, error=str(e))
            # 出错时返回空权限集（拒绝访问）
            return set()


def require_permissions(permissions: list):
    """權限要求裝飾器"""
    return Depends(PermissionChecker(permissions))


# CORS 配置
def get_cors_config() -> Dict[str, Any]:
    """獲取CORS配置"""
    return {
        "allow_origins": settings.CORS_ORIGINS,
        "allow_credentials": settings.CORS_ALLOW_CREDENTIALS,
        "allow_methods": settings.CORS_ALLOW_METHODS,
        "allow_headers": settings.CORS_ALLOW_HEADERS,
    }
