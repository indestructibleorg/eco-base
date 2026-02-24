# =============================================================================
# User Model
# =============================================================================

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, String, Boolean, DateTime, Integer, Text, ForeignKey
from sqlalchemy.orm import relationship

from app.db.base import Base


class User(Base):
    """用戶模型"""
    
    __tablename__ = "users"
    
    # 基本信息
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=True)
    
    # 個人信息
    full_name = Column(String(255), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    
    # 狀態
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # 時間戳
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    verified_at = Column(DateTime(timezone=True), nullable=True)
    
    # 關聯
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    provider_configs = relationship("UserProviderConfig", back_populates="user", cascade="all, delete-orphan")
    requests = relationship("RequestLog", back_populates="user")
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"


class ApiKey(Base):
    """API密鑰模型"""
    
    __tablename__ = "api_keys"
    
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    hashed_key = Column(String(255), unique=True, nullable=False, index=True)
    
    # 權限
    permissions = Column(Text, nullable=True)  # JSON格式存儲
    
    # 限額
    rate_limit = Column(Integer, default=100, nullable=False)  # 每分鐘請求數
    monthly_quota = Column(Integer, default=10000, nullable=False)
    monthly_used = Column(Integer, default=0, nullable=False)
    
    # 狀態
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    
    # 關聯
    user = relationship("User", back_populates="api_keys")
    
    def __repr__(self) -> str:
        return f"<ApiKey(id={self.id}, name={self.name})>"


class UserProviderConfig(Base):
    """用戶第三方平台配置模型"""
    
    __tablename__ = "user_provider_configs"
    
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    provider_id = Column(String(50), nullable=False, index=True)
    
    # 加密存儲的配置
    encrypted_config = Column(Text, nullable=False)
    
    # 狀態
    is_active = Column(Boolean, default=True, nullable=False)
    
    # 元數據
    provider_name = Column(String(100), nullable=True)
    description = Column(String(500), nullable=True)
    
    # 關聯
    user = relationship("User", back_populates="provider_configs")
    
    __table_args__ = (
        # 每個用戶每個提供者只能有一個配置
        {"sqlite_autoincrement": True},
    )
    
    def __repr__(self) -> str:
        return f"<UserProviderConfig(user_id={self.user_id}, provider={self.provider_id})>"
