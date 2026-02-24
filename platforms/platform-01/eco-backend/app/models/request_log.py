# =============================================================================
# Request Log Model
# =============================================================================

from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text, Float, ForeignKey, Index
from sqlalchemy.orm import relationship

from app.db.base import Base


class RequestLog(Base):
    """請求日誌模型"""
    
    __tablename__ = "request_logs"
    
    # 請求信息
    request_id = Column(String(36), unique=True, nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    
    # 請求詳情
    method = Column(String(10), nullable=False)
    path = Column(String(500), nullable=False, index=True)
    query_params = Column(Text, nullable=True)
    request_body = Column(Text, nullable=True)
    client_ip = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # 響應信息
    status_code = Column(Integer, nullable=False, index=True)
    response_body = Column(Text, nullable=True)
    
    # 性能指標
    duration_ms = Column(Float, nullable=False, index=True)
    
    # 提供者調用信息
    provider_id = Column(String(50), nullable=True, index=True)
    provider_duration_ms = Column(Float, nullable=True)
    provider_error = Column(Text, nullable=True)
    
    # 錯誤信息
    error_code = Column(String(50), nullable=True, index=True)
    error_message = Column(Text, nullable=True)
    
    # 時間戳
    created_at = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # 關聯
    user = relationship("User", back_populates="requests")
    
    # 索引
    __table_args__ = (
        Index("ix_request_logs_user_created", "user_id", "created_at"),
        Index("ix_request_logs_path_created", "path", "created_at"),
        Index("ix_request_logs_provider_created", "provider_id", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<RequestLog(id={self.id}, path={self.path}, status={self.status_code})>"


class ProviderCallLog(Base):
    """提供者調用日誌模型"""
    
    __tablename__ = "provider_call_logs"
    
    # 關聯請求
    request_id = Column(String(36), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    
    # 提供者信息
    provider_id = Column(String(50), nullable=False, index=True)
    capability = Column(String(50), nullable=False, index=True)
    operation = Column(String(100), nullable=False)
    
    # 請求詳情
    request_payload = Column(Text, nullable=True)
    
    # 響應詳情
    response_payload = Column(Text, nullable=True)
    success = Column(Integer, nullable=False, default=0)  # 0=False, 1=True
    
    # 性能指標
    duration_ms = Column(Float, nullable=False)
    
    # 錯誤信息
    error_code = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    
    # 重試信息
    retry_count = Column(Integer, default=0)
    
    # 時間戳
    created_at = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # 索引
    __table_args__ = (
        Index("ix_provider_logs_user_provider", "user_id", "provider_id"),
        Index("ix_provider_logs_capability_created", "capability", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<ProviderCallLog(provider={self.provider_id}, operation={self.operation})>"
