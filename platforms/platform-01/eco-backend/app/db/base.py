# =============================================================================
# Eco-Backend Database Base
# =============================================================================
# SQLAlchemy 基礎配置
# =============================================================================

from datetime import datetime
from typing import Any, AsyncGenerator
from sqlalchemy import Column, DateTime, String, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, declared_attr
import uuid

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("database")


# 創建異步引擎
engine = create_async_engine(
    settings.database_async_url,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    echo=settings.DATABASE_ECHO,
    future=True,
)

# 創建異步會話工廠
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


class CustomBase:
    """自定義基礎模型類"""
    
    # 自動生成表名
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()
    
    # 通用字段
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def to_dict(self) -> dict:
        """轉換為字典"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.id})>"


# 聲明式基類
Base = declarative_base(cls=CustomBase)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """獲取數據庫會話 (依賴函數)"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()


async def init_db() -> None:
    """初始化數據庫"""
    async with engine.begin() as conn:
        # 在開發環境創建所有表
        if settings.is_development:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("database_tables_created")


async def close_db() -> None:
    """關閉數據庫連接"""
    await engine.dispose()
    logger.info("database_connection_closed")
