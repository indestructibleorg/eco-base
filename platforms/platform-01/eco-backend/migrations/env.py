# =============================================================================
# Alembic Environment
# =============================================================================

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

from app.core.config import settings
from app.db.base import Base

# 導入所有模型以確保它們被註冊
from app.models.user import User, ApiKey, UserProviderConfig
from app.models.request_log import RequestLog, ProviderCallLog

# Alembic配置對象
config = context.config

# 設置日誌
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 目標元數據
target_metadata = Base.metadata

# 從應用配置獲取數據庫URL
def get_url():
    return settings.DATABASE_URL


def run_migrations_offline() -> None:
    """離線運行遷移"""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """執行遷移"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """異步運行遷移"""
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """在線運行遷移"""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
