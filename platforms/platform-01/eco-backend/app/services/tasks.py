# =============================================================================
# Background Tasks
# =============================================================================

import asyncio
from datetime import datetime, timedelta
from typing import Optional

from app.core.celery import celery_app
from app.core.logging import get_logger
from app.services.platform_integration_service import platform_integration_service

logger = get_logger("tasks")


@celery_app.task(bind=True, max_retries=3)
def process_provider_request(
    self,
    provider_id: str,
    operation: str,
    payload: dict,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
):
    """處理提供者請求任務"""
    
    async def _process():
        """异步处理"""
        try:
            logger.info(
                "processing_provider_request",
                provider_id=provider_id,
                operation=operation,
                user_id=user_id,
                request_id=request_id,
            )
            
            # 调用平台集成框架
            result = None
            
            if operation == "persist":
                # 数据持久化
                result = await platform_integration_service.persist_data(
                    table=payload.get("table", "default"),
                    data=payload.get("data", {}),
                    provider=provider_id,
                )
            elif operation == "query":
                # 数据查询
                result = await platform_integration_service.query_data(
                    table=payload.get("table", "default"),
                    filters=payload.get("filters"),
                    provider=provider_id,
                )
            elif operation == "vector_search":
                # 向量搜索
                result = await platform_integration_service.vector_search(
                    index=payload.get("index", "default"),
                    vector=payload.get("vector", []),
                    top_k=payload.get("top_k", 5),
                    provider=provider_id,
                )
            elif operation == "chat":
                # 聊天补全
                result = await platform_integration_service.chat_completion(
                    messages=payload.get("messages", []),
                    model=payload.get("model"),
                    temperature=payload.get("temperature", 0.7),
                    provider=provider_id,
                )
            elif operation == "agent_task":
                # 智能体任务
                result = await platform_integration_service.run_agent_task(
                    task=payload.get("task", ""),
                    context=payload.get("context"),
                    provider=provider_id,
                )
            elif operation == "create_pr":
                # 创建 PR
                result = await platform_integration_service.create_pull_request(
                    title=payload.get("title", ""),
                    body=payload.get("body", ""),
                    head_branch=payload.get("head_branch", ""),
                    base_branch=payload.get("base_branch", "main"),
                    provider=provider_id,
                )
            elif operation == "send_notification":
                # 发送通知
                result = await platform_integration_service.send_notification(
                    message=payload.get("message", ""),
                    channel=payload.get("channel"),
                    provider=provider_id,
                )
            elif operation == "deploy":
                # 部署
                result = await platform_integration_service.deploy(
                    project=payload.get("project", ""),
                    environment=payload.get("environment", "production"),
                    provider=provider_id,
                )
            else:
                logger.warning(
                    "unknown_operation",
                    operation=operation,
                    provider_id=provider_id,
                )
                return {
                    "status": "error",
                    "error": f"Unknown operation: {operation}",
                    "provider_id": provider_id,
                }
            
            if result and result.success:
                return {
                    "status": "success",
                    "provider_id": provider_id,
                    "operation": operation,
                    "data": result.data,
                }
            else:
                error_msg = result.error if result else "Operation failed"
                raise Exception(error_msg)
        
        except Exception as exc:
            logger.error(
                "provider_request_failed",
                provider_id=provider_id,
                operation=operation,
                error=str(exc),
                retry_count=self.request.retries,
            )
            raise
    
    try:
        # 运行异步任务
        return asyncio.run(_process())
    
    except Exception as exc:
        # 重試
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))
        raise


@celery_app.task
def cleanup_old_logs(days: int = 30):
    """清理舊日誌"""
    from app.db.base import AsyncSessionLocal
    from app.models.request_log import RequestLog, ProviderCallLog
    from sqlalchemy import delete
    
    async def _cleanup():
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        async with AsyncSessionLocal() as session:
            # 清理請求日誌
            result = await session.execute(
                delete(RequestLog).where(RequestLog.created_at < cutoff_date)
            )
            request_logs_deleted = result.rowcount
            
            # 清理提供者調用日誌
            result = await session.execute(
                delete(ProviderCallLog).where(ProviderCallLog.created_at < cutoff_date)
            )
            provider_logs_deleted = result.rowcount
            
            await session.commit()
            
            logger.info(
                "old_logs_cleaned",
                request_logs_deleted=request_logs_deleted,
                provider_logs_deleted=provider_logs_deleted,
                cutoff_date=cutoff_date.isoformat(),
            )
    
    import asyncio
    asyncio.run(_cleanup())


@celery_app.task
def reset_monthly_quotas():
    """重置月度配額"""
    from app.db.base import AsyncSessionLocal
    from app.models.user import ApiKey
    from sqlalchemy import update
    
    async def _reset():
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                update(ApiKey).values(monthly_used=0)
            )
            await session.commit()
            
            logger.info(
                "monthly_quotas_reset",
                keys_reset=result.rowcount,
            )
    
    import asyncio
    asyncio.run(_reset())


@celery_app.task(bind=True, max_retries=3)
def send_notification(
    self,
    channel: str,
    message: str,
    user_id: Optional[str] = None,
    provider: str = "slack",
):
    """發送通知"""
    
    async def _send():
        """异步发送"""
        logger.info(
            "sending_notification",
            channel=channel,
            user_id=user_id,
            provider=provider,
        )
        
        # 调用协作通信适配器发送通知
        result = await platform_integration_service.send_notification(
            message=message,
            channel=channel,
            provider=provider,
        )
        
        if result.success:
            return {
                "status": "sent",
                "channel": channel,
                "provider": provider,
                "data": result.data,
            }
        else:
            raise Exception(result.error or "Failed to send notification")
    
    try:
        return asyncio.run(_send())
    except Exception as exc:
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=60)
        raise


@celery_app.task(bind=True, max_retries=5)
def sync_provider_data(
    self,
    provider_id: str,
    sync_type: str,
    user_id: Optional[str] = None,
    table: Optional[str] = None,
    filters: Optional[dict] = None,
):
    """同步提供者數據"""
    
    async def _sync():
        """异步同步"""
        try:
            logger.info(
                "syncing_provider_data",
                provider_id=provider_id,
                sync_type=sync_type,
                user_id=user_id,
            )
            
            # 实现数据同步逻辑
            result = None
            
            if sync_type == "query":
                # 查询同步
                result = await platform_integration_service.query_data(
                    table=table or "default",
                    filters=filters,
                    provider=provider_id,
                )
            elif sync_type == "persist":
                # 持久化同步
                result = await platform_integration_service.persist_data(
                    table=table or "default",
                    data=filters or {},
                    provider=provider_id,
                )
            else:
                logger.warning(
                    "unknown_sync_type",
                    sync_type=sync_type,
                    provider_id=provider_id,
                )
                return {
                    "status": "error",
                    "error": f"Unknown sync type: {sync_type}",
                    "provider_id": provider_id,
                }
            
            if result and result.success:
                return {
                    "status": "success",
                    "provider_id": provider_id,
                    "sync_type": sync_type,
                    "data": result.data,
                }
            else:
                error_msg = result.error if result else "Sync failed"
                raise Exception(error_msg)
        
        except Exception as exc:
            logger.error(
                "sync_provider_data_failed",
                provider_id=provider_id,
                sync_type=sync_type,
                error=str(exc),
                retry_count=self.request.retries,
            )
            raise
    
    try:
        return asyncio.run(_sync())
    except Exception as exc:
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=300)
        raise
