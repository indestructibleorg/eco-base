# =============================================================================
# Auto-Remediation Engine
# =============================================================================
# 自動修復引擎 - Phase 1 基礎閉環核心組件
# 支持多種修復動作: Pod 重啟、緩存清理、提供者切換、限流調整
# =============================================================================

import asyncio
from typing import Dict, Any, Optional, List, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from abc import ABC, abstractmethod
import uuid

from app.core.logging import get_logger
from app.core.events import EventType, emit_event
from app.core.cache import cache_manager

logger = get_logger("remediator")


class RemediationStatus(Enum):
    """修復狀態"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class RemediationType(Enum):
    """修復類型"""
    RESTART_POD = auto()
    CLEAR_CACHE = auto()
    SWITCH_PROVIDER = auto()
    SCALE_UP = auto()
    SCALE_DOWN = auto()
    ENABLE_CIRCUIT_BREAKER = auto()
    DISABLE_CIRCUIT_BREAKER = auto()
    ROLLBACK_DEPLOYMENT = auto()


@dataclass
class RemediationResult:
    """修復結果"""
    action_id: str
    action_type: RemediationType
    status: RemediationStatus
    message: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def duration_seconds(self) -> float:
        """獲取執行時長"""
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.name,
            "status": self.status.value,
            "message": self.message,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
            "error": self.error
        }


@dataclass
class RemediationAction:
    """修復動作定義"""
    action_type: RemediationType
    target: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 60
    retry_count: int = 3
    retry_delay_seconds: int = 5


class RemediationActionHandler(ABC):
    """修復動作處理器基類"""
    
    @abstractmethod
    async def execute(self, action: RemediationAction) -> RemediationResult:
        """執行修復動作"""
        pass
    
    @abstractmethod
    async def validate(self, action: RemediationAction) -> bool:
        """驗證動作參數"""
        pass
    
    @abstractmethod
    async def rollback(self, action: RemediationAction) -> RemediationResult:
        """回滾動作"""
        pass


class RestartPodHandler(RemediationActionHandler):
    """重啟 Pod 處理器"""
    
    def __init__(self, k8s_client=None):
        self.k8s_client = k8s_client
        self._deleted_pods: Dict[str, Dict[str, Any]] = {}
    
    async def validate(self, action: RemediationAction) -> bool:
        """驗證參數"""
        required = ["namespace", "pod_name"]
        return all(param in action.parameters for param in required)
    
    async def execute(self, action: RemediationAction) -> RemediationResult:
        """執行 Pod 重啟"""
        action_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        
        namespace = action.parameters.get("namespace", "default")
        pod_name = action.parameters["pod_name"]
        label_selector = action.parameters.get("label_selector")
        
        try:
            if self.k8s_client:
                # 實際 K8s 操作
                if pod_name:
                    await self._restart_specific_pod(namespace, pod_name)
                elif label_selector:
                    await self._restart_pods_by_label(namespace, label_selector)
                else:
                    raise ValueError("Either pod_name or label_selector required")
            else:
                # 模擬操作
                logger.info(
                    "pod_restart_simulated",
                    namespace=namespace,
                    pod_name=pod_name,
                    label_selector=label_selector
                )
            
            return RemediationResult(
                action_id=action_id,
                action_type=RemediationType.RESTART_POD,
                status=RemediationStatus.SUCCESS,
                message=f"Pod {pod_name or label_selector} restarted successfully",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                metadata={"namespace": namespace, "pod_name": pod_name}
            )
            
        except Exception as e:
            logger.error("pod_restart_failed", error=str(e))
            return RemediationResult(
                action_id=action_id,
                action_type=RemediationType.RESTART_POD,
                status=RemediationStatus.FAILED,
                message=f"Failed to restart pod: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                error=str(e)
            )
    
    async def _restart_specific_pod(self, namespace: str, pod_name: str) -> None:
        """重啟特定 Pod"""
        # 保存 Pod 信息用於回滾
        self._deleted_pods[pod_name] = {
            "namespace": namespace,
            "deleted_at": datetime.utcnow()
        }
        
        # 刪除 Pod (Deployment 會自動重建)
        if self.k8s_client:
            await self.k8s_client.delete_namespaced_pod(
                name=pod_name,
                namespace=namespace
            )
    
    async def _restart_pods_by_label(self, namespace: str, label_selector: str) -> None:
        """根據標籤重啟 Pod"""
        if self.k8s_client:
            pods = await self.k8s_client.list_namespaced_pod(
                namespace=namespace,
                label_selector=label_selector
            )
            for pod in pods.items:
                await self._restart_specific_pod(namespace, pod.metadata.name)
    
    async def rollback(self, action: RemediationAction) -> RemediationResult:
        """回滾: Pod 重啟無法直接回滾，但 Deployment 會自動重建"""
        return RemediationResult(
            action_id=str(uuid.uuid4()),
            action_type=RemediationType.RESTART_POD,
            status=RemediationStatus.SUCCESS,
            message="Pod restart rollback: Deployment will recreate pods automatically",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )


class ClearCacheHandler(RemediationActionHandler):
    """清理緩存處理器"""
    
    async def validate(self, action: RemediationAction) -> bool:
        """驗證參數"""
        # pattern 是可選的
        return True
    
    async def execute(self, action: RemediationAction) -> RemediationResult:
        """執行緩存清理"""
        action_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        
        pattern = action.parameters.get("pattern", "*")
        
        try:
            # 獲取清理前的緩存統計
            keys_before = await cache_manager.keys(pattern)
            count_before = len(keys_before)
            
            # 清理緩存
            deleted_count = await cache_manager.delete_pattern(pattern)
            
            return RemediationResult(
                action_id=action_id,
                action_type=RemediationType.CLEAR_CACHE,
                status=RemediationStatus.SUCCESS,
                message=f"Cleared {deleted_count} cache keys matching '{pattern}'",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                metadata={
                    "pattern": pattern,
                    "keys_before": count_before,
                    "deleted_count": deleted_count
                }
            )
            
        except Exception as e:
            logger.error("cache_clear_failed", error=str(e))
            return RemediationResult(
                action_id=action_id,
                action_type=RemediationType.CLEAR_CACHE,
                status=RemediationStatus.FAILED,
                message=f"Failed to clear cache: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                error=str(e)
            )
    
    async def rollback(self, action: RemediationAction) -> RemediationResult:
        """回滾: 緩存清理無法直接回滾"""
        return RemediationResult(
            action_id=str(uuid.uuid4()),
            action_type=RemediationType.CLEAR_CACHE,
            status=RemediationStatus.SUCCESS,
            message="Cache clear rollback: Cache will be repopulated automatically",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )


class SwitchProviderHandler(RemediationActionHandler):
    """切換提供者處理器"""
    
    def __init__(self, config_store=None):
        self.config_store = config_store
        self._previous_provider: Dict[str, str] = {}
    
    async def validate(self, action: RemediationAction) -> bool:
        """驗證參數"""
        required = ["domain", "new_provider"]
        return all(param in action.parameters for param in required)
    
    async def execute(self, action: RemediationAction) -> RemediationResult:
        """執行提供者切換"""
        action_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        
        domain = action.parameters["domain"]
        new_provider = action.parameters["new_provider"]
        
        try:
            # 保存當前提供者
            current_provider = await self._get_current_provider(domain)
            self._previous_provider[domain] = current_provider
            
            # 切換提供者
            await self._switch_provider(domain, new_provider)
            
            return RemediationResult(
                action_id=action_id,
                action_type=RemediationType.SWITCH_PROVIDER,
                status=RemediationStatus.SUCCESS,
                message=f"Switched {domain} provider from {current_provider} to {new_provider}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                metadata={
                    "domain": domain,
                    "previous_provider": current_provider,
                    "new_provider": new_provider
                }
            )
            
        except Exception as e:
            logger.error("provider_switch_failed", error=str(e))
            return RemediationResult(
                action_id=action_id,
                action_type=RemediationType.SWITCH_PROVIDER,
                status=RemediationStatus.FAILED,
                message=f"Failed to switch provider: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                error=str(e)
            )
    
    async def _get_current_provider(self, domain: str) -> str:
        """獲取當前提供者"""
        # 從配置存儲獲取
        if self.config_store:
            return await self.config_store.get(f"provider.{domain}", "default")
        return "default"
    
    async def _switch_provider(self, domain: str, provider: str) -> None:
        """切換提供者"""
        if self.config_store:
            await self.config_store.set(f"provider.{domain}", provider)
        
        logger.info(
            "provider_switched",
            domain=domain,
            provider=provider
        )
    
    async def rollback(self, action: RemediationAction) -> RemediationResult:
        """回滾提供者切換"""
        domain = action.parameters["domain"]
        previous = self._previous_provider.get(domain)
        
        if previous:
            await self._switch_provider(domain, previous)
            return RemediationResult(
                action_id=str(uuid.uuid4()),
                action_type=RemediationType.SWITCH_PROVIDER,
                status=RemediationStatus.ROLLED_BACK,
                message=f"Rolled back {domain} provider to {previous}",
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
        
        return RemediationResult(
            action_id=str(uuid.uuid4()),
            action_type=RemediationType.SWITCH_PROVIDER,
            status=RemediationStatus.FAILED,
            message="Rollback failed: no previous provider recorded",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )


class ScaleHandler(RemediationActionHandler):
    """擴縮容處理器"""
    
    def __init__(self, k8s_client=None):
        self.k8s_client = k8s_client
        self._previous_replicas: Dict[str, int] = {}
    
    async def validate(self, action: RemediationAction) -> bool:
        """驗證參數"""
        required = ["namespace", "deployment"]
        return all(param in action.parameters for param in required)
    
    async def execute(self, action: RemediationAction) -> RemediationResult:
        """執行擴縮容"""
        action_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        
        namespace = action.parameters["namespace"]
        deployment = action.parameters["deployment"]
        replicas_delta = action.parameters.get("replicas_delta", 1)
        min_replicas = action.parameters.get("min_replicas", 1)
        max_replicas = action.parameters.get("max_replicas", 10)
        
        try:
            # 獲取當前副本數
            current_replicas = await self._get_replicas(namespace, deployment)
            self._previous_replicas[f"{namespace}/{deployment}"] = current_replicas
            
            # 計算新副本數
            new_replicas = max(min_replicas, min(max_replicas, current_replicas + replicas_delta))
            
            # 執行擴縮容
            await self._scale_deployment(namespace, deployment, new_replicas)
            
            return RemediationResult(
                action_id=action_id,
                action_type=RemediationType.SCALE_UP if replicas_delta > 0 else RemediationType.SCALE_DOWN,
                status=RemediationStatus.SUCCESS,
                message=f"Scaled {deployment} from {current_replicas} to {new_replicas} replicas",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                metadata={
                    "namespace": namespace,
                    "deployment": deployment,
                    "previous_replicas": current_replicas,
                    "new_replicas": new_replicas
                }
            )
            
        except Exception as e:
            logger.error("scale_failed", error=str(e))
            return RemediationResult(
                action_id=action_id,
                action_type=RemediationType.SCALE_UP,
                status=RemediationStatus.FAILED,
                message=f"Failed to scale: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                error=str(e)
            )
    
    async def _get_replicas(self, namespace: str, deployment: str) -> int:
        """獲取當前副本數"""
        if self.k8s_client:
            dep = await self.k8s_client.read_namespaced_deployment(
                name=deployment,
                namespace=namespace
            )
            return dep.spec.replicas or 1
        return 1
    
    async def _scale_deployment(self, namespace: str, deployment: str, replicas: int) -> None:
        """擴縮容 Deployment"""
        if self.k8s_client:
            patch = {"spec": {"replicas": replicas}}
            await self.k8s_client.patch_namespaced_deployment_scale(
                name=deployment,
                namespace=namespace,
                body=patch
            )
        
        logger.info(
            "deployment_scaled",
            namespace=namespace,
            deployment=deployment,
            replicas=replicas
        )
    
    async def rollback(self, action: RemediationAction) -> RemediationResult:
        """回滾擴縮容"""
        namespace = action.parameters["namespace"]
        deployment = action.parameters["deployment"]
        key = f"{namespace}/{deployment}"
        previous = self._previous_replicas.get(key)
        
        if previous is not None:
            await self._scale_deployment(namespace, deployment, previous)
            return RemediationResult(
                action_id=str(uuid.uuid4()),
                action_type=RemediationType.SCALE_UP,
                status=RemediationStatus.ROLLED_BACK,
                message=f"Rolled back {deployment} to {previous} replicas",
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
        
        return RemediationResult(
            action_id=str(uuid.uuid4()),
            action_type=RemediationType.SCALE_UP,
            status=RemediationStatus.FAILED,
            message="Rollback failed: no previous replica count recorded",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )


class Remediator:
    """
    自動修復引擎
    
    統一管理所有修復動作
    """
    
    def __init__(self):
        self.handlers: Dict[RemediationType, RemediationActionHandler] = {}
        self._action_history: List[RemediationResult] = []
        self._max_history = 100
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """註冊默認處理器"""
        self.register_handler(RemediationType.RESTART_POD, RestartPodHandler())
        self.register_handler(RemediationType.CLEAR_CACHE, ClearCacheHandler())
        self.register_handler(RemediationType.SWITCH_PROVIDER, SwitchProviderHandler())
        self.register_handler(RemediationType.SCALE_UP, ScaleHandler())
        self.register_handler(RemediationType.SCALE_DOWN, ScaleHandler())
    
    def register_handler(
        self,
        action_type: RemediationType,
        handler: RemediationActionHandler
    ) -> None:
        """註冊處理器"""
        self.handlers[action_type] = handler
        logger.info("remediation_handler_registered", action_type=action_type.name)
    
    async def execute(self, action: RemediationAction) -> RemediationResult:
        """
        執行修復動作
        
        Args:
            action: 修復動作定義
            
        Returns:
            修復結果
        """
        handler = self.handlers.get(action.action_type)
        
        if not handler:
            return RemediationResult(
                action_id=str(uuid.uuid4()),
                action_type=action.action_type,
                status=RemediationStatus.FAILED,
                message=f"No handler registered for {action.action_type.name}",
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
        
        # 驗證參數
        if not await handler.validate(action):
            return RemediationResult(
                action_id=str(uuid.uuid4()),
                action_type=action.action_type,
                status=RemediationStatus.FAILED,
                message="Action validation failed",
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
        
        # 執行動作（帶重試）
        result = await self._execute_with_retry(action, handler)
        
        # 記錄歷史
        self._action_history.append(result)
        if len(self._action_history) > self._max_history:
            self._action_history.pop(0)
        
        # 發布事件
        await self._emit_remediation_event(result)
        
        return result
    
    async def _execute_with_retry(
        self,
        action: RemediationAction,
        handler: RemediationActionHandler
    ) -> RemediationResult:
        """帶重試的執行"""
        last_error = None
        
        for attempt in range(action.retry_count):
            try:
                result = await asyncio.wait_for(
                    handler.execute(action),
                    timeout=action.timeout_seconds
                )
                
                if result.status == RemediationStatus.SUCCESS:
                    return result
                    
            except asyncio.TimeoutError:
                last_error = "Timeout"
                logger.warning(
                    "remediation_timeout",
                    action_type=action.action_type.name,
                    attempt=attempt + 1
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "remediation_attempt_failed",
                    action_type=action.action_type.name,
                    attempt=attempt + 1,
                    error=str(e)
                )
            
            if attempt < action.retry_count - 1:
                await asyncio.sleep(action.retry_delay_seconds)
        
        # 所有重試失敗
        return RemediationResult(
            action_id=str(uuid.uuid4()),
            action_type=action.action_type,
            status=RemediationStatus.FAILED,
            message=f"All {action.retry_count} attempts failed",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            error=last_error
        )
    
    async def rollback(self, action_id: str) -> Optional[RemediationResult]:
        """回滾指定動作"""
        # 查找歷史記錄
        for result in reversed(self._action_history):
            if result.action_id == action_id:
                # 查找對應的處理器
                handler = self.handlers.get(result.action_type)
                if handler:
                    # 重建動作對象
                    action = RemediationAction(
                        action_type=result.action_type,
                        target=result.metadata.get("target", ""),
                        parameters=result.metadata
                    )
                    return await handler.rollback(action)
        
        return None
    
    async def _emit_remediation_event(self, result: RemediationResult) -> None:
        """發布修復事件"""
        event_type = (
            EventType.PROVIDER_ERROR
            if result.status == RemediationStatus.FAILED
            else EventType.USER_LOGIN  # 復用現有事件類型
        )
        
        await emit_event(
            event_type,
            {
                "action_id": result.action_id,
                "action_type": result.action_type.name,
                "status": result.status.value,
                "message": result.message,
                "duration_seconds": result.duration_seconds
            },
            source="remediator"
        )
    
    def get_action_history(
        self,
        action_type: Optional[RemediationType] = None,
        status: Optional[RemediationStatus] = None,
        limit: int = 50
    ) -> List[RemediationResult]:
        """獲取動作歷史"""
        results = self._action_history
        
        if action_type:
            results = [r for r in results if r.action_type == action_type]
        
        if status:
            results = [r for r in results if r.status == status]
        
        return results[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """獲取統計信息"""
        total = len(self._action_history)
        if total == 0:
            return {
                "total_actions": 0,
                "success_rate": 0.0,
                "avg_duration_seconds": 0.0
            }
        
        successful = len([r for r in self._action_history if r.status == RemediationStatus.SUCCESS])
        avg_duration = sum(r.duration_seconds for r in self._action_history) / total
        
        return {
            "total_actions": total,
            "successful": successful,
            "failed": len([r for r in self._action_history if r.status == RemediationStatus.FAILED]),
            "success_rate": successful / total,
            "avg_duration_seconds": round(avg_duration, 2),
            "by_type": self._group_by_type()
        }
    
    def _group_by_type(self) -> Dict[str, Dict[str, int]]:
        """按類型分組統計"""
        stats: Dict[str, Dict[str, int]] = {}
        
        for result in self._action_history:
            type_name = result.action_type.name
            if type_name not in stats:
                stats[type_name] = {"total": 0, "success": 0, "failed": 0}
            
            stats[type_name]["total"] += 1
            if result.status == RemediationStatus.SUCCESS:
                stats[type_name]["success"] += 1
            else:
                stats[type_name]["failed"] += 1
        
        return stats


# 全局自動修復引擎實例
remediator = Remediator()
