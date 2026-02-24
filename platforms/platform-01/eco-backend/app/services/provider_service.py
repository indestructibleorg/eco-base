# =============================================================================
# Provider Service
# =============================================================================
# 提供者業務邏輯服務層
# =============================================================================

from datetime import datetime
from typing import Optional, Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import ProviderError, CircuitBreakerError
from app.core.metrics import record_provider_call
from app.utils.encryption import decrypt_provider_config
from app.services.platform_integration_service import platform_integration_service

logger = get_logger("provider_service")


# 提供者註冊表
PROVIDERS_REGISTRY = {
    "alpha-persistence": {
        "domain": "DATA_PERSISTENCE",
        "capabilities": ["query", "mutate", "subscribe", "vector_search"],
    },
    "beta-persistence": {
        "domain": "DATA_PERSISTENCE",
        "capabilities": ["query", "mutate", "sql"],
    },
    "gamma-cognitive": {
        "domain": "COGNITIVE_COMPUTE",
        "capabilities": ["generate", "stream", "embed", "function_call", "multimodal"],
    },
    "delta-cognitive": {
        "domain": "COGNITIVE_COMPUTE",
        "capabilities": ["generate", "stream", "long_context", "coding"],
    },
    "epsilon-cognitive": {
        "domain": "COGNITIVE_COMPUTE",
        "capabilities": ["generate", "stream", "reasoning"],
    },
    "zeta-code": {
        "domain": "CODE_ENGINEERING",
        "capabilities": ["complete", "explain", "refactor", "review", "generate_tests"],
    },
    "eta-code": {
        "domain": "CODE_ENGINEERING",
        "capabilities": ["review"],
    },
    "theta-code": {
        "domain": "CODE_ENGINEERING",
        "capabilities": ["complete", "generate_tests"],
    },
    "iota-collaboration": {
        "domain": "COLLABORATION",
        "capabilities": ["message", "channel", "summarize", "workflow"],
    },
    "kappa-collaboration": {
        "domain": "COLLABORATION",
        "capabilities": ["message", "workflow", "search"],
    },
    "lambda-visual": {
        "domain": "VISUAL_DESIGN",
        "capabilities": ["components", "export", "generate", "inspect", "prototype"],
    },
    "mu-visual": {
        "domain": "VISUAL_DESIGN",
        "capabilities": ["components", "export", "inspect"],
    },
    "omicron-deployment": {
        "domain": "DEPLOYMENT",
        "capabilities": ["build", "deploy", "preview", "rollback"],
    },
    "pi-deployment": {
        "domain": "DEPLOYMENT",
        "capabilities": ["build", "cache"],
    },
    "rho-deployment": {
        "domain": "DEPLOYMENT",
        "capabilities": ["build", "deploy", "plan", "state"],
    },
}


class CircuitBreaker:
    """熔斷器"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def record_success(self):
        """記錄成功"""
        self.failures = 0
        self.state = "closed"
    
    def record_failure(self):
        """記錄失敗"""
        self.failures += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failures >= self.failure_threshold:
            self.state = "open"
    
    def can_execute(self) -> bool:
        """檢查是否可以執行"""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # 檢查是否超過恢復時間
            if self.last_failure_time:
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = "half-open"
                    return True
            return False
        
        return True  # half-open


class ProviderService:
    """提供者服務"""
    
    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def _get_circuit_breaker(self, provider_id: str) -> CircuitBreaker:
        """獲取或創建熔斷器"""
        if provider_id not in self._circuit_breakers:
            self._circuit_breakers[provider_id] = CircuitBreaker(
                failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
            )
        return self._circuit_breakers[provider_id]
    
    async def call_provider(
        self,
        provider_id: str,
        operation: str,
        payload: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        調用提供者
        
        Args:
            provider_id: 提供者ID
            operation: 操作名稱
            payload: 請求數據
            user_id: 用戶ID (可選)
            
        Returns:
            提供者響應
            
        Raises:
            CircuitBreakerError: 熔斷器打開
            ProviderError: 提供者調用失敗
        """
        # 檢查熔斷器
        circuit_breaker = self._get_circuit_breaker(provider_id)
        if not circuit_breaker.can_execute():
            raise CircuitBreakerError(provider_id)
        
        # 獲取提供者配置
        config = self._get_provider_config(provider_id, user_id)
        if not config:
            raise ProviderError(provider_id, "Configuration not found")
        
        start_time = datetime.utcnow()
        
        try:
            # 調用提供者
            result = await self._execute_call(
                provider_id=provider_id,
                operation=operation,
                payload=payload,
                config=config,
            )
            
            # 記錄成功
            circuit_breaker.record_success()
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            record_provider_call(
                provider=provider_id,
                operation=operation,
                duration=duration,
                success=True
            )
            
            return result
            
        except Exception as e:
            # 記錄失敗
            circuit_breaker.record_failure()
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            record_provider_call(
                provider=provider_id,
                operation=operation,
                duration=duration,
                success=False,
                error_type=type(e).__name__
            )
            
            raise ProviderError(provider_id, str(e))
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _execute_call(
        self,
        provider_id: str,
        operation: str,
        payload: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """執行提供者調用 (帶重試)"""
        # 集成 eco-platform-integrations 框架
        
        result = None
        
        # 根据操作类型调用相应的服务
        if operation in ["query", "mutate", "vector_search"]:
            # 数据持久化操作
            if operation == "query":
                result = await platform_integration_service.query_data(
                    table=payload.get("table", "default"),
                    filters=payload.get("filters"),
                    provider=provider_id,
                )
            elif operation == "mutate":
                result = await platform_integration_service.persist_data(
                    table=payload.get("table", "default"),
                    data=payload.get("data", {}),
                    provider=provider_id,
                )
            elif operation == "vector_search":
                result = await platform_integration_service.vector_search(
                    index=payload.get("index", "default"),
                    vector=payload.get("vector", []),
                    top_k=payload.get("top_k", 5),
                    provider=provider_id,
                )
        
        elif operation in ["generate", "stream", "embed"]:
            # 认知计算操作
            if operation == "generate":
                result = await platform_integration_service.chat_completion(
                    messages=payload.get("messages", []),
                    model=payload.get("model"),
                    temperature=payload.get("temperature", 0.7),
                    provider=provider_id,
                )
            elif operation == "agent_task":
                result = await platform_integration_service.run_agent_task(
                    task=payload.get("task", ""),
                    context=payload.get("context"),
                    provider=provider_id,
                )
        
        elif operation in ["complete", "explain", "refactor", "review"]:
            # 代码工程操作
            if operation == "review":
                result = await platform_integration_service.review_code(
                    pr_number=payload.get("pr_number", 0),
                    comments=payload.get("comments", []),
                    provider=provider_id,
                )
            elif operation == "create_pr":
                result = await platform_integration_service.create_pull_request(
                    title=payload.get("title", ""),
                    body=payload.get("body", ""),
                    head_branch=payload.get("head_branch", ""),
                    base_branch=payload.get("base_branch", "main"),
                    provider=provider_id,
                )
        
        elif operation in ["send_message", "trigger_workflow"]:
            # 协作通信操作
            if operation == "send_message":
                result = await platform_integration_service.send_notification(
                    message=payload.get("message", ""),
                    channel=payload.get("channel"),
                    provider=provider_id,
                )
            elif operation == "trigger_workflow":
                result = await platform_integration_service.trigger_workflow(
                    workflow_id=payload.get("workflow_id", ""),
                    inputs=payload.get("inputs"),
                    provider=provider_id,
                )
        
        elif operation == "deploy":
            # 部署交付操作
            result = await platform_integration_service.deploy(
                project=payload.get("project", ""),
                environment=payload.get("environment", "production"),
                provider=provider_id,
            )
        
        else:
            # 未知操作类型
            logger.warning(
                "unknown_provider_operation",
                provider_id=provider_id,
                operation=operation,
            )
            return {
                "success": False,
                "error": f"Unknown operation: {operation}",
                "provider": provider_id,
            }
        
        # 返回结果
        if result:
            return {
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "provider": result.provider or provider_id,
                "duration_ms": result.duration_ms,
            }
        else:
            return {
                "success": False,
                "error": "Operation failed",
                "provider": provider_id,
            }
    
    def _get_provider_config(
        self,
        provider_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """獲取提供者配置"""
        # 優先從全局配置獲取
        global_configs = settings.get_provider_configs()
        config = global_configs.get(provider_id, {})
        
        if any(v for v in config.values() if v is not None):
            return config
        
        return None
    
    def list_providers(
        self,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """列出所有可用的提供者"""
        providers = []
        
        for provider_id, info in PROVIDERS_REGISTRY.items():
            if domain and info["domain"] != domain.upper():
                continue
            
            # 檢查配置是否可用
            config = settings.get_provider_configs().get(provider_id, {})
            is_available = any(v for v in config.values() if v is not None)
            
            providers.append({
                "provider_id": provider_id,
                "domain": info["domain"],
                "capabilities": info["capabilities"],
                "is_available": is_available,
            })
        
        return providers
    
    def get_provider(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """獲取提供者詳情"""
        info = PROVIDERS_REGISTRY.get(provider_id)
        if not info:
            return None
        
        config = settings.get_provider_configs().get(provider_id, {})
        is_available = any(v for v in config.values() if v is not None)
        
        return {
            "provider_id": provider_id,
            "domain": info["domain"],
            "capabilities": info["capabilities"],
            "is_available": is_available,
        }
    
    def check_health(self, provider_id: str) -> Dict[str, Any]:
        """檢查提供者健康狀態"""
        config = settings.get_provider_configs().get(provider_id, {})
        has_config = any(v for v in config.values() if v is not None)
        
        circuit_breaker = self._get_circuit_breaker(provider_id)
        
        return {
            "provider_id": provider_id,
            "healthy": has_config and circuit_breaker.state != "open",
            "circuit_state": circuit_breaker.state,
            "configured": has_config,
        }


# 全局服務實例
provider_service = ProviderService()
