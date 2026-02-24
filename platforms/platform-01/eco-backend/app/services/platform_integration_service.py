from __future__ import annotations

# =============================================================================
# Platform Integration Service V2 (Hard Constraints)
# =============================================================================
# 硬约束版本：服务必须在应用启动时初始化成功
# 任何初始化失败都导致应用启动失败
# =============================================================================

import sys
import os
from typing import Dict, Any, List, Optional, AsyncIterator, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# 添加 eco-platform-integrations 到路径
PLATFORM_INTEGRATIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "..", "eco-platform-integrations"
)
if os.path.exists(PLATFORM_INTEGRATIONS_PATH):
    sys.path.insert(0, PLATFORM_INTEGRATIONS_PATH)

# 硬约束: eco-platform-integrations 必须可用
try:
    from eco_platform_integrations import (
        EcoPlatformService,
        eco_service,
        CapabilityDomain,
        OperationResult,
        QuerySpec,
        MutationSpec,
        InferenceRequest,
        AgentTask,
        CodeContext,
        MessagePayload,
        WorkflowTrigger,
        DeploySpec,
    )
    from eco_platform_integrations import register_all_adapters, get_provider_config
    PLATFORM_INTEGRATIONS_AVAILABLE = True
except ImportError as e:
    PLATFORM_INTEGRATIONS_AVAILABLE = False
    IMPORT_ERROR = str(e)

from app.core.logging import get_logger

logger = get_logger("platform_integration")


class PlatformIntegrationError(Exception):
    """平台集成错误"""
    pass


class ServiceNotInitializedError(PlatformIntegrationError):
    """服务未初始化错误"""
    pass


class ProviderConfigError(PlatformIntegrationError):
    """提供商配置错误"""
    pass


class ProviderUnavailableError(PlatformIntegrationError):
    """提供商不可用错误"""
    pass


@dataclass(frozen=True)
class IntegrationResult:
    """
    集成操作结果
    
    不变量:
    - success=True 时 data 必须不为 None
    - success=False 时 error 必须不为 None
    """
    success: bool
    data: Any = None
    error: Optional[str] = None
    provider: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', datetime.utcnow())
        # 验证不变量
        if self.success and self.data is None:
            raise ValueError("IntegrationResult: success=True requires data != None")
        if not self.success and self.error is None:
            raise ValueError("IntegrationResult: success=False requires error != None")


class PlatformIntegrationService:
    """
    平台集成服务 (硬约束版本)
    
    硬约束:
    1. 服务必须在应用启动时初始化成功
    2. 任何初始化失败都导致应用启动失败
    3. 所有公共方法必须有类型注解
    4. 所有异常必须向上传播，禁止吞掉
    
    Usage:
        service = PlatformIntegrationService()
        await service.initialize(config)  # 失败则抛异常
        
        # 后续调用不需要检查初始化状态
        result = await service.query_data(...)  # 内部硬检查
    """
    
    def __init__(self) -> None:
        self._service: Optional[EcoPlatformService] = None
        self._initialized: bool = False
        self._config: Dict[str, Any] = {}
        self._health_check_callbacks: List[Callable[[], Awaitable[Dict[str, Any]]]] = []
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        初始化平台集成服务
        
        硬约束:
        - 初始化失败 raise PlatformIntegrationError，不返回 False
        - 所有依赖必须验证可用性
        - 初始化成功后 _initialized = True
        
        Args:
            config: 配置字典，包含各平台的 API key 等
            
        Raises:
            PlatformIntegrationError: 初始化失败
            ProviderConfigError: 提供商配置错误
            ProviderUnavailableError: 提供商不可用
        """
        # 硬约束: eco-platform-integrations 必须可用
        if not PLATFORM_INTEGRATIONS_AVAILABLE:
            raise PlatformIntegrationError(
                f"eco-platform-integrations not available: {IMPORT_ERROR}. "
                f"Install: pip install -e ./eco-platform-integrations"
            )
        
        self._service = eco_service
        self._config = config
        
        # 注册所有适配器 - 失败则抛异常
        try:
            register_all_adapters()
        except Exception as e:
            raise PlatformIntegrationError(f"Failed to register adapters: {e}") from e
        
        # 配置各提供商 - 失败则抛异常
        await self._configure_providers(config)
        
        # 验证所有必需提供商可用 - 失败则抛异常
        await self._validate_providers()
        
        self._initialized = True
        logger.info("platform_integration_service_initialized")
    
    async def _configure_providers(self, config: Dict[str, Any]) -> None:
        """
        配置各平台提供商
        
        Raises:
            ProviderConfigError: 配置错误
        """
        # 配置 Supabase
        if supabase_config := config.get("supabase"):
            provider_config = get_provider_config("supabase")
            provider_config.api_key = supabase_config.get("api_key")
            provider_config.url = supabase_config.get("url")
            
            if not provider_config.api_key:
                raise ProviderConfigError("supabase.api_key is required")
            if not provider_config.url:
                raise ProviderConfigError("supabase.url is required")
        
        # 配置 Pinecone
        if pinecone_config := config.get("pinecone"):
            provider_config = get_provider_config("pinecone")
            provider_config.api_key = pinecone_config.get("api_key")
            provider_config.environment = pinecone_config.get("environment")
            
            if not provider_config.api_key:
                raise ProviderConfigError("pinecone.api_key is required")
        
        # 配置 OpenAI
        if openai_config := config.get("openai"):
            provider_config = get_provider_config("openai")
            provider_config.api_key = openai_config.get("api_key")
            provider_config.model = openai_config.get("model", "gpt-4")
            
            if not provider_config.api_key:
                raise ProviderConfigError("openai.api_key is required")
        
        # 配置 GitHub
        if github_config := config.get("github"):
            provider_config = get_provider_config("github")
            provider_config.api_key = github_config.get("api_key")
            provider_config.owner = github_config.get("owner")
            provider_config.repo = github_config.get("repo")
            
            if not provider_config.api_key:
                raise ProviderConfigError("github.api_key is required")
        
        # 配置 Slack
        if slack_config := config.get("slack"):
            provider_config = get_provider_config("slack")
            provider_config.api_key = slack_config.get("api_key")
            provider_config.channel = slack_config.get("channel")
            
            if not provider_config.api_key:
                raise ProviderConfigError("slack.api_key is required")
        
        # 配置 Vercel
        if vercel_config := config.get("vercel"):
            provider_config = get_provider_config("vercel")
            provider_config.api_key = vercel_config.get("api_key")
            provider_config.team_id = vercel_config.get("team_id")
            
            if not provider_config.api_key:
                raise ProviderConfigError("vercel.api_key is required")
        
        logger.info("providers_configured", provider_count=len(config))
    
    async def _validate_providers(self) -> None:
        """
        验证所有配置的提供商可用
        
        Raises:
            ProviderUnavailableError: 提供商不可用
        """
        health = await self.health_check()
        
        if health["status"] != "healthy":
            unhealthy = [
                name for name, status in health.get("providers", {}).items()
                if status.get("status") != "healthy"
            ]
            raise ProviderUnavailableError(f"Providers unhealthy: {unhealthy}")
    
    def _ensure_initialized(self) -> EcoPlatformService:
        """
        运行时检查，失败直接抛异常
        
        Returns:
            EcoPlatformService 实例
            
        Raises:
            ServiceNotInitializedError: 服务未初始化
        """
        if not self._initialized or self._service is None:
            raise ServiceNotInitializedError(
                "PlatformIntegrationService not initialized. "
                "Call initialize() during application startup."
            )
        return self._service
    
    # =====================================================================
    # 数据持久化 (Supabase, Pinecone)
    # =====================================================================
    
    async def persist_data(
        self,
        table: str,
        data: Dict[str, Any],
        provider: str = "supabase",
    ) -> IntegrationResult:
        """
        持久化数据
        
        Args:
            table: 表名
            data: 数据
            provider: 提供商 (supabase)
            
        Returns:
            IntegrationResult
            
        Raises:
            ServiceNotInitializedError: 服务未初始化
            PlatformIntegrationError: 操作失败
        """
        service = self._ensure_initialized()
        start = datetime.utcnow()
        
        mutation = MutationSpec(
            table=table,
            operation="insert",
            data=data,
        )
        
        result: OperationResult = await service.execute(
            domain=CapabilityDomain.DATA_PERSISTENCE,
            operation="mutate",
            params={"mutation": mutation},
        )
        
        duration = (datetime.utcnow() - start).total_seconds() * 1000
        
        return IntegrationResult(
            success=result.success,
            data=result.data,
            error=result.error,
            provider=provider,
            duration_ms=duration,
        )
    
    async def query_data(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        provider: str = "supabase",
    ) -> IntegrationResult:
        """
        查询数据
        
        Args:
            table: 表名
            filters: 过滤条件
            provider: 提供商
            
        Returns:
            IntegrationResult
            
        Raises:
            ServiceNotInitializedError: 服务未初始化
            PlatformIntegrationError: 操作失败
        """
        service = self._ensure_initialized()
        start = datetime.utcnow()
        
        query = QuerySpec(
            table=table,
            filters=filters or {},
        )
        
        result: OperationResult = await service.execute(
            domain=CapabilityDomain.DATA_PERSISTENCE,
            operation="query",
            params={"query": query},
        )
        
        duration = (datetime.utcnow() - start).total_seconds() * 1000
        
        return IntegrationResult(
            success=result.success,
            data=result.data,
            error=result.error,
            provider=provider,
            duration_ms=duration,
        )
    
    async def vector_search(
        self,
        index: str,
        vector: List[float],
        top_k: int = 5,
        provider: str = "pinecone",
    ) -> IntegrationResult:
        """
        向量搜索
        
        Args:
            index: 索引名
            vector: 查询向量
            top_k: 返回结果数
            provider: 提供商
            
        Returns:
            IntegrationResult
            
        Raises:
            ServiceNotInitializedError: 服务未初始化
            PlatformIntegrationError: 操作失败
        """
        service = self._ensure_initialized()
        start = datetime.utcnow()
        
        result: OperationResult = await service.execute(
            domain=CapabilityDomain.DATA_PERSISTENCE,
            operation="vector_search",
            params={
                "index": index,
                "vector": vector,
                "top_k": top_k,
            },
        )
        
        duration = (datetime.utcnow() - start).total_seconds() * 1000
        
        return IntegrationResult(
            success=result.success,
            data=result.data,
            error=result.error,
            provider=provider,
            duration_ms=duration,
        )
    
    # =====================================================================
    # 认知计算 (OpenAI, Anthropic)
    # =====================================================================
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        provider: str = "openai",
    ) -> IntegrationResult:
        """
        聊天补全
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度
            provider: 提供商
            
        Returns:
            IntegrationResult
            
        Raises:
            ServiceNotInitializedError: 服务未初始化
            PlatformIntegrationError: 操作失败
        """
        service = self._ensure_initialized()
        start = datetime.utcnow()
        
        request = InferenceRequest(
            prompt=messages[-1].get("content", "") if messages else "",
            model=model,
            parameters={
                "temperature": temperature,
                "messages": messages,
            },
        )
        
        result: OperationResult = await service.execute(
            domain=CapabilityDomain.COGNITIVE_COMPUTE,
            operation="infer",
            params={"request": request},
        )
        
        duration = (datetime.utcnow() - start).total_seconds() * 1000
        
        return IntegrationResult(
            success=result.success,
            data=result.data,
            error=result.error,
            provider=provider,
            duration_ms=duration,
        )
    
    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """
        流式聊天补全
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度
            
        Yields:
            文本块
            
        Raises:
            ServiceNotInitializedError: 服务未初始化
            PlatformIntegrationError: 操作失败
        """
        service = self._ensure_initialized()
        
        request = InferenceRequest(
            prompt=messages[-1].get("content", "") if messages else "",
            model=model,
            parameters={
                "temperature": temperature,
                "messages": messages,
                "stream": True,
            },
        )
        
        async for chunk in service.stream(
            domain=CapabilityDomain.COGNITIVE_COMPUTE,
            operation="infer_stream",
            params={"request": request},
        ):
            if chunk.text:
                yield chunk.text
    
    async def run_agent_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        provider: str = "openai",
    ) -> IntegrationResult:
        """
        运行智能体任务
        
        Args:
            task: 任务描述
            context: 上下文
            provider: 提供商
            
        Returns:
            IntegrationResult
            
        Raises:
            ServiceNotInitializedError: 服务未初始化
            PlatformIntegrationError: 操作失败
        """
        service = self._ensure_initialized()
        start = datetime.utcnow()
        
        agent_task = AgentTask(
            task=task,
            context=context or {},
        )
        
        result: OperationResult = await service.execute(
            domain=CapabilityDomain.COGNITIVE_COMPUTE,
            operation="agent_task",
            params={"task": agent_task},
        )
        
        duration = (datetime.utcnow() - start).total_seconds() * 1000
        
        return IntegrationResult(
            success=result.success,
            data=result.data,
            error=result.error,
            provider=provider,
            duration_ms=duration,
        )
    
    # =====================================================================
    # 代码工程 (GitHub)
    # =====================================================================
    
    async def create_pull_request(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
        provider: str = "github",
    ) -> IntegrationResult:
        """
        创建 Pull Request
        
        Args:
            title: PR 标题
            body: PR 内容
            head_branch: 源分支
            base_branch: 目标分支
            provider: 提供商
            
        Returns:
            IntegrationResult
            
        Raises:
            ServiceNotInitializedError: 服务未初始化
            PlatformIntegrationError: 操作失败
        """
        service = self._ensure_initialized()
        start = datetime.utcnow()
        
        result: OperationResult = await service.execute(
            domain=CapabilityDomain.CODE_ENGINEERING,
            operation="create_pr",
            params={
                "title": title,
                "body": body,
                "head": head_branch,
                "base": base_branch,
            },
        )
        
        duration = (datetime.utcnow() - start).total_seconds() * 1000
        
        return IntegrationResult(
            success=result.success,
            data=result.data,
            error=result.error,
            provider=provider,
            duration_ms=duration,
        )
    
    async def review_code(
        self,
        pr_number: int,
        comments: List[Dict[str, Any]],
        provider: str = "github",
    ) -> IntegrationResult:
        """
        代码审查
        
        Args:
            pr_number: PR 编号
            comments: 评论列表
            provider: 提供商
            
        Returns:
            IntegrationResult
            
        Raises:
            ServiceNotInitializedError: 服务未初始化
            PlatformIntegrationError: 操作失败
        """
        service = self._ensure_initialized()
        start = datetime.utcnow()
        
        result: OperationResult = await service.execute(
            domain=CapabilityDomain.CODE_ENGINEERING,
            operation="review",
            params={
                "pr_number": pr_number,
                "comments": comments,
            },
        )
        
        duration = (datetime.utcnow() - start).total_seconds() * 1000
        
        return IntegrationResult(
            success=result.success,
            data=result.data,
            error=result.error,
            provider=provider,
            duration_ms=duration,
        )
    
    # =====================================================================
    # 协作通信 (Slack)
    # =====================================================================
    
    async def send_notification(
        self,
        message: str,
        channel: Optional[str] = None,
        provider: str = "slack",
    ) -> IntegrationResult:
        """
        发送通知
        
        Args:
            message: 消息内容
            channel: 频道
            provider: 提供商
            
        Returns:
            IntegrationResult
            
        Raises:
            ServiceNotInitializedError: 服务未初始化
            PlatformIntegrationError: 操作失败
        """
        service = self._ensure_initialized()
        start = datetime.utcnow()
        
        payload = MessagePayload(
            content=message,
            channel=channel,
        )
        
        result: OperationResult = await service.execute(
            domain=CapabilityDomain.COLLABORATION,
            operation="send_message",
            params={"payload": payload},
        )
        
        duration = (datetime.utcnow() - start).total_seconds() * 1000
        
        return IntegrationResult(
            success=result.success,
            data=result.data,
            error=result.error,
            provider=provider,
            duration_ms=duration,
        )
    
    async def trigger_workflow(
        self,
        workflow_id: str,
        inputs: Optional[Dict[str, Any]] = None,
        provider: str = "slack",
    ) -> IntegrationResult:
        """
        触发工作流
        
        Args:
            workflow_id: 工作流 ID
            inputs: 输入参数
            provider: 提供商
            
        Returns:
            IntegrationResult
            
        Raises:
            ServiceNotInitializedError: 服务未初始化
            PlatformIntegrationError: 操作失败
        """
        service = self._ensure_initialized()
        start = datetime.utcnow()
        
        trigger = WorkflowTrigger(
            workflow_id=workflow_id,
            inputs=inputs or {},
        )
        
        result: OperationResult = await service.execute(
            domain=CapabilityDomain.COLLABORATION,
            operation="trigger_workflow",
            params={"trigger": trigger},
        )
        
        duration = (datetime.utcnow() - start).total_seconds() * 1000
        
        return IntegrationResult(
            success=result.success,
            data=result.data,
            error=result.error,
            provider=provider,
            duration_ms=duration,
        )
    
    # =====================================================================
    # 部署交付 (Vercel)
    # =====================================================================
    
    async def deploy(
        self,
        project: str,
        environment: str = "production",
        provider: str = "vercel",
    ) -> IntegrationResult:
        """
        部署项目
        
        Args:
            project: 项目名称
            environment: 环境
            provider: 提供商
            
        Returns:
            IntegrationResult
            
        Raises:
            ServiceNotInitializedError: 服务未初始化
            PlatformIntegrationError: 操作失败
        """
        service = self._ensure_initialized()
        start = datetime.utcnow()
        
        deploy_spec = DeploySpec(
            project=project,
            environment=environment,
        )
        
        result: OperationResult = await service.execute(
            domain=CapabilityDomain.DEPLOYMENT,
            operation="deploy",
            params={"spec": deploy_spec},
        )
        
        duration = (datetime.utcnow() - start).total_seconds() * 1000
        
        return IntegrationResult(
            success=result.success,
            data=result.data,
            error=result.error,
            provider=provider,
            duration_ms=duration,
        )
    
    # =====================================================================
    # 健康检查
    # =====================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态字典
        """
        if not PLATFORM_INTEGRATIONS_AVAILABLE:
            return {
                "status": "unavailable",
                "available": False,
                "error": IMPORT_ERROR,
            }
        
        if not self._initialized or self._service is None:
            return {
                "status": "not_initialized",
                "available": True,
                "initialized": False,
            }
        
        try:
            result = await self._service.health_check()
            return {
                "status": "healthy" if result.healthy else "unhealthy",
                "available": True,
                "initialized": True,
                "providers": result.provider_status if hasattr(result, 'provider_status') else {},
            }
        except Exception as e:
            return {
                "status": "error",
                "available": True,
                "initialized": True,
                "error": str(e),
            }
    
    def register_health_check(self, callback: Callable[[], Awaitable[Dict[str, Any]]]) -> None:
        """注册健康检查回调"""
        self._health_check_callbacks.append(callback)


# 全局服务实例
platform_integration_service_v2 = PlatformIntegrationService()


__all__ = [
    "PlatformIntegrationService",
    "platform_integration_service_v2",
    "IntegrationResult",
    "PlatformIntegrationError",
    "ServiceNotInitializedError",
    "ProviderConfigError",
    "ProviderUnavailableError",
]
