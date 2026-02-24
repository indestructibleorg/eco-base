# =============================================================================
# Service Facade
# =============================================================================
# 統一服務門面
# 提供簡潔的 API 調用所有第三方能力
# =============================================================================

from typing import Any, AsyncIterator, Dict, List, Optional, Union
import logging

from core.interfaces import (
    CapabilityContext, OperationResult, StreamChunk,
    CapabilityDomain, QuerySpec, MutationSpec, InferenceRequest,
    AgentTask, CodeContext, ReviewRequest, MessagePayload,
    WorkflowTrigger, DocumentSpec, KnowledgeQuery,
    DeploySpec, BuildSpec, LearningPath, ExerciseSubmission
)
from registry.provider_registry import ProviderRegistry, register_all_adapters
from config.platform_configs import get_provider_config, validate_config

logger = logging.getLogger(__name__)


class EcoPlatformService:
    """
    Eco-Platform 統一服務門面
    
    提供對所有第三方能力的統一訪問接口
    """
    
    def __init__(self):
        self._registry = ProviderRegistry()
        self._initialized = False
    
    async def initialize(self, provider_ids: Optional[List[str]] = None):
        """
        初始化服務
        
        Args:
            provider_ids: 要初始化的提供者ID列表，None表示全部
        """
        if self._initialized:
            return
        
        # 註冊所有適配器
        register_all_adapters()
        
        # 獲取要初始化的提供者
        if provider_ids is None:
            from config.platform_configs import get_all_configs
            provider_ids = list(get_all_configs().keys())
        
        # 創建提供者實例
        for provider_id in provider_ids:
            config = get_provider_config(provider_id)
            
            if not validate_config(provider_id, config):
                logger.warning(f"Skipping {provider_id}: incomplete configuration")
                continue
            
            try:
                self._registry.create_provider(provider_id, config)
                logger.info(f"Initialized provider: {provider_id}")
            except Exception as e:
                logger.error(f"Failed to initialize {provider_id}: {e}")
        
        self._initialized = True
    
    # =====================================================================
    # 數據持久化服務
    # =====================================================================
    
    async def query_data(
        self, 
        table: str, 
        filters: Dict[str, Any] = None,
        provider: str = 'alpha-persistence',
        ctx: Optional[CapabilityContext] = None
    ) -> OperationResult:
        """
        查詢數據
        
        Args:
            table: 表名
            filters: 過濾條件
            provider: 提供者ID
            ctx: 上下文
            
        Returns:
            查詢結果
        """
        from core.interfaces import IDataPersistenceProvider
        
        provider_instance = self._registry.get_provider(provider)
        if not isinstance(provider_instance, IDataPersistenceProvider):
            return OperationResult(success=False, error_message=f'Provider {provider} not found or invalid type')
        
        spec = QuerySpec(table=table, filters=filters)
        ctx = ctx or CapabilityContext(request_id=self._generate_request_id())
        
        return await provider_instance.query(spec, ctx)
    
    async def mutate_data(
        self,
        operation: str,
        table: str,
        data: Dict[str, Any],
        conditions: Dict[str, Any] = None,
        provider: str = 'alpha-persistence',
        ctx: Optional[CapabilityContext] = None
    ) -> OperationResult:
        """
        變更數據
        
        Args:
            operation: 操作類型 (insert, update, delete, upsert)
            table: 表名
            data: 數據
            conditions: 條件
            provider: 提供者ID
            ctx: 上下文
            
        Returns:
            操作結果
        """
        from core.interfaces import IDataPersistenceProvider
        
        provider_instance = self._registry.get_provider(provider)
        if not isinstance(provider_instance, IDataPersistenceProvider):
            return OperationResult(success=False, error_message=f'Provider {provider} not found or invalid type')
        
        spec = MutationSpec(operation=operation, table=table, data=data, conditions=conditions)
        ctx = ctx or CapabilityContext(request_id=self._generate_request_id())
        
        return await provider_instance.mutate(spec, ctx)
    
    async def vector_search(
        self,
        table: str,
        vector: List[float],
        top_k: int = 10,
        provider: str = 'alpha-persistence',
        ctx: Optional[CapabilityContext] = None
    ) -> OperationResult:
        """
        向量搜索
        
        Args:
            table: 表名
            vector: 查詢向量
            top_k: 返回數量
            provider: 提供者ID
            ctx: 上下文
            
        Returns:
            搜索結果
        """
        from core.interfaces import IDataPersistenceProvider
        
        provider_instance = self._registry.get_provider(provider)
        if not isinstance(provider_instance, IDataPersistenceProvider):
            return OperationResult(success=False, error_message=f'Provider {provider} not found or invalid type')
        
        ctx = ctx or CapabilityContext(request_id=self._generate_request_id())
        return await provider_instance.vector_search(table, vector, top_k, ctx)
    
    # =====================================================================
    # 認知計算服務
    # =====================================================================
    
    async def generate_text(
        self,
        prompt: str,
        context: List[Dict[str, str]] = None,
        parameters: Dict[str, Any] = None,
        provider: str = 'gamma-cognitive',
        ctx: Optional[CapabilityContext] = None
    ) -> OperationResult:
        """
        生成文本
        
        Args:
            prompt: 提示詞
            context: 對話上下文
            parameters: 生成參數
            provider: 提供者ID
            ctx: 上下文
            
        Returns:
            生成結果
        """
        from core.interfaces import ICognitiveComputeProvider
        
        provider_instance = self._registry.get_provider(provider)
        if not isinstance(provider_instance, ICognitiveComputeProvider):
            return OperationResult(success=False, error_message=f'Provider {provider} not found or invalid type')
        
        request = InferenceRequest(
            prompt=prompt,
            context=context,
            parameters=parameters
        )
        ctx = ctx or CapabilityContext(request_id=self._generate_request_id())
        
        return await provider_instance.generate(request, ctx)
    
    async def generate_text_stream(
        self,
        prompt: str,
        context: List[Dict[str, str]] = None,
        parameters: Dict[str, Any] = None,
        provider: str = 'gamma-cognitive',
        ctx: Optional[CapabilityContext] = None
    ) -> AsyncIterator[StreamChunk]:
        """
        流式生成文本
        
        Args:
            prompt: 提示詞
            context: 對話上下文
            parameters: 生成參數
            provider: 提供者ID
            ctx: 上下文
            
        Yields:
            流式數據塊
        """
        from core.interfaces import ICognitiveComputeProvider
        
        provider_instance = self._registry.get_provider(provider)
        if not isinstance(provider_instance, ICognitiveComputeProvider):
            yield StreamChunk(content={'error': f'Provider {provider} not found'}, is_final=True)
            return
        
        request = InferenceRequest(
            prompt=prompt,
            context=context,
            parameters=parameters
        )
        ctx = ctx or CapabilityContext(request_id=self._generate_request_id())
        
        async for chunk in provider_instance.generate_stream(request, ctx):
            yield chunk
    
    async def embed_texts(
        self,
        texts: List[str],
        provider: str = 'gamma-cognitive',
        ctx: Optional[CapabilityContext] = None
    ) -> OperationResult:
        """
        文本嵌入
        
        Args:
            texts: 文本列表
            provider: 提供者ID
            ctx: 上下文
            
        Returns:
            嵌入結果
        """
        from core.interfaces import ICognitiveComputeProvider
        
        provider_instance = self._registry.get_provider(provider)
        if not isinstance(provider_instance, ICognitiveComputeProvider):
            return OperationResult(success=False, error_message=f'Provider {provider} not found or invalid type')
        
        ctx = ctx or CapabilityContext(request_id=self._generate_request_id())
        return await provider_instance.embed(texts, ctx)
    
    async def function_call(
        self,
        prompt: str,
        functions: List[Dict],
        provider: str = 'gamma-cognitive',
        ctx: Optional[CapabilityContext] = None
    ) -> OperationResult:
        """
        函數調用
        
        Args:
            prompt: 提示詞
            functions: 可用函數定義
            provider: 提供者ID
            ctx: 上下文
            
        Returns:
            調用結果
        """
        from core.interfaces import ICognitiveComputeProvider
        
        provider_instance = self._registry.get_provider(provider)
        if not isinstance(provider_instance, ICognitiveComputeProvider):
            return OperationResult(success=False, error_message=f'Provider {provider} not found or invalid type')
        
        ctx = ctx or CapabilityContext(request_id=self._generate_request_id())
        return await provider_instance.function_call(prompt, functions, ctx)
    
    # =====================================================================
    # 代碼工程服務
    # =====================================================================
    
    async def complete_code(
        self,
        code: str,
        language: str,
        cursor_position: int = None,
        provider: str = 'zeta-code',
        ctx: Optional[CapabilityContext] = None
    ) -> OperationResult:
        """
        代碼補全
        
        Args:
            code: 代碼內容
            language: 編程語言
            cursor_position: 光標位置
            provider: 提供者ID
            ctx: 上下文
            
        Returns:
            補全結果
        """
        from core.interfaces import ICodeEngineeringProvider
        
        provider_instance = self._registry.get_provider(provider)
        if not isinstance(provider_instance, ICodeEngineeringProvider):
            return OperationResult(success=False, error_message=f'Provider {provider} not found or invalid type')
        
        context = CodeContext(
            content=code,
            language=language,
            cursor_position=cursor_position
        )
        ctx = ctx or CapabilityContext(request_id=self._generate_request_id())
        
        return await provider_instance.complete(context, ctx)
    
    async def explain_code(
        self,
        code: str,
        language: str,
        provider: str = 'zeta-code',
        ctx: Optional[CapabilityContext] = None
    ) -> OperationResult:
        """
        解釋代碼
        
        Args:
            code: 代碼內容
            language: 編程語言
            provider: 提供者ID
            ctx: 上下文
            
        Returns:
            解釋結果
        """
        from core.interfaces import ICodeEngineeringProvider
        
        provider_instance = self._registry.get_provider(provider)
        if not isinstance(provider_instance, ICodeEngineeringProvider):
            return OperationResult(success=False, error_message=f'Provider {provider} not found or invalid type')
        
        ctx = ctx or CapabilityContext(request_id=self._generate_request_id())
        return await provider_instance.explain(code, language, ctx)
    
    async def review_code(
        self,
        code: str,
        language: str,
        review_type: str = 'general',
        provider: str = 'zeta-code',
        ctx: Optional[CapabilityContext] = None
    ) -> OperationResult:
        """
        審查代碼
        
        Args:
            code: 代碼內容
            language: 編程語言
            review_type: 審查類型
            provider: 提供者ID
            ctx: 上下文
            
        Returns:
            審查結果
        """
        from core.interfaces import ICodeEngineeringProvider
        
        provider_instance = self._registry.get_provider(provider)
        if not isinstance(provider_instance, ICodeEngineeringProvider):
            return OperationResult(success=False, error_message=f'Provider {provider} not found or invalid type')
        
        request = ReviewRequest(code=code, language=language, review_type=review_type)
        ctx = ctx or CapabilityContext(request_id=self._generate_request_id())
        
        return await provider_instance.review(request, ctx)
    
    # =====================================================================
    # 協作通信服務
    # =====================================================================
    
    async def send_message(
        self,
        channel: str,
        content: str,
        attachments: List[Dict] = None,
        provider: str = 'iota-collaboration',
        ctx: Optional[CapabilityContext] = None
    ) -> OperationResult:
        """
        發送消息
        
        Args:
            channel: 頻道
            content: 內容
            attachments: 附件
            provider: 提供者ID
            ctx: 上下文
            
        Returns:
            發送結果
        """
        from core.interfaces import ICollaborationProvider
        
        provider_instance = self._registry.get_provider(provider)
        if not isinstance(provider_instance, ICollaborationProvider):
            return OperationResult(success=False, error_message=f'Provider {provider} not found or invalid type')
        
        payload = MessagePayload(channel=channel, content=content, attachments=attachments)
        ctx = ctx or CapabilityContext(request_id=self._generate_request_id())
        
        return await provider_instance.send_message(payload, ctx)
    
    async def summarize_channel(
        self,
        channel: str,
        provider: str = 'iota-collaboration',
        ctx: Optional[CapabilityContext] = None
    ) -> OperationResult:
        """
        摘要頻道對話
        
        Args:
            channel: 頻道
            provider: 提供者ID
            ctx: 上下文
            
        Returns:
            摘要結果
        """
        from core.interfaces import ICollaborationProvider
        from datetime import datetime, timedelta
        
        provider_instance = self._registry.get_provider(provider)
        if not isinstance(provider_instance, ICollaborationProvider):
            return OperationResult(success=False, error_message=f'Provider {provider} not found or invalid type')
        
        since = datetime.utcnow() - timedelta(hours=24)
        ctx = ctx or CapabilityContext(request_id=self._generate_request_id())
        
        return await provider_instance.summarize_conversation(channel, since, ctx)
    
    # =====================================================================
    # 部署交付服務
    # =====================================================================
    
    async def deploy(
        self,
        artifact_path: str,
        environment: str,
        version: str,
        config_overrides: Dict[str, Any] = None,
        provider: str = 'omicron-deployment',
        ctx: Optional[CapabilityContext] = None
    ) -> OperationResult:
        """
        部署應用
        
        Args:
            artifact_path: 制品路徑
            environment: 環境
            version: 版本
            config_overrides: 配置覆蓋
            provider: 提供者ID
            ctx: 上下文
            
        Returns:
            部署結果
        """
        from core.interfaces import IDeploymentProvider
        
        provider_instance = self._registry.get_provider(provider)
        if not isinstance(provider_instance, IDeploymentProvider):
            return OperationResult(success=False, error_message=f'Provider {provider} not found or invalid type')
        
        spec = DeploySpec(
            artifact_path=artifact_path,
            environment=environment,
            version=version,
            config_overrides=config_overrides
        )
        ctx = ctx or CapabilityContext(request_id=self._generate_request_id())
        
        return await provider_instance.deploy(spec, ctx)
    
    # =====================================================================
    # 輔助方法
    # =====================================================================
    
    def _generate_request_id(self) -> str:
        """生成請求ID"""
        import uuid
        return str(uuid.uuid4())
    
    def get_available_providers(self, domain: Optional[CapabilityDomain] = None) -> List[str]:
        """
        獲取可用提供者
        
        Args:
            domain: 能力領域，None表示全部
            
        Returns:
            提供者ID列表
        """
        if domain:
            return self._registry.get_provider_ids_by_domain(domain)
        return list(self._registry.list_all_providers().keys())
    
    async def health_check(self) -> Dict[str, Any]:
        """
        健康檢查
        
        Returns:
            健康狀態報告
        """
        return self._registry.health_check_all()


# 全局服務實例
eco_service = EcoPlatformService()
