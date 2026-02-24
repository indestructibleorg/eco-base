# =============================================================================
# Provider Registry
# =============================================================================
# 能力提供者註冊中心
# 負責管理所有第三方平台的適配器實例
# =============================================================================

from typing import Any, Dict, List, Optional, Type, TypeVar
import importlib
import logging

from core.interfaces import (
    ICapabilityProvider, CapabilityDomain,
    IDataPersistenceProvider, ICognitiveComputeProvider,
    ICodeEngineeringProvider, ICollaborationProvider,
    IVisualDesignProvider, IKnowledgeManagementProvider,
    IDeploymentProvider, ILearningProvider
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=ICapabilityProvider)


class ProviderRegistry:
    """
    能力提供者註冊中心
    
    使用單例模式確保全局唯一實例
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._providers: Dict[str, ICapabilityProvider] = {}
            cls._instance._domain_map: Dict[CapabilityDomain, List[str]] = {
                domain: [] for domain in CapabilityDomain
            }
            cls._instance._adapter_classes: Dict[str, Type] = {}
        return cls._instance
    
    def register_adapter_class(self, provider_id: str, adapter_class: Type[T]) -> None:
        """
        註冊適配器類
        
        Args:
            provider_id: 提供者唯一標識
            adapter_class: 適配器類
        """
        self._adapter_classes[provider_id] = adapter_class
        logger.info(f"Registered adapter class: {provider_id}")
    
    def create_provider(self, provider_id: str, config: Dict[str, Any]) -> ICapabilityProvider:
        """
        創建提供者實例
        
        Args:
            provider_id: 提供者唯一標識
            config: 配置參數
            
        Returns:
            提供者實例
        """
        if provider_id not in self._adapter_classes:
            raise ValueError(f"Unknown provider: {provider_id}")
        
        adapter_class = self._adapter_classes[provider_id]
        provider = adapter_class(config)
        
        # 註冊實例
        self._providers[provider_id] = provider
        self._domain_map[provider.domain].append(provider_id)
        
        logger.info(f"Created provider instance: {provider_id} ({provider.domain.name})")
        return provider
    
    def get_provider(self, provider_id: str) -> Optional[ICapabilityProvider]:
        """
        獲取提供者實例
        
        Args:
            provider_id: 提供者唯一標識
            
        Returns:
            提供者實例或 None
        """
        return self._providers.get(provider_id)
    
    def get_providers_by_domain(self, domain: CapabilityDomain) -> List[ICapabilityProvider]:
        """
        根據領域獲取所有提供者
        
        Args:
            domain: 能力領域
            
        Returns:
            提供者實例列表
        """
        provider_ids = self._domain_map.get(domain, [])
        return [self._providers[pid] for pid in provider_ids if pid in self._providers]
    
    def get_provider_ids_by_domain(self, domain: CapabilityDomain) -> List[str]:
        """
        根據領域獲取所有提供者ID
        
        Args:
            domain: 能力領域
            
        Returns:
            提供者ID列表
        """
        return self._domain_map.get(domain, [])
    
    def list_all_providers(self) -> Dict[str, str]:
        """
        列出所有已註冊的提供者
        
        Returns:
            {provider_id: domain_name}
        """
        return {
            pid: provider.domain.name 
            for pid, provider in self._providers.items()
        }
    
    def health_check_all(self) -> Dict[str, Any]:
        """
        對所有提供者執行健康檢查
        
        Returns:
            健康狀態報告
        """
        import asyncio
        
        async def check_provider(pid: str, provider: ICapabilityProvider):
            try:
                result = await provider.health_check()
                return pid, result.success, result.error_message
            except Exception as e:
                return pid, False, str(e)
        
        async def run_checks():
            tasks = [
                check_provider(pid, provider)
                for pid, provider in self._providers.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        try:
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(run_checks())
        except RuntimeError:
            results = asyncio.run(run_checks())
        
        return {
            pid: {'healthy': healthy, 'error': error}
            for pid, healthy, error in results
        }
    
    def remove_provider(self, provider_id: str) -> bool:
        """
        移除提供者
        
        Args:
            provider_id: 提供者唯一標識
            
        Returns:
            是否成功移除
        """
        if provider_id not in self._providers:
            return False
        
        provider = self._providers[provider_id]
        domain = provider.domain
        
        del self._providers[provider_id]
        self._domain_map[domain].remove(provider_id)
        
        logger.info(f"Removed provider: {provider_id}")
        return True


# =============================================================================
# 預定義適配器註冊
# =============================================================================

def register_all_adapters():
    """
    註冊所有內置適配器
    """
    registry = ProviderRegistry()
    
    # 數據持久化適配器
    from adapters.data_persistence import AlphaPersistenceAdapter, BetaPersistenceAdapter
    registry.register_adapter_class('alpha-persistence', AlphaPersistenceAdapter)
    registry.register_adapter_class('beta-persistence', BetaPersistenceAdapter)
    
    # 認知計算適配器
    from adapters.cognitive_compute import GammaCognitiveAdapter, DeltaCognitiveAdapter, EpsilonCognitiveAdapter
    registry.register_adapter_class('gamma-cognitive', GammaCognitiveAdapter)
    registry.register_adapter_class('delta-cognitive', DeltaCognitiveAdapter)
    registry.register_adapter_class('epsilon-cognitive', EpsilonCognitiveAdapter)
    
    # 代碼工程適配器
    from adapters.code_engineering import ZetaCodeAdapter, EtaCodeAdapter, ThetaCodeAdapter
    registry.register_adapter_class('zeta-code', ZetaCodeAdapter)
    registry.register_adapter_class('eta-code', EtaCodeAdapter)
    registry.register_adapter_class('theta-code', ThetaCodeAdapter)
    
    # 協作通信適配器
    from adapters.collaboration import IotaCollaborationAdapter, KappaCollaborationAdapter
    registry.register_adapter_class('iota-collaboration', IotaCollaborationAdapter)
    registry.register_adapter_class('kappa-collaboration', KappaCollaborationAdapter)
    
    # 視覺設計適配器
    from adapters.visual_design import LambdaVisualAdapter, MuVisualAdapter
    registry.register_adapter_class('lambda-visual', LambdaVisualAdapter)
    registry.register_adapter_class('mu-visual', MuVisualAdapter)
    
    # 知識管理適配器
    from adapters.knowledge_mgmt import NuKnowledgeAdapter, XiKnowledgeAdapter
    registry.register_adapter_class('nu-knowledge', NuKnowledgeAdapter)
    registry.register_adapter_class('xi-knowledge', XiKnowledgeAdapter)
    
    # 部署交付適配器
    from adapters.deployment import OmicronDeploymentAdapter, PiDeploymentAdapter, RhoDeploymentAdapter
    registry.register_adapter_class('omicron-deployment', OmicronDeploymentAdapter)
    registry.register_adapter_class('pi-deployment', PiDeploymentAdapter)
    registry.register_adapter_class('rho-deployment', RhoDeploymentAdapter)
    
    # 學習教育適配器
    from adapters.learning import SigmaLearningAdapter, TauLearningAdapter, UpsilonLearningAdapter
    registry.register_adapter_class('sigma-learning', SigmaLearningAdapter)
    registry.register_adapter_class('tau-learning', TauLearningAdapter)
    registry.register_adapter_class('upsilon-learning', UpsilonLearningAdapter)
    
    logger.info("All adapters registered successfully")


# 全局註冊中心實例
registry = ProviderRegistry()
