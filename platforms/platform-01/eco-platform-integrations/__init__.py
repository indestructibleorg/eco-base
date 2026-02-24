# =============================================================================
# Eco-Platform Integration Framework
# =============================================================================
# 企業級第三方平台能力集成框架
# =============================================================================

__version__ = "1.0.0"
__author__ = "Eco-Base Team"

# 導出核心接口
from core.interfaces import (
    # 基礎接口
    ICapabilityProvider,
    CapabilityContext,
    OperationResult,
    StreamChunk,
    CapabilityDomain,
    ProviderTier,
    
    # 數據持久化
    IDataPersistenceProvider,
    QuerySpec,
    MutationSpec,
    
    # 認知計算
    ICognitiveComputeProvider,
    InferenceRequest,
    AgentTask,
    
    # 代碼工程
    ICodeEngineeringProvider,
    CodeContext,
    ReviewRequest,
    
    # 協作通信
    ICollaborationProvider,
    MessagePayload,
    WorkflowTrigger,
    
    # 視覺設計
    IVisualDesignProvider,
    DesignAsset,
    DesignSystem,
    
    # 知識管理
    IKnowledgeManagementProvider,
    DocumentSpec,
    KnowledgeQuery,
    
    # 部署交付
    IDeploymentProvider,
    DeploySpec,
    BuildSpec,
    
    # 學習教育
    ILearningProvider,
    LearningPath,
    ExerciseSubmission,
)

# 導出服務門面
from core.service_facade import EcoPlatformService, eco_service

# 導出註冊中心
from registry.provider_registry import ProviderRegistry, register_all_adapters, registry

# 導出配置
from config.platform_configs import (
    get_provider_config,
    validate_config,
    get_all_configs,
)

__all__ = [
    # 版本
    '__version__',
    
    # 基礎接口
    'ICapabilityProvider',
    'CapabilityContext',
    'OperationResult',
    'StreamChunk',
    'CapabilityDomain',
    'ProviderTier',
    
    # 領域接口
    'IDataPersistenceProvider',
    'ICognitiveComputeProvider',
    'ICodeEngineeringProvider',
    'ICollaborationProvider',
    'IVisualDesignProvider',
    'IKnowledgeManagementProvider',
    'IDeploymentProvider',
    'ILearningProvider',
    
    # 數據結構
    'QuerySpec',
    'MutationSpec',
    'InferenceRequest',
    'AgentTask',
    'CodeContext',
    'ReviewRequest',
    'MessagePayload',
    'WorkflowTrigger',
    'DesignAsset',
    'DesignSystem',
    'DocumentSpec',
    'KnowledgeQuery',
    'DeploySpec',
    'BuildSpec',
    'LearningPath',
    'ExerciseSubmission',
    
    # 服務
    'EcoPlatformService',
    'eco_service',
    
    # 註冊中心
    'ProviderRegistry',
    'register_all_adapters',
    'registry',
    
    # 配置
    'get_provider_config',
    'validate_config',
    'get_all_configs',
]
