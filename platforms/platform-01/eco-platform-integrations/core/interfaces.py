# =============================================================================
# Eco-Platform Integration Framework - Core Interfaces
# =============================================================================
# 此模組定義所有第三方能力集成的抽象接口
# 遵循依賴倒置原則：核心業務邏輯依賴抽象，而非具體實現
# =============================================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, TypeVar, Generic, Union
from enum import Enum, auto
import asyncio
from datetime import datetime


# =============================================================================
# 通用數據結構
# =============================================================================

@dataclass
class CapabilityContext:
    """能力調用上下文"""
    request_id: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class OperationResult:
    """操作結果封裝"""
    success: bool
    data: Any = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: float = 0.0
    provider_info: Dict[str, Any] = None


@dataclass
class StreamChunk:
    """流式數據塊"""
    content: Any
    is_final: bool = False
    metadata: Dict[str, Any] = None


# =============================================================================
# 能力領域枚舉 (抽象分類，無平台名稱)
# =============================================================================

class CapabilityDomain(Enum):
    """能力領域分類"""
    DATA_PERSISTENCE = auto()      # 數據持久化
    COGNITIVE_COMPUTE = auto()     # 認知計算
    CODE_ENGINEERING = auto()      # 代碼工程
    COLLABORATION = auto()         # 協作通信
    VISUAL_DESIGN = auto()         # 視覺設計
    KNOWLEDGE_MGMT = auto()        # 知識管理
    DEPLOYMENT = auto()            # 部署交付
    LEARNING = auto()              # 學習教育


class ProviderTier(Enum):
    """提供者層級"""
    FREE = auto()
    STANDARD = auto()
    PREMIUM = auto()
    ENTERPRISE = auto()


# =============================================================================
# 基礎能力接口
# =============================================================================

class ICapabilityProvider(ABC):
    """能力提供者基礎接口"""
    
    @property
    @abstractmethod
    def domain(self) -> CapabilityDomain:
        """返回所屬能力領域"""
        pass
    
    @property
    @abstractmethod
    def provider_id(self) -> str:
        """返回提供者唯一標識 (抽象ID，非平台名稱)"""
        pass
    
    @abstractmethod
    async def health_check(self) -> OperationResult:
        """健康檢查"""
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> List[str]:
        """獲取支持的能力列表"""
        pass


# =============================================================================
# 領域一：數據持久化 (Data Persistence)
# =============================================================================

@dataclass
class QuerySpec:
    """查詢規格"""
    table: str
    filters: Dict[str, Any] = None
    ordering: List[str] = None
    limit: Optional[int] = None
    offset: Optional[int] = None


@dataclass
class MutationSpec:
    """變更規格"""
    operation: str  # insert, update, delete, upsert
    table: str
    data: Dict[str, Any]
    conditions: Dict[str, Any] = None


class IDataPersistenceProvider(ICapabilityProvider):
    """數據持久化提供者接口"""
    
    @abstractmethod
    async def query(self, spec: QuerySpec, ctx: CapabilityContext) -> OperationResult:
        """執行查詢"""
        pass
    
    @abstractmethod
    async def mutate(self, spec: MutationSpec, ctx: CapabilityContext) -> OperationResult:
        """執行變更"""
        pass
    
    @abstractmethod
    async def subscribe(self, table: str, ctx: CapabilityContext) -> AsyncIterator[StreamChunk]:
        """訂閱實時變更"""
        pass
    
    @abstractmethod
    async def execute_sql(self, sql: str, params: List[Any], ctx: CapabilityContext) -> OperationResult:
        """執行原生SQL"""
        pass
    
    @abstractmethod
    async def vector_search(self, table: str, vector: List[float], top_k: int, ctx: CapabilityContext) -> OperationResult:
        """向量相似度搜索"""
        pass


# =============================================================================
# 領域二：認知計算 (Cognitive Computing)
# =============================================================================

@dataclass
class InferenceRequest:
    """推理請求"""
    prompt: str
    context: List[Dict[str, str]] = None  # 對話歷史
    parameters: Dict[str, Any] = None     # 溫度、最大token等
    modalities: List[str] = None          # text, image, audio, video


@dataclass
class AgentTask:
    """代理任務"""
    task_type: str
    description: str
    inputs: Dict[str, Any]
    expected_output_format: str = "text"


class ICognitiveComputeProvider(ICapabilityProvider):
    """認知計算提供者接口"""
    
    @abstractmethod
    async def generate(self, request: InferenceRequest, ctx: CapabilityContext) -> OperationResult:
        """單次生成"""
        pass
    
    @abstractmethod
    async def generate_stream(self, request: InferenceRequest, ctx: CapabilityContext) -> AsyncIterator[StreamChunk]:
        """流式生成"""
        pass
    
    @abstractmethod
    async def function_call(self, prompt: str, functions: List[Dict], ctx: CapabilityContext) -> OperationResult:
        """函數調用"""
        pass
    
    @abstractmethod
    async def execute_agent_task(self, task: AgentTask, ctx: CapabilityContext) -> OperationResult:
        """執行代理任務"""
        pass
    
    @abstractmethod
    async def embed(self, texts: List[str], ctx: CapabilityContext) -> OperationResult:
        """文本嵌入"""
        pass
    
    @abstractmethod
    async def multimodal_process(self, inputs: Dict[str, Any], ctx: CapabilityContext) -> OperationResult:
        """多模態處理 (文本+圖像+音頻)"""
        pass


# =============================================================================
# 領域三：代碼工程 (Code Engineering)
# =============================================================================

@dataclass
class CodeContext:
    """代碼上下文"""
    content: str
    language: str
    file_path: Optional[str] = None
    cursor_position: Optional[int] = None
    selection_range: Optional[tuple] = None


@dataclass
class ReviewRequest:
    """審查請求"""
    code: str
    language: str
    review_type: str = "general"  # general, security, performance, style


class ICodeEngineeringProvider(ICapabilityProvider):
    """代碼工程提供者接口"""
    
    @abstractmethod
    async def complete(self, context: CodeContext, ctx: CapabilityContext) -> OperationResult:
        """代碼補全"""
        pass
    
    @abstractmethod
    async def explain(self, code: str, language: str, ctx: CapabilityContext) -> OperationResult:
        """代碼解釋"""
        pass
    
    @abstractmethod
    async def refactor(self, code: str, language: str, instruction: str, ctx: CapabilityContext) -> OperationResult:
        """代碼重構"""
        pass
    
    @abstractmethod
    async def review(self, request: ReviewRequest, ctx: CapabilityContext) -> OperationResult:
        """代碼審查"""
        pass
    
    @abstractmethod
    async def generate_tests(self, code: str, language: str, ctx: CapabilityContext) -> OperationResult:
        """生成測試"""
        pass
    
    @abstractmethod
    async def translate_language(self, code: str, source_lang: str, target_lang: str, ctx: CapabilityContext) -> OperationResult:
        """跨語言轉換"""
        pass
    
    @abstractmethod
    async def search_repository(self, query: str, repo_id: Optional[str], ctx: CapabilityContext) -> OperationResult:
        """倉庫搜索"""
        pass


# =============================================================================
# 領域四：協作通信 (Collaboration)
# =============================================================================

@dataclass
class MessagePayload:
    """消息載荷"""
    channel: str
    content: str
    attachments: List[Dict] = None
    thread_id: Optional[str] = None
    mentions: List[str] = None


@dataclass
class WorkflowTrigger:
    """工作流觸發器"""
    event_type: str
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]


class ICollaborationProvider(ICapabilityProvider):
    """協作通信提供者接口"""
    
    @abstractmethod
    async def send_message(self, payload: MessagePayload, ctx: CapabilityContext) -> OperationResult:
        """發送消息"""
        pass
    
    @abstractmethod
    async def create_channel(self, name: str, members: List[str], ctx: CapabilityContext) -> OperationResult:
        """創建頻道"""
        pass
    
    @abstractmethod
    async def summarize_conversation(self, channel: str, since: datetime, ctx: CapabilityContext) -> OperationResult:
        """對話摘要"""
        pass
    
    @abstractmethod
    async def setup_workflow(self, trigger: WorkflowTrigger, ctx: CapabilityContext) -> OperationResult:
        """設置自動化工作流"""
        pass
    
    @abstractmethod
    async def search_knowledge(self, query: str, ctx: CapabilityContext) -> OperationResult:
        """企業知識搜索"""
        pass


# =============================================================================
# 領域五：視覺設計 (Visual Design)
# =============================================================================

@dataclass
class DesignAsset:
    """設計資源"""
    asset_type: str  # component, style, prototype
    name: str
    content: Any
    metadata: Dict[str, Any] = None


@dataclass
class DesignSystem:
    """設計系統"""
    name: str
    components: List[DesignAsset]
    styles: List[DesignAsset]


class IVisualDesignProvider(ICapabilityProvider):
    """視覺設計提供者接口"""
    
    @abstractmethod
    async def get_components(self, system_id: Optional[str], ctx: CapabilityContext) -> OperationResult:
        """獲取組件庫"""
        pass
    
    @abstractmethod
    async def export_asset(self, asset_id: str, format: str, ctx: CapabilityContext) -> OperationResult:
        """導出設計資源"""
        pass
    
    @abstractmethod
    async def generate_from_description(self, description: str, asset_type: str, ctx: CapabilityContext) -> OperationResult:
        """根據描述生成設計"""
        pass
    
    @abstractmethod
    async def inspect_design(self, design_id: str, ctx: CapabilityContext) -> OperationResult:
        """設計檢視 (獲取CSS/樣式)"""
        pass
    
    @abstractmethod
    async def create_prototype(self, screens: List[str], transitions: List[Dict], ctx: CapabilityContext) -> OperationResult:
        """創建交互原型"""
        pass


# =============================================================================
# 領域六：知識管理 (Knowledge Management)
# =============================================================================

@dataclass
class DocumentSpec:
    """文檔規格"""
    title: str
    content: str
    format: str = "markdown"
    parent_id: Optional[str] = None
    tags: List[str] = None


@dataclass
class KnowledgeQuery:
    """知識查詢"""
    query: str
    filters: Dict[str, Any] = None
    top_k: int = 10


class IKnowledgeManagementProvider(ICapabilityProvider):
    """知識管理提供者接口"""
    
    @abstractmethod
    async def create_document(self, spec: DocumentSpec, ctx: CapabilityContext) -> OperationResult:
        """創建文檔"""
        pass
    
    @abstractmethod
    async def update_document(self, doc_id: str, spec: DocumentSpec, ctx: CapabilityContext) -> OperationResult:
        """更新文檔"""
        pass
    
    @abstractmethod
    async def query_knowledge(self, query: KnowledgeQuery, ctx: CapabilityContext) -> OperationResult:
        """知識查詢"""
        pass
    
    @abstractmethod
    async def sync_from_git(self, repo_url: str, branch: str, ctx: CapabilityContext) -> OperationResult:
        """從Git同步文檔"""
        pass
    
    @abstractmethod
    async def export_to_format(self, doc_id: str, target_format: str, ctx: CapabilityContext) -> OperationResult:
        """導出為其他格式 (PDF/EPUB)"""
        pass


# =============================================================================
# 領域七：部署交付 (Deployment)
# =============================================================================

@dataclass
class DeploySpec:
    """部署規格"""
    artifact_path: str
    environment: str
    version: str
    config_overrides: Dict[str, Any] = None


@dataclass
class BuildSpec:
    """構建規格"""
    source_path: str
    target_platform: str
    cache_key: Optional[str] = None
    parallel_jobs: int = 1


class IDeploymentProvider(ICapabilityProvider):
    """部署交付提供者接口"""
    
    @abstractmethod
    async def build(self, spec: BuildSpec, ctx: CapabilityContext) -> OperationResult:
        """構建制品"""
        pass
    
    @abstractmethod
    async def deploy(self, spec: DeploySpec, ctx: CapabilityContext) -> OperationResult:
        """部署到環境"""
        pass
    
    @abstractmethod
    async def get_deployment_status(self, deployment_id: str, ctx: CapabilityContext) -> OperationResult:
        """獲取部署狀態"""
        pass
    
    @abstractmethod
    async def rollback(self, deployment_id: str, ctx: CapabilityContext) -> OperationResult:
        """回滾部署"""
        pass
    
    @abstractmethod
    async def preview_deployment(self, branch: str, ctx: CapabilityContext) -> OperationResult:
        """預覽部署 (Preview URL)"""
        pass


# =============================================================================
# 領域八：學習教育 (Learning)
# =============================================================================

@dataclass
class LearningPath:
    """學習路徑"""
    topic: str
    skill_level: str  # beginner, intermediate, advanced
    estimated_hours: int
    modules: List[Dict[str, Any]]


@dataclass
class ExerciseSubmission:
    """練習提交"""
    exercise_id: str
    code: str
    language: str


class ILearningProvider(ICapabilityProvider):
    """學習教育提供者接口"""
    
    @abstractmethod
    async def get_learning_path(self, topic: str, skill_level: str, ctx: CapabilityContext) -> OperationResult:
        """獲取學習路徑"""
        pass
    
    @abstractmethod
    async def submit_exercise(self, submission: ExerciseSubmission, ctx: CapabilityContext) -> OperationResult:
        """提交練習"""
        pass
    
    @abstractmethod
    async def get_hint(self, exercise_id: str, ctx: CapabilityContext) -> OperationResult:
        """獲取提示"""
        pass
    
    @abstractmethod
    async def track_progress(self, user_id: str, ctx: CapabilityContext) -> OperationResult:
        """追蹤學習進度"""
        pass
