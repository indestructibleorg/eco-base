"""
闭环系统入口

渐进式闭环系统 - Phase 1 + Phase 2 + Phase 3

Phase 1: 基础闭环
- 异常检测引擎
- 自动修复引擎
- 规则引擎
- 闭环指标

Phase 2: 智能闭环
- 根因分析引擎 (RCA)
- 智能告警路由
- 容量预测与自动调整
- 决策工作流引擎

Phase 3: 自治闭环
- 自学习优化引擎
- 多目标决策优化
- 知识图谱系统
- 预测性修复系统
- 跨系统协同
- 人机协作界面
"""

__version__ = "3.0.0"

# Phase 1 组件
from .detector import (
    AnomalyDetector,
    Anomaly,
    AnomalyType,
    DetectionAlgorithm,
)

from .remediator import (
    AutoRemediator,
    RemediationAction,
    RemediationResult,
    RemediationStatus,
    RemediationType,
)

from .rules import (
    RuleEngine,
    Rule,
    RuleCondition,
    RuleAction,
)

from .metrics import (
    ClosedLoopMetrics,
    MTTRTracker,
    EffectivenessMetrics,
)

from .core import (
    ClosedLoopController,
    Phase2Controller,
    LoopState,
)

# Phase 2 组件
from .rca import (
    EventCollector,
    Event,
    EventType,
    EventSeverity,
    CorrelationAnalyzer,
    CorrelationResult,
    RootCauseIdentifier,
    RootCauseAnalysis,
    RootCause,
    RootCauseCategory,
    ReportGenerator,
    RCAReport,
    ReportFormat,
)

from .alert import (
    SmartAlertRouter,
    AlertAggregator,
    Alert,
    AlertSeverity,
    AlertStatus,
    NotificationChannel,
    RoutingRule,
    OnCallSchedule,
    create_default_router,
)

from .capacity import (
    ForecastEngine,
    ForecastResult,
    ForecastModel,
    CapacityPlanner,
    CapacityPlan,
    ScalingRecommendation,
    ActionType,
    ResourceType,
)

from .workflow import (
    WorkflowEngine,
    WorkflowDefinition,
    WorkflowInstance,
    WorkflowStep,
    ApprovalRequest,
    WorkflowStatus,
    ActionType as WorkflowActionType,
    ApprovalLevel,
)

# Phase 3 组件
from .learning import (
    PPOPolicyLearner,
    BayesianOptimizer,
    EffectEvaluator,
    OnlineLearner,
)

from .optimizer import (
    CostModelBuilder,
    CostOptimizer,
    RiskEngine,
    NSGAIIOptimizer,
    DecisionRecommender,
)

from .knowledge import (
    EntityExtractor,
    Entity,
    EntityType,
    RelationBuilder,
    Relationship,
    GNNReasoner,
    KnowledgeQueryInterface,
    QueryResult,
)

from .predictive import (
    FailurePredictor,
    LSTMPredictor,
    FailurePrediction,
    ImpactAnalyzer,
    ServiceImpact,
    PrepairPlanner,
    MaintenanceWindow,
    PrepairPlan,
)

from .orchestration import (
    TopologyBuilder,
    ServiceNode,
    TopologyEdge,
    ConsensusEngine,
    ConsensusDecision,
    CascadeController,
    CascadeAction,
    ServiceState,
)

from .human import (
    XAIExplainer,
    DecisionExplanation,
    FeatureImportance,
    ApprovalWorkflowEngine,
    ApprovalRequest,
    ApprovalStatus,
    ExpertKnowledgeSystem,
    ExpertRule,
)

# 治理模块 (强制治理规范)
from .governance import (
    # Decision Contract
    Decision,
    DecisionContractManager,
    AnomalyEvidence,
    RootCause,
    Risk,
    RiskLevel,
    Action,
    Rollback,
    Verify,
    Approval,
    ApprovalLevel,
    
    # Idempotent Action
    IdempotentAction,
    ActionState,
    ActionType,
    ActionResult,
    ActionFactory,
    ActionStateMachine,
    
    # Approval Gate
    ApprovalGate,
    GateResult,
    RiskAssessmentEngine,
    EmergencyBypass,
    
    # Verification Gate
    VerificationGate,
    VerificationStatus,
    VerificationConfig,
    RollbackTrigger,
    
    # Audit Trail
    AuditTrail,
    Evidence,
    EvidenceType,
    EvidenceChain,
    EvidenceCollector,
    
    # Fault Domain
    FaultDomainManager,
    FaultDomain,
    ServiceStatus,
    DegradationLevel,
    CircuitBreaker,
    CircuitBreakerState,
    KillSwitch,
    RetryPolicy,
    RetryExecutor,
)

__all__ = [
    # 版本
    '__version__',
    
    # Phase 1 - 异常检测
    'AnomalyDetector',
    'Anomaly',
    'AnomalyType',
    'DetectionAlgorithm',
    
    # Phase 1 - 自动修复
    'AutoRemediator',
    'RemediationAction',
    'RemediationResult',
    'RemediationStatus',
    'RemediationType',
    
    # Phase 1 - 规则引擎
    'RuleEngine',
    'Rule',
    'RuleCondition',
    'RuleAction',
    
    # Phase 1 - 指标
    'ClosedLoopMetrics',
    'MTTRTracker',
    'EffectivenessMetrics',
    
    # Phase 1 - 核心控制器
    'ClosedLoopController',
    'Phase2Controller',
    'LoopState',
    
    # Phase 2 - RCA
    'EventCollector',
    'Event',
    'EventType',
    'EventSeverity',
    'CorrelationAnalyzer',
    'CorrelationResult',
    'RootCauseIdentifier',
    'RootCauseAnalysis',
    'RootCause',
    'RootCauseCategory',
    'ReportGenerator',
    'RCAReport',
    'ReportFormat',
    
    # Phase 2 - 告警路由
    'SmartAlertRouter',
    'AlertAggregator',
    'Alert',
    'AlertSeverity',
    'AlertStatus',
    'NotificationChannel',
    'RoutingRule',
    'OnCallSchedule',
    'create_default_router',
    
    # Phase 2 - 容量管理
    'ForecastEngine',
    'ForecastResult',
    'ForecastModel',
    'CapacityPlanner',
    'CapacityPlan',
    'ScalingRecommendation',
    'ActionType',
    'ResourceType',
    
    # Phase 2 - 工作流
    'WorkflowEngine',
    'WorkflowDefinition',
    'WorkflowInstance',
    'WorkflowStep',
    'ApprovalRequest',
    'WorkflowStatus',
    'WorkflowActionType',
    'ApprovalLevel',
    
    # Phase 3 - 自学习优化
    'PPOPolicyLearner',
    'BayesianOptimizer',
    'EffectEvaluator',
    'OnlineLearner',
    
    # Phase 3 - 多目标决策优化
    'CostModelBuilder',
    'CostOptimizer',
    'RiskEngine',
    'NSGAIIOptimizer',
    'DecisionRecommender',
    
    # Phase 3 - 知识图谱
    'EntityExtractor',
    'Entity',
    'EntityType',
    'RelationBuilder',
    'Relationship',
    'GNNReasoner',
    'KnowledgeQueryInterface',
    'QueryResult',
    
    # Phase 3 - 预测性修复
    'FailurePredictor',
    'LSTMPredictor',
    'FailurePrediction',
    'ImpactAnalyzer',
    'ServiceImpact',
    'PrepairPlanner',
    'MaintenanceWindow',
    'PrepairPlan',
    
    # Phase 3 - 跨系统协同
    'TopologyBuilder',
    'ServiceNode',
    'TopologyEdge',
    'ConsensusEngine',
    'ConsensusDecision',
    'CascadeController',
    'CascadeAction',
    'ServiceState',
    
    # Phase 3 - 人机协作
    'XAIExplainer',
    'DecisionExplanation',
    'FeatureImportance',
    'ApprovalWorkflowEngine',
    'ApprovalRequest',
    'ApprovalStatus',
    'ExpertKnowledgeSystem',
    'ExpertRule',
    
    # 治理模块 - 决策契约
    'Decision',
    'DecisionContractManager',
    'AnomalyEvidence',
    'RootCause',
    'Risk',
    'RiskLevel',
    'Action',
    'Rollback',
    'Verify',
    'Approval',
    'ApprovalLevel',
    
    # 治理模块 - 幂等动作
    'IdempotentAction',
    'ActionState',
    'ActionType',
    'ActionResult',
    'ActionFactory',
    'ActionStateMachine',
    
    # 治理模块 - 审批门檻
    'ApprovalGate',
    'GateResult',
    'RiskAssessmentEngine',
    'EmergencyBypass',
    
    # 治理模块 - 闭环验证
    'VerificationGate',
    'VerificationStatus',
    'VerificationConfig',
    'RollbackTrigger',
    
    # 治理模块 - 证据链
    'AuditTrail',
    'Evidence',
    'EvidenceType',
    'EvidenceChain',
    'EvidenceCollector',
    
    # 治理模块 - 故障域
    'FaultDomainManager',
    'FaultDomain',
    'ServiceStatus',
    'DegradationLevel',
    'CircuitBreaker',
    'CircuitBreakerState',
    'KillSwitch',
    'RetryPolicy',
    'RetryExecutor',
]
