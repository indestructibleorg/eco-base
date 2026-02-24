"""
人机协作界面模块

提供以下功能:
- 可解释 AI (XAI)
- 交互式审批工作流
- 专家知识集成
"""

from .xai_explainer import (
    XAIExplainer,
    SHAPExplainer,
    LIMEExplainer,
    DecisionPathExplainer,
    CounterfactualExplainer,
    DecisionExplanation,
    FeatureImportance
)

from .approval_workflow import (
    ApprovalWorkflowEngine,
    ApprovalRequest,
    ApprovalRecord,
    ApprovalStep,
    ApprovalLevel,
    ApprovalStatus,
    ApprovalUrgency
)

from .expert_knowledge import (
    ExpertKnowledgeSystem,
    ExpertRule,
    CaseStudy,
    FeedbackRecord,
    RuleInjector,
    CaseLibrary,
    FeedbackLearner
)

__all__ = [
    # XAI Explainer
    'XAIExplainer',
    'SHAPExplainer',
    'LIMEExplainer',
    'DecisionPathExplainer',
    'CounterfactualExplainer',
    'DecisionExplanation',
    'FeatureImportance',
    
    # Approval Workflow
    'ApprovalWorkflowEngine',
    'ApprovalRequest',
    'ApprovalRecord',
    'ApprovalStep',
    'ApprovalLevel',
    'ApprovalStatus',
    'ApprovalUrgency',
    
    # Expert Knowledge
    'ExpertKnowledgeSystem',
    'ExpertRule',
    'CaseStudy',
    'FeedbackRecord',
    'RuleInjector',
    'CaseLibrary',
    'FeedbackLearner'
]
