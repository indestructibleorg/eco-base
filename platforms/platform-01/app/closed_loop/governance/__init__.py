"""
治理模块入口

强制治理规范核心组件
- 决策契约 (Decision Contract)
- 幂等动作 (Idempotent Action)
- 审批门檻 (Approval Gate)
- 闭环验证 (Verification Gate)
- 证据链 (Audit Trail)
- 故障域与降级 (Fault Domain)
"""

from .decision_contract import (
    Decision,
    DecisionContractManager,
    Version,
    AnomalyEvidence,
    RootCause,
    Evidence,
    Risk,
    RiskFactor,
    RiskLevel,
    Action,
    RollbackAction,
    Rollback,
    Verify,
    VerifyCheck,
    Approval,
    Escalation,
    Metadata,
    ApprovalLevel,
)

from .idempotent_action import (
    IdempotentAction,
    ActionState,
    ActionType,
    ActionResult,
    VerificationResult,
    RestartServiceAction,
    ScaleAction,
    ActionFactory,
    ActionStateMachine,
)

from .approval_gate import (
    ApprovalGate,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalLevel,
    GateResult,
    RiskAssessmentEngine,
    RiskLevel,
    EmergencyBypass,
)

from .verification_gate import (
    VerificationGate,
    VerificationResult,
    VerificationStatus,
    VerificationStrategy,
    MetricCheck,
    VerificationConfig,
    RollbackTrigger,
    VerificationReporter,
)

from .audit_trail import (
    AuditTrail,
    AuditTrailStorage,
    Evidence,
    EvidenceType,
    EvidenceChain,
    InputEvidence,
    DecisionEvidence,
    ExecutionEvidence,
    VerificationEvidence,
    EvidenceCollector,
    StorageType,
)

from .fault_domain import (
    FaultDomainManager,
    FaultDomain,
    ServiceStatus,
    DegradationLevel,
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    FaultIsolator,
    GracefulDegradation,
    KillSwitch,
    RetryPolicy,
    RetryExecutor,
    KillSwitchEnabledError,
    ServiceIsolatedError,
)

__all__ = [
    # Decision Contract
    'Decision',
    'DecisionContractManager',
    'Version',
    'AnomalyEvidence',
    'RootCause',
    'Evidence',
    'Risk',
    'RiskFactor',
    'RiskLevel',
    'Action',
    'RollbackAction',
    'Rollback',
    'Verify',
    'VerifyCheck',
    'Approval',
    'Escalation',
    'Metadata',
    'ApprovalLevel',
    
    # Idempotent Action
    'IdempotentAction',
    'ActionState',
    'ActionType',
    'ActionResult',
    'VerificationResult',
    'RestartServiceAction',
    'ScaleAction',
    'ActionFactory',
    'ActionStateMachine',
    
    # Approval Gate
    'ApprovalGate',
    'ApprovalRequest',
    'ApprovalStatus',
    'GateResult',
    'RiskAssessmentEngine',
    'EmergencyBypass',
    
    # Verification Gate
    'VerificationGate',
    'VerificationStatus',
    'VerificationStrategy',
    'MetricCheck',
    'VerificationConfig',
    'RollbackTrigger',
    'VerificationReporter',
    
    # Audit Trail
    'AuditTrail',
    'AuditTrailStorage',
    'Evidence',
    'EvidenceType',
    'EvidenceChain',
    'InputEvidence',
    'DecisionEvidence',
    'ExecutionEvidence',
    'VerificationEvidence',
    'EvidenceCollector',
    'StorageType',
    
    # Fault Domain
    'FaultDomainManager',
    'FaultDomain',
    'ServiceStatus',
    'DegradationLevel',
    'CircuitBreaker',
    'CircuitBreakerState',
    'CircuitBreakerConfig',
    'CircuitBreakerOpenError',
    'FaultIsolator',
    'GracefulDegradation',
    'KillSwitch',
    'RetryPolicy',
    'RetryExecutor',
    'KillSwitchEnabledError',
    'ServiceIsolatedError',
]
