"""
决策契约系统 (Decision Contract)

强制治理规范核心组件
每个决策必须输出标准化的 decision.json
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any
import uuid


class RiskLevel(Enum):
    """风险等级"""
    CRITICAL = "critical"    # 0.8-1.0
    HIGH = "high"            # 0.6-0.8
    MEDIUM = "medium"        # 0.3-0.6
    LOW = "low"              # 0-0.3


class ApprovalLevel(Enum):
    """审批级别"""
    L1 = "L1"  # On-call
    L2 = "L2"  # SRE Lead
    L3 = "L3"  # VP


class ActionState(Enum):
    """动作状态"""
    PENDING = "pending"
    APPROVED = "approved"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    VERIFIED = "verified"
    ROLLBACK_INITIATED = "rollback_initiated"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


@dataclass
class Version:
    """版本信息"""
    schema: str = "1.0.0"
    model: str = ""
    topology: str = ""
    rules: str = ""


@dataclass
class AnomalyEvidence:
    """异常证据"""
    anomaly_id: str
    anomaly_type: str
    severity: str
    confidence: float
    affected_services: List[str]


@dataclass
class RootCause:
    """根因"""
    cause: str
    confidence: float
    evidence: List[str]


@dataclass
class Evidence:
    """证据"""
    anomaly: AnomalyEvidence
    root_causes: List[RootCause]
    input_snapshot_hash: str


@dataclass
class RiskFactor:
    """风险因子"""
    name: str
    impact: str
    weight: float


@dataclass
class Risk:
    """风险评估"""
    level: str
    score: float
    factors: List[RiskFactor]
    requires_approval: bool


@dataclass
class Action:
    """执行动作"""
    action_id: str
    action_type: str
    target: str
    params: Dict[str, Any]
    estimated_duration: int
    order: int


@dataclass
class RollbackAction:
    """回滚动作"""
    action_id: str
    action_type: str
    target: str
    backup_ref: str


@dataclass
class AutoTrigger:
    """自动触发条件"""
    enabled: bool
    conditions: List[Dict[str, str]]


@dataclass
class Rollback:
    """回滚方案"""
    enabled: bool
    actions: List[RollbackAction]
    auto_trigger: AutoTrigger


@dataclass
class VerifyCheck:
    """验证检查项"""
    metric: str
    expected: str
    duration: str


@dataclass
class Verify:
    """验证方案"""
    enabled: bool
    checks: List[VerifyCheck]
    timeout: int


@dataclass
class Escalation:
    """升级配置"""
    enabled: bool
    after: str
    to: str


@dataclass
class Approval:
    """审批配置"""
    required: bool
    level: str
    approvers: List[str]
    timeout: int
    escalation: Escalation


@dataclass
class Metadata:
    """元数据"""
    source: str
    triggered_by: str
    correlation_id: str


@dataclass
class Decision:
    """
    决策契约 - 标准化的决策输出
    """
    decision_id: str
    trace_id: str
    timestamp: str
    version: Version
    evidence: Evidence
    risk: Risk
    actions: List[Action]
    rollback: Rollback
    verify: Verify
    approval: Approval
    metadata: Metadata
    
    @classmethod
    def create(
        cls,
        trace_id: str,
        anomaly: AnomalyEvidence,
        root_causes: List[RootCause],
        input_hash: str,
        actions: List[Action],
        risk_score: float,
        version: Optional[Version] = None
    ) -> "Decision":
        """创建决策"""
        decision_id = f"dec_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 确定风险等级
        risk_level = cls._calculate_risk_level(risk_score)
        requires_approval = risk_score >= 0.3
        
        # 确定审批级别
        approval_level = cls._calculate_approval_level(risk_score)
        
        return cls(
            decision_id=decision_id,
            trace_id=trace_id,
            timestamp=datetime.now().isoformat(),
            version=version or Version(),
            evidence=Evidence(
                anomaly=anomaly,
                root_causes=root_causes,
                input_snapshot_hash=input_hash
            ),
            risk=Risk(
                level=risk_level.value,
                score=risk_score,
                factors=[],  # 由调用方填充
                requires_approval=requires_approval
            ),
            actions=actions,
            rollback=Rollback(
                enabled=True,
                actions=[],  # 由调用方填充
                auto_trigger=AutoTrigger(
                    enabled=True,
                    conditions=[
                        {"metric": "error_rate", "threshold": "> 0.05", "duration": "5m"}
                    ]
                )
            ),
            verify=Verify(
                enabled=True,
                checks=[
                    VerifyCheck(metric="cpu_usage", expected="< 70%", duration="3m"),
                    VerifyCheck(metric="error_rate", expected="< 0.01", duration="5m")
                ],
                timeout=600
            ),
            approval=Approval(
                required=requires_approval,
                level=approval_level.value,
                approvers=cls._get_approvers(approval_level),
                timeout=1800,
                escalation=Escalation(
                    enabled=True,
                    after="15m",
                    to="vp_engineering"
                )
            ),
            metadata=Metadata(
                source="closed_loop_controller",
                triggered_by="anomaly_detector",
                correlation_id=trace_id
            )
        )
    
    @staticmethod
    def _calculate_risk_level(score: float) -> RiskLevel:
        """计算风险等级"""
        if score >= 0.8:
            return RiskLevel.CRITICAL
        elif score >= 0.6:
            return RiskLevel.HIGH
        elif score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    @staticmethod
    def _calculate_approval_level(score: float) -> ApprovalLevel:
        """计算审批级别"""
        if score >= 0.8:
            return ApprovalLevel.L3
        elif score >= 0.6:
            return ApprovalLevel.L2
        else:
            return ApprovalLevel.L1
    
    @staticmethod
    def _get_approvers(level: ApprovalLevel) -> List[str]:
        """获取审批人列表"""
        approvers_map = {
            ApprovalLevel.L1: ["oncall_db"],
            ApprovalLevel.L2: ["oncall_db", "sre_lead"],
            ApprovalLevel.L3: ["oncall_db", "sre_lead", "vp_engineering"]
        }
        return approvers_map.get(level, ["oncall_db"])
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def compute_hash(self) -> str:
        """计算决策哈希"""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def validate(self) -> List[str]:
        """
        验证决策契约完整性
        返回缺失的必填字段列表
        """
        errors = []
        
        # 必填字段检查
        if not self.decision_id:
            errors.append("decision_id is required")
        if not self.trace_id:
            errors.append("trace_id is required")
        if not self.timestamp:
            errors.append("timestamp is required")
        if not self.version:
            errors.append("version is required")
        if not self.evidence:
            errors.append("evidence is required")
        if not self.risk:
            errors.append("risk is required")
        if not self.actions:
            errors.append("actions is required")
        if not self.rollback:
            errors.append("rollback is required")
        if not self.verify:
            errors.append("verify is required")
        
        # 高风险必须审批
        if self.risk.score >= 0.3 and not self.approval.required:
            errors.append("high risk decision must require approval")
        
        return errors
    
    def is_valid(self) -> bool:
        """检查决策是否有效"""
        return len(self.validate()) == 0


class DecisionContractManager:
    """
    决策契约管理器
    负责决策的创建、验证、存储和查询
    """
    
    def __init__(self, storage: Optional[Any] = None):
        self.storage = storage
        self._decisions: Dict[str, Decision] = {}
    
    def create_decision(
        self,
        trace_id: str,
        anomaly: AnomalyEvidence,
        root_causes: List[RootCause],
        input_hash: str,
        actions: List[Action],
        risk_score: float,
        version: Optional[Version] = None
    ) -> Decision:
        """创建决策"""
        decision = Decision.create(
            trace_id=trace_id,
            anomaly=anomaly,
            root_causes=root_causes,
            input_hash=input_hash,
            actions=actions,
            risk_score=risk_score,
            version=version
        )
        
        # 验证
        errors = decision.validate()
        if errors:
            raise ValueError(f"Invalid decision: {errors}")
        
        # 存储
        self._decisions[decision.decision_id] = decision
        
        return decision
    
    def get_decision(self, decision_id: str) -> Optional[Decision]:
        """获取决策"""
        return self._decisions.get(decision_id)
    
    def get_decisions_by_trace(self, trace_id: str) -> List[Decision]:
        """根据trace_id获取决策列表"""
        return [d for d in self._decisions.values() if d.trace_id == trace_id]
    
    def list_decisions(
        self,
        risk_level: Optional[RiskLevel] = None,
        limit: int = 100
    ) -> List[Decision]:
        """列出决策"""
        decisions = list(self._decisions.values())
        
        if risk_level:
            decisions = [d for d in decisions if d.risk.level == risk_level.value]
        
        # 按时间倒序
        decisions.sort(key=lambda d: d.timestamp, reverse=True)
        
        return decisions[:limit]
