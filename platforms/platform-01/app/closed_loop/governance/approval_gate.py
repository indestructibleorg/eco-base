"""
强制审批门檻系统 (Approval Gate)

强制治理规范核心组件
高风险操作必须审批，未审批自动拦截
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """审批状态"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ESCALATED = "escalated"


class ApprovalLevel(Enum):
    """审批级别"""
    L1 = "L1"  # On-call
    L2 = "L2"  # SRE Lead
    L3 = "L3"  # VP


class RiskLevel(Enum):
    """风险等级"""
    CRITICAL = "critical"    # 0.8-1.0
    HIGH = "high"            # 0.6-0.8
    MEDIUM = "medium"        # 0.3-0.6
    LOW = "low"              # 0-0.3


@dataclass
class ApprovalRequest:
    """审批请求"""
    request_id: str
    decision_id: str
    level: ApprovalLevel
    approvers: List[str]
    requested_at: datetime
    timeout_seconds: int
    escalation: Optional[Dict] = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """检查是否已过期"""
        elapsed = (datetime.now() - self.requested_at).total_seconds()
        return elapsed > self.timeout_seconds
    
    def time_remaining(self) -> int:
        """获取剩余时间（秒）"""
        elapsed = (datetime.now() - self.requested_at).total_seconds()
        remaining = self.timeout_seconds - int(elapsed)
        return max(0, remaining)


@dataclass
class GateResult:
    """门檻结果"""
    allowed: bool
    reason: str
    approval_request: Optional[ApprovalRequest] = None
    auto_execute: bool = False
    risk_level: Optional[RiskLevel] = None
    risk_score: float = 0.0


class RiskAssessmentEngine:
    """风险评估引擎"""
    
    # 风险分级配置
    RISK_LEVELS = {
        RiskLevel.CRITICAL: {"min": 0.8, "max": 1.0, "requires_approval": True, "level": ApprovalLevel.L3},
        RiskLevel.HIGH: {"min": 0.6, "max": 0.8, "requires_approval": True, "level": ApprovalLevel.L2},
        RiskLevel.MEDIUM: {"min": 0.3, "max": 0.6, "requires_approval": True, "level": ApprovalLevel.L1},
        RiskLevel.LOW: {"min": 0.0, "max": 0.3, "requires_approval": False, "level": None},
    }
    
    # 风险因子权重
    RISK_FACTORS = {
        "blast_radius": 0.3,      # 影响范围
        "failure_probability": 0.25,  # 失败概率
        "recovery_difficulty": 0.2,   # 恢复难度
        "data_impact": 0.15,      # 数据影响
        "business_impact": 0.1,   # 业务影响
    }
    
    def calculate_risk(
        self,
        affected_services: List[str],
        action_type: str,
        blast_radius: int = 1,
        failure_probability: float = 0.1,
        recovery_difficulty: float = 0.5,
        data_impact: float = 0.0,
        business_impact: float = 0.0
    ) -> Dict[str, Any]:
        """
        计算风险分数
        
        Args:
            affected_services: 受影响服务列表
            action_type: 动作类型
            blast_radius: 影响范围 (服务数量)
            failure_probability: 失败概率 (0-1)
            recovery_difficulty: 恢复难度 (0-1)
            data_impact: 数据影响 (0-1)
            business_impact: 业务影响 (0-1)
        
        Returns:
            风险评估结果
        """
        # 标准化影响范围
        normalized_blast = min(blast_radius / 10, 1.0)
        
        # 计算加权风险分数
        score = (
            normalized_blast * self.RISK_FACTORS["blast_radius"] +
            failure_probability * self.RISK_FACTORS["failure_probability"] +
            recovery_difficulty * self.RISK_FACTORS["recovery_difficulty"] +
            data_impact * self.RISK_FACTORS["data_impact"] +
            business_impact * self.RISK_FACTORS["business_impact"]
        )
        
        # 确定风险等级
        risk_level = self._get_risk_level(score)
        
        # 生成风险因子详情
        factors = [
            {"name": "blast_radius", "value": blast_radius, "weight": self.RISK_FACTORS["blast_radius"]},
            {"name": "failure_probability", "value": failure_probability, "weight": self.RISK_FACTORS["failure_probability"]},
            {"name": "recovery_difficulty", "value": recovery_difficulty, "weight": self.RISK_FACTORS["recovery_difficulty"]},
            {"name": "data_impact", "value": data_impact, "weight": self.RISK_FACTORS["data_impact"]},
            {"name": "business_impact", "value": business_impact, "weight": self.RISK_FACTORS["business_impact"]},
        ]
        
        return {
            "score": round(score, 2),
            "level": risk_level.value,
            "requires_approval": self.RISK_LEVELS[risk_level]["requires_approval"],
            "approval_level": self.RISK_LEVELS[risk_level]["level"].value if self.RISK_LEVELS[risk_level]["level"] else None,
            "factors": factors
        }
    
    def _get_risk_level(self, score: float) -> RiskLevel:
        """根据分数确定风险等级"""
        for level, config in self.RISK_LEVELS.items():
            if config["min"] <= score < config["max"]:
                return level
        return RiskLevel.LOW
    
    def get_high_risk_actions(self) -> List[str]:
        """获取高风险动作类型列表"""
        return [
            "database_migration",
            "config_change_global",
            "service_shutdown",
            "network_change",
            "security_policy_change"
        ]


class ApprovalGate:
    """
    强制审批门檻
    
    核心规则:
    - 高风险操作必须审批
    - 未审批自动拦截
    - 审批超时自动升级
    - 审批记录永久保存
    """
    
    def __init__(self, storage: Optional[Any] = None):
        self.risk_engine = RiskAssessmentEngine()
        self.storage = storage
        
        # 审批请求存储
        self._approval_requests: Dict[str, ApprovalRequest] = {}
        
        # 回调函数
        self._approval_callbacks: List[Callable[[ApprovalRequest], None]] = []
        self._escalation_callbacks: List[Callable[[ApprovalRequest], None]] = []
        
        # 启动后台任务
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动审批门檻"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Approval gate started")
    
    async def stop(self):
        """停止审批门檻"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Approval gate stopped")
    
    async def evaluate(self, decision: Dict[str, Any]) -> GateResult:
        """
        评估决策是否需要审批
        
        Args:
            decision: 决策字典
        
        Returns:
            GateResult: 门檻结果
        """
        decision_id = decision.get("decision_id", "unknown")
        risk = decision.get("risk", {})
        risk_score = risk.get("score", 0.0)
        risk_level_str = risk.get("level", "low")
        
        try:
            risk_level = RiskLevel(risk_level_str)
        except ValueError:
            risk_level = RiskLevel.LOW
        
        # 获取风险配置
        risk_config = self.risk_engine.RISK_LEVELS.get(risk_level)
        
        if not risk_config:
            logger.error(f"Unknown risk level: {risk_level}")
            return GateResult(
                allowed=False,
                reason="Invalid risk configuration",
                risk_level=risk_level,
                risk_score=risk_score
            )
        
        # 检查是否需要审批
        if not risk_config["requires_approval"]:
            logger.info(f"Decision {decision_id} - low risk, auto-approved")
            return GateResult(
                allowed=True,
                reason="Low risk, auto-approved",
                auto_execute=True,
                risk_level=risk_level,
                risk_score=risk_score
            )
        
        # 需要审批
        approval_level = risk_config["level"]
        approvers = self._get_approvers(approval_level)
        
        # 创建审批请求
        request = ApprovalRequest(
            request_id=f"apr_{datetime.now().strftime('%Y%m%d%H%M%S')}_{decision_id[-8:]}",
            decision_id=decision_id,
            level=approval_level,
            approvers=approvers,
            requested_at=datetime.now(),
            timeout_seconds=1800,  # 30分钟
            escalation={
                "enabled": True,
                "after_seconds": 900,  # 15分钟后升级
                "escalate_to": self._get_escalation_target(approval_level)
            },
            metadata={
                "risk_score": risk_score,
                "risk_level": risk_level.value,
                "affected_services": decision.get("evidence", {}).get("anomaly", {}).get("affected_services", [])
            }
        )
        
        # 存储审批请求
        self._approval_requests[request.request_id] = request
        
        # 发送审批通知
        await self._send_approval_notification(request)
        
        logger.info(
            f"Decision {decision_id} requires approval - "
            f"level={approval_level.value}, approvers={approvers}"
        )
        
        return GateResult(
            allowed=False,
            reason=f"Requires {approval_level.value} approval",
            approval_request=request,
            auto_execute=False,
            risk_level=risk_level,
            risk_score=risk_score
        )
    
    async def approve(
        self,
        request_id: str,
        approver: str,
        comment: Optional[str] = None
    ) -> bool:
        """
        批准请求
        
        Args:
            request_id: 审批请求ID
            approver: 审批人
            comment: 审批意见
        
        Returns:
            是否成功
        """
        request = self._approval_requests.get(request_id)
        if not request:
            logger.error(f"Approval request not found: {request_id}")
            return False
        
        if request.status != ApprovalStatus.PENDING:
            logger.warning(f"Approval request {request_id} is not pending")
            return False
        
        if approver not in request.approvers:
            logger.warning(f"Approver {approver} not in allowed list: {request.approvers}")
            return False
        
        # 更新状态
        request.status = ApprovalStatus.APPROVED
        request.approved_by = approver
        request.approved_at = datetime.now()
        request.metadata["approval_comment"] = comment
        
        logger.info(f"Approval request {request_id} approved by {approver}")
        
        # 触发回调
        for callback in self._approval_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(request)
                else:
                    callback(request)
            except Exception as e:
                logger.warning(f"Approval callback failed: {e}")
        
        return True
    
    async def reject(
        self,
        request_id: str,
        approver: str,
        reason: str
    ) -> bool:
        """
        拒绝请求
        
        Args:
            request_id: 审批请求ID
            approver: 审批人
            reason: 拒绝原因
        
        Returns:
            是否成功
        """
        request = self._approval_requests.get(request_id)
        if not request:
            logger.error(f"Approval request not found: {request_id}")
            return False
        
        if request.status != ApprovalStatus.PENDING:
            logger.warning(f"Approval request {request_id} is not pending")
            return False
        
        # 更新状态
        request.status = ApprovalStatus.REJECTED
        request.approved_by = approver
        request.approved_at = datetime.now()
        request.rejection_reason = reason
        
        logger.info(f"Approval request {request_id} rejected by {approver}: {reason}")
        
        return True
    
    def get_approval_status(self, request_id: str) -> Optional[ApprovalStatus]:
        """获取审批状态"""
        request = self._approval_requests.get(request_id)
        if request:
            # 检查是否过期
            if request.status == ApprovalStatus.PENDING and request.is_expired():
                request.status = ApprovalStatus.EXPIRED
            return request.status
        return None
    
    def get_approval_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """获取审批请求"""
        return self._approval_requests.get(request_id)
    
    def list_pending_requests(
        self,
        approver: Optional[str] = None
    ) -> List[ApprovalRequest]:
        """列出待审批请求"""
        requests = []
        for req in self._approval_requests.values():
            if req.status == ApprovalStatus.PENDING:
                if approver is None or approver in req.approvers:
                    requests.append(req)
        return requests
    
    def register_approval_callback(self, callback: Callable):
        """注册审批回调"""
        self._approval_callbacks.append(callback)
    
    def register_escalation_callback(self, callback: Callable):
        """注册升级回调"""
        self._escalation_callbacks.append(callback)
    
    def _get_approvers(self, level: ApprovalLevel) -> List[str]:
        """获取审批人列表"""
        approvers_map = {
            ApprovalLevel.L1: ["oncall_db"],
            ApprovalLevel.L2: ["sre_lead", "oncall_db"],
            ApprovalLevel.L3: ["vp_engineering", "sre_lead"]
        }
        return approvers_map.get(level, ["oncall_db"])
    
    def _get_escalation_target(self, level: ApprovalLevel) -> str:
        """获取升级目标"""
        escalation_map = {
            ApprovalLevel.L1: "sre_lead",
            ApprovalLevel.L2: "vp_engineering",
            ApprovalLevel.L3: "cto"
        }
        return escalation_map.get(level, "sre_lead")
    
    async def _send_approval_notification(self, request: ApprovalRequest):
        """发送审批通知"""
        # 模拟发送通知
        logger.info(
            f"Sending approval notification - "
            f"request={request.request_id}, "
            f"approvers={request.approvers}, "
            f"timeout={request.timeout_seconds}s"
        )
    
    async def _cleanup_loop(self):
        """清理过期请求的循环"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次
                
                expired_count = 0
                for request in list(self._approval_requests.values()):
                    if request.status == ApprovalStatus.PENDING and request.is_expired():
                        request.status = ApprovalStatus.EXPIRED
                        expired_count += 1
                        
                        logger.info(f"Approval request {request.request_id} expired")
                
                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} expired approval requests")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Approval cleanup error: {e}")


class EmergencyBypass:
    """
    紧急绕过机制
    
    紧急情况下可强制绕过审批
    需事后24小时内补审批
    """
    
    def __init__(self):
        self._bypass_records: List[Dict] = []
    
    async def bypass(
        self,
        decision_id: str,
        reason: str,
        operator: str
    ) -> Dict[str, Any]:
        """
        紧急绕过审批
        
        Args:
            decision_id: 决策ID
            reason: 绕过原因
            operator: 操作人
        
        Returns:
            绕过记录
        """
        record = {
            "bypass_id": f"bypass_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "decision_id": decision_id,
            "reason": reason,
            "operator": operator,
            "timestamp": datetime.now().isoformat(),
            "requires_followup": True,
            "followup_deadline": (datetime.now() + timedelta(hours=24)).isoformat(),
            "followup_completed": False
        }
        
        self._bypass_records.append(record)
        
        logger.warning(
            f"EMERGENCY BYPASS - decision={decision_id}, "
            f"operator={operator}, reason={reason}"
        )
        
        return record
    
    def get_pending_followups(self) -> List[Dict]:
        """获取待跟进的绕过记录"""
        return [
            r for r in self._bypass_records
            if r["requires_followup"] and not r["followup_completed"]
        ]
    
    def complete_followup(self, bypass_id: str) -> bool:
        """完成跟进"""
        for record in self._bypass_records:
            if record["bypass_id"] == bypass_id:
                record["followup_completed"] = True
                record["followup_completed_at"] = datetime.now().isoformat()
                return True
        return False
