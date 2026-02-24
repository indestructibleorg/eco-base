# =============================================================================
# Approval Service
# =============================================================================
# 审批服务 - 实现人工审批流程
# 支持审批请求创建、状态跟踪、超时处理
# =============================================================================

import asyncio
import hashlib
import json
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from uuid import uuid4

from app.core.logging import get_logger

logger = get_logger("approval_service")


class ApprovalStatus(Enum):
    """审批状态"""
    PENDING = "pending"           # 待审批
    APPROVED = "approved"         # 已批准
    REJECTED = "rejected"         # 已拒绝
    EXPIRED = "expired"           # 已过期
    CANCELLED = "cancelled"       # 已取消


class ApprovalPriority(Enum):
    """审批优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ApprovalRequest:
    """审批请求"""
    request_id: str
    rule_name: str
    action_type: str
    action_parameters: Dict[str, Any]
    
    # 风险信息
    risk_level: str
    risk_score: float
    risk_reasons: List[str] = field(default_factory=list)
    
    # 上下文信息
    context: Dict[str, Any] = field(default_factory=dict)
    facts_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # 审批配置
    priority: ApprovalPriority = ApprovalPriority.MEDIUM
    timeout_minutes: int = 30
    approvers: List[str] = field(default_factory=list)  # 指定审批人列表
    
    # 状态
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    approval_comment: Optional[str] = None
    rejected_at: Optional[datetime] = None
    rejected_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    
    # 元数据
    decision_hash: str = ""  # 用于验证审批内容与决策一致
    
    def __post_init__(self):
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(minutes=self.timeout_minutes)
    
    def compute_hash(self) -> str:
        """计算决策哈希"""
        content = json.dumps({
            "rule_name": self.rule_name,
            "action_type": self.action_type,
            "action_parameters": self.action_parameters,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data["status"] = self.status.value
        data["priority"] = self.priority.value
        data["priority_label"] = self.priority.name
        return data
    
    def is_expired(self) -> bool:
        """检查是否已过期"""
        return datetime.utcnow() > self.expires_at
    
    def time_remaining(self) -> timedelta:
        """获取剩余时间"""
        remaining = self.expires_at - datetime.utcnow()
        return max(remaining, timedelta(0))


@dataclass
class ApprovalResult:
    """审批结果"""
    request_id: str
    approved: bool
    status: ApprovalStatus
    message: str
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None


class ApprovalStore:
    """审批存储基类"""
    
    async def save(self, request: ApprovalRequest) -> bool:
        raise NotImplementedError
    
    async def load(self, request_id: str) -> Optional[ApprovalRequest]:
        raise NotImplementedError
    
    async def list_pending(self) -> List[ApprovalRequest]:
        raise NotImplementedError
    
    async def list_by_rule(self, rule_name: str, limit: int = 100) -> List[ApprovalRequest]:
        raise NotImplementedError


class InMemoryApprovalStore(ApprovalStore):
    """内存审批存储"""
    
    def __init__(self):
        self._requests: Dict[str, ApprovalRequest] = {}
    
    async def save(self, request: ApprovalRequest) -> bool:
        self._requests[request.request_id] = request
        return True
    
    async def load(self, request_id: str) -> Optional[ApprovalRequest]:
        return self._requests.get(request_id)
    
    async def list_pending(self) -> List[ApprovalRequest]:
        return [
            req for req in self._requests.values()
            if req.status == ApprovalStatus.PENDING
        ]
    
    async def list_by_rule(self, rule_name: str, limit: int = 100) -> List[ApprovalRequest]:
        requests = [
            req for req in self._requests.values()
            if req.rule_name == rule_name
        ]
        return sorted(requests, key=lambda r: r.created_at, reverse=True)[:limit]


class ApprovalService:
    """
    审批服务
    
    管理审批流程：创建请求、处理审批、超时检查
    """
    
    def __init__(self, store: Optional[ApprovalStore] = None):
        self.store = store or InMemoryApprovalStore()
        self._notification_handlers: List[Callable[[ApprovalRequest], Awaitable[None]]] = []
        self._expiry_check_task: Optional[asyncio.Task] = None
        self._running = False
    
    def register_notification_handler(self, handler: Callable[[ApprovalRequest], Awaitable[None]]) -> None:
        """注册审批通知处理器"""
        self._notification_handlers.append(handler)
    
    async def start(self):
        """启动审批服务"""
        self._running = True
        self._expiry_check_task = asyncio.create_task(self._expiry_check_loop())
        logger.info("approval_service_started")
    
    async def stop(self):
        """停止审批服务"""
        self._running = False
        if self._expiry_check_task:
            self._expiry_check_task.cancel()
            try:
                await self._expiry_check_task
            except asyncio.CancelledError:
                pass
        logger.info("approval_service_stopped")
    
    async def create_request(
        self,
        rule_name: str,
        action_type: str,
        action_parameters: Dict[str, Any],
        risk_level: str,
        risk_score: float,
        context: Optional[Dict[str, Any]] = None,
        facts_snapshot: Optional[Dict[str, Any]] = None,
        priority: ApprovalPriority = ApprovalPriority.MEDIUM,
        timeout_minutes: int = 30,
        approvers: Optional[List[str]] = None,
    ) -> ApprovalRequest:
        """
        创建审批请求
        
        Args:
            rule_name: 规则名称
            action_type: 动作类型
            action_parameters: 动作参数
            risk_level: 风险等级
            risk_score: 风险分数
            context: 上下文信息
            facts_snapshot: 事实快照
            priority: 优先级
            timeout_minutes: 超时时间（分钟）
            approvers: 指定审批人列表
            
        Returns:
            创建的审批请求
        """
        request = ApprovalRequest(
            request_id=f"apr_{uuid4().hex[:16]}",
            rule_name=rule_name,
            action_type=action_type,
            action_parameters=action_parameters,
            risk_level=risk_level,
            risk_score=risk_score,
            risk_reasons=self._extract_risk_reasons(risk_level, risk_score, context),
            context=context or {},
            facts_snapshot=facts_snapshot or {},
            priority=priority,
            timeout_minutes=timeout_minutes,
            approvers=approvers or [],
        )
        
        # 计算决策哈希
        request.decision_hash = request.compute_hash()
        
        # 保存请求
        await self.store.save(request)
        
        logger.info(
            "approval_request_created",
            request_id=request.request_id,
            rule_name=rule_name,
            action_type=action_type,
            risk_level=risk_level,
            risk_score=risk_score,
        )
        
        # 发送通知
        await self._notify(request)
        
        return request
    
    async def approve(
        self,
        request_id: str,
        approved_by: str,
        comment: Optional[str] = None,
    ) -> ApprovalResult:
        """
        批准请求
        
        Args:
            request_id: 请求ID
            approved_by: 审批人
            comment: 审批意见
            
        Returns:
            审批结果
        """
        request = await self.store.load(request_id)
        
        if not request:
            return ApprovalResult(
                request_id=request_id,
                approved=False,
                status=ApprovalStatus.CANCELLED,
                message="Request not found",
            )
        
        if request.status != ApprovalStatus.PENDING:
            return ApprovalResult(
                request_id=request_id,
                approved=False,
                status=request.status,
                message=f"Request is already {request.status.value}",
            )
        
        if request.is_expired():
            request.status = ApprovalStatus.EXPIRED
            await self.store.save(request)
            return ApprovalResult(
                request_id=request_id,
                approved=False,
                status=ApprovalStatus.EXPIRED,
                message="Request has expired",
            )
        
        # 执行批准
        request.status = ApprovalStatus.APPROVED
        request.approved_by = approved_by
        request.approved_at = datetime.utcnow()
        request.approval_comment = comment
        
        await self.store.save(request)
        
        logger.info(
            "approval_request_approved",
            request_id=request_id,
            approved_by=approved_by,
            rule_name=request.rule_name,
        )
        
        return ApprovalResult(
            request_id=request_id,
            approved=True,
            status=ApprovalStatus.APPROVED,
            message="Request approved",
            approved_by=approved_by,
            approved_at=request.approved_at,
        )
    
    async def reject(
        self,
        request_id: str,
        rejected_by: str,
        reason: str,
    ) -> ApprovalResult:
        """
        拒绝请求
        
        Args:
            request_id: 请求ID
            rejected_by: 拒绝人
            reason: 拒绝原因
            
        Returns:
            审批结果
        """
        request = await self.store.load(request_id)
        
        if not request:
            return ApprovalResult(
                request_id=request_id,
                approved=False,
                status=ApprovalStatus.CANCELLED,
                message="Request not found",
            )
        
        if request.status != ApprovalStatus.PENDING:
            return ApprovalResult(
                request_id=request_id,
                approved=False,
                status=request.status,
                message=f"Request is already {request.status.value}",
            )
        
        # 执行拒绝
        request.status = ApprovalStatus.REJECTED
        request.rejected_by = rejected_by
        request.rejected_at = datetime.utcnow()
        request.rejection_reason = reason
        
        await self.store.save(request)
        
        logger.info(
            "approval_request_rejected",
            request_id=request_id,
            rejected_by=rejected_by,
            reason=reason,
            rule_name=request.rule_name,
        )
        
        return ApprovalResult(
            request_id=request_id,
            approved=False,
            status=ApprovalStatus.REJECTED,
            message="Request rejected",
            rejection_reason=reason,
        )
    
    async def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """获取审批请求"""
        return await self.store.load(request_id)
    
    async def check_status(self, request_id: str) -> Optional[ApprovalStatus]:
        """检查审批状态"""
        request = await self.store.load(request_id)
        return request.status if request else None
    
    async def wait_for_approval(
        self,
        request_id: str,
        timeout_seconds: Optional[float] = None,
        check_interval: float = 1.0,
    ) -> ApprovalResult:
        """
        等待审批结果
        
        Args:
            request_id: 请求ID
            timeout_seconds: 超时时间（秒）
            check_interval: 检查间隔（秒）
            
        Returns:
            审批结果
        """
        start_time = datetime.utcnow()
        
        while True:
            request = await self.store.load(request_id)
            
            if not request:
                return ApprovalResult(
                    request_id=request_id,
                    approved=False,
                    status=ApprovalStatus.CANCELLED,
                    message="Request not found",
                )
            
            if request.status != ApprovalStatus.PENDING:
                return ApprovalResult(
                    request_id=request_id,
                    approved=request.status == ApprovalStatus.APPROVED,
                    status=request.status,
                    message=f"Request {request.status.value}",
                    approved_by=request.approved_by,
                    approved_at=request.approved_at,
                    rejection_reason=request.rejection_reason,
                )
            
            if request.is_expired():
                request.status = ApprovalStatus.EXPIRED
                await self.store.save(request)
                return ApprovalResult(
                    request_id=request_id,
                    approved=False,
                    status=ApprovalStatus.EXPIRED,
                    message="Request expired",
                )
            
            if timeout_seconds:
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                if elapsed >= timeout_seconds:
                    return ApprovalResult(
                        request_id=request_id,
                        approved=False,
                        status=ApprovalStatus.PENDING,
                        message="Wait timeout",
                    )
            
            await asyncio.sleep(check_interval)
    
    async def _notify(self, request: ApprovalRequest) -> None:
        """发送审批通知"""
        for handler in self._notification_handlers:
            try:
                await handler(request)
            except Exception as e:
                logger.error("notification_handler_failed", error=str(e))
    
    async def _expiry_check_loop(self) -> None:
        """过期检查循环"""
        while self._running:
            try:
                pending = await self.store.list_pending()
                for request in pending:
                    if request.is_expired():
                        request.status = ApprovalStatus.EXPIRED
                        await self.store.save(request)
                        logger.info(
                            "approval_request_expired",
                            request_id=request.request_id,
                            rule_name=request.rule_name,
                        )
                await asyncio.sleep(60)  # 每分钟检查一次
            except Exception as e:
                logger.error("expiry_check_failed", error=str(e))
                await asyncio.sleep(60)
    
    def _extract_risk_reasons(
        self,
        risk_level: str,
        risk_score: float,
        context: Optional[Dict[str, Any]],
    ) -> List[str]:
        """提取风险原因"""
        reasons = []
        
        if risk_level in ["HIGH", "CRITICAL"]:
            reasons.append(f"高风险操作 (等级: {risk_level})")
        
        if risk_score > 0.8:
            reasons.append(f"风险评分过高 ({risk_score:.2f})")
        
        affected_services = context.get("affected_services", []) if context else []
        if len(affected_services) > 3:
            reasons.append(f"影响服务过多 ({len(affected_services)} 个)")
        
        return reasons


# 全局审批服务实例
approval_service = ApprovalService()


__all__ = [
    "ApprovalStatus",
    "ApprovalPriority",
    "ApprovalRequest",
    "ApprovalResult",
    "ApprovalStore",
    "InMemoryApprovalStore",
    "ApprovalService",
    "approval_service",
]
