"""
审批工作流系统
多级审批、紧急通道、批量审批
"""

import time
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ApprovalLevel(Enum):
    """审批级别"""
    AUTO = "auto"  # 自动审批
    L1 = "level_1"  # 一级审批
    L2 = "level_2"  # 二级审批
    L3 = "level_3"  # 三级审批 (最高)


class ApprovalStatus(Enum):
    """审批状态"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalUrgency(Enum):
    """审批紧急程度"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ApprovalStep:
    """审批步骤"""
    step_id: str
    level: ApprovalLevel
    approvers: List[str]
    required_approvals: int
    timeout_minutes: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_id': self.step_id,
            'level': self.level.value,
            'approvers': self.approvers,
            'required_approvals': self.required_approvals,
            'timeout_minutes': self.timeout_minutes
        }


@dataclass
class ApprovalRecord:
    """审批记录"""
    record_id: str
    approver: str
    decision: str  # approved, rejected
    comment: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'record_id': self.record_id,
            'approver': self.approver,
            'decision': self.decision,
            'comment': self.comment,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ApprovalRequest:
    """审批请求"""
    request_id: str
    action_type: str
    target: str
    parameters: Dict[str, Any]
    requester: str
    urgency: ApprovalUrgency
    approval_level: ApprovalLevel
    status: ApprovalStatus
    current_step: int
    steps: List[ApprovalStep]
    records: List[ApprovalRecord]
    created_at: datetime
    expires_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'action_type': self.action_type,
            'target': self.target,
            'parameters': self.parameters,
            'requester': self.requester,
            'urgency': self.urgency.value,
            'approval_level': self.approval_level.value,
            'status': self.status.value,
            'current_step': self.current_step,
            'steps': [s.to_dict() for s in self.steps],
            'records': [r.to_dict() for r in self.records],
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'metadata': self.metadata
        }


class ApprovalWorkflowEngine:
    """审批工作流引擎"""
    
    def __init__(self):
        # 预定义的审批流程
        self.workflow_templates = {
            'restart_service': {
                'levels': [ApprovalLevel.L1],
                'timeout': 30
            },
            'scale_service': {
                'levels': [ApprovalLevel.L1],
                'timeout': 15
            },
            'rollback_deployment': {
                'levels': [ApprovalLevel.L1, ApprovalLevel.L2],
                'timeout': 15
            },
            'config_update': {
                'levels': [ApprovalLevel.L2],
                'timeout': 60
            },
            'emergency_action': {
                'levels': [ApprovalLevel.AUTO],
                'timeout': 0
            }
        }
        
        # 审批人配置
        self.approvers = {
            ApprovalLevel.L1: ['sre_1', 'sre_2', 'sre_3'],
            ApprovalLevel.L2: ['team_lead_1', 'team_lead_2'],
            ApprovalLevel.L3: ['manager_1', 'manager_2']
        }
        
        # 存储
        self.requests: Dict[str, ApprovalRequest] = {}
        
    def create_request(self, action_type: str, target: str,
                      parameters: Dict[str, Any],
                      requester: str,
                      urgency: ApprovalUrgency = ApprovalUrgency.NORMAL,
                      custom_level: ApprovalLevel = None) -> ApprovalRequest:
        """创建审批请求"""
        request_id = f"req_{int(time.time() * 1000)}_{requester}"
        
        # 确定审批级别
        template = self.workflow_templates.get(action_type, {'levels': [ApprovalLevel.L1], 'timeout': 30})
        
        if urgency == ApprovalUrgency.CRITICAL:
            approval_level = ApprovalLevel.AUTO
        elif custom_level:
            approval_level = custom_level
        else:
            approval_level = template['levels'][0]
        
        # 创建审批步骤
        steps = self._create_steps(template['levels'])
        
        # 计算过期时间
        timeout = template['timeout']
        if urgency == ApprovalUrgency.HIGH:
            timeout = timeout // 2
        
        request = ApprovalRequest(
            request_id=request_id,
            action_type=action_type,
            target=target,
            parameters=parameters,
            requester=requester,
            urgency=urgency,
            approval_level=approval_level,
            status=ApprovalStatus.PENDING,
            current_step=0,
            steps=steps,
            records=[],
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=timeout),
            metadata={
                'risk_score': parameters.get('risk_score', 0.5),
                'estimated_impact': parameters.get('impact', 'medium')
            }
        )
        
        self.requests[request_id] = request
        
        # 如果是自动审批，立即处理
        if approval_level == ApprovalLevel.AUTO:
            self._auto_approve(request)
        
        logger.info(f"Created approval request: {request_id} for {action_type} on {target}")
        
        return request
    
    def _create_steps(self, levels: List[ApprovalLevel]) -> List[ApprovalStep]:
        """创建审批步骤"""
        steps = []
        
        for i, level in enumerate(levels):
            if level == ApprovalLevel.AUTO:
                continue
            
            step = ApprovalStep(
                step_id=f"step_{i+1}",
                level=level,
                approvers=self.approvers.get(level, []),
                required_approvals=1,
                timeout_minutes=30
            )
            steps.append(step)
        
        return steps
    
    def _auto_approve(self, request: ApprovalRequest):
        """自动审批"""
        request.status = ApprovalStatus.APPROVED
        
        record = ApprovalRecord(
            record_id=f"rec_auto_{request.request_id}",
            approver='system',
            decision='approved',
            comment='Auto-approved based on critical urgency',
            timestamp=datetime.now()
        )
        request.records.append(record)
        
        logger.info(f"Auto-approved request: {request.request_id}")
    
    def approve(self, request_id: str, approver: str, 
               comment: str = "") -> Dict[str, Any]:
        """批准请求"""
        request = self.requests.get(request_id)
        if not request:
            return {'error': 'Request not found'}
        
        if request.status != ApprovalStatus.PENDING:
            return {'error': f'Request is not pending, current status: {request.status}'}
        
        # 检查审批人权限
        current_step = request.steps[request.current_step] if request.steps else None
        if current_step and approver not in current_step.approvers:
            return {'error': 'Approver not authorized for this step'}
        
        # 记录审批
        record = ApprovalRecord(
            record_id=f"rec_{int(time.time())}",
            approver=approver,
            decision='approved',
            comment=comment,
            timestamp=datetime.now()
        )
        request.records.append(record)
        
        # 检查是否完成当前步骤
        step_approvals = sum(1 for r in request.records 
                           if r.decision == 'approved' and 
                           request.current_step < len(request.steps) and
                           r.approver in request.steps[request.current_step].approvers)
        
        if current_step and step_approvals >= current_step.required_approvals:
            request.current_step += 1
            
            # 检查是否完成所有步骤
            if request.current_step >= len(request.steps):
                request.status = ApprovalStatus.APPROVED
                logger.info(f"Request {request_id} fully approved")
        
        return {
            'success': True,
            'request_id': request_id,
            'status': request.status.value,
            'current_step': request.current_step,
            'total_steps': len(request.steps)
        }
    
    def reject(self, request_id: str, approver: str, 
              comment: str = "") -> Dict[str, Any]:
        """拒绝请求"""
        request = self.requests.get(request_id)
        if not request:
            return {'error': 'Request not found'}
        
        if request.status != ApprovalStatus.PENDING:
            return {'error': f'Request is not pending, current status: {request.status}'}
        
        # 记录审批
        record = ApprovalRecord(
            record_id=f"rec_{int(time.time())}",
            approver=approver,
            decision='rejected',
            comment=comment,
            timestamp=datetime.now()
        )
        request.records.append(record)
        
        request.status = ApprovalStatus.REJECTED
        
        logger.info(f"Request {request_id} rejected by {approver}")
        
        return {
            'success': True,
            'request_id': request_id,
            'status': request.status.value
        }
    
    def escalate(self, request_id: str, reason: str = "") -> Dict[str, Any]:
        """升级请求"""
        request = self.requests.get(request_id)
        if not request:
            return {'error': 'Request not found'}
        
        # 升级审批级别
        level_order = [ApprovalLevel.L1, ApprovalLevel.L2, ApprovalLevel.L3]
        
        current_idx = level_order.index(request.approval_level) if request.approval_level in level_order else 0
        
        if current_idx < len(level_order) - 1:
            request.approval_level = level_order[current_idx + 1]
            request.status = ApprovalStatus.ESCALATED
            
            # 添加新的审批步骤
            new_step = ApprovalStep(
                step_id=f"step_escalated",
                level=request.approval_level,
                approvers=self.approvers.get(request.approval_level, []),
                required_approvals=1,
                timeout_minutes=60
            )
            request.steps.append(new_step)
            
            logger.info(f"Request {request_id} escalated to {request.approval_level.value}")
            
            return {
                'success': True,
                'request_id': request_id,
                'new_level': request.approval_level.value
            }
        
        return {'error': 'Cannot escalate further'}
    
    def batch_approve(self, request_ids: List[str], approver: str,
                     comment: str = "") -> Dict[str, Any]:
        """批量批准"""
        results = []
        
        for request_id in request_ids:
            result = self.approve(request_id, approver, comment)
            results.append({
                'request_id': request_id,
                'result': result
            })
        
        successful = sum(1 for r in results if 'error' not in r['result'])
        
        return {
            'total': len(request_ids),
            'successful': successful,
            'failed': len(request_ids) - successful,
            'details': results
        }
    
    def get_pending_requests(self, approver: str = None) -> List[ApprovalRequest]:
        """获取待处理请求"""
        pending = []
        
        for request in self.requests.values():
            if request.status == ApprovalStatus.PENDING:
                if approver:
                    # 检查审批人是否有权限
                    current_step = request.steps[request.current_step] if request.steps else None
                    if current_step and approver in current_step.approvers:
                        pending.append(request)
                else:
                    pending.append(request)
        
        # 按紧急程度排序
        urgency_order = {
            ApprovalUrgency.CRITICAL: 0,
            ApprovalUrgency.HIGH: 1,
            ApprovalUrgency.NORMAL: 2,
            ApprovalUrgency.LOW: 3
        }
        
        pending.sort(key=lambda r: urgency_order.get(r.urgency, 2))
        
        return pending
    
    def check_expired_requests(self) -> List[str]:
        """检查过期请求"""
        expired = []
        
        for request in self.requests.values():
            if request.status == ApprovalStatus.PENDING and datetime.now() > request.expires_at:
                request.status = ApprovalStatus.EXPIRED
                expired.append(request.request_id)
                logger.info(f"Request {request.request_id} expired")
        
        return expired
