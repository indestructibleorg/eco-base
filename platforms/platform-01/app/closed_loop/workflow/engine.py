"""
决策工作流引擎
实现自动化决策流程和人工审批工作流
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Set, Union
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """工作流状态"""
    PENDING = auto()          # 等待执行
    RUNNING = auto()          # 执行中
    WAITING_APPROVAL = auto() # 等待审批
    APPROVED = auto()         # 已批准
    REJECTED = auto()         # 已拒绝
    COMPLETED = auto()        # 已完成
    FAILED = auto()           # 失败
    CANCELLED = auto()        # 已取消
    TIMEOUT = auto()          # 超时


class ActionType(Enum):
    """动作类型"""
    AUTO_REMEDIATE = "auto_remediate"      # 自动修复
    SCALE_UP = "scale_up"                  # 扩容
    SCALE_DOWN = "scale_down"              # 缩容
    RESTART = "restart"                    # 重启
    ROLLBACK = "rollback"                  # 回滚
    NOTIFY = "notify"                      # 通知
    ESCALATE = "escalate"                  # 升级
    CUSTOM = "custom"                      # 自定义


class ApprovalLevel(Enum):
    """审批级别"""
    NONE = 0          # 无需审批
    LEVEL_1 = 1       # 一级审批 (值班工程师)
    LEVEL_2 = 2       # 二级审批 (团队负责人)
    LEVEL_3 = 3       # 三级审批 (部门经理)
    LEVEL_4 = 4       # 四级审批 (高级管理层)


@dataclass
class WorkflowStep:
    """工作流步骤"""
    step_id: str
    name: str
    action_type: ActionType
    params: Dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False
    approval_level: ApprovalLevel = ApprovalLevel.NONE
    timeout_seconds: int = 300
    retry_count: int = 3
    retry_delay: int = 10
    condition: Optional[str] = None  # 执行条件表达式
    next_steps: List[str] = field(default_factory=list)
    on_failure: Optional[str] = None  # 失败时跳转的步骤


@dataclass
class WorkflowDefinition:
    """工作流定义"""
    workflow_id: str
    name: str
    description: str
    version: str = "1.0"
    steps: Dict[str, WorkflowStep] = field(default_factory=dict)
    start_step: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


@dataclass
class StepExecution:
    """步骤执行记录"""
    step_id: str
    status: WorkflowStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_attempt: int = 0
    logs: List[str] = field(default_factory=list)


@dataclass
class WorkflowInstance:
    """工作流实例"""
    instance_id: str
    workflow_id: str
    status: WorkflowStatus
    context: Dict[str, Any] = field(default_factory=dict)
    current_step: Optional[str] = None
    step_executions: Dict[str, StepExecution] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    triggered_by: str = "system"
    priority: int = 5  # 1-10, 10为最高优先级


@dataclass
class ApprovalRequest:
    """审批请求"""
    request_id: str
    instance_id: str
    step_id: str
    level: ApprovalLevel
    title: str
    description: str
    requester: str
    requested_at: datetime
    status: WorkflowStatus = WorkflowStatus.WAITING_APPROVAL
    approver: Optional[str] = None
    decision: Optional[str] = None
    comment: Optional[str] = None
    decided_at: Optional[datetime] = None
    timeout_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=24))


class WorkflowEngine:
    """工作流引擎"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.instances: Dict[str, WorkflowInstance] = {}
        self.approval_requests: Dict[str, ApprovalRequest] = {}
        self.action_handlers: Dict[ActionType, Callable] = {}
        self.notification_handlers: List[Callable] = []
        self.running = False
        self._task_queue: List[tuple] = []  # 优先级队列 (priority, timestamp, instance_id)
        self._lock = asyncio.Lock()
        self._worker_task: Optional[asyncio.Task] = None
        
        # 注册默认动作处理器
        self._register_default_handlers()
        
        logger.info("工作流引擎初始化完成")
    
    def _register_default_handlers(self):
        """注册默认动作处理器"""
        self.action_handlers[ActionType.NOTIFY] = self._handle_notify
        self.action_handlers[ActionType.ESCALATE] = self._handle_escalate
        logger.info("默认动作处理器注册完成")
    
    def register_action_handler(self, action_type: ActionType, handler: Callable):
        """注册动作处理器"""
        self.action_handlers[action_type] = handler
        logger.info(f"动作处理器注册: {action_type.value}")
    
    def register_notification_handler(self, handler: Callable):
        """注册通知处理器"""
        self.notification_handlers.append(handler)
        logger.info("通知处理器注册完成")
    
    # ==================== 工作流定义管理 ====================
    
    def create_workflow(self, name: str, description: str, 
                       version: str = "1.0", tags: List[str] = None) -> WorkflowDefinition:
        """创建工作流定义"""
        workflow = WorkflowDefinition(
            workflow_id=str(uuid.uuid4()),
            name=name,
            description=description,
            version=version,
            tags=tags or []
        )
        self.workflows[workflow.workflow_id] = workflow
        logger.info(f"工作流创建: {name} (ID: {workflow.workflow_id})")
        return workflow
    
    def add_workflow_step(self, workflow_id: str, step: WorkflowStep):
        """添加工作流步骤"""
        if workflow_id not in self.workflows:
            raise ValueError(f"工作流不存在: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        workflow.steps[step.step_id] = step
        
        # 如果是第一个步骤，设为起始步骤
        if not workflow.start_step:
            workflow.start_step = step.step_id
        
        workflow.updated_at = datetime.now()
        logger.info(f"步骤添加: {step.name} -> {workflow.name}")
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """获取工作流定义"""
        return self.workflows.get(workflow_id)
    
    def list_workflows(self, tags: List[str] = None) -> List[WorkflowDefinition]:
        """列出工作流定义"""
        workflows = list(self.workflows.values())
        if tags:
            workflows = [w for w in workflows if any(t in w.tags for t in tags)]
        return workflows
    
    # ==================== 工作流实例执行 ====================
    
    async def start_workflow(self, workflow_id: str, context: Dict[str, Any] = None,
                            triggered_by: str = "system", priority: int = 5) -> str:
        """启动工作流实例"""
        if workflow_id not in self.workflows:
            raise ValueError(f"工作流不存在: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        if not workflow.start_step:
            raise ValueError("工作流没有定义起始步骤")
        
        instance = WorkflowInstance(
            instance_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            context=context or {},
            triggered_by=triggered_by,
            priority=priority
        )
        
        self.instances[instance.instance_id] = instance
        
        # 加入优先级队列
        heapq.heappush(self._task_queue, (-priority, datetime.now().timestamp(), instance.instance_id))
        
        logger.info(f"工作流启动: {instance.instance_id} (优先级: {priority})")
        
        # 触发执行
        asyncio.create_task(self._execute_workflow(instance.instance_id))
        
        return instance.instance_id
    
    async def _execute_workflow(self, instance_id: str):
        """执行工作流"""
        instance = self.instances.get(instance_id)
        if not instance:
            logger.error(f"工作流实例不存在: {instance_id}")
            return
        
        workflow = self.workflows.get(instance.workflow_id)
        if not workflow:
            logger.error(f"工作流定义不存在: {instance.workflow_id}")
            instance.status = WorkflowStatus.FAILED
            return
        
        instance.status = WorkflowStatus.RUNNING
        instance.started_at = datetime.now()
        instance.current_step = workflow.start_step
        
        logger.info(f"工作流执行开始: {instance_id}")
        
        try:
            while instance.current_step and instance.status == WorkflowStatus.RUNNING:
                step = workflow.steps.get(instance.current_step)
                if not step:
                    logger.error(f"步骤不存在: {instance.current_step}")
                    instance.status = WorkflowStatus.FAILED
                    break
                
                # 执行步骤
                await self._execute_step(instance, step)
                
                # 检查是否需要等待审批
                if instance.status == WorkflowStatus.WAITING_APPROVAL:
                    break
                
                # 确定下一步
                step_execution = instance.step_executions.get(step.step_id)
                if step_execution and step_execution.status == WorkflowStatus.COMPLETED:
                    # 执行成功，进入下一步
                    if step.next_steps:
                        # 根据条件选择下一步
                        instance.current_step = await self._select_next_step(
                            instance, step, step_execution.result
                        )
                    else:
                        instance.current_step = None
                elif step_execution and step_execution.status == WorkflowStatus.FAILED:
                    # 执行失败
                    if step.on_failure:
                        instance.current_step = step.on_failure
                    else:
                        instance.status = WorkflowStatus.FAILED
                        break
            
            # 工作流完成
            if instance.status == WorkflowStatus.RUNNING and not instance.current_step:
                instance.status = WorkflowStatus.COMPLETED
                instance.completed_at = datetime.now()
                logger.info(f"工作流完成: {instance_id}")
                await self._notify_workflow_completion(instance)
        
        except Exception as e:
            logger.exception(f"工作流执行异常: {instance_id}")
            instance.status = WorkflowStatus.FAILED
            await self._notify_workflow_failure(instance, str(e))
    
    async def _execute_step(self, instance: WorkflowInstance, step: WorkflowStep):
        """执行单个步骤"""
        logger.info(f"执行步骤: {step.name} (实例: {instance.instance_id})")
        
        step_execution = StepExecution(
            step_id=step.step_id,
            status=WorkflowStatus.RUNNING,
            started_at=datetime.now()
        )
        instance.step_executions[step.step_id] = step_execution
        
        # 检查是否需要审批
        if step.requires_approval:
            await self._create_approval_request(instance, step)
            instance.status = WorkflowStatus.WAITING_APPROVAL
            step_execution.status = WorkflowStatus.WAITING_APPROVAL
            return
        
        # 执行动作
        handler = self.action_handlers.get(step.action_type)
        if not handler:
            step_execution.status = WorkflowStatus.FAILED
            step_execution.error = f"未找到动作处理器: {step.action_type.value}"
            return
        
        # 重试逻辑
        for attempt in range(step.retry_count):
            try:
                step_execution.retry_attempt = attempt + 1
                result = await handler(instance, step)
                step_execution.result = result
                step_execution.status = WorkflowStatus.COMPLETED
                step_execution.completed_at = datetime.now()
                step_execution.logs.append(f"执行成功 (尝试 {attempt + 1})")
                logger.info(f"步骤执行成功: {step.name}")
                return
            
            except Exception as e:
                logger.warning(f"步骤执行失败 (尝试 {attempt + 1}/{step.retry_count}): {e}")
                step_execution.logs.append(f"执行失败 (尝试 {attempt + 1}): {str(e)}")
                if attempt < step.retry_count - 1:
                    await asyncio.sleep(step.retry_delay)
        
        # 所有重试失败
        step_execution.status = WorkflowStatus.FAILED
        step_execution.error = f"执行失败，已重试 {step.retry_count} 次"
        logger.error(f"步骤执行失败: {step.name}")
    
    async def _select_next_step(self, instance: WorkflowInstance, 
                                current_step: WorkflowStep, 
                                result: Any) -> Optional[str]:
        """选择下一步"""
        if not current_step.next_steps:
            return None
        
        if len(current_step.next_steps) == 1:
            return current_step.next_steps[0]
        
        # 多分支条件选择
        for next_step_id in current_step.next_steps:
            next_step = self.workflows[instance.workflow_id].steps.get(next_step_id)
            if next_step and next_step.condition:
                # 评估条件
                if self._evaluate_condition(next_step.condition, instance.context, result):
                    return next_step_id
        
        # 默认返回第一个
        return current_step.next_steps[0]
    
    def _evaluate_condition(self, condition: str, context: Dict, result: Any) -> bool:
        """评估条件表达式"""
        try:
            # 简单的条件评估，实际应用中可以使用更复杂的表达式引擎
            eval_context = {
                'context': context,
                'result': result,
                'True': True,
                'False': False
            }
            return eval(condition, {"__builtins__": {}}, eval_context)
        except Exception as e:
            logger.warning(f"条件评估失败: {condition}, 错误: {e}")
            return False
    
    # ==================== 审批管理 ====================
    
    async def _create_approval_request(self, instance: WorkflowInstance, step: WorkflowStep):
        """创建审批请求"""
        request = ApprovalRequest(
            request_id=str(uuid.uuid4()),
            instance_id=instance.instance_id,
            step_id=step.step_id,
            level=step.approval_level,
            title=f"审批请求: {step.name}",
            description=f"工作流实例 {instance.instance_id} 请求执行步骤: {step.name}",
            requester=instance.triggered_by,
            requested_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(seconds=step.timeout_seconds)
        )
        
        self.approval_requests[request.request_id] = request
        
        logger.info(f"审批请求创建: {request.request_id} (级别: {step.approval_level.name})")
        
        # 发送通知
        await self._notify_approval_request(request)
    
    async def approve_request(self, request_id: str, approver: str, 
                             comment: str = "") -> bool:
        """批准请求"""
        request = self.approval_requests.get(request_id)
        if not request or request.status != WorkflowStatus.WAITING_APPROVAL:
            return False
        
        request.status = WorkflowStatus.APPROVED
        request.approver = approver
        request.decision = "approved"
        request.comment = comment
        request.decided_at = datetime.now()
        
        # 继续工作流执行
        instance = self.instances.get(request.instance_id)
        if instance:
            instance.status = WorkflowStatus.RUNNING
            step = self.workflows[instance.workflow_id].steps.get(request.step_id)
            if step:
                asyncio.create_task(self._continue_after_approval(instance, step))
        
        logger.info(f"审批请求已批准: {request_id} by {approver}")
        return True
    
    async def reject_request(self, request_id: str, approver: str, 
                            comment: str = "") -> bool:
        """拒绝请求"""
        request = self.approval_requests.get(request_id)
        if not request or request.status != WorkflowStatus.WAITING_APPROVAL:
            return False
        
        request.status = WorkflowStatus.REJECTED
        request.approver = approver
        request.decision = "rejected"
        request.comment = comment
        request.decided_at = datetime.now()
        
        # 工作流失败
        instance = self.instances.get(request.instance_id)
        if instance:
            instance.status = WorkflowStatus.FAILED
            step_execution = instance.step_executions.get(request.step_id)
            if step_execution:
                step_execution.status = WorkflowStatus.REJECTED
        
        logger.info(f"审批请求已拒绝: {request_id} by {approver}")
        return True
    
    async def _continue_after_approval(self, instance: WorkflowInstance, step: WorkflowStep):
        """审批后继续执行"""
        handler = self.action_handlers.get(step.action_type)
        if handler:
            try:
                result = await handler(instance, step)
                step_execution = instance.step_executions[step.step_id]
                step_execution.result = result
                step_execution.status = WorkflowStatus.COMPLETED
                step_execution.completed_at = datetime.now()
                
                # 继续工作流
                if step.next_steps:
                    instance.current_step = await self._select_next_step(
                        instance, step, result
                    )
                else:
                    instance.current_step = None
                    instance.status = WorkflowStatus.COMPLETED
                    instance.completed_at = datetime.now()
                
                # 如果还有后续步骤，继续执行
                if instance.current_step and instance.status == WorkflowStatus.RUNNING:
                    await self._execute_workflow(instance.instance_id)
                    
            except Exception as e:
                logger.exception(f"审批后执行失败: {e}")
                step_execution = instance.step_executions.get(step.step_id)
                if step_execution:
                    step_execution.status = WorkflowStatus.FAILED
                    step_execution.error = str(e)
    
    def get_pending_approvals(self, level: ApprovalLevel = None) -> List[ApprovalRequest]:
        """获取待审批请求"""
        requests = [
            r for r in self.approval_requests.values()
            if r.status == WorkflowStatus.WAITING_APPROVAL
        ]
        if level:
            requests = [r for r in requests if r.level == level]
        return sorted(requests, key=lambda x: x.requested_at)
    
    def get_approval_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """获取审批请求"""
        return self.approval_requests.get(request_id)
    
    # ==================== 工作流管理 ====================
    
    async def cancel_workflow(self, instance_id: str, reason: str = "") -> bool:
        """取消工作流实例"""
        instance = self.instances.get(instance_id)
        if not instance:
            return False
        
        if instance.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, 
                               WorkflowStatus.CANCELLED]:
            logger.warning(f"无法取消已完成/失败/已取消的工作流: {instance_id}")
            return False
        
        instance.status = WorkflowStatus.CANCELLED
        instance.completed_at = datetime.now()
        instance.context['cancel_reason'] = reason
        
        # 取消相关的审批请求
        for request in self.approval_requests.values():
            if request.instance_id == instance_id and request.status == WorkflowStatus.WAITING_APPROVAL:
                request.status = WorkflowStatus.CANCELLED
        
        logger.info(f"工作流已取消: {instance_id}, 原因: {reason}")
        await self._notify_workflow_cancellation(instance, reason)
        return True
    
    def get_workflow_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """获取工作流实例"""
        return self.instances.get(instance_id)
    
    def list_workflow_instances(self, status: WorkflowStatus = None,
                                workflow_id: str = None) -> List[WorkflowInstance]:
        """列出工作流实例"""
        instances = list(self.instances.values())
        if status:
            instances = [i for i in instances if i.status == status]
        if workflow_id:
            instances = [i for i in instances if i.workflow_id == workflow_id]
        return sorted(instances, key=lambda x: x.created_at, reverse=True)
    
    def get_workflow_stats(self, workflow_id: str = None) -> Dict[str, Any]:
        """获取工作流统计"""
        instances = self.instances.values()
        if workflow_id:
            instances = [i for i in instances if i.workflow_id == workflow_id]
        
        total = len(instances)
        if total == 0:
            return {"total": 0}
        
        completed = sum(1 for i in instances if i.status == WorkflowStatus.COMPLETED)
        failed = sum(1 for i in instances if i.status == WorkflowStatus.FAILED)
        cancelled = sum(1 for i in instances if i.status == WorkflowStatus.CANCELLED)
        running = sum(1 for i in instances if i.status == WorkflowStatus.RUNNING)
        pending = sum(1 for i in instances if i.status == WorkflowStatus.PENDING)
        waiting_approval = sum(1 for i in instances if i.status == WorkflowStatus.WAITING_APPROVAL)
        
        # 计算平均执行时间
        execution_times = []
        for i in instances:
            if i.started_at and i.completed_at:
                execution_times.append((i.completed_at - i.started_at).total_seconds())
        
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # 成功率
        completed_instances = [i for i in instances if i.status == WorkflowStatus.COMPLETED]
        success_rate = len(completed_instances) / total * 100 if total > 0 else 0
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "cancelled": cancelled,
            "running": running,
            "pending": pending,
            "waiting_approval": waiting_approval,
            "success_rate": round(success_rate, 2),
            "avg_execution_time_seconds": round(avg_execution_time, 2),
            "pending_approvals": len(self.get_pending_approvals())
        }
    
    # ==================== 默认动作处理器 ====================
    
    async def _handle_notify(self, instance: WorkflowInstance, step: WorkflowStep) -> Any:
        """处理通知动作"""
        message = step.params.get('message', '工作流通知')
        channels = step.params.get('channels', ['log'])
        
        for handler in self.notification_handlers:
            try:
                await handler({
                    'type': 'workflow_notification',
                    'instance_id': instance.instance_id,
                    'step': step.name,
                    'message': message,
                    'channels': channels,
                    'context': instance.context
                })
            except Exception as e:
                logger.warning(f"通知处理器失败: {e}")
        
        return {'notified': True, 'channels': channels}
    
    async def _handle_escalate(self, instance: WorkflowInstance, step: WorkflowStep) -> Any:
        """处理升级动作"""
        escalate_to = step.params.get('escalate_to', 'manager')
        reason = step.params.get('reason', '工作流升级')
        
        logger.info(f"工作流升级: {instance.instance_id} -> {escalate_to}")
        
        for handler in self.notification_handlers:
            try:
                await handler({
                    'type': 'workflow_escalation',
                    'instance_id': instance.instance_id,
                    'escalate_to': escalate_to,
                    'reason': reason,
                    'context': instance.context
                })
            except Exception as e:
                logger.warning(f"升级通知失败: {e}")
        
        return {'escalated': True, 'to': escalate_to}
    
    # ==================== 通知 ====================
    
    async def _notify_approval_request(self, request: ApprovalRequest):
        """通知审批请求"""
        for handler in self.notification_handlers:
            try:
                await handler({
                    'type': 'approval_request',
                    'request_id': request.request_id,
                    'title': request.title,
                    'description': request.description,
                    'level': request.level.name,
                    'requester': request.requester,
                    'timeout_at': request.timeout_at.isoformat()
                })
            except Exception as e:
                logger.warning(f"审批通知失败: {e}")
    
    async def _notify_workflow_completion(self, instance: WorkflowInstance):
        """通知工作流完成"""
        for handler in self.notification_handlers:
            try:
                await handler({
                    'type': 'workflow_completed',
                    'instance_id': instance.instance_id,
                    'workflow_id': instance.workflow_id,
                    'duration': (instance.completed_at - instance.started_at).total_seconds() if instance.started_at else 0,
                    'context': instance.context
                })
            except Exception as e:
                logger.warning(f"完成通知失败: {e}")
    
    async def _notify_workflow_failure(self, instance: WorkflowInstance, error: str):
        """通知工作流失败"""
        for handler in self.notification_handlers:
            try:
                await handler({
                    'type': 'workflow_failed',
                    'instance_id': instance.instance_id,
                    'workflow_id': instance.workflow_id,
                    'error': error,
                    'context': instance.context
                })
            except Exception as e:
                logger.warning(f"失败通知失败: {e}")
    
    async def _notify_workflow_cancellation(self, instance: WorkflowInstance, reason: str):
        """通知工作流取消"""
        for handler in self.notification_handlers:
            try:
                await handler({
                    'type': 'workflow_cancelled',
                    'instance_id': instance.instance_id,
                    'workflow_id': instance.workflow_id,
                    'reason': reason,
                    'context': instance.context
                })
            except Exception as e:
                logger.warning(f"取消通知失败: {e}")
    
    # ==================== 预定义工作流 ====================
    
    def create_auto_remediation_workflow(self, remediation_handler: Callable = None) -> str:
        """创建自动修复工作流"""
        workflow = self.create_workflow(
            name="自动修复工作流",
            description="检测问题后自动执行修复操作",
            tags=["auto", "remediation"]
        )
        
        # 步骤1: 通知检测
        self.add_workflow_step(workflow.workflow_id, WorkflowStep(
            step_id="notify_detection",
            name="通知问题检测",
            action_type=ActionType.NOTIFY,
            params={'message': '检测到问题，开始自动修复流程', 'channels': ['log', 'slack']},
            next_steps=["attempt_remediation"]
        ))
        
        # 步骤2: 尝试修复
        self.add_workflow_step(workflow.workflow_id, WorkflowStep(
            step_id="attempt_remediation",
            name="执行自动修复",
            action_type=ActionType.AUTO_REMEDIATE,
            retry_count=3,
            retry_delay=5,
            next_steps=["verify_remediation"],
            on_failure="escalate_issue"
        ))
        
        if remediation_handler:
            self.register_action_handler(ActionType.AUTO_REMEDIATE, remediation_handler)
        
        # 步骤3: 验证修复
        self.add_workflow_step(workflow.workflow_id, WorkflowStep(
            step_id="verify_remediation",
            name="验证修复结果",
            action_type=ActionType.NOTIFY,
            params={'message': '修复完成，正在验证', 'channels': ['log']},
            next_steps=["complete"]
        ))
        
        # 步骤4: 完成
        self.add_workflow_step(workflow.workflow_id, WorkflowStep(
            step_id="complete",
            name="流程完成",
            action_type=ActionType.NOTIFY,
            params={'message': '自动修复流程完成', 'channels': ['log', 'slack']}
        ))
        
        # 失败处理: 升级
        self.add_workflow_step(workflow.workflow_id, WorkflowStep(
            step_id="escalate_issue",
            name="升级问题",
            action_type=ActionType.ESCALATE,
            params={'escalate_to': 'oncall', 'reason': '自动修复失败'},
            requires_approval=True,
            approval_level=ApprovalLevel.LEVEL_1
        ))
        
        logger.info(f"自动修复工作流创建完成: {workflow.workflow_id}")
        return workflow.workflow_id
    
    def create_scaling_workflow(self, scale_handler: Callable = None) -> str:
        """创建自动扩缩容工作流"""
        workflow = self.create_workflow(
            name="自动扩缩容工作流",
            description="根据负载自动调整资源",
            tags=["auto", "scaling"]
        )
        
        # 步骤1: 评估当前状态
        self.add_workflow_step(workflow.workflow_id, WorkflowStep(
            step_id="assess_state",
            name="评估当前状态",
            action_type=ActionType.NOTIFY,
            params={'message': '评估当前系统状态', 'channels': ['log']},
            next_steps=["decide_action"]
        ))
        
        # 步骤2: 决定动作
        self.add_workflow_step(workflow.workflow_id, WorkflowStep(
            step_id="decide_action",
            name="决定扩缩容动作",
            action_type=ActionType.CUSTOM,
            params={'decision_logic': 'threshold_based'},
            next_steps=["execute_scaling"]
        ))
        
        # 步骤3: 执行扩缩容
        self.add_workflow_step(workflow.workflow_id, WorkflowStep(
            step_id="execute_scaling",
            name="执行扩缩容",
            action_type=ActionType.SCALE_UP,
            requires_approval=True,
            approval_level=ApprovalLevel.LEVEL_2,
            timeout_seconds=600,
            next_steps=["verify_scaling"],
            on_failure="notify_failure"
        ))
        
        if scale_handler:
            self.register_action_handler(ActionType.SCALE_UP, scale_handler)
            self.register_action_handler(ActionType.SCALE_DOWN, scale_handler)
        
        # 步骤4: 验证
        self.add_workflow_step(workflow.workflow_id, WorkflowStep(
            step_id="verify_scaling",
            name="验证扩缩容结果",
            action_type=ActionType.NOTIFY,
            params={'message': '扩缩容完成，验证中', 'channels': ['log']}
        ))
        
        # 失败通知
        self.add_workflow_step(workflow.workflow_id, WorkflowStep(
            step_id="notify_failure",
            name="通知失败",
            action_type=ActionType.NOTIFY,
            params={'message': '扩缩容失败', 'channels': ['log', 'slack', 'pagerduty']}
        ))
        
        logger.info(f"自动扩缩容工作流创建完成: {workflow.workflow_id}")
        return workflow.workflow_id
    
    # ==================== 系统管理 ====================
    
    async def start(self):
        """启动引擎"""
        self.running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("工作流引擎已启动")
    
    async def stop(self):
        """停止引擎"""
        self.running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("工作流引擎已停止")
    
    async def _worker_loop(self):
        """工作线程循环"""
        while self.running:
            try:
                # 处理超时审批
                await self._check_approval_timeouts()
                
                # 处理队列中的任务
                await self._process_task_queue()
                
                await asyncio.sleep(1)
            except Exception as e:
                logger.exception(f"工作线程异常: {e}")
    
    async def _check_approval_timeouts(self):
        """检查审批超时"""
        now = datetime.now()
        for request in list(self.approval_requests.values()):
            if request.status == WorkflowStatus.WAITING_APPROVAL and now > request.timeout_at:
                request.status = WorkflowStatus.TIMEOUT
                
                # 相关工作流失败
                instance = self.instances.get(request.instance_id)
                if instance and instance.status == WorkflowStatus.WAITING_APPROVAL:
                    instance.status = WorkflowStatus.FAILED
                    step_execution = instance.step_executions.get(request.step_id)
                    if step_execution:
                        step_execution.status = WorkflowStatus.TIMEOUT
                        step_execution.error = "审批超时"
                
                logger.warning(f"审批请求超时: {request.request_id}")
    
    async def _process_task_queue(self):
        """处理任务队列"""
        # 优先级队列自动排序，这里可以添加额外的调度逻辑
        pass
    
    def export_workflow(self, workflow_id: str) -> Optional[str]:
        """导出工作流定义为JSON"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None
        
        def serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, (WorkflowStep, WorkflowDefinition)):
                return asdict(obj)
            raise TypeError(f"不可序列化类型: {type(obj)}")
        
        return json.dumps(workflow, default=serialize, indent=2, ensure_ascii=False)
    
    def import_workflow(self, json_str: str) -> Optional[str]:
        """从JSON导入工作流定义"""
        try:
            data = json.loads(json_str)
            
            workflow = WorkflowDefinition(
                workflow_id=str(uuid.uuid4()),  # 生成新ID
                name=data['name'],
                description=data['description'],
                version=data.get('version', '1.0'),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                tags=data.get('tags', [])
            )
            
            # 导入步骤
            for step_data in data.get('steps', {}).values():
                step = WorkflowStep(
                    step_id=step_data['step_id'],
                    name=step_data['name'],
                    action_type=ActionType(step_data['action_type']),
                    params=step_data.get('params', {}),
                    requires_approval=step_data.get('requires_approval', False),
                    approval_level=ApprovalLevel(step_data.get('approval_level', 0)),
                    timeout_seconds=step_data.get('timeout_seconds', 300),
                    retry_count=step_data.get('retry_count', 3),
                    retry_delay=step_data.get('retry_delay', 10),
                    condition=step_data.get('condition'),
                    next_steps=step_data.get('next_steps', []),
                    on_failure=step_data.get('on_failure')
                )
                workflow.steps[step.step_id] = step
            
            workflow.start_step = data.get('start_step', '')
            
            self.workflows[workflow.workflow_id] = workflow
            logger.info(f"工作流导入成功: {workflow.workflow_id}")
            return workflow.workflow_id
            
        except Exception as e:
            logger.error(f"工作流导入失败: {e}")
            return None
    
    def create_workflow(self, workflow_def: Dict[str, Any], description: str = "") -> Dict[str, Any]:
        """创建工作流 (简化接口)"""
        workflow_id = workflow_def.get('id', str(uuid.uuid4()))
        
        # 创建工作流定义
        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=workflow_def.get('name', 'Unnamed Workflow'),
            description=description or workflow_def.get('description', ''),
            version='1.0',
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # 添加步骤
        states = workflow_def.get('states', [])
        transitions = workflow_def.get('transitions', [])
        
        for i, state in enumerate(states):
            step = WorkflowStep(
                step_id=state,
                name=state,
                action_type=ActionType.CUSTOM,
                params={},
                next_steps=[]
            )
            workflow.steps[state] = step
        
        # 设置转换
        for trans in transitions:
            from_state = trans.get('from')
            to_state = trans.get('to')
            if from_state in workflow.steps:
                workflow.steps[from_state].next_steps.append(to_state)
        
        # 设置起始步骤
        if states:
            workflow.start_step = states[0]
        
        # 存储工作流
        self.workflows[workflow_id] = workflow
        
        # 创建实例
        instance_id = str(uuid.uuid4())
        instance = WorkflowInstance(
            instance_id=instance_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            context={},
            step_executions={},
            created_at=datetime.now()
        )
        self.instances[instance_id] = instance
        
        return {
            'instance_id': instance_id,
            'workflow_id': workflow_id,
            'status': instance.status.value
        }
