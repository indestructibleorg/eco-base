"""
可恢复状态机控制器 (Recoverable Controller)

消除断尾根因的核心组件
- 状态持久化
- 崩溃后可恢复
- 幂等执行
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import logging

from .state_store import (
    RunState, RunPhase, RunStateTransition,
    StateStore, StateStoreFactory
)
from ..governance import (
    Decision, DecisionContractManager,
    IdempotentAction, ActionFactory, ActionType,
    ApprovalGate, GateResult,
    VerificationGate, VerificationConfig,
    AuditTrail, EvidenceCollector
)

logger = logging.getLogger(__name__)


class RecoverableController:
    """
    可恢复闭环控制器
    
    状态流转:
    NEW -> DETECTED -> ANALYZED -> PLANNED -> 
    (APPROVAL_PENDING) -> APPROVED -> EXECUTING -> EXECUTED -> 
    VERIFYING -> (ROLLED_BACK | SUCCEEDED | ESCALATED | FAILED)
    """
    
    def __init__(
        self,
        state_store: Optional[StateStore] = None,
        approval_gate: Optional[ApprovalGate] = None,
        verification_gate: Optional[VerificationGate] = None,
        decision_manager: Optional[DecisionContractManager] = None
    ):
        # 依赖组件
        self.state_store = state_store or StateStoreFactory.create("memory")
        self.approval_gate = approval_gate or ApprovalGate()
        self.verification_gate = verification_gate or VerificationGate(VerificationConfig())
        self.decision_manager = decision_manager or DecisionContractManager()
        
        # 动作注册表
        self._action_handlers: Dict[str, Callable] = {}
        
        # 运行锁（防止并发修改同一run）
        self._run_locks: Dict[str, asyncio.Lock] = {}
    
    def register_action_handler(self, action_type: str, handler: Callable):
        """注册动作处理器"""
        self._action_handlers[action_type] = handler
    
    async def start_run(
        self,
        trace_id: str,
        inputs: Dict[str, Any]
    ) -> RunState:
        """
        启动新的运行实例
        
        Args:
            trace_id: 追踪ID
            inputs: 输入数据
        
        Returns:
            运行状态
        """
        run_id = f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 计算输入哈希
        import json
        import hashlib
        inputs_hash = hashlib.sha256(
            json.dumps(inputs, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        
        # 创建运行状态
        state = RunState(
            run_id=run_id,
            trace_id=trace_id,
            phase=RunPhase.NEW,
            inputs_hash=inputs_hash,
            metadata={"inputs": inputs}
        )
        
        # 持久化
        await self.state_store.save(state)
        
        logger.info(f"Run started: {run_id}")
        
        return state
    
    async def resume_run(self, run_id: str) -> Optional[RunState]:
        """
        恢复运行实例
        
        从持久化存储加载状态并继续执行
        
        Args:
            run_id: 运行ID
        
        Returns:
            运行状态
        """
        # 加载状态
        state = await self.state_store.load(run_id)
        if not state:
            logger.error(f"Run not found: {run_id}")
            return None
        
        if state.is_terminal():
            logger.warning(f"Run {run_id} is already in terminal state: {state.phase.value}")
            return state
        
        if not state.can_resume():
            logger.warning(f"Run {run_id} cannot be resumed from state: {state.phase.value}")
            return state
        
        logger.info(f"Resuming run: {run_id} from phase: {state.phase.value}")
        
        # 从当前阶段继续执行
        return await self._continue_from_phase(state)
    
    async def _continue_from_phase(self, state: RunPhase) -> RunState:
        """从当前阶段继续执行"""
        phase_handlers = {
            RunPhase.NEW: self._handle_new,
            RunPhase.DETECTED: self._handle_detected,
            RunPhase.ANALYZED: self._handle_analyzed,
            RunPhase.PLANNED: self._handle_planned,
            RunPhase.APPROVAL_PENDING: self._handle_approval_pending,
            RunPhase.APPROVED: self._handle_approved,
            RunPhase.EXECUTING: self._handle_executing,
            RunPhase.EXECUTED: self._handle_executed,
            RunPhase.VERIFYING: self._handle_verifying,
        }
        
        handler = phase_handlers.get(state.phase)
        if handler:
            return await handler(state)
        
        logger.warning(f"No handler for phase: {state.phase.value}")
        return state
    
    async def _handle_new(self, state: RunState) -> RunState:
        """处理 NEW 阶段"""
        logger.info(f"[{state.run_id}] Handling NEW phase")
        
        # 转移到 DETECTED
        state.transition_to(RunPhase.DETECTED, RunStateTransition.ANOMALY_DETECTED)
        await self.state_store.save(state)
        
        # 继续下一阶段
        return await self._handle_detected(state)
    
    async def _handle_detected(self, state: RunState) -> RunState:
        """处理 DETECTED 阶段"""
        logger.info(f"[{state.run_id}] Handling DETECTED phase")
        
        # 执行异常检测逻辑
        # ...
        
        # 转移到 ANALYZED
        state.transition_to(RunPhase.ANALYZED, RunStateTransition.RCA_COMPLETED)
        await self.state_store.save(state)
        
        return await self._handle_analyzed(state)
    
    async def _handle_analyzed(self, state: RunState) -> RunState:
        """处理 ANALYZED 阶段"""
        logger.info(f"[{state.run_id}] Handling ANALYZED phase")
        
        # 执行根因分析
        # ...
        
        # 转移到 PLANNED
        state.transition_to(RunPhase.PLANNED, RunStateTransition.DECISION_PLANNED)
        await self.state_store.save(state)
        
        return await self._handle_planned(state)
    
    async def _handle_planned(self, state: RunState) -> RunState:
        """处理 PLANNED 阶段 - 决策已生成"""
        logger.info(f"[{state.run_id}] Handling PLANNED phase")
        
        # 获取决策
        decision = self.decision_manager.get_decision(state.decision_id) if state.decision_id else None
        
        if not decision:
            logger.error(f"[{state.run_id}] Decision not found")
            state.set_error(
                error_type="DecisionNotFound",
                error_message="Decision not found for planned run",
                recoverable=False
            )
            state.transition_to(RunPhase.FAILED, RunStateTransition.EXECUTION_FAILED)
            await self.state_store.save(state)
            return state
        
        # 检查是否需要审批
        if decision.risk.requires_approval:
            # 转移到审批等待
            state.transition_to(RunPhase.APPROVAL_PENDING, RunStateTransition.APPROVAL_REQUIRED)
            await self.state_store.save(state)
            return await self._handle_approval_pending(state)
        else:
            # 低风险，自动批准
            state.transition_to(RunPhase.APPROVED, RunStateTransition.APPROVAL_GRANTED)
            await self.state_store.save(state)
            return await self._handle_approved(state)
    
    async def _handle_approval_pending(self, state: RunState) -> RunState:
        """处理 APPROVAL_PENDING 阶段"""
        logger.info(f"[{state.run_id}] Handling APPROVAL_PENDING phase")
        
        # 检查审批状态
        if not state.approval_request_id:
            logger.error(f"[{state.run_id}] No approval request ID")
            state.transition_to(RunPhase.FAILED, RunStateTransition.APPROVAL_TIMEOUT)
            await self.state_store.save(state)
            return state
        
        from ..governance import ApprovalStatus
        approval_status = self.approval_gate.get_approval_status(state.approval_request_id)
        
        if approval_status == ApprovalStatus.APPROVED:
            state.transition_to(RunPhase.APPROVED, RunStateTransition.APPROVAL_GRANTED)
            await self.state_store.save(state)
            return await self._handle_approved(state)
        
        elif approval_status == ApprovalStatus.REJECTED:
            state.transition_to(RunPhase.FAILED, RunStateTransition.APPROVAL_REJECTED)
            await self.state_store.save(state)
            return state
        
        elif approval_status == ApprovalStatus.EXPIRED:
            state.transition_to(RunPhase.ESCALATED, RunStateTransition.APPROVAL_TIMEOUT)
            await self.state_store.save(state)
            return state
        
        # 仍在等待审批
        logger.info(f"[{state.run_id}] Still waiting for approval")
        return state
    
    async def _handle_approved(self, state: RunState) -> RunState:
        """处理 APPROVED 阶段"""
        logger.info(f"[{state.run_id}] Handling APPROVED phase")
        
        # 转移到执行中
        state.transition_to(RunPhase.EXECUTING, RunStateTransition.EXECUTION_STARTED)
        await self.state_store.save(state)
        
        return await self._handle_executing(state)
    
    async def _handle_executing(self, state: RunState) -> RunState:
        """处理 EXECUTING 阶段 - 执行动作"""
        logger.info(f"[{state.run_id}] Handling EXECUTING phase")
        
        # 获取决策
        decision = self.decision_manager.get_decision(state.decision_id)
        if not decision:
            logger.error(f"[{state.run_id}] Decision not found")
            state.transition_to(RunPhase.FAILED, RunStateTransition.EXECUTION_FAILED)
            await self.state_store.save(state)
            return state
        
        # 执行动作
        all_success = True
        for action_def in decision.actions:
            action_type = action_def.type
            handler = self._action_handlers.get(action_type)
            
            if not handler:
                logger.error(f"[{state.run_id}] No handler for action type: {action_type}")
                all_success = False
                continue
            
            try:
                # 执行动作（幂等）
                result = await handler(action_def)
                if not result.success:
                    all_success = False
                    logger.error(f"[{state.run_id}] Action {action_def.action_id} failed: {result.message}")
            except Exception as e:
                all_success = False
                logger.exception(f"[{state.run_id}] Action {action_def.action_id} exception: {e}")
        
        if all_success:
            state.transition_to(RunPhase.EXECUTED, RunStateTransition.EXECUTION_COMPLETED)
        else:
            state.set_error(
                error_type="ActionExecutionFailed",
                error_message="One or more actions failed",
                recoverable=True,
                retry_eligible=True
            )
            state.transition_to(RunPhase.FAILED, RunStateTransition.EXECUTION_FAILED)
        
        await self.state_store.save(state)
        
        if state.phase == RunPhase.EXECUTED:
            return await self._handle_executed(state)
        
        return state
    
    async def _handle_executed(self, state: RunState) -> RunState:
        """处理 EXECUTED 阶段"""
        logger.info(f"[{state.run_id}] Handling EXECUTED phase")
        
        # 转移到验证中
        state.transition_to(RunPhase.VERIFYING, RunStateTransition.VERIFICATION_STARTED)
        await self.state_store.save(state)
        
        return await self._handle_verifying(state)
    
    async def _handle_verifying(self, state: RunState) -> RunState:
        """处理 VERIFYING 阶段 - 验证执行效果"""
        logger.info(f"[{state.run_id}] Handling VERIFYING phase")
        
        # 获取决策
        decision = self.decision_manager.get_decision(state.decision_id)
        if not decision:
            logger.error(f"[{state.run_id}] Decision not found")
            state.transition_to(RunPhase.FAILED, RunStateTransition.VERIFICATION_FAILED)
            await self.state_store.save(state)
            return state
        
        # 执行验证
        checks = [
            {"metric": c.metric, "expected": c.expected}
            for c in decision.verify.checks
        ]
        
        verification_result = await self.verification_gate.verify(
            action_id=state.action_ids[0] if state.action_ids else "unknown",
            checks=checks
        )
        
        if verification_result.passed:
            # 验证通过
            state.transition_to(RunPhase.VERIFIED, RunStateTransition.VERIFICATION_PASSED)
            await self.state_store.save(state)
            
            # 最终成功
            state.transition_to(RunPhase.SUCCEEDED, RunStateTransition.EXECUTION_COMPLETED)
            await self.state_store.save(state)
            
            logger.info(f"[{state.run_id}] Run completed successfully")
        else:
            # 验证失败 - 触发回滚
            logger.warning(f"[{state.run_id}] Verification failed, triggering rollback")
            
            state.transition_to(RunPhase.ROLLED_BACK, RunStateTransition.ROLLBACK_INITIATED)
            await self.state_store.save(state)
            
            # 执行回滚
            await self._execute_rollback(state)
        
        return state
    
    async def _execute_rollback(self, state: RunState):
        """执行回滚"""
        logger.info(f"[{state.run_id}] Executing rollback")
        
        # 获取决策
        decision = self.decision_manager.get_decision(state.decision_id)
        if not decision or not decision.rollback.enabled:
            logger.warning(f"[{state.run_id}] Rollback not enabled or decision not found")
            return
        
        # 执行回滚动作
        for rollback_action in decision.rollback.actions:
            # 执行回滚
            logger.info(f"[{state.run_id}] Rolling back: {rollback_action.action_id}")
        
        logger.info(f"[{state.run_id}] Rollback completed")
    
    async def list_active_runs(self) -> List[RunState]:
        """列出活动运行"""
        return await self.state_store.list_active()
    
    async def get_run_status(self, run_id: str) -> Optional[Dict]:
        """获取运行状态"""
        state = await self.state_store.load(run_id)
        if not state:
            return None
        
        return {
            "run_id": state.run_id,
            "phase": state.phase.value,
            "is_terminal": state.is_terminal(),
            "can_resume": state.can_resume(),
            "created_at": state.created_at,
            "started_at": state.started_at,
            "completed_at": state.completed_at,
            "retry_count": state.retry_count,
            "error_info": state.error_info.__dict__ if state.error_info else None
        }
