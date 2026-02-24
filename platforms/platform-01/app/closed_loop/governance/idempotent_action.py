"""
幂等动作系统 (Idempotent Action)

强制治理规范核心组件
每个 Action 必须实现 apply/rollback/verify 三段式接口
"""

import asyncio
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


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


class ActionType(Enum):
    """动作类型"""
    RESTART_SERVICE = "restart_service"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SWITCH_TRAFFIC = "switch_traffic"
    CLEAR_CACHE = "clear_cache"
    ROLLBACK_CONFIG = "rollback_config"
    CUSTOM_SCRIPT = "custom_script"


@dataclass
class ActionResult:
    """动作执行结果"""
    action_id: str
    state: ActionState
    success: bool
    message: str
    output: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: float = 0.0
    error: Optional[str] = None
    
    @classmethod
    def success_result(
        cls,
        action_id: str,
        message: str = "",
        output: Optional[Dict] = None
    ) -> "ActionResult":
        return cls(
            action_id=action_id,
            state=ActionState.SUCCESS,
            success=True,
            message=message,
            output=output or {}
        )
    
    @classmethod
    def failed_result(
        cls,
        action_id: str,
        error: str,
        message: str = ""
    ) -> "ActionResult":
        return cls(
            action_id=action_id,
            state=ActionState.FAILED,
            success=False,
            message=message,
            error=error
        )
    
    @classmethod
    def skipped_result(cls, action_id: str, reason: str) -> "ActionResult":
        return cls(
            action_id=action_id,
            state=ActionState.VERIFIED,  # 跳过视为已验证
            success=True,
            message=f"Skipped: {reason}"
        )


@dataclass
class VerificationResult:
    """验证结果"""
    action_id: str
    passed: bool
    checks: List[Dict[str, Any]]
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @classmethod
    def success(cls, action_id: str, checks: List[Dict]) -> "VerificationResult":
        return cls(
            action_id=action_id,
            passed=True,
            checks=checks,
            message="All verification checks passed"
        )
    
    @classmethod
    def failed(
        cls,
        action_id: str,
        checks: List[Dict],
        message: str
    ) -> "VerificationResult":
        return cls(
            action_id=action_id,
            passed=False,
            checks=checks,
            message=message
        )


class IdempotentAction(ABC):
    """
    幂等动作基类
    
    每个 Action 必须实现:
    - apply(): 执行动作（必须幂等）
    - rollback(): 回滚动作（必须幂等）
    - verify(): 验证动作效果
    """
    
    def __init__(
        self,
        action_id: str,
        action_type: ActionType,
        target: str,
        params: Dict[str, Any]
    ):
        self.action_id = action_id
        self.action_type = action_type
        self.target = target
        self.params = params
        self.state = ActionState.PENDING
        self._execution_record: Optional[ActionResult] = None
        self._rollback_record: Optional[ActionResult] = None
        self._verification_record: Optional[VerificationResult] = None
    
    # ==================== 核心接口 ====================
    
    async def apply(self) -> ActionResult:
        """
        执行动作 - 必须幂等
        相同 action_id 多次调用结果一致
        """
        start_time = datetime.now()
        
        # 1. 检查是否已执行成功
        if self._is_executed_successfully():
            logger.info(f"Action {self.action_id} already executed successfully, returning cached result")
            return self._get_execution_result()
        
        # 2. 检查是否正在执行
        if self.state == ActionState.EXECUTING:
            logger.warning(f"Action {self.action_id} is already executing")
            return ActionResult(
                action_id=self.action_id,
                state=ActionState.EXECUTING,
                success=False,
                message="Action is already executing"
            )
        
        # 3. 执行动作
        self.state = ActionState.EXECUTING
        try:
            result = await self._do_apply()
            self.state = ActionState.SUCCESS if result.success else ActionState.FAILED
            self._execution_record = result
            
            # 计算执行时间
            duration = (datetime.now() - start_time).total_seconds() * 1000
            result.duration_ms = duration
            
            logger.info(f"Action {self.action_id} executed: {result.state.value}")
            return result
            
        except Exception as e:
            self.state = ActionState.FAILED
            error_msg = str(e)
            logger.exception(f"Action {self.action_id} failed: {error_msg}")
            
            result = ActionResult.failed_result(
                action_id=self.action_id,
                error=error_msg,
                message="Execution failed with exception"
            )
            result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._execution_record = result
            return result
    
    async def rollback(self) -> ActionResult:
        """
        回滚动作 - 必须幂等
        """
        start_time = datetime.now()
        
        # 1. 检查是否已执行
        if not self._is_executed():
            logger.info(f"Action {self.action_id} not executed, skip rollback")
            return ActionResult.skipped_result(self.action_id, "Not executed")
        
        # 2. 检查是否已回滚
        if self._is_rolled_back():
            logger.info(f"Action {self.action_id} already rolled back, returning cached result")
            return self._get_rollback_result()
        
        # 3. 执行回滚
        self.state = ActionState.ROLLBACK_INITIATED
        try:
            result = await self._do_rollback()
            self.state = ActionState.ROLLED_BACK if result.success else ActionState.FAILED
            self._rollback_record = result
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            result.duration_ms = duration
            
            logger.info(f"Action {self.action_id} rolled back: {result.state.value}")
            return result
            
        except Exception as e:
            self.state = ActionState.FAILED
            error_msg = str(e)
            logger.exception(f"Action {self.action_id} rollback failed: {error_msg}")
            
            result = ActionResult.failed_result(
                action_id=self.action_id,
                error=error_msg,
                message="Rollback failed with exception"
            )
            result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._rollback_record = result
            return result
    
    async def verify(self) -> VerificationResult:
        """
        验证动作效果
        """
        # 1. 检查是否已执行
        if not self._is_executed():
            return VerificationResult.failed(
                action_id=self.action_id,
                checks=[],
                message="Action not executed, cannot verify"
            )
        
        # 2. 执行验证
        try:
            result = await self._do_verify()
            self._verification_record = result
            
            if result.passed:
                self.state = ActionState.VERIFIED
            
            logger.info(f"Action {self.action_id} verified: {result.passed}")
            return result
            
        except Exception as e:
            logger.exception(f"Action {self.action_id} verification failed: {e}")
            return VerificationResult.failed(
                action_id=self.action_id,
                checks=[],
                message=f"Verification failed: {str(e)}"
            )
    
    # ==================== 抽象方法 - 子类必须实现 ====================
    
    @abstractmethod
    async def _do_apply(self) -> ActionResult:
        """实际执行逻辑 - 子类实现"""
        pass
    
    @abstractmethod
    async def _do_rollback(self) -> ActionResult:
        """实际回滚逻辑 - 子类实现"""
        pass
    
    @abstractmethod
    async def _do_verify(self) -> VerificationResult:
        """实际验证逻辑 - 子类实现"""
        pass
    
    # ==================== 状态查询 ====================
    
    def _is_executed(self) -> bool:
        """检查是否已执行"""
        return self._execution_record is not None
    
    def _is_executed_successfully(self) -> bool:
        """检查是否已成功执行"""
        return (
            self._execution_record is not None and
            self._execution_record.success and
            self.state in [ActionState.SUCCESS, ActionState.VERIFIED]
        )
    
    def _is_rolled_back(self) -> bool:
        """检查是否已回滚"""
        return (
            self._rollback_record is not None and
            self._rollback_record.success and
            self.state == ActionState.ROLLED_BACK
        )
    
    def _get_execution_result(self) -> Optional[ActionResult]:
        """获取执行结果"""
        return self._execution_record
    
    def _get_rollback_result(self) -> Optional[ActionResult]:
        """获取回滚结果"""
        return self._rollback_record
    
    def get_state(self) -> ActionState:
        """获取当前状态"""
        return self.state
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "target": self.target,
            "params": self.params,
            "state": self.state.value,
            "execution": self._execution_record.__dict__ if self._execution_record else None,
            "rollback": self._rollback_record.__dict__ if self._rollback_record else None,
            "verification": self._verification_record.__dict__ if self._verification_record else None
        }


# ==================== 具体动作实现 ====================

class RestartServiceAction(IdempotentAction):
    """重启服务动作"""
    
    def __init__(self, action_id: str, target: str, params: Dict[str, Any]):
        super().__init__(action_id, ActionType.RESTART_SERVICE, target, params)
    
    async def _do_apply(self) -> ActionResult:
        """执行重启"""
        graceful = self.params.get("graceful", True)
        
        # 模拟重启逻辑
        await asyncio.sleep(1)
        
        return ActionResult.success_result(
            action_id=self.action_id,
            message=f"Service {self.target} restarted (graceful={graceful})",
            output={"restart_time": datetime.now().isoformat()}
        )
    
    async def _do_rollback(self) -> ActionResult:
        """回滚重启 - 通常不需要回滚"""
        return ActionResult.success_result(
            action_id=self.action_id,
            message=f"Service restart rollback - no action needed for {self.target}"
        )
    
    async def _do_verify(self) -> VerificationResult:
        """验证重启效果"""
        # 模拟验证逻辑
        checks = [
            {"metric": "service_health", "expected": "healthy", "actual": "healthy", "passed": True},
            {"metric": "error_rate", "expected": "< 0.01", "actual": "0.005", "passed": True}
        ]
        
        all_passed = all(c["passed"] for c in checks)
        
        if all_passed:
            return VerificationResult.success(self.action_id, checks)
        else:
            return VerificationResult.failed(
                self.action_id,
                checks,
                "Some verification checks failed"
            )


class ScaleAction(IdempotentAction):
    """扩缩容动作"""
    
    def __init__(self, action_id: str, target: str, params: Dict[str, Any]):
        action_type = ActionType.SCALE_UP if params.get("direction") == "up" else ActionType.SCALE_DOWN
        super().__init__(action_id, action_type, target, params)
        self._original_replicas: Optional[int] = None
    
    async def _do_apply(self) -> ActionResult:
        """执行扩缩容"""
        replicas = self.params.get("replicas", 1)
        
        # 保存原始副本数（用于回滚）
        self._original_replicas = self.params.get("original_replicas", 1)
        
        # 模拟扩缩容逻辑
        await asyncio.sleep(0.5)
        
        return ActionResult.success_result(
            action_id=self.action_id,
            message=f"Service {self.target} scaled to {replicas} replicas",
            output={
                "target_replicas": replicas,
                "original_replicas": self._original_replicas
            }
        )
    
    async def _do_rollback(self) -> ActionResult:
        """回滚扩缩容 - 恢复到原始副本数"""
        if self._original_replicas is None:
            return ActionResult.failed_result(
                action_id=self.action_id,
                error="Original replicas not recorded",
                message="Cannot rollback scale action"
            )
        
        # 模拟回滚逻辑
        await asyncio.sleep(0.5)
        
        return ActionResult.success_result(
            action_id=self.action_id,
            message=f"Service {self.target} scaled back to {self._original_replicas} replicas",
            output={"restored_replicas": self._original_replicas}
        )
    
    async def _do_verify(self) -> VerificationResult:
        """验证扩缩容效果"""
        checks = [
            {"metric": "replica_count", "expected": str(self.params.get("replicas")), "actual": str(self.params.get("replicas")), "passed": True},
            {"metric": "pod_health", "expected": "all_ready", "actual": "all_ready", "passed": True}
        ]
        
        return VerificationResult.success(self.action_id, checks)


class ActionFactory:
    """动作工厂"""
    
    _registry: Dict[ActionType, type] = {
        ActionType.RESTART_SERVICE: RestartServiceAction,
        ActionType.SCALE_UP: ScaleAction,
        ActionType.SCALE_DOWN: ScaleAction,
    }
    
    @classmethod
    def register(cls, action_type: ActionType, action_class: type):
        """注册动作类型"""
        cls._registry[action_type] = action_class
    
    @classmethod
    def create(
        cls,
        action_type: ActionType,
        action_id: str,
        target: str,
        params: Dict[str, Any]
    ) -> IdempotentAction:
        """创建动作实例"""
        action_class = cls._registry.get(action_type)
        if not action_class:
            raise ValueError(f"Unknown action type: {action_type}")
        
        return action_class(action_id, target, params)


class ActionStateMachine:
    """
    动作状态机
    管理动作的状态流转
    """
    
    # 状态转移规则
    TRANSITIONS: Dict[ActionState, List[ActionState]] = {
        ActionState.PENDING: [ActionState.APPROVED, ActionState.REJECTED],
        ActionState.APPROVED: [ActionState.EXECUTING],
        ActionState.EXECUTING: [ActionState.SUCCESS, ActionState.FAILED],
        ActionState.SUCCESS: [ActionState.VERIFIED, ActionState.ROLLBACK_INITIATED],
        ActionState.FAILED: [ActionState.ROLLBACK_INITIATED],
        ActionState.VERIFIED: [],  # 终态
        ActionState.ROLLBACK_INITIATED: [ActionState.ROLLED_BACK, ActionState.FAILED],
        ActionState.ROLLED_BACK: [],  # 终态
        ActionState.REJECTED: [],  # 终态
    }
    
    def __init__(self, action: IdempotentAction):
        self.action = action
    
    def can_transition(self, new_state: ActionState) -> bool:
        """检查是否可以转移到新状态"""
        current = self.action.get_state()
        allowed = self.TRANSITIONS.get(current, [])
        return new_state in allowed
    
    def transition(self, new_state: ActionState) -> bool:
        """执行状态转移"""
        if not self.can_transition(new_state):
            logger.warning(
                f"Invalid state transition: {self.action.get_state().value} -> {new_state.value}"
            )
            return False
        
        self.action.state = new_state
        logger.info(f"Action {self.action.action_id} transitioned to {new_state.value}")
        return True
    
    def get_allowed_transitions(self) -> List[ActionState]:
        """获取允许的转移状态"""
        return self.TRANSITIONS.get(self.action.get_state(), [])
    
    def is_terminal(self) -> bool:
        """检查是否为终态"""
        return len(self.get_allowed_transitions()) == 0
