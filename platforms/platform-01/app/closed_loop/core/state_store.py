"""
状态存储系统 (State Store)

可恢复状态机核心组件
每个 phase/step 写入可持久化的 run state
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class RunPhase(Enum):
    """运行阶段"""
    NEW = "NEW"
    DETECTED = "DETECTED"
    ANALYZED = "ANALYZED"
    PLANNED = "PLANNED"
    APPROVAL_PENDING = "APPROVAL_PENDING"
    APPROVED = "APPROVED"
    EXECUTING = "EXECUTING"
    EXECUTED = "EXECUTED"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"
    ROLLED_BACK = "ROLLED_BACK"
    SUCCEEDED = "SUCCEEDED"
    ESCALATED = "ESCALATED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RunStateTransition(Enum):
    """状态转移原因"""
    ANOMALY_DETECTED = "anomaly_detected"
    RCA_COMPLETED = "rca_completed"
    DECISION_PLANNED = "decision_planned"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_REJECTED = "approval_rejected"
    APPROVAL_TIMEOUT = "approval_timeout"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_PASSED = "verification_passed"
    VERIFICATION_FAILED = "verification_failed"
    ROLLBACK_INITIATED = "rollback_initiated"
    ROLLBACK_COMPLETED = "rollback_completed"
    ESCALATION_TRIGGERED = "escalation_triggered"
    RUN_CANCELLED = "run_cancelled"


@dataclass
class PhaseTiming:
    """阶段耗时"""
    phase: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class StateHistoryEntry:
    """状态历史记录"""
    phase: str
    timestamp: str
    transition_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorInfo:
    """错误信息"""
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    recoverable: bool = False
    retry_eligible: bool = False


@dataclass
class RunState:
    """
    运行状态 - 可持久化
    
    包含所有恢复运行所需的信息
    """
    run_id: str
    trace_id: str
    phase: RunPhase
    previous_phase: Optional[RunPhase] = None
    state_history: List[StateHistoryEntry] = field(default_factory=list)
    
    # 时间戳
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    phase_timings: Dict[str, float] = field(default_factory=dict)
    
    # 哈希与引用
    inputs_hash: str = ""
    artifacts_uri: Dict[str, str] = field(default_factory=dict)
    
    # 关联ID
    decision_id: Optional[str] = None
    action_ids: List[str] = field(default_factory=list)
    approval_request_id: Optional[str] = None
    verification_id: Optional[str] = None
    
    # 执行信息
    retry_count: int = 0
    error_info: Optional[ErrorInfo] = None
    
    # 版本
    version: str = "1.0.0"
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['phase'] = self.phase.value
        data['previous_phase'] = self.previous_phase.value if self.previous_phase else None
        return data
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def compute_hash(self) -> str:
        """计算状态哈希"""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def transition_to(
        self,
        new_phase: RunPhase,
        reason: RunStateTransition,
        metadata: Optional[Dict] = None
    ):
        """状态转移"""
        # 记录历史
        self.state_history.append(StateHistoryEntry(
            phase=self.phase.value,
            timestamp=datetime.now().isoformat(),
            transition_reason=reason.value,
            metadata=metadata or {}
        ))
        
        # 记录阶段耗时
        if self.phase != RunPhase.NEW:
            current_time = datetime.now()
            last_entry = self.state_history[-1] if self.state_history else None
            if last_entry:
                last_time = datetime.fromisoformat(last_entry.timestamp)
                duration = (current_time - last_time).total_seconds() * 1000
                self.phase_timings[self.phase.value] = duration
        
        # 执行转移
        self.previous_phase = self.phase
        self.phase = new_phase
        self.last_updated = datetime.now().isoformat()
        
        # 记录开始/完成时间
        if new_phase == RunPhase.EXECUTING and not self.started_at:
            self.started_at = datetime.now().isoformat()
        
        if new_phase in [RunPhase.SUCCEEDED, RunPhase.FAILED, RunPhase.ROLLED_BACK, RunPhase.ESCALATED]:
            self.completed_at = datetime.now().isoformat()
        
        logger.info(f"Run {self.run_id} transitioned: {self.previous_phase.value} -> {new_phase.value} ({reason.value})")
    
    def is_terminal(self) -> bool:
        """检查是否为终态"""
        return self.phase in [
            RunPhase.SUCCEEDED,
            RunPhase.FAILED,
            RunPhase.ROLLED_BACK,
            RunPhase.ESCALATED,
            RunPhase.CANCELLED
        ]
    
    def can_resume(self) -> bool:
        """检查是否可恢复"""
        return (
            not self.is_terminal() and
            self.phase not in [RunPhase.NEW, RunPhase.CANCELLED]
        )
    
    def set_error(
        self,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        recoverable: bool = False,
        retry_eligible: bool = False
    ):
        """设置错误信息"""
        self.error_info = ErrorInfo(
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            recoverable=recoverable,
            retry_eligible=retry_eligible
        )


class StateStore:
    """
    状态存储基类
    
    支持多种存储后端
    """
    
    async def save(self, state: RunState) -> bool:
        """保存状态"""
        raise NotImplementedError
    
    async def load(self, run_id: str) -> Optional[RunState]:
        """加载状态"""
        raise NotImplementedError
    
    async def list_active(self) -> List[RunState]:
        """列出活动状态"""
        raise NotImplementedError
    
    async def delete(self, run_id: str) -> bool:
        """删除状态"""
        raise NotImplementedError


class InMemoryStateStore(StateStore):
    """内存状态存储（用于测试）"""
    
    def __init__(self):
        self._states: Dict[str, RunState] = {}
    
    async def save(self, state: RunState) -> bool:
        """保存状态"""
        self._states[state.run_id] = state
        logger.debug(f"State saved: {state.run_id}")
        return True
    
    async def load(self, run_id: str) -> Optional[RunState]:
        """加载状态"""
        return self._states.get(run_id)
    
    async def list_active(self) -> List[RunState]:
        """列出活动状态"""
        return [
            state for state in self._states.values()
            if not state.is_terminal()
        ]
    
    async def delete(self, run_id: str) -> bool:
        """删除状态"""
        if run_id in self._states:
            del self._states[run_id]
            return True
        return False


class FileStateStore(StateStore):
    """文件状态存储"""
    
    def __init__(self, base_path: str = "/tmp/closed_loop_states"):
        self.base_path = base_path
        import os
        os.makedirs(base_path, exist_ok=True)
    
    async def save(self, state: RunState) -> bool:
        """保存状态到文件"""
        import os
        filepath = os.path.join(self.base_path, f"{state.run_id}.json")
        
        try:
            with open(filepath, 'w') as f:
                f.write(state.to_json())
            logger.debug(f"State saved to file: {filepath}")
            return True
        except Exception as e:
            logger.exception(f"Failed to save state: {e}")
            return False
    
    async def load(self, run_id: str) -> Optional[RunState]:
        """从文件加载状态"""
        import os
        filepath = os.path.join(self.base_path, f"{run_id}.json")
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # 重建 RunState 对象
            return self._deserialize(data)
        except Exception as e:
            logger.exception(f"Failed to load state: {e}")
            return None
    
    async def list_active(self) -> List[RunState]:
        """列出活动状态"""
        import os
        import glob
        
        active_states = []
        pattern = os.path.join(self.base_path, "run_*.json")
        
        for filepath in glob.glob(pattern):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                state = self._deserialize(data)
                if not state.is_terminal():
                    active_states.append(state)
            except Exception as e:
                logger.warning(f"Failed to load state from {filepath}: {e}")
        
        return active_states
    
    async def delete(self, run_id: str) -> bool:
        """删除状态文件"""
        import os
        filepath = os.path.join(self.base_path, f"{run_id}.json")
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
    
    def _deserialize(self, data: Dict) -> RunState:
        """反序列化 RunState"""
        state = RunState(
            run_id=data['run_id'],
            trace_id=data['trace_id'],
            phase=RunPhase(data['phase']),
            previous_phase=RunPhase(data['previous_phase']) if data.get('previous_phase') else None,
            created_at=data['created_at'],
            started_at=data.get('started_at'),
            completed_at=data.get('completed_at'),
            last_updated=data['last_updated'],
            inputs_hash=data.get('inputs_hash', ''),
            artifacts_uri=data.get('artifacts_uri', {}),
            decision_id=data.get('decision_id'),
            action_ids=data.get('action_ids', []),
            approval_request_id=data.get('approval_request_id'),
            verification_id=data.get('verification_id'),
            retry_count=data.get('retry_count', 0),
            version=data.get('version', '1.0.0'),
            metadata=data.get('metadata', {})
        )
        
        # 恢复历史记录
        for entry_data in data.get('state_history', []):
            state.state_history.append(StateHistoryEntry(**entry_data))
        
        # 恢复阶段耗时
        state.phase_timings = data.get('phase_timings', {})
        
        # 恢复错误信息
        if data.get('error_info'):
            state.error_info = ErrorInfo(**data['error_info'])
        
        return state


class StateStoreFactory:
    """状态存储工厂"""
    
    @staticmethod
    def create(store_type: str, **kwargs) -> StateStore:
        """创建状态存储实例"""
        if store_type == "memory":
            return InMemoryStateStore()
        elif store_type == "file":
            return FileStateStore(**kwargs)
        else:
            raise ValueError(f"Unknown store type: {store_type}")
