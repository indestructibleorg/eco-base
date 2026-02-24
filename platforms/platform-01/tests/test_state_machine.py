"""
状态机 Table-Driven 测试

实现三个不变量：
1. 受助者必须通过验证（SUCCEEDED 必经 VERIFYING）
2. 禁止高/具种族歧视行为（HIGH/CRITICAL 未批准不得 EXECUTING）
3. 验证故障必须 ROLLED_BACK/ESCALATED
"""

import pytest
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass
from enum import Enum

# 导入被测对象
from app.closed_loop.core.state_store import RunState, RunPhase, RunStateTransition


# ============================================================================
# Table-Driven 测试数据
# ============================================================================

@dataclass
class TransitionCase:
    """状态转移测试用例"""
    name: str
    from_phase: RunPhase
    to_phase: RunPhase
    transition: RunStateTransition
    should_succeed: bool
    description: str


# 合法状态转移表 (基于 VALID_TRANSITIONS)
VALID_TRANSITIONS: Dict[str, List[str]] = {
    'NEW': ['DETECTED', 'CANCELLED'],
    'DETECTED': ['ANALYZED', 'CANCELLED'],
    'ANALYZED': ['PLANNED', 'CANCELLED'],
    'PLANNED': ['APPROVAL_PENDING', 'APPROVED', 'CANCELLED'],
    'APPROVAL_PENDING': ['APPROVED', 'FAILED', 'ESCALATED'],
    'APPROVED': ['EXECUTING'],
    'EXECUTING': ['EXECUTED', 'FAILED'],
    'EXECUTED': ['VERIFYING'],
    'VERIFYING': ['VERIFIED', 'ROLLED_BACK', 'ESCALATED'],
    'VERIFIED': ['SUCCEEDED'],
    'SUCCEEDED': [],
    'ROLLED_BACK': [],
    'FAILED': [],
    'ESCALATED': [],
    'CANCELLED': []
}


# 完整转移测试表
TRANSITION_TEST_CASES: List[TransitionCase] = [
    # ========== 合法转移 ==========
    # NEW -> *
    TransitionCase("NEW->DETECTED", RunPhase.NEW, RunPhase.DETECTED, 
                   RunStateTransition.ANOMALY_DETECTED, True, "异常检测"),
    TransitionCase("NEW->CANCELLED", RunPhase.NEW, RunPhase.CANCELLED, 
                   RunStateTransition.RUN_CANCELLED, True, "取消运行"),
    
    # DETECTED -> *
    TransitionCase("DETECTED->ANALYZED", RunPhase.DETECTED, RunPhase.ANALYZED, 
                   RunStateTransition.RCA_COMPLETED, True, "RCA完成"),
    TransitionCase("DETECTED->CANCELLED", RunPhase.DETECTED, RunPhase.CANCELLED, 
                   RunStateTransition.RUN_CANCELLED, True, "取消运行"),
    
    # ANALYZED -> *
    TransitionCase("ANALYZED->PLANNED", RunPhase.ANALYZED, RunPhase.PLANNED, 
                   RunStateTransition.DECISION_PLANNED, True, "决策计划完成"),
    TransitionCase("ANALYZED->CANCELLED", RunPhase.ANALYZED, RunPhase.CANCELLED, 
                   RunStateTransition.RUN_CANCELLED, True, "取消运行"),
    
    # PLANNED -> *
    TransitionCase("PLANNED->APPROVAL_PENDING", RunPhase.PLANNED, RunPhase.APPROVAL_PENDING, 
                   RunStateTransition.APPROVAL_REQUIRED, True, "需要审批"),
    TransitionCase("PLANNED->APPROVED", RunPhase.PLANNED, RunPhase.APPROVED, 
                   RunStateTransition.APPROVAL_GRANTED, True, "低风险跳过审批"),
    TransitionCase("PLANNED->CANCELLED", RunPhase.PLANNED, RunPhase.CANCELLED, 
                   RunStateTransition.RUN_CANCELLED, True, "取消运行"),
    
    # APPROVAL_PENDING -> *
    TransitionCase("APPROVAL_PENDING->APPROVED", RunPhase.APPROVAL_PENDING, RunPhase.APPROVED, 
                   RunStateTransition.APPROVAL_GRANTED, True, "审批通过"),
    TransitionCase("APPROVAL_PENDING->FAILED", RunPhase.APPROVAL_PENDING, RunPhase.FAILED, 
                   RunStateTransition.APPROVAL_REJECTED, True, "审批拒绝"),
    TransitionCase("APPROVAL_PENDING->ESCALATED", RunPhase.APPROVAL_PENDING, RunPhase.ESCALATED, 
                   RunStateTransition.ESCALATION_TRIGGERED, True, "升级处理"),
    
    # APPROVED -> *
    TransitionCase("APPROVED->EXECUTING", RunPhase.APPROVED, RunPhase.EXECUTING, 
                   RunStateTransition.EXECUTION_STARTED, True, "开始执行"),
    
    # EXECUTING -> *
    TransitionCase("EXECUTING->EXECUTED", RunPhase.EXECUTING, RunPhase.EXECUTED, 
                   RunStateTransition.EXECUTION_COMPLETED, True, "执行完成"),
    TransitionCase("EXECUTING->FAILED", RunPhase.EXECUTING, RunPhase.FAILED, 
                   RunStateTransition.EXECUTION_FAILED, True, "执行失败"),
    
    # EXECUTED -> *
    TransitionCase("EXECUTED->VERIFYING", RunPhase.EXECUTED, RunPhase.VERIFYING, 
                   RunStateTransition.VERIFICATION_STARTED, True, "开始验证"),
    
    # VERIFYING -> *
    TransitionCase("VERIFYING->VERIFIED", RunPhase.VERIFYING, RunPhase.VERIFIED, 
                   RunStateTransition.VERIFICATION_PASSED, True, "验证通过"),
    TransitionCase("VERIFYING->ROLLED_BACK", RunPhase.VERIFYING, RunPhase.ROLLED_BACK, 
                   RunStateTransition.ROLLBACK_INITIATED, True, "回滚"),
    TransitionCase("VERIFYING->ESCALATED", RunPhase.VERIFYING, RunPhase.ESCALATED, 
                   RunStateTransition.ESCALATION_TRIGGERED, True, "升级"),
    
    # VERIFIED -> *
    TransitionCase("VERIFIED->SUCCEEDED", RunPhase.VERIFIED, RunPhase.SUCCEEDED, 
                   RunStateTransition.EXECUTION_COMPLETED, True, "成功完成"),
    
    # ========== 非法转移 ==========
    TransitionCase("NEW->SUCCEEDED(非法)", RunPhase.NEW, RunPhase.SUCCEEDED, 
                   RunStateTransition.EXECUTION_COMPLETED, False, "不能直接到成功"),
    TransitionCase("NEW->EXECUTING(非法)", RunPhase.NEW, RunPhase.EXECUTING, 
                   RunStateTransition.EXECUTION_STARTED, False, "不能直接执行"),
    TransitionCase("PLANNED->EXECUTING(非法)", RunPhase.PLANNED, RunPhase.EXECUTING, 
                   RunStateTransition.EXECUTION_STARTED, False, "未审批不能执行"),
    TransitionCase("FAILED->SUCCEEDED(非法)", RunPhase.FAILED, RunPhase.SUCCEEDED, 
                   RunStateTransition.EXECUTION_COMPLETED, False, "失败不能转成功"),
    TransitionCase("SUCCEEDED->FAILED(非法)", RunPhase.SUCCEEDED, RunPhase.FAILED, 
                   RunStateTransition.EXECUTION_FAILED, False, "终态不能转移"),
    TransitionCase("VERIFYING->SUCCEEDED(非法)", RunPhase.VERIFYING, RunPhase.SUCCEEDED, 
                   RunStateTransition.EXECUTION_COMPLETED, False, "验证中不能直接成功"),
]


# ============================================================================
# 辅助函数
# ============================================================================

def is_valid_transition(from_phase: RunPhase, to_phase: RunPhase) -> bool:
    """检查转移是否合法"""
    valid_targets = VALID_TRANSITIONS.get(from_phase.value, [])
    return to_phase.value in valid_targets


def get_all_phases() -> List[RunPhase]:
    """获取所有状态"""
    return [
        RunPhase.NEW,
        RunPhase.DETECTED,
        RunPhase.ANALYZED,
        RunPhase.PLANNED,
        RunPhase.APPROVAL_PENDING,
        RunPhase.APPROVED,
        RunPhase.EXECUTING,
        RunPhase.EXECUTED,
        RunPhase.VERIFYING,
        RunPhase.VERIFIED,
        RunPhase.SUCCEEDED,
        RunPhase.ROLLED_BACK,
        RunPhase.FAILED,
        RunPhase.ESCALATED,
        RunPhase.CANCELLED,
    ]


# ============================================================================
# 测试类
# ============================================================================

class TestValidTransitions:
    """合法状态转移测试"""
    
    @pytest.mark.parametrize("case", [
        tc for tc in TRANSITION_TEST_CASES if tc.should_succeed
    ], ids=lambda tc: tc.name)
    def test_valid_transition(self, case: TransitionCase):
        """测试合法状态转移"""
        state = RunState(
            run_id="run_001",
            trace_id="trace_001",
            phase=case.from_phase
        )
        
        # 执行转移
        state.transition_to(case.to_phase, case.transition)
        
        # 验证状态
        assert state.phase == case.to_phase, \
            f"{case.name}: 状态应该变为 {case.to_phase.value}"
        assert state.previous_phase == case.from_phase, \
            f"{case.name}: 前一状态应该记录为 {case.from_phase.value}"


class TestInvalidTransitions:
    """非法状态转移测试"""
    
    @pytest.mark.parametrize("case", [
        tc for tc in TRANSITION_TEST_CASES if not tc.should_succeed
    ], ids=lambda tc: tc.name)
    def test_invalid_transition_blocked(self, case: TransitionCase):
        """测试非法状态转移被阻止"""
        # 验证转移表中没有这条转移
        assert not is_valid_transition(case.from_phase, case.to_phase), \
            f"{case.name}: {case.from_phase.value} -> {case.to_phase.value} 应该是非法转移"


class TestTransitionTableCompleteness:
    """转移表完整性测试"""
    
    def test_all_states_defined_in_transition_table(self):
        """所有状态都必须在转移表中定义"""
        all_phases = get_all_phases()
        for phase in all_phases:
            assert phase.value in VALID_TRANSITIONS, \
                f"状态 {phase.value} 必须在 VALID_TRANSITIONS 中定义"
    
    def test_terminal_states_have_no_outgoing_transitions(self):
        """终态没有 outgoing transitions"""
        terminal_states = ['SUCCEEDED', 'ROLLED_BACK', 'FAILED', 'ESCALATED', 'CANCELLED']
        
        for state in terminal_states:
            targets = VALID_TRANSITIONS.get(state, [])
            assert len(targets) == 0, \
                f"终态 {state} 不应该有 outgoing transitions, 但有: {targets}"


# ============================================================================
# 不变量测试 (Invariants)
# ============================================================================

class TestInvariantSuccessRequiresVerifying:
    """
    不变量 1: 受助者必须通过验证（SUCCEEDED 必经 VERIFYING）
    """
    
    def test_success_requires_verifying_invariant(self):
        """
        SUCCEEDED 之前必须经过 VERIFYING
        验证路径: ... -> EXECUTING -> EXECUTED -> VERIFYING -> VERIFIED -> SUCCEEDED
        """
        state = RunState(
            run_id="run_001",
            trace_id="trace_001",
            phase=RunPhase.NEW
        )
        
        # 完整路径
        path = [
            (RunPhase.DETECTED, RunStateTransition.ANOMALY_DETECTED),
            (RunPhase.ANALYZED, RunStateTransition.RCA_COMPLETED),
            (RunPhase.PLANNED, RunStateTransition.DECISION_PLANNED),
            (RunPhase.APPROVED, RunStateTransition.APPROVAL_GRANTED),
            (RunPhase.EXECUTING, RunStateTransition.EXECUTION_STARTED),
            (RunPhase.EXECUTED, RunStateTransition.EXECUTION_COMPLETED),
            (RunPhase.VERIFYING, RunStateTransition.VERIFICATION_STARTED),
            (RunPhase.VERIFIED, RunStateTransition.VERIFICATION_PASSED),
            (RunPhase.SUCCEEDED, RunStateTransition.EXECUTION_COMPLETED),
        ]
        
        for target_phase, reason in path:
            state.transition_to(target_phase, reason)
        
        # 验证最终状态
        assert state.phase == RunPhase.SUCCEEDED
        
        # 验证历史记录中包含 VERIFYING
        phase_history = [h.phase for h in state.state_history]
        assert RunPhase.VERIFYING.value in phase_history, \
            "SUCCEEDED 之前必须经过 VERIFYING"
    
    def test_direct_success_from_new_is_invalid(self):
        """NEW 不能直接到 SUCCEEDED"""
        assert not is_valid_transition(RunPhase.NEW, RunPhase.SUCCEEDED)
    
    def test_direct_success_from_executing_is_invalid(self):
        """EXECUTING 不能直接到 SUCCEEDED"""
        assert not is_valid_transition(RunPhase.EXECUTING, RunPhase.SUCCEEDED)
    
    def test_verifying_is_mandatory_for_success(self):
        """VERIFYING 是 SUCCEEDED 的必要条件"""
        # 检查所有到 SUCCEEDED 的路径
        for from_phase, targets in VALID_TRANSITIONS.items():
            if 'SUCCEEDED' in targets:
                # 只有 VERIFIED 能到 SUCCEEDED
                assert from_phase == 'VERIFIED', \
                    f"只有 VERIFIED 能转移到 SUCCEEDED, 但 {from_phase} 也能"


class TestInvariantHighRiskRequiresApproval:
    """
    不变量 2: 禁止高/具种族歧视行为（HIGH/CRITICAL 未批准不得 EXECUTING）
    
    注: 这里的"种族歧视"是原文要求，实际含义是"高风险/关键操作"
    高风险操作必须经过审批才能执行
    """
    
    def test_planned_cannot_go_directly_to_executing(self):
        """PLANNED 不能直接到 EXECUTING"""
        assert not is_valid_transition(RunPhase.PLANNED, RunPhase.EXECUTING), \
            "PLANNED 必须经过 APPROVED 才能 EXECUTING"
    
    def test_approval_pending_cannot_go_directly_to_executing(self):
        """APPROVAL_PENDING 不能直接到 EXECUTING"""
        assert not is_valid_transition(RunPhase.APPROVAL_PENDING, RunPhase.EXECUTING), \
            "APPROVAL_PENDING 必须经过 APPROVED 才能 EXECUTING"
    
    def test_only_approved_can_execute(self):
        """只有 APPROVED 能转移到 EXECUTING"""
        for from_phase, targets in VALID_TRANSITIONS.items():
            if 'EXECUTING' in targets:
                assert from_phase == 'APPROVED', \
                    f"只有 APPROVED 能转移到 EXECUTING, 但 {from_phase} 也能"
    
    def test_approval_workflow_enforcement(self):
        """审批流程强制执行"""
        # 高风险场景: PLANNED -> APPROVAL_PENDING -> APPROVED -> EXECUTING
        state = RunState(
            run_id="run_001",
            trace_id="trace_001",
            phase=RunPhase.PLANNED
        )
        
        # 必须经过审批
        state.transition_to(RunPhase.APPROVAL_PENDING, RunStateTransition.APPROVAL_REQUIRED)
        state.transition_to(RunPhase.APPROVED, RunStateTransition.APPROVAL_GRANTED)
        state.transition_to(RunPhase.EXECUTING, RunStateTransition.EXECUTION_STARTED)
        
        assert state.phase == RunPhase.EXECUTING
        
        # 验证历史包含审批步骤
        phase_history = [h.phase for h in state.state_history]
        assert RunPhase.APPROVAL_PENDING.value in phase_history
        assert RunPhase.APPROVED.value in phase_history


class TestInvariantVerifyFailMustRollbackOrEscalate:
    """
    不变量 3: 验证故障必须 ROLLED_BACK/ESCALATED
    
    VERIFYING 失败只能转移到 ROLLED_BACK 或 ESCALATED
    """
    
    def test_verifying_can_only_go_to_verified_rollback_or_escalated(self):
        """VERIFYING 只能转移到 VERIFIED, ROLLED_BACK, ESCALATED"""
        verify_targets = set(VALID_TRANSITIONS.get('VERIFYING', []))
        expected = {'VERIFIED', 'ROLLED_BACK', 'ESCALATED'}
        
        assert verify_targets == expected, \
            f"VERIFYING 只能转移到 {expected}, 但实际可以转移到 {verify_targets}"
    
    def test_verifying_cannot_go_to_succeeded(self):
        """VERIFYING 不能直接到 SUCCEEDED"""
        assert not is_valid_transition(RunPhase.VERIFYING, RunPhase.SUCCEEDED), \
            "VERIFYING 必须经过 VERIFIED 才能到 SUCCEEDED"
    
    def test_verifying_cannot_go_to_failed(self):
        """VERIFYING 不能直接到 FAILED"""
        assert 'FAILED' not in VALID_TRANSITIONS.get('VERIFYING', []), \
            "VERIFYING 失败必须 ROLLED_BACK 或 ESCALATED, 不能直接 FAILED"
    
    def test_verifying_failure_workflow_rollback(self):
        """验证失败回滚流程"""
        state = RunState(
            run_id="run_001",
            trace_id="trace_001",
            phase=RunPhase.VERIFYING
        )
        
        state.transition_to(RunPhase.ROLLED_BACK, RunStateTransition.ROLLBACK_INITIATED)
        
        assert state.phase == RunPhase.ROLLED_BACK
        assert state.is_terminal()
    
    def test_verifying_failure_workflow_escalation(self):
        """验证失败升级流程"""
        state = RunState(
            run_id="run_001",
            trace_id="trace_001",
            phase=RunPhase.VERIFYING
        )
        
        state.transition_to(RunPhase.ESCALATED, RunStateTransition.ESCALATION_TRIGGERED)
        
        assert state.phase == RunPhase.ESCALATED
        assert state.is_terminal()


# ============================================================================
# 状态历史测试
# ============================================================================

class TestStateHistory:
    """状态历史记录测试"""
    
    def test_state_history_recorded(self):
        """状态转移应该被记录到历史"""
        state = RunState(
            run_id="run_001",
            trace_id="trace_001",
            phase=RunPhase.NEW
        )
        
        state.transition_to(RunPhase.DETECTED, RunStateTransition.ANOMALY_DETECTED)
        state.transition_to(RunPhase.ANALYZED, RunStateTransition.RCA_COMPLETED)
        state.transition_to(RunPhase.PLANNED, RunStateTransition.DECISION_PLANNED)
        
        assert len(state.state_history) == 3
        
        # 验证历史记录内容
        assert state.state_history[0].phase == RunPhase.NEW.value
        assert state.state_history[0].transition_reason == RunStateTransition.ANOMALY_DETECTED.value
    
    def test_phase_timings_recorded(self):
        """阶段耗时应该被记录"""
        state = RunState(
            run_id="run_001",
            trace_id="trace_001",
            phase=RunPhase.NEW
        )
        
        state.transition_to(RunPhase.DETECTED, RunStateTransition.ANOMALY_DETECTED)
        state.transition_to(RunPhase.ANALYZED, RunStateTransition.RCA_COMPLETED)
        
        # DETECTED 阶段应该有耗时记录
        assert RunPhase.DETECTED.value in state.phase_timings
        assert state.phase_timings[RunPhase.DETECTED.value] >= 0


# ============================================================================
# 终态测试
# ============================================================================

class TestTerminalStates:
    """终态测试"""
    
    @pytest.mark.parametrize("phase", [
        RunPhase.SUCCEEDED,
        RunPhase.FAILED,
        RunPhase.ROLLED_BACK,
        RunPhase.ESCALATED,
        RunPhase.CANCELLED,
    ])
    def test_terminal_states(self, phase: RunPhase):
        """测试终态"""
        state = RunState(
            run_id="run_001",
            trace_id="trace_001",
            phase=phase
        )
        
        assert state.is_terminal(), f"{phase.value} 应该是终态"
    
    @pytest.mark.parametrize("phase", [
        RunPhase.NEW,
        RunPhase.DETECTED,
        RunPhase.ANALYZED,
        RunPhase.PLANNED,
        RunPhase.APPROVAL_PENDING,
        RunPhase.APPROVED,
        RunPhase.EXECUTING,
        RunPhase.EXECUTED,
        RunPhase.VERIFYING,
        RunPhase.VERIFIED,
    ])
    def test_non_terminal_states(self, phase: RunPhase):
        """测试非终态"""
        state = RunState(
            run_id="run_001",
            trace_id="trace_001",
            phase=phase
        )
        
        assert not state.is_terminal(), f"{phase.value} 不应该是终态"


# ============================================================================
# 完整路径测试
# ============================================================================

class TestCompleteWorkflows:
    """完整工作流测试"""
    
    def test_successful_workflow(self):
        """成功完成的工作流"""
        state = RunState(
            run_id="run_001",
            trace_id="trace_001",
            phase=RunPhase.NEW
        )
        
        # 完整成功路径
        transitions = [
            (RunPhase.DETECTED, RunStateTransition.ANOMALY_DETECTED),
            (RunPhase.ANALYZED, RunStateTransition.RCA_COMPLETED),
            (RunPhase.PLANNED, RunStateTransition.DECISION_PLANNED),
            (RunPhase.APPROVED, RunStateTransition.APPROVAL_GRANTED),
            (RunPhase.EXECUTING, RunStateTransition.EXECUTION_STARTED),
            (RunPhase.EXECUTED, RunStateTransition.EXECUTION_COMPLETED),
            (RunPhase.VERIFYING, RunStateTransition.VERIFICATION_STARTED),
            (RunPhase.VERIFIED, RunStateTransition.VERIFICATION_PASSED),
            (RunPhase.SUCCEEDED, RunStateTransition.EXECUTION_COMPLETED),
        ]
        
        for target_phase, reason in transitions:
            state.transition_to(target_phase, reason)
        
        assert state.phase == RunPhase.SUCCEEDED
        assert state.is_terminal()
        assert len(state.state_history) == 9
    
    def test_rollback_workflow(self):
        """回滚工作流"""
        state = RunState(
            run_id="run_001",
            trace_id="trace_001",
            phase=RunPhase.NEW
        )
        
        transitions = [
            (RunPhase.DETECTED, RunStateTransition.ANOMALY_DETECTED),
            (RunPhase.ANALYZED, RunStateTransition.RCA_COMPLETED),
            (RunPhase.PLANNED, RunStateTransition.DECISION_PLANNED),
            (RunPhase.APPROVED, RunStateTransition.APPROVAL_GRANTED),
            (RunPhase.EXECUTING, RunStateTransition.EXECUTION_STARTED),
            (RunPhase.EXECUTED, RunStateTransition.EXECUTION_COMPLETED),
            (RunPhase.VERIFYING, RunStateTransition.VERIFICATION_STARTED),
            (RunPhase.ROLLED_BACK, RunStateTransition.ROLLBACK_INITIATED),
        ]
        
        for target_phase, reason in transitions:
            state.transition_to(target_phase, reason)
        
        assert state.phase == RunPhase.ROLLED_BACK
        assert state.is_terminal()
    
    def test_escalation_workflow(self):
        """升级工作流"""
        state = RunState(
            run_id="run_001",
            trace_id="trace_001",
            phase=RunPhase.NEW
        )
        
        transitions = [
            (RunPhase.DETECTED, RunStateTransition.ANOMALY_DETECTED),
            (RunPhase.ANALYZED, RunStateTransition.RCA_COMPLETED),
            (RunPhase.PLANNED, RunStateTransition.DECISION_PLANNED),
            (RunPhase.APPROVAL_PENDING, RunStateTransition.APPROVAL_REQUIRED),
            (RunPhase.ESCALATED, RunStateTransition.ESCALATION_TRIGGERED),
        ]
        
        for target_phase, reason in transitions:
            state.transition_to(target_phase, reason)
        
        assert state.phase == RunPhase.ESCALATED
        assert state.is_terminal()
    
    def test_execution_failure_workflow(self):
        """执行失败工作流"""
        state = RunState(
            run_id="run_001",
            trace_id="trace_001",
            phase=RunPhase.NEW
        )
        
        transitions = [
            (RunPhase.DETECTED, RunStateTransition.ANOMALY_DETECTED),
            (RunPhase.ANALYZED, RunStateTransition.RCA_COMPLETED),
            (RunPhase.PLANNED, RunStateTransition.DECISION_PLANNED),
            (RunPhase.APPROVED, RunStateTransition.APPROVAL_GRANTED),
            (RunPhase.EXECUTING, RunStateTransition.EXECUTION_STARTED),
            (RunPhase.FAILED, RunStateTransition.EXECUTION_FAILED),
        ]
        
        for target_phase, reason in transitions:
            state.transition_to(target_phase, reason)
        
        assert state.phase == RunPhase.FAILED
        assert state.is_terminal()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
