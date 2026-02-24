#!/usr/bin/env python3
"""
Idempotency E2E Tests

幂等 E2E 測試：
- 同 run_id 重跑 5 次
- action_id 重跑
- 並發重跑
"""

import pytest
import asyncio
from app.closed_loop.governance import (
    RestartServiceAction,
    ScaleAction,
    ActionState,
    ActionStateMachine
)


class TestActionIdempotency:
    """Action 幂等性測試"""
    
    @pytest.mark.asyncio
    async def test_restart_action_idempotent(self):
        """I-02: 重啟服務 action 幂等性"""
        action = RestartServiceAction(
            action_id="act_restart_idem_001",
            target="order-service",
            params={"graceful": True}
        )
        
        # 第一次執行
        result1 = await action.apply()
        assert result1.success, "第一次執行應該成功"
        
        # 第二次執行（應該幂等）
        result2 = await action.apply()
        assert result2.success, "第二次執行應該成功（幂等）"
        
        # 驗證狀態
        assert action.get_state() == ActionState.SUCCESS
    
    @pytest.mark.asyncio
    async def test_scale_action_idempotent(self):
        """Scale action 幂等性"""
        action = ScaleAction(
            action_id="act_scale_idem_001",
            target="api-gateway",
            params={"replicas": 5, "direction": "up"}
        )
        
        # 多次執行
        for i in range(3):
            result = await action.apply()
            assert result.success, f"第 {i+1} 次執行應該成功"
        
        # 驗證狀態
        assert action.get_state() == ActionState.SUCCESS
    
    @pytest.mark.asyncio
    async def test_rollback_idempotent(self):
        """回滾操作幂等性"""
        action = RestartServiceAction(
            action_id="act_rollback_idem_001",
            target="payment-service",
            params={}
        )
        
        # 先執行
        await action.apply()
        
        # 第一次回滾
        result1 = await action.rollback()
        assert result1.success, "第一次回滾應該成功"
        
        # 第二次回滾（應該幂等）
        result2 = await action.rollback()
        assert result2.success, "第二次回滾應該成功（幂等）"
        
        # 驗證狀態
        assert action.get_state() == ActionState.ROLLED_BACK


class TestReplayIdempotency:
    """重跑幂等性測試"""
    
    @pytest.mark.asyncio
    async def test_same_action_id_replay_5_times(self):
        """I-01: 同 action_id 重跑 5 次"""
        action = RestartServiceAction(
            action_id="act_replay_001",
            target="inventory-service",
            params={}
        )
        
        results = []
        for i in range(5):
            result = await action.apply()
            results.append(result)
        
        # 所有執行都應該成功
        assert all(r.success for r in results), "所有重跑都應該成功"
        
        # 狀態應該是 SUCCESS
        assert action.get_state() == ActionState.SUCCESS


class TestConcurrentIdempotency:
    """並發幂等性測試"""
    
    @pytest.mark.asyncio
    async def test_concurrent_apply_same_action(self):
        """I-03: 兩個 worker 同時處理同 action"""
        action = RestartServiceAction(
            action_id="act_concurrent_001",
            target="user-service",
            params={}
        )
        
        async def execute():
            return await action.apply()
        
        # 並發執行
        results = await asyncio.gather(
            execute(),
            execute(),
            execute(),
            return_exceptions=True
        )
        
        # 至少一個成功
        success_count = sum(
            1 for r in results
            if not isinstance(r, Exception) and getattr(r, 'success', False)
        )
        
        assert success_count >= 1, "至少一個並發請求應該成功"


class TestStateMachineTransitions:
    """狀態機轉移測試"""
    
    def test_valid_transitions(self):
        """測試合法狀態轉移"""
        action = RestartServiceAction("act_sm_001", "svc", {})
        sm = ActionStateMachine(action)
        
        # PENDING -> APPROVED
        action.state = ActionState.PENDING
        assert sm.can_transition(ActionState.APPROVED)
        
        # APPROVED -> EXECUTING
        action.state = ActionState.APPROVED
        assert sm.can_transition(ActionState.EXECUTING)
        
        # EXECUTING -> SUCCESS
        action.state = ActionState.EXECUTING
        assert sm.can_transition(ActionState.SUCCESS)
    
    def test_invalid_transitions(self):
        """測試非法狀態轉移被拒絕"""
        action = RestartServiceAction("act_sm_002", "svc", {})
        sm = ActionStateMachine(action)
        
        # PENDING -> SUCCESS (非法)
        action.state = ActionState.PENDING
        assert not sm.can_transition(ActionState.SUCCESS)
        
        # SUCCESS -> EXECUTING (非法)
        action.state = ActionState.SUCCESS
        assert not sm.can_transition(ActionState.EXECUTING)
    
    def test_terminal_states(self):
        """測試終態"""
        action = RestartServiceAction("act_sm_003", "svc", {})
        sm = ActionStateMachine(action)
        
        terminal_states = [
            ActionState.SUCCESS,
            ActionState.FAILED,
            ActionState.ROLLED_BACK,
            ActionState.REJECTED
        ]
        
        for state in terminal_states:
            action.state = state
            assert sm.is_terminal(), f"{state.value} 應該是終態"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
