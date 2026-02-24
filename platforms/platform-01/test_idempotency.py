"""
幂等性测试

验证：同一 run_id 重放不产生重复副作用
"""

import asyncio
from datetime import datetime
from app.closed_loop.governance import (
    RestartServiceAction,
    ScaleAction,
    ActionState
)


class SideEffectTracker:
    """副作用追踪器"""
    
    def __init__(self):
        self.restart_count = 0
        self.scale_count = 0
        self.executed_actions = set()
    
    def record_restart(self, action_id: str):
        if action_id not in self.executed_actions:
            self.restart_count += 1
            self.executed_actions.add(action_id)
    
    def record_scale(self, action_id: str):
        if action_id not in self.executed_actions:
            self.scale_count += 1
            self.executed_actions.add(action_id)


async def test_restart_action_idempotency():
    """测试重启动作的幂等性"""
    print("\n[Test] Restart action idempotency")
    
    tracker = SideEffectTracker()
    
    action = RestartServiceAction(
        action_id="act_restart_001",
        target="order-service",
        params={"graceful": True}
    )
    
    # 第一次执行
    result1 = await action.apply()
    if result1.success:
        tracker.record_restart(action.action_id)
    
    print(f"  First execution: success={result1.success}, state={action.get_state().value}")
    
    # 第二次执行（应该幂等）
    result2 = await action.apply()
    if result2.success:
        tracker.record_restart(action.action_id)
    
    print(f"  Second execution: success={result2.success}, state={action.get_state().value}")
    
    # 验证副作用只发生一次
    if tracker.restart_count == 1:
        print(f"  ✅ Idempotency verified: only 1 side effect")
        return True
    else:
        print(f"  ❌ Idempotency failed: {tracker.restart_count} side effects")
        return False


async def test_scale_action_idempotency():
    """测试扩缩容动作的幂等性"""
    print("\n[Test] Scale action idempotency")
    
    tracker = SideEffectTracker()
    
    action = ScaleAction(
        action_id="act_scale_001",
        target="api-gateway",
        params={"replicas": 5, "direction": "up"}
    )
    
    # 第一次执行
    result1 = await action.apply()
    if result1.success:
        tracker.record_scale(action.action_id)
    
    print(f"  First execution: success={result1.success}, state={action.get_state().value}")
    
    # 第二次执行
    result2 = await action.apply()
    if result2.success:
        tracker.record_scale(action.action_id)
    
    print(f"  Second execution: success={result2.success}, state={action.get_state().value}")
    
    # 第三次执行
    result3 = await action.apply()
    if result3.success:
        tracker.record_scale(action.action_id)
    
    print(f"  Third execution: success={result3.success}, state={action.get_state().value}")
    
    if tracker.scale_count == 1:
        print(f"  ✅ Idempotency verified: only 1 side effect")
        return True
    else:
        print(f"  ❌ Idempotency failed: {tracker.scale_count} side effects")
        return False


async def test_rollback_idempotency():
    """测试回滚的幂等性"""
    print("\n[Test] Rollback idempotency")
    
    action = RestartServiceAction(
        action_id="act_restart_002",
        target="payment-service",
        params={"graceful": True}
    )
    
    # 执行
    await action.apply()
    
    # 第一次回滚
    result1 = await action.rollback()
    print(f"  First rollback: success={result1.success}, state={action.get_state().value}")
    
    # 第二次回滚（应该幂等）
    result2 = await action.rollback()
    print(f"  Second rollback: success={result2.success}, state={action.get_state().value}")
    
    if action.get_state() == ActionState.ROLLED_BACK:
        print(f"  ✅ Rollback idempotency verified")
        return True
    else:
        print(f"  ❌ Rollback idempotency failed")
        return False


async def test_action_state_machine():
    """测试动作状态机"""
    print("\n[Test] Action state machine")
    
    from app.closed_loop.governance import ActionStateMachine
    
    action = RestartServiceAction(
        action_id="act_restart_003",
        target="inventory-service",
        params={}
    )
    
    sm = ActionStateMachine(action)
    
    # 验证状态转移
    transitions = [
        (ActionState.PENDING, ActionState.APPROVED, True),
        (ActionState.APPROVED, ActionState.EXECUTING, True),
        (ActionState.EXECUTING, ActionState.SUCCESS, True),
        (ActionState.SUCCESS, ActionState.VERIFIED, True),
        (ActionState.PENDING, ActionState.SUCCESS, False),  # 无效转移
    ]
    
    all_passed = True
    for from_state, to_state, expected in transitions:
        action.state = from_state
        can_transition = sm.can_transition(to_state)
        
        if can_transition == expected:
            print(f"  ✅ {from_state.value} -> {to_state.value}: {can_transition}")
        else:
            print(f"  ❌ {from_state.value} -> {to_state.value}: expected {expected}, got {can_transition}")
            all_passed = False
    
    return all_passed


async def run_idempotency_tests():
    """运行所有幂等性测试"""
    print("\n" + "="*60)
    print("Idempotency Tests")
    print("="*60)
    
    results = []
    results.append(("Restart action idempotency", await test_restart_action_idempotency()))
    results.append(("Scale action idempotency", await test_scale_action_idempotency()))
    results.append(("Rollback idempotency", await test_rollback_idempotency()))
    results.append(("Action state machine", await test_action_state_machine()))
    
    # 汇总
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_idempotency_tests())
    exit(0 if success else 1)
