"""
Crash-Resume 测试

验证：模拟中途crash，重启后必须从正确状态续跑
"""

import asyncio
import os
from datetime import datetime
from app.closed_loop.core.state_store import (
    RunState, RunPhase, RunStateTransition,
    FileStateStore
)
from app.closed_loop.core.recoverable_controller import RecoverableController


async def test_crash_at_executing():
    """测试在执行阶段崩溃后恢复"""
    print("\n[Test] Crash at EXECUTING phase")
    
    # 创建状态存储
    store = FileStateStore(base_path="/tmp/test_crash_resume")
    
    # 创建控制器
    controller = RecoverableController(state_store=store)
    
    # 启动运行
    trace_id = f"trace_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    state = await controller.start_run(trace_id, {"test": "data"})
    run_id = state.run_id
    
    print(f"  Started run: {run_id}")
    
    # 手动推进到 EXECUTING 阶段
    state.transition_to(RunPhase.DETECTED, RunStateTransition.ANOMALY_DETECTED)
    await store.save(state)
    state.transition_to(RunPhase.ANALYZED, RunStateTransition.RCA_COMPLETED)
    await store.save(state)
    state.transition_to(RunPhase.PLANNED, RunStateTransition.DECISION_PLANNED)
    await store.save(state)
    state.transition_to(RunPhase.APPROVED, RunStateTransition.APPROVAL_GRANTED)
    await store.save(state)
    state.transition_to(RunPhase.EXECUTING, RunStateTransition.EXECUTION_STARTED)
    await store.save(state)
    
    print(f"  State saved at EXECUTING")
    
    # 模拟崩溃：创建新的控制器实例
    new_controller = RecoverableController(state_store=store)
    
    # 恢复运行
    resumed_state = await new_controller.resume_run(run_id)
    
    if resumed_state and resumed_state.phase == RunPhase.EXECUTING:
        print(f"  ✅ Successfully resumed from EXECUTING")
        return True
    else:
        print(f"  ❌ Failed to resume. Current phase: {resumed_state.phase.value if resumed_state else 'None'}")
        return False


async def test_crash_at_verifying():
    """测试在验证阶段崩溃后恢复"""
    print("\n[Test] Crash at VERIFYING phase")
    
    store = FileStateStore(base_path="/tmp/test_crash_resume")
    controller = RecoverableController(state_store=store)
    
    trace_id = f"trace_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    state = await controller.start_run(trace_id, {"test": "data"})
    run_id = state.run_id
    
    # 推进到 VERIFYING 阶段
    state.transition_to(RunPhase.DETECTED, RunStateTransition.ANOMALY_DETECTED)
    await store.save(state)
    state.transition_to(RunPhase.ANALYZED, RunStateTransition.RCA_COMPLETED)
    await store.save(state)
    state.transition_to(RunPhase.PLANNED, RunStateTransition.DECISION_PLANNED)
    await store.save(state)
    state.transition_to(RunPhase.APPROVED, RunStateTransition.APPROVAL_GRANTED)
    await store.save(state)
    state.transition_to(RunPhase.EXECUTING, RunStateTransition.EXECUTION_STARTED)
    await store.save(state)
    state.transition_to(RunPhase.EXECUTED, RunStateTransition.EXECUTION_COMPLETED)
    await store.save(state)
    state.transition_to(RunPhase.VERIFYING, RunStateTransition.VERIFICATION_STARTED)
    await store.save(state)
    
    print(f"  State saved at VERIFYING")
    
    # 模拟崩溃后恢复
    new_controller = RecoverableController(state_store=store)
    resumed_state = await new_controller.resume_run(run_id)
    
    if resumed_state and resumed_state.phase == RunPhase.VERIFYING:
        print(f"  ✅ Successfully resumed from VERIFYING")
        return True
    else:
        print(f"  ❌ Failed to resume. Current phase: {resumed_state.phase.value if resumed_state else 'None'}")
        return False


async def test_state_persistence():
    """测试状态持久化"""
    print("\n[Test] State persistence")
    
    store = FileStateStore(base_path="/tmp/test_crash_resume")
    
    trace_id = f"trace_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    state = RunState(
        run_id=f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}_test",
        trace_id=trace_id,
        phase=RunPhase.PLANNED,
        inputs_hash="abc123"
    )
    
    # 保存状态
    await store.save(state)
    
    # 加载状态
    loaded = await store.load(state.run_id)
    
    if loaded and loaded.run_id == state.run_id and loaded.phase == RunPhase.PLANNED:
        print(f"  ✅ State persisted correctly")
        return True
    else:
        print(f"  ❌ State not persisted correctly")
        return False


async def run_crash_resume_tests(iterations: int = 20):
    """运行所有 crash-resume 测试"""
    print("\n" + "="*60)
    print("Crash-Resume Tests")
    print("="*60)
    
    results = []
    
    # 运行单次测试
    results.append(("State persistence", await test_state_persistence()))
    results.append(("Crash at EXECUTING", await test_crash_at_executing()))
    results.append(("Crash at VERIFYING", await test_crash_at_verifying()))
    
    # 连续运行多次
    print(f"\n[Stress Test] Running {iterations} iterations...")
    success_count = 0
    
    for i in range(iterations):
        store = FileStateStore(base_path=f"/tmp/test_crash_resume_{i}")
        controller = RecoverableController(state_store=store)
        
        trace_id = f"trace_{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        state = await controller.start_run(trace_id, {"iteration": i})
        
        # 推进到执行阶段
        state.transition_to(RunPhase.DETECTED, RunStateTransition.ANOMALY_DETECTED)
        await store.save(state)
        state.transition_to(RunPhase.EXECUTING, RunStateTransition.EXECUTION_STARTED)
        await store.save(state)
        
        # 恢复
        new_controller = RecoverableController(state_store=store)
        resumed = await new_controller.resume_run(state.run_id)
        
        if resumed and resumed.phase == RunPhase.EXECUTING:
            success_count += 1
    
    stress_passed = success_count == iterations
    results.append((f"Stress test ({iterations} iterations)", stress_passed))
    
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
    
    if stress_passed:
        print(f"  ✅ Stress test: {success_count}/{iterations} passed")
    else:
        print(f"  ❌ Stress test: {success_count}/{iterations} passed")
    
    # 清理
    import shutil
    for i in range(iterations):
        path = f"/tmp/test_crash_resume_{i}"
        if os.path.exists(path):
            shutil.rmtree(path)
    
    return passed == total and stress_passed


if __name__ == "__main__":
    success = asyncio.run(run_crash_resume_tests(iterations=20))
    exit(0 if success else 1)
