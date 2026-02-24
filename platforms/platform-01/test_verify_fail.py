"""
Verify-Fail 测试

验证：verify fail 必定触发 rollback 或 escalate（无例外）
"""

import asyncio
from datetime import datetime
from app.closed_loop.governance import (
    VerificationGate,
    VerificationConfig,
    VerificationStatus,
    RollbackTrigger
)


async def test_verify_fail_triggers_rollback():
    """测试验证失败触发回滚"""
    print("\n[Test] Verify fail triggers rollback")
    
    # 创建验证门檻，配置为验证失败时自动回滚
    config = VerificationConfig(
        enabled=True,
        timeout_seconds=10,
        check_interval_seconds=1,
        auto_rollback=True
    )
    
    gate = VerificationGate(config)
    
    # 设置验证失败的条件（期望的值与实际不符）
    checks = [
        {"metric": "cpu_usage", "expected": "< 10%"},  # 实际 65%，会失败
        {"metric": "error_rate", "expected": "< 0.001"}  # 实际 0.005，会失败
    ]
    
    print(f"  Running verification with failing checks...")
    result = await gate.verify("action_fail_001", checks)
    
    print(f"  Verification status: {result.status.value}")
    print(f"  Pass ratio: {result.pass_ratio}")
    print(f"  Trigger rollback: {result.trigger_rollback}")
    
    if result.status == VerificationStatus.FAILED and result.trigger_rollback:
        print(f"  ✅ Verify fail correctly triggers rollback")
        return True
    else:
        print(f"  ❌ Verify fail did not trigger rollback")
        return False


async def test_verify_pass_no_rollback():
    """测试验证通过不触发回滚"""
    print("\n[Test] Verify pass no rollback")
    
    config = VerificationConfig(
        enabled=True,
        timeout_seconds=10,
        check_interval_seconds=1,
        auto_rollback=True
    )
    
    gate = VerificationGate(config)
    
    # 设置验证通过的条件
    checks = [
        {"metric": "cpu_usage", "expected": "< 100%"},  # 实际 65%，会通过
        {"metric": "error_rate", "expected": "< 0.01"}  # 实际 0.005，会通过
    ]
    
    print(f"  Running verification with passing checks...")
    result = await gate.verify("action_pass_001", checks)
    
    print(f"  Verification status: {result.status.value}")
    print(f"  Pass ratio: {result.pass_ratio}")
    print(f"  Trigger rollback: {result.trigger_rollback}")
    
    if result.status == VerificationStatus.PASSED and not result.trigger_rollback:
        print(f"  ✅ Verify pass correctly does not trigger rollback")
        return True
    else:
        print(f"  ❌ Unexpected behavior")
        return False


async def test_rollback_trigger_check():
    """测试回滚触发器"""
    print("\n[Test] Rollback trigger check")
    
    config = VerificationConfig(auto_rollback=True)
    gate = VerificationGate(config)
    
    # 先执行一个失败的验证
    checks = [{"metric": "cpu_usage", "expected": "< 10%"}]
    result = await gate.verify("action_trigger_001", checks)
    
    # 检查是否应该触发回滚
    should_trigger = gate.should_trigger_rollback(result.verification_id)
    
    print(f"  Verification failed: {result.status == VerificationStatus.FAILED}")
    print(f"  Should trigger rollback: {should_trigger}")
    
    if should_trigger:
        print(f"  ✅ Rollback trigger correctly detected")
        return True
    else:
        print(f"  ❌ Rollback trigger not detected")
        return False


async def test_verify_timeout():
    """测试验证超时"""
    print("\n[Test] Verify timeout")
    
    config = VerificationConfig(
        enabled=True,
        timeout_seconds=2,  # 短超时
        check_interval_seconds=1,
        consecutive_success_period_seconds=10  # 长成功期，会超时
    )
    
    gate = VerificationGate(config)
    
    checks = [{"metric": "cpu_usage", "expected": "< 100%"}]
    
    print(f"  Running verification with short timeout...")
    result = await gate.verify("action_timeout_001", checks)
    
    print(f"  Verification status: {result.status.value}")
    
    if result.status == VerificationStatus.TIMEOUT:
        print(f"  ✅ Timeout correctly detected")
        return True
    else:
        print(f"  ❌ Timeout not detected")
        return False


async def test_verify_disabled():
    """测试验证禁用"""
    print("\n[Test] Verify disabled")
    
    config = VerificationConfig(enabled=False)
    gate = VerificationGate(config)
    
    checks = [{"metric": "cpu_usage", "expected": "< 100%"}]
    
    result = await gate.verify("action_disabled_001", checks)
    
    print(f"  Verification status: {result.status.value}")
    
    if result.status == VerificationStatus.SKIPPED:
        print(f"  ✅ Verification correctly skipped when disabled")
        return True
    else:
        print(f"  ❌ Unexpected status")
        return False


async def run_verify_fail_tests():
    """运行所有 verify-fail 测试"""
    print("\n" + "="*60)
    print("Verify-Fail Tests")
    print("="*60)
    
    results = []
    results.append(("Verify fail triggers rollback", await test_verify_fail_triggers_rollback()))
    results.append(("Verify pass no rollback", await test_verify_pass_no_rollback()))
    results.append(("Rollback trigger check", await test_rollback_trigger_check()))
    results.append(("Verify timeout", await test_verify_timeout()))
    results.append(("Verify disabled", await test_verify_disabled()))
    
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
    success = asyncio.run(run_verify_fail_tests())
    exit(0 if success else 1)
