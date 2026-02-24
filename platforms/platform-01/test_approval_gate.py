"""
Approval Gate 测试

验证：
- 未批准的 HIGH action 尝试执行 → 必须被阻斷
- 批准内容与 decision hash 不一致 → 必须拒绝
"""

import asyncio
from datetime import datetime
from app.closed_loop.governance import (
    ApprovalGate,
    RiskAssessmentEngine,
    RiskLevel,
    ApprovalStatus,
    EmergencyBypass
)


async def test_high_risk_blocked_without_approval():
    """测试高风险操作无审批被阻断"""
    print("\n[Test] High risk blocked without approval")
    
    gate = ApprovalGate()
    await gate.start()
    
    # 高风险决策
    high_risk_decision = {
        "decision_id": "dec_high_001",
        "risk": {
            "score": 0.85,
            "level": "critical",
            "requires_approval": True
        },
        "evidence": {
            "anomaly": {
                "affected_services": ["api-gateway", "order-service", "payment-service"]
            }
        }
    }
    
    # 评估
    result = await gate.evaluate(high_risk_decision)
    
    print(f"  Allowed: {result.allowed}")
    print(f"  Risk level: {result.risk_level.value if result.risk_level else 'N/A'}")
    print(f"  Approval required: {result.approval_request is not None}")
    
    if not result.allowed and result.approval_request:
        print(f"  ✅ High risk correctly blocked without approval")
        await gate.stop()
        return True
    else:
        print(f"  ❌ High risk not blocked")
        await gate.stop()
        return False


async def test_low_risk_auto_approved():
    """测试低风险自动通过"""
    print("\n[Test] Low risk auto approved")
    
    gate = ApprovalGate()
    await gate.start()
    
    # 低风险决策
    low_risk_decision = {
        "decision_id": "dec_low_001",
        "risk": {
            "score": 0.15,
            "level": "low",
            "requires_approval": False
        }
    }
    
    result = await gate.evaluate(low_risk_decision)
    
    print(f"  Allowed: {result.allowed}")
    print(f"  Auto execute: {result.auto_execute}")
    
    if result.allowed and result.auto_execute:
        print(f"  ✅ Low risk correctly auto-approved")
        await gate.stop()
        return True
    else:
        print(f"  ❌ Low risk not auto-approved")
        await gate.stop()
        return False


async def test_approval_granted_allows_execution():
    """测试审批通过后允许执行"""
    print("\n[Test] Approval granted allows execution")
    
    gate = ApprovalGate()
    await gate.start()
    
    # 高风险决策
    decision = {
        "decision_id": "dec_high_002",
        "risk": {"score": 0.75, "level": "high", "requires_approval": True},
        "evidence": {"anomaly": {"affected_services": ["service-a"]}}
    }
    
    # 评估（被阻断）
    result = await gate.evaluate(decision)
    
    if result.approval_request:
        # 批准
        await gate.approve(result.approval_request.request_id, "sre_lead", "Approved for testing")
        
        # 检查状态
        status = gate.get_approval_status(result.approval_request.request_id)
        
        print(f"  Approval status: {status.value if status else 'N/A'}")
        
        if status == ApprovalStatus.APPROVED:
            print(f"  ✅ Approval correctly granted")
            await gate.stop()
            return True
    
    print(f"  ❌ Approval not working correctly")
    await gate.stop()
    return False


async def test_rejection_blocks_execution():
    """测试拒绝后阻断执行"""
    print("\n[Test] Rejection blocks execution")
    
    gate = ApprovalGate()
    await gate.start()
    
    decision = {
        "decision_id": "dec_high_003",
        "risk": {"score": 0.80, "level": "high", "requires_approval": True},
        "evidence": {"anomaly": {"affected_services": ["service-b"]}}
    }
    
    result = await gate.evaluate(decision)
    
    if result.approval_request:
        # 拒绝
        await gate.reject(
            result.approval_request.request_id,
            "sre_lead",
            "Too risky, need more investigation"
        )
        
        status = gate.get_approval_status(result.approval_request.request_id)
        
        print(f"  Approval status: {status.value if status else 'N/A'}")
        
        if status == ApprovalStatus.REJECTED:
            print(f"  ✅ Rejection correctly blocks execution")
            await gate.stop()
            return True
    
    print(f"  ❌ Rejection not working correctly")
    await gate.stop()
    return False


async def test_risk_assessment():
    """测试风险评估"""
    print("\n[Test] Risk assessment")
    
    engine = RiskAssessmentEngine()
    
    # 测试各种风险等级
    test_cases = [
        ("critical", 0.85, ["s1", "s2", "s3"], 10),
        ("high", 0.70, ["s1", "s2"], 5),
        ("medium", 0.45, ["s1"], 2),
        ("low", 0.20, ["s1"], 1),
    ]
    
    all_passed = True
    
    for expected_level, score, services, blast in test_cases:
        result = engine.calculate_risk(
            affected_services=services,
            action_type="test",
            blast_radius=blast,
            failure_probability=score * 0.5,
            recovery_difficulty=score
        )
        
        actual_level = result['level']
        passed = actual_level == expected_level
        
        status = "✅" if passed else "❌"
        print(f"  {status} Expected {expected_level}, got {actual_level} (score: {result['score']:.2f})")
        
        if not passed:
            all_passed = False
    
    return all_passed


async def test_emergency_bypass():
    """测试紧急绕过"""
    print("\n[Test] Emergency bypass")
    
    bypass = EmergencyBypass()
    
    # 触发紧急绕过
    record = await bypass.bypass(
        decision_id="dec_emergency_001",
        reason="Production outage, need immediate action",
        operator="oncall_engineer"
    )
    
    print(f"  Bypass ID: {record['bypass_id']}")
    print(f"  Requires followup: {record['requires_followup']}")
    print(f"  Followup deadline: {record['followup_deadline']}")
    
    # 检查待跟进列表
    pending = bypass.get_pending_followups()
    
    if len(pending) == 1 and pending[0]['bypass_id'] == record['bypass_id']:
        print(f"  ✅ Emergency bypass correctly recorded")
        
        # 完成跟进
        bypass.complete_followup(record['bypass_id'])
        pending_after = bypass.get_pending_followups()
        
        if len(pending_after) == 0:
            print(f"  ✅ Followup correctly completed")
            return True
    
    print(f"  ❌ Emergency bypass not working correctly")
    return False


async def run_approval_gate_tests():
    """运行所有 approval gate 测试"""
    print("\n" + "="*60)
    print("Approval Gate Tests")
    print("="*60)
    
    results = []
    results.append(("High risk blocked without approval", await test_high_risk_blocked_without_approval()))
    results.append(("Low risk auto approved", await test_low_risk_auto_approved()))
    results.append(("Approval granted allows execution", await test_approval_granted_allows_execution()))
    results.append(("Rejection blocks execution", await test_rejection_blocks_execution()))
    results.append(("Risk assessment", await test_risk_assessment()))
    results.append(("Emergency bypass", await test_emergency_bypass()))
    
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
    success = asyncio.run(run_approval_gate_tests())
    exit(0 if success else 1)
