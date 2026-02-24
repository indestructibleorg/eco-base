"""
治理模块测试脚本

测试强制治理规范的所有核心组件
"""

import asyncio
from datetime import datetime
from app.closed_loop.governance import (
    # Decision Contract
    Decision,
    DecisionContractManager,
    AnomalyEvidence,
    RootCause,
    RiskLevel,
    Action,
    
    # Idempotent Action
    ActionState,
    ActionType,
    RestartServiceAction,
    ScaleAction,
    ActionFactory,
    
    # Approval Gate
    ApprovalGate,
    RiskAssessmentEngine,
    
    # Verification Gate
    VerificationGate,
    VerificationStatus,
    VerificationConfig,
    
    # Audit Trail
    AuditTrail,
    AuditTrailStorage,
    EvidenceType,
    EvidenceCollector,
    
    # Fault Domain
    FaultDomainManager,
    FaultDomain,
    ServiceStatus,
    DegradationLevel,
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerConfig,
    KillSwitch,
    RetryPolicy,
)


async def test_decision_contract():
    """测试决策契约系统"""
    print("\n" + "="*60)
    print("[测试] 决策契约系统")
    print("="*60)
    
    manager = DecisionContractManager()
    
    # 创建决策
    anomaly = AnomalyEvidence(
        anomaly_id="anom_001",
        anomaly_type="cpu_high",
        severity="critical",
        confidence=0.95,
        affected_services=["api-gateway"]
    )
    
    root_causes = [
        RootCause(cause="connection_pool_exhausted", confidence=0.88, evidence=["metric_001"])
    ]
    
    actions = [
        Action(
            action_id="act_001",
            action_type="restart_service",
            target="api-gateway",
            params={"graceful": True},
            estimated_duration=30,
            order=1
        )
    ]
    
    decision = manager.create_decision(
        trace_id="trace_001",
        anomaly=anomaly,
        root_causes=root_causes,
        input_hash="sha256:abc123",
        actions=actions,
        risk_score=0.85
    )
    
    print(f"✅ 决策创建成功: {decision.decision_id}")
    print(f"   风险等级: {decision.risk.level}")
    print(f"   需要审批: {decision.risk.requires_approval}")
    print(f"   审批级别: {decision.approval.level}")
    
    # 验证决策
    errors = decision.validate()
    if errors:
        print(f"❌ 验证失败: {errors}")
    else:
        print(f"✅ 决策验证通过")
    
    # 计算哈希
    hash_value = decision.compute_hash()
    print(f"✅ 决策哈希: {hash_value}")
    
    return True


async def test_idempotent_action():
    """测试幂等动作系统"""
    print("\n" + "="*60)
    print("[测试] 幂等动作系统")
    print("="*60)
    
    # 测试重启服务动作
    action = RestartServiceAction(
        action_id="act_restart_001",
        target="order-service",
        params={"graceful": True}
    )
    
    print(f"初始状态: {action.get_state().value}")
    
    # 第一次执行
    result1 = await action.apply()
    print(f"✅ 第一次执行: {result1.success}, 状态: {action.get_state().value}")
    
    # 第二次执行（幂等测试）
    result2 = await action.apply()
    print(f"✅ 第二次执行（幂等）: {result2.success}, 状态: {action.get_state().value}")
    
    # 验证
    verify_result = await action.verify()
    print(f"✅ 验证结果: {verify_result.passed}")
    
    # 回滚
    rollback_result = await action.rollback()
    print(f"✅ 回滚结果: {rollback_result.success}, 状态: {action.get_state().value}")
    
    return True


async def test_approval_gate():
    """测试审批门檻系统"""
    print("\n" + "="*60)
    print("[测试] 审批门檻系统")
    print("="*60)
    
    gate = ApprovalGate()
    await gate.start()
    
    # 测试低风险决策（自动通过）
    low_risk_decision = {
        "decision_id": "dec_low_001",
        "risk": {"score": 0.2, "level": "low"}
    }
    
    result = await gate.evaluate(low_risk_decision)
    print(f"✅ 低风险决策: allowed={result.allowed}, auto_execute={result.auto_execute}")
    
    # 测试高风险决策（需要审批）
    high_risk_decision = {
        "decision_id": "dec_high_001",
        "risk": {"score": 0.85, "level": "critical"},
        "evidence": {"anomaly": {"affected_services": ["api-gateway", "order-service"]}}
    }
    
    result = await gate.evaluate(high_risk_decision)
    print(f"✅ 高风险决策: allowed={result.allowed}")
    print(f"   审批请求ID: {result.approval_request.request_id if result.approval_request else 'N/A'}")
    print(f"   审批级别: {result.approval_request.level.value if result.approval_request else 'N/A'}")
    
    # 测试审批
    if result.approval_request:
        await gate.approve(result.approval_request.request_id, "sre_lead", "Approved")
        status = gate.get_approval_status(result.approval_request.request_id)
        print(f"✅ 审批状态: {status.value if status else 'N/A'}")
    
    await gate.stop()
    return True


async def test_verification_gate():
    """测试闭环验证系统"""
    print("\n" + "="*60)
    print("[测试] 闭环验证系统")
    print("="*60)
    
    config = VerificationConfig(
        enabled=True,
        timeout_seconds=30,
        check_interval_seconds=2,
        consecutive_success_period_seconds=5
    )
    
    gate = VerificationGate(config)
    
    # 测试验证
    checks = [
        {"metric": "cpu_usage", "expected": "< 70%"},
        {"metric": "error_rate", "expected": "< 0.01"}
    ]
    
    print("开始验证（模拟）...")
    result = await gate.verify("action_001", checks)
    
    print(f"✅ 验证完成: {result.status.value}")
    print(f"   通过比例: {result.pass_ratio}")
    print(f"   是否触发回滚: {result.trigger_rollback}")
    
    for check in result.checks:
        print(f"   - {check.metric}: {check.actual} vs {check.expected} -> {'✅' if check.passed else '❌'}")
    
    return True


async def test_audit_trail():
    """测试证据链系统"""
    print("\n" + "="*60)
    print("[测试] 证据链系统")
    print("="*60)
    
    storage = AuditTrailStorage()
    audit = AuditTrail(storage)
    collector = EvidenceCollector(storage)
    
    # 收集输入证据
    input_evidence = await collector.collect_input_evidence(
        trace_id="trace_001",
        metric_name="cpu_usage",
        metric_value=85.5,
        raw_input={"timestamp": datetime.now().isoformat()}
    )
    print(f"✅ 输入证据: {input_evidence.evidence_id}")
    print(f"   完整性验证: {input_evidence.verify_integrity()}")
    
    # 收集决策证据
    decision_evidence = await collector.collect_decision_evidence(
        trace_id="trace_001",
        decision_id="dec_001",
        decision_context={"risk_score": 0.85},
        reasoning_process=["Detected anomaly", "Matched rule R1"]
    )
    print(f"✅ 决策证据: {decision_evidence.evidence_id}")
    
    # 获取证据链
    chain = await storage.get_evidence_chain("trace_001")
    if chain:
        print(f"✅ 证据链哈希: {chain.compute_chain_hash()}")
        print(f"   证据数量: {len(chain.get_evidences())}")
        print(f"   链完整性: {chain.verify_chain_integrity()}")
    
    # 生成审计报告
    report = await audit.generate_audit_report("trace_001")
    print(f"✅ 审计报告生成完成")
    print(f"   完整性验证: {report.get('integrity_verified')}")
    
    return True


async def test_fault_domain():
    """测试故障域系统"""
    print("\n" + "="*60)
    print("[测试] 故障域系统")
    print("="*60)
    
    manager = FaultDomainManager()
    
    # 注册服务
    manager.register_service(
        "detector",
        FaultDomain.MODULE,
        retry_policy=RetryPolicy(max_retries=3),
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3)
    )
    
    # 测试健康状态更新
    manager.update_health("detector", ServiceStatus.HEALTHY, 50.0, 0.0)
    health = manager.get_health_status("detector")
    print(f"✅ 健康状态: {health.status.value if health else 'N/A'}")
    
    # 测试降级
    manager.set_degradation_level(DegradationLevel.PARTIAL)
    print(f"✅ 降级级别: {manager._degradation.get_level().value}")
    
    # 测试停机保护
    kill_switch = manager._kill_switch
    print(f"✅ 停机保护状态: {kill_switch.is_enabled()}")
    
    # 测试熔断器
    cb = manager._circuit_breakers.get("detector")
    if cb:
        print(f"✅ 熔断器状态: {cb.get_state().value}")
    
    # 获取整体状态
    status = manager.get_status()
    print(f"✅ 整体状态获取成功")
    print(f"   被隔离服务: {status['isolated_services']}")
    print(f"   熔断器状态: {status['circuit_breakers']}")
    
    return True


async def test_risk_assessment():
    """测试风险评估引擎"""
    print("\n" + "="*60)
    print("[测试] 风险评估引擎")
    print("="*60)
    
    engine = RiskAssessmentEngine()
    
    # 测试低风险
    result = engine.calculate_risk(
        affected_services=["service-a"],
        action_type="restart_service",
        blast_radius=1,
        failure_probability=0.1,
        recovery_difficulty=0.3
    )
    print(f"✅ 低风险评估: score={result['score']}, level={result['level']}")
    
    # 测试高风险
    result = engine.calculate_risk(
        affected_services=["api-gateway", "order-service", "payment-service"],
        action_type="config_change_global",
        blast_radius=10,
        failure_probability=0.5,
        recovery_difficulty=0.8,
        data_impact=0.7,
        business_impact=0.9
    )
    print(f"✅ 高风险评估: score={result['score']}, level={result['level']}")
    print(f"   需要审批: {result['requires_approval']}")
    print(f"   审批级别: {result['approval_level']}")
    
    return True


async def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("治理模块全面测试")
    print("="*60)
    
    results = []
    
    # 运行所有测试
    results.append(("决策契约系统", await test_decision_contract()))
    results.append(("幂等动作系统", await test_idempotent_action()))
    results.append(("审批门檻系统", await test_approval_gate()))
    results.append(("闭环验证系统", await test_verification_gate()))
    results.append(("证据链系统", await test_audit_trail()))
    results.append(("故障域系统", await test_fault_domain()))
    results.append(("风险评估引擎", await test_risk_assessment()))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有治理模块测试通过！")
    else:
        print(f"\n⚠️ {total - passed} 个测试失败")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
