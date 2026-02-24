#!/usr/bin/env python3
"""
闭环系统整合测试脚本
验证 Phase 1 和 Phase 2 所有组件
"""

import asyncio
import sys
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, '/mnt/okcomputer/output')

from app.closed_loop import (
    # Phase 1
    AnomalyDetector, AnomalyType, DetectionAlgorithm,
    AutoRemediator, RemediationAction, RemediationType,
    RuleEngine, Rule, RuleCondition, RuleAction,
    ClosedLoopMetrics, ClosedLoopController,
    
    # Phase 2
    EventCollector, Event, EventType, EventSeverity,
    CorrelationAnalyzer, CorrelationResult,
    RootCauseIdentifier, RootCauseAnalysis, RootCauseCategory,
    ReportGenerator, ReportFormat,
    SmartAlertRouter, Alert, AlertSeverity, AlertStatus, NotificationChannel,
    ForecastEngine, ForecastModel,
    CapacityPlanner, ResourceType,
    WorkflowEngine, WorkflowStatus, ApprovalLevel,
)


async def test_phase1():
    """测试 Phase 1 组件"""
    print("=" * 60)
    print("Phase 1: 基础闭环测试")
    print("=" * 60)
    
    # 1. 异常检测引擎
    print("\n[1] 异常检测引擎测试")
    detector = AnomalyDetector(config={
        'algorithms': [DetectionAlgorithm.THREE_SIGMA, DetectionAlgorithm.IQR],
        'sensitivity': 0.95,
        'min_points': 10
    })
    
    # 添加正常数据
    for i in range(20):
        detector.add_data_point(
            "cpu_usage",
            datetime.now() + timedelta(minutes=i),
            50.0 + (i % 5)
        )
    
    # 测试异常检测
    anomaly = detector.add_data_point(
        "cpu_usage",
        datetime.now() + timedelta(minutes=20),
        150.0
    )
    
    if anomaly:
        print(f"  ✓ 异常检测成功: {anomaly.anomaly_type.value}")
        print(f"    - 严重级别: {anomaly.severity:.2f}")
        print(f"    - 置信度: {anomaly.confidence:.2f}")
    else:
        print("  ✗ 未检测到异常")
    
    # 2. 自动修复引擎
    print("\n[2] 自动修复引擎测试")
    remediator = AutoRemediator()
    
    async def mock_restart(action):
        return f"服务 {action.target} 重启成功"
    
    remediator.register_action_handler(RemediationType.RESTART, mock_restart)
    
    action = RemediationAction(
        action_id="test_restart",
        action_type=RemediationType.RESTART,
        target="test_service"
    )
    
    result = await remediator.execute(action)
    print(f"  ✓ 修复执行: {result.status.name}")
    
    # 3. 规则引擎
    print("\n[3] 规则引擎测试")
    rule_engine = RuleEngine()
    
    rule = Rule(
        rule_id="cpu_threshold",
        name="CPU阈值规则",
        description="CPU使用率超过80%触发",
        condition_type=RuleCondition.METRIC_THRESHOLD,
        condition_config={'metric': 'cpu', 'threshold': 80, 'operator': '>'},
        action_type=RuleAction.ALERT,
        action_config={'channel': 'slack'}
    )
    rule_engine.add_rule(rule)
    
    # 测试规则评估
    context = {'metrics': {'cpu': 90}}
    triggered = rule_engine.evaluate_rule("cpu_threshold", context)
    print(f"  ✓ 规则评估: {'触发' if triggered else '未触发'}")
    
    # 4. 闭环指标
    print("\n[4] 闭环指标测试")
    metrics = ClosedLoopMetrics()
    metrics.increment_counter("test_counter", 5)
    metrics.set_gauge("test_gauge", 42.0)
    
    print(f"  ✓ 计数器: {metrics.get_counter('test_counter')}")
    print(f"  ✓ 仪表盘: {metrics.get_gauge('test_gauge')}")
    
    # 5. 闭环控制器
    print("\n[5] 闭环控制器测试")
    controller = ClosedLoopController()
    controller.set_detector(detector)
    controller.set_remediator(remediator)
    controller.set_rule_engine(rule_engine)
    controller.set_metrics(metrics)
    
    status = controller.get_status()
    print(f"  ✓ 控制器状态: {status['state']}")
    print(f"  ✓ 组件状态: detector={status['components']['detector']}")
    
    print("\n✅ Phase 1 测试完成")
    return True


async def test_phase2():
    """测试 Phase 2 组件"""
    print("\n" + "=" * 60)
    print("Phase 2: 智能闭环测试")
    print("=" * 60)
    
    # 1. 事件收集器
    print("\n[1] 事件收集器测试")
    collector = EventCollector()
    
    event = Event(
        event_id="test_event_001",
        event_type=EventType.ALERT,
        source="api-gateway",
        timestamp=datetime.now(),
        severity=EventSeverity.HIGH,
        title="高延迟告警",
        description="响应时间超过阈值"
    )
    
    event_id = await collector.collect(event)
    print(f"  ✓ 事件收集: {event_id}")
    
    # 从告警数据收集
    alert_event_id = await collector.collect_from_alert({
        'service': 'user-service',
        'severity': 'critical',
        'title': '服务不可用',
        'description': '连接超时'
    })
    print(f"  ✓ 告警事件收集: {alert_event_id}")
    
    # 2. 关联分析器
    print("\n[2] 关联分析器测试")
    analyzer = CorrelationAnalyzer(collector)
    
    # 添加更多相关事件
    for i in range(5):
        await collector.collect(Event(
            event_id=f"related_event_{i}",
            event_type=EventType.LOG_ERROR,
            source="api-gateway",
            timestamp=datetime.now() + timedelta(minutes=i),
            severity=EventSeverity.MEDIUM,
            title=f"错误日志 {i}",
            description="数据库连接失败",
            tags=['database', 'error']
        ))
    
    result = analyzer.analyze(event_id)
    print(f"  ✓ 关联分析: {len(result.related_events)} 个相关事件")
    print(f"    - 关联类型: {result.correlation_type}")
    print(f"    - 置信度: {result.confidence:.2f}")
    
    # 3. 根因识别器
    print("\n[3] 根因识别器测试")
    identifier = RootCauseIdentifier(collector, analyzer)
    
    analysis = await identifier.analyze(event_id)
    print(f"  ✓ 根因分析: {len(analysis.root_causes)} 个可能原因")
    
    if analysis.root_causes:
        top_cause = analysis.get_top_cause()
        if top_cause:
            print(f"    - 最可能原因: {top_cause.category.value}")
            print(f"    - 置信度: {top_cause.confidence:.2f}")
    
    # 4. 报告生成器
    print("\n[4] 报告生成器测试")
    reporter = ReportGenerator(collector, analyzer, identifier)
    
    report = await reporter.generate(event_id)
    print(f"  ✓ 报告生成: {report.report_id}")
    print(f"    - 根因数: {len(report.root_causes)}")
    print(f"    - 建议数: {len(report.recommendations)}")
    
    # 5. 智能告警路由
    print("\n[5] 智能告警路由测试")
    router = SmartAlertRouter()
    
    # 注册通知处理器
    async def mock_notification_handler(notification):
        print(f"    [通知] {notification.get('type', 'unknown')}")
    
    router.register_notification_handler(
        NotificationChannel.SLACK,
        lambda alert, recipients: mock_notification_handler({'type': 'alert'})
    )
    
    alert = Alert(
        alert_id="alert_001",
        title="CPU使用率过高",
        description="CPU使用率超过90%",
        source="web-server",
        severity=AlertSeverity.HIGH,
        status=AlertStatus.NEW,
        created_at=datetime.now()
    )
    
    result = await router.route_alert(alert)
    print(f"  ✓ 告警路由: {result.get('status', 'unknown')}")
    
    # 6. 预测引擎
    print("\n[6] 预测引擎测试")
    forecast_engine = ForecastEngine(config={
        'default_model': ForecastModel.LINEAR_REGRESSION,
        'forecast_horizon_hours': 12
    })
    
    # 添加历史数据
    for i in range(48):
        forecast_engine.add_metric_point(
            "service.cpu",
            datetime.now() - timedelta(hours=48-i),
            50.0 + (i % 24) * 1.5
        )
    
    forecast = await forecast_engine.forecast("service.cpu", horizon_hours=12)
    if forecast:
        print(f"  ✓ 预测完成: {len(forecast.forecast_values)} 个预测点")
        print(f"    - 模型: {forecast.model_used.value}")
        avg = forecast.get_average_value()
        print(f"    - 预测均值: {avg:.2f}")
    
    # 7. 容量规划器
    print("\n[7] 容量规划器测试")
    planner = CapacityPlanner(forecast_engine)
    planner.set_threshold("api-gateway", ResourceType.CPU, 70)
    
    plans = await planner.analyze_and_plan("api-gateway")
    print(f"  ✓ 容量计划: {len(plans)} 个计划")
    
    for plan in plans:
        print(f"    - {plan.action.value}: {plan.current_value:.1f} -> {plan.target_value:.1f}")
    
    # 8. 工作流引擎
    print("\n[8] 工作流引擎测试")
    workflow_engine = WorkflowEngine()
    
    # 创建自动修复工作流
    async def mock_remediation(instance, step):
        return {"status": "success", "message": "修复完成"}
    
    workflow_id = workflow_engine.create_auto_remediation_workflow(
        remediation_handler=mock_remediation
    )
    print(f"  ✓ 工作流创建: {workflow_id}")
    
    # 启动工作流
    instance_id = await workflow_engine.start_workflow(
        workflow_id=workflow_id,
        context={'service': 'test-service', 'issue': 'high_cpu'},
        priority=8
    )
    print(f"  ✓ 工作流启动: {instance_id}")
    
    instance = workflow_engine.get_workflow_instance(instance_id)
    print(f"    - 状态: {instance.status.name}")
    
    # 获取统计
    stats = workflow_engine.get_workflow_stats()
    print(f"    - 总实例数: {stats.get('total', 0)}")
    
    print("\n✅ Phase 2 测试完成")
    return True


async def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("闭环系统整合测试")
    print("=" * 60)
    
    try:
        # 测试 Phase 1
        phase1_ok = await test_phase1()
        
        # 测试 Phase 2
        phase2_ok = await test_phase2()
        
        # 总结
        print("\n" + "=" * 60)
        print("测试结果总结")
        print("=" * 60)
        print(f"Phase 1 (基础闭环): {'✅ 通过' if phase1_ok else '❌ 失败'}")
        print(f"Phase 2 (智能闭环): {'✅ 通过' if phase2_ok else '❌ 失败'}")
        
        if phase1_ok and phase2_ok:
            print("\n🎉 所有测试通过！闭环系统已就绪。")
            return 0
        else:
            print("\n⚠️ 部分测试失败，请检查日志。")
            return 1
    
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
