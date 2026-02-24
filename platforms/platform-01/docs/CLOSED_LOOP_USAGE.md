# 渐进式闭环系统使用手册

## 概述

本系统实现了渐进式闭环自动化运维，分为两个阶段：

- **Phase 1**: 基础闭环 - 异常检测、自动修复、规则引擎
- **Phase 2**: 智能闭环 - 根因分析、智能告警路由、容量预测、决策工作流

## 快速开始

### 1. 基础闭环 (Phase 1)

```python
import asyncio
from datetime import datetime
from app.closed_loop import (
    AnomalyDetector,
    AutoRemediator,
    RuleEngine,
    ClosedLoopMetrics,
    ClosedLoopController,
    Rule,
    RuleCondition,
    RuleAction,
    RemediationAction,
    RemediationType,
)

# 创建组件
detector = AnomalyDetector()
remediator = AutoRemediator()
rule_engine = RuleEngine()
metrics = ClosedLoopMetrics()

# 创建控制器
controller = ClosedLoopController()
controller.set_detector(detector)
controller.set_remediator(remediator)
controller.set_rule_engine(rule_engine)
controller.set_metrics(metrics)

# 注册自定义修复处理器
async def custom_restart(action):
    print(f"重启服务: {action.target}")
    return "重启成功"

remediator.register_action_handler(
    RemediationType.RESTART, 
    custom_restart
)

# 添加规则
rule = Rule(
    rule_id="cpu_high",
    name="CPU高使用率",
    condition_type=RuleCondition.METRIC_THRESHOLD,
    condition_config={'metric': 'cpu', 'threshold': 80, 'operator': '>'},
    action_type=RuleAction.REMEDIATE,
    action_config={'action_type': 'restart'}
)
rule_engine.add_rule(rule)

# 启动系统
async def main():
    await controller.start()
    
    # 处理指标数据
    await controller.process_metric("cpu", 85.0, datetime.now())
    
    await controller.stop()

asyncio.run(main())
```

### 2. 智能闭环 (Phase 2)

```python
from app.closed_loop import (
    EventCollector,
    CorrelationAnalyzer,
    RootCauseIdentifier,
    ReportGenerator,
    SmartAlertRouter,
    ForecastEngine,
    CapacityPlanner,
    WorkflowEngine,
)

# 创建RCA组件
event_collector = EventCollector()
correlation_analyzer = CorrelationAnalyzer(event_collector)
root_cause_identifier = RootCauseIdentifier(
    event_collector, 
    correlation_analyzer
)
rca_engine = ReportGenerator(
    event_collector,
    correlation_analyzer,
    root_cause_identifier
)

# 创建告警路由器
alert_router = SmartAlertRouter()

# 创建容量管理器
forecast_engine = ForecastEngine()
capacity_planner = CapacityPlanner(forecast_engine)

# 创建工作流引擎
workflow_engine = WorkflowEngine()

# 创建Phase 2控制器
from app.closed_loop import Phase2Controller

phase2 = Phase2Controller(
    base_controller=controller,
    rca_engine=rca_engine,
    alert_router=alert_router,
    capacity_manager=capacity_planner,
    workflow_engine=workflow_engine
)
```

## 异常检测引擎

### 支持的算法

- **3-sigma**: 基于标准差的异常检测
- **IQR**: 四分位距检测
- **MAD**: 中位数绝对偏差
- **Z-score**: Z分数检测
- **百分比变化**: 基于变化率的检测
- **阈值**: 简单阈值检测

### 使用示例

```python
from app.closed_loop import AnomalyDetector, DetectionAlgorithm

# 配置检测器
detector = AnomalyDetector(config={
    'algorithms': [
        DetectionAlgorithm.THREE_SIGMA,
        DetectionAlgorithm.IQR
    ],
    'sensitivity': 0.95,
    'min_points': 10
})

# 注册指标
detector.register_metric("cpu_usage", window_size=100)

# 添加数据点
anomaly = detector.add_data_point(
    "cpu_usage",
    datetime.now(),
    95.0
)

if anomaly:
    print(f"检测到异常: {anomaly.anomaly_type.value}")
    print(f"严重级别: {anomaly.severity}")
```

## 规则引擎

### 规则条件类型

- `metric_threshold`: 指标阈值
- `anomaly_detected`: 异常检测
- `time_based`: 时间条件
- `composite`: 复合条件

### 规则动作类型

- `alert`: 发送告警
- `remediate`: 执行修复
- `escalate`: 升级处理
- `suppress`: 抑制告警
- `workflow`: 触发工作流

### 规则配置示例

```yaml
rules:
  - id: cpu_high_usage
    name: CPU 高使用率
    enabled: true
    priority: 8
    cooldown_minutes: 5
    condition:
      type: metric_threshold
      config:
        metric: cpu_usage
        threshold: 80
        operator: '>'
    action:
      type: alert
      config:
        channel: slack
        message: "CPU使用率过高"
```

## 根因分析引擎 (RCA)

### 事件收集

```python
from app.closed_loop import EventCollector, Event, EventType, EventSeverity

collector = EventCollector()

# 从告警创建事件
event_id = await collector.collect_from_alert({
    'service': 'api-gateway',
    'severity': 'high',
    'title': '高延迟告警',
    'description': '响应时间超过阈值'
})

# 查询事件
events = collector.query_events(
    start_time=datetime.now() - timedelta(hours=1),
    event_type=EventType.ALERT,
    limit=100
)
```

### 关联分析

```python
from app.closed_loop import CorrelationAnalyzer

analyzer = CorrelationAnalyzer(collector)

# 分析事件关联
result = analyzer.analyze(event_id)
print(f"关联事件数: {len(result.related_events)}")
print(f"关联类型: {result.correlation_type}")

# 事件聚类
clusters = analyzer.cluster_events()
for cluster in clusters:
    print(f"簇 {cluster.cluster_id}: {len(cluster.events)} 个事件")
```

### 根因识别

```python
from app.closed_loop import RootCauseIdentifier

identifier = RootCauseIdentifier(collector, analyzer)

# 执行根因分析
analysis = await identifier.analyze(event_id)

# 获取最可能的根因
top_cause = analysis.get_top_cause(min_confidence=0.5)
if top_cause:
    print(f"根因: {top_cause.description}")
    print(f"置信度: {top_cause.confidence}")
    print(f"建议动作: {top_cause.suggested_actions}")
```

### 报告生成

```python
from app.closed_loop import ReportGenerator, ReportFormat

reporter = ReportGenerator(collector, analyzer, identifier)

# 生成报告
report = await reporter.generate(event_id)

# 导出为不同格式
markdown = reporter.export(report, ReportFormat.MARKDOWN)
json_report = reporter.export(report, ReportFormat.JSON)
html = reporter.export(report, ReportFormat.HTML)
```

## 智能告警路由

### 基本使用

```python
from app.closed_loop import SmartAlertRouter, create_default_router

# 使用默认配置
router = create_default_router()

# 或自定义配置
router = SmartAlertRouter(config={
    'aggregation': {
        'aggregation_window_minutes': 5,
        'max_group_size': 50
    },
    'immediate_delivery': True
})

# 注册通知处理器
async def slack_handler(alert, recipients):
    print(f"发送Slack通知: {alert.title}")

router.register_notification_handler(
    NotificationChannel.SLACK, 
    slack_handler
)

# 路由告警
result = await router.route_alert(alert)
```

### 值班安排

```python
from app.closed_loop import OnCallSchedule

schedule = OnCallSchedule(
    schedule_id="primary",
    name="主值班表",
    rotations=[
        {
            'start': datetime(2024, 1, 1),
            'end': datetime(2024, 1, 8),
            'person': 'engineer1@company.com'
        },
        {
            'start': datetime(2024, 1, 8),
            'end': datetime(2024, 1, 15),
            'person': 'engineer2@company.com'
        }
    ]
)

router.add_schedule(schedule)
```

## 容量预测与自动调整

### 预测引擎

```python
from app.closed_loop import ForecastEngine, ForecastModel

engine = ForecastEngine(config={
    'default_model': ForecastModel.HOLT_WINTERS,
    'forecast_horizon_hours': 24,
    'history_window_days': 7
})

# 添加历史数据
for i in range(168):  # 7天 * 24小时
    engine.add_metric_point(
        "service.cpu",
        datetime.now() - timedelta(hours=168-i),
        50.0 + (i % 24) * 2  # 模拟日周期
    )

# 执行预测
forecast = await engine.forecast("service.cpu", horizon_hours=24)

print(f"预测平均值: {forecast.get_average_value():.2f}")
peak, peak_time = forecast.get_peak_value()
print(f"预测峰值: {peak:.2f} 在 {peak_time}")
```

### 容量规划

```python
from app.closed_loop import CapacityPlanner, ResourceType

planner = CapacityPlanner(engine)

# 设置阈值
planner.set_threshold("api-gateway", ResourceType.CPU, 70)
planner.set_threshold("api-gateway", ResourceType.MEMORY, 80)

# 分析并制定计划
plans = await planner.analyze_and_plan("api-gateway")

for plan in plans:
    print(f"计划: {plan.action.value} {plan.resource_type.value}")
    print(f"当前: {plan.current_value} -> 目标: {plan.target_value}")
    print(f"原因: {plan.reason}")
```

### 扩缩容建议

```python
# 获取扩缩容建议
recommendation = await planner.recommend_scaling(
    service="api-gateway",
    current_instances=5
)

if recommendation:
    print(f"建议实例数: {recommendation.recommended_instances}")
    print(f"置信度: {recommendation.confidence}")
    print(f"原因: {recommendation.reason}")
    print(f"成本影响: {recommendation.cost_impact}")
```

## 决策工作流引擎

### 创建工作流

```python
from app.closed_loop import WorkflowEngine, WorkflowStep
from app.closed_loop.workflow.engine import ActionType, ApprovalLevel

engine = WorkflowEngine()

# 创建自动修复工作流
workflow_id = engine.create_auto_remediation_workflow(
    remediation_handler=custom_remediation
)

# 或自定义工作流
workflow = engine.create_workflow(
    name="扩容审批工作流",
    description="自动扩容需要审批",
    tags=["scaling", "approval"]
)

# 添加步骤
engine.add_workflow_step(workflow.workflow_id, WorkflowStep(
    step_id="notify",
    name="通知扩容请求",
    action_type=ActionType.NOTIFY,
    params={'message': '请求扩容', 'channels': ['slack']},
    next_steps=["approval"]
))

engine.add_workflow_step(workflow.workflow_id, WorkflowStep(
    step_id="approval",
    name="等待审批",
    action_type=ActionType.SCALE_UP,
    requires_approval=True,
    approval_level=ApprovalLevel.LEVEL_2,
    timeout_seconds=3600,
    next_steps=["execute"]
))
```

### 执行工作流

```python
# 启动工作流实例
instance_id = await engine.start_workflow(
    workflow_id=workflow_id,
    context={'service': 'api-gateway', 'target_instances': 10},
    priority=8
)

# 获取实例状态
instance = engine.get_workflow_instance(instance_id)
print(f"状态: {instance.status.name}")

# 审批请求
pending = engine.get_pending_approvals()
for req in pending:
    print(f"待审批: {req.title}")
    # 批准
    await engine.approve_request(req.request_id, "manager@company.com")
```

## 性能指标

### 闭环延迟

```python
from app.closed_loop.metrics import (
    record_detection_latency,
    record_remediation_latency,
    record_full_loop_latency
)

# 记录延迟
record_detection_latency(metrics, detection_time_ms)
record_remediation_latency(metrics, remediation_time_ms)
record_full_loop_latency(metrics, full_loop_time_ms)

# 获取统计
timer_stats = metrics.get_timer_stats('full_loop_latency_ms')
print(f"平均延迟: {timer_stats['mean']:.2f}ms")
print(f"P95延迟: {timer_stats['p95']:.2f}ms")
```

### MTTR 追踪

```python
from app.closed_loop import MTTRTracker

tracker = MTTRTracker()

# 记录事件
tracker.record_incident(
    incident_id="incident_001",
    start_time=datetime.now()
)

# 解决事件
tracker.resolve_incident(
    incident_id="incident_001",
    end_time=datetime.now() + timedelta(minutes=15)
)

# 计算MTTR
mttr = tracker.calculate_mttr(hours=24)
print(f"平均恢复时间: {mttr:.2f} 分钟")
```

## 配置参考

### 异常检测配置

```python
config = {
    'algorithms': [
        DetectionAlgorithm.THREE_SIGMA,
        DetectionAlgorithm.IQR,
        DetectionAlgorithm.MAD
    ],
    'sensitivity': 0.95,
    'min_points': 10,
    'thresholds': {
        'upper': 90,
        'lower': 10
    },
    'pct_change_threshold': 50
}
```

### 告警路由配置

```python
config = {
    'aggregation': {
        'aggregation_window_minutes': 5,
        'max_group_size': 50
    },
    'immediate_delivery': True
}
```

### 预测引擎配置

```python
config = {
    'default_model': ForecastModel.HOLT_WINTERS,
    'forecast_horizon_hours': 24,
    'history_window_days': 7,
    'model_params': {
        'seasonal_periods': 24,
        'trend': 'add',
        'seasonal': 'add'
    }
}
```

## 最佳实践

1. **渐进式部署**: 先部署Phase 1，稳定后再启用Phase 2功能
2. **阈值调优**: 根据实际业务调整检测阈值，避免误报
3. **规则管理**: 定期审查和更新规则，删除无效规则
4. **监控闭环**: 监控闭环系统自身的性能指标
5. **人工介入**: 关键操作保留人工审批环节
6. **知识积累**: 利用RCA结果持续优化规则和修复策略

## 故障排查

### 常见问题

1. **异常检测不准确**
   - 检查历史数据量是否充足
   - 调整算法参数和阈值
   - 考虑使用多种算法组合

2. **规则不触发**
   - 检查规则是否启用
   - 检查冷却时间设置
   - 验证条件配置

3. **告警风暴**
   - 启用告警聚合
   - 调整抑制窗口
   - 优化路由规则

4. **预测不准确**
   - 增加历史数据量
   - 选择合适的预测模型
   - 考虑数据季节性

## API 参考

详见各模块的 `__init__.py` 文件获取完整的导出接口列表。
