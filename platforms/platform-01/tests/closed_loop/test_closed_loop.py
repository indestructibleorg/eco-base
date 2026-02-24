"""
闭环系统测试
"""

import asyncio
import pytest
from datetime import datetime, timedelta

from app.closed_loop import (
    # Phase 1
    AnomalyDetector,
    AnomalyType,
    DetectionAlgorithm,
    AutoRemediator,
    RemediationAction,
    RemediationType,
    RuleEngine,
    RuleCondition,
    RuleAction,
    ClosedLoopMetrics,
    ClosedLoopController,
    
    # Phase 2
    EventCollector,
    EventType,
    EventSeverity,
    CorrelationAnalyzer,
    RootCauseIdentifier,
    ReportGenerator,
    ReportFormat,
    SmartAlertRouter,
    AlertSeverity,
    AlertStatus,
    ForecastEngine,
    ForecastModel,
    CapacityPlanner,
    WorkflowEngine,
    WorkflowStatus,
    ActionType as WorkflowActionType,
    ApprovalLevel,
)


# ==================== Phase 1 测试 ====================

class TestAnomalyDetector:
    """异常检测引擎测试"""
    
    def test_detector_initialization(self):
        """测试检测器初始化"""
        detector = AnomalyDetector()
        assert detector is not None
        assert len(detector.algorithms) > 0
    
    def test_metric_registration(self):
        """测试指标注册"""
        detector = AnomalyDetector()
        detector.register_metric("test_metric")
        assert "test_metric" in detector.metrics
    
    def test_normal_data_no_anomaly(self):
        """测试正常数据不产生异常"""
        detector = AnomalyDetector()
        
        # 添加正常数据
        for i in range(20):
            detector.add_data_point(
                "test_metric",
                datetime.now() + timedelta(minutes=i),
                50.0 + (i % 5)  # 正常波动
            )
        
        # 添加一个正常值
        anomaly = detector.add_data_point(
            "test_metric",
            datetime.now() + timedelta(minutes=20),
            52.0
        )
        
        assert anomaly is None
    
    def test_anomaly_detection(self):
        """测试异常检测"""
        detector = AnomalyDetector()
        
        # 添加正常数据
        for i in range(20):
            detector.add_data_point(
                "test_metric",
                datetime.now() + timedelta(minutes=i),
                50.0
            )
        
        # 添加异常值
        anomaly = detector.add_data_point(
            "test_metric",
            datetime.now() + timedelta(minutes=20),
            150.0  # 异常高值
        )
        
        assert anomaly is not None
        assert anomaly.metric_name == "test_metric"
        assert anomaly.anomaly_type in [AnomalyType.SPIKE, AnomalyType.UNKNOWN]


class TestAutoRemediator:
    """自动修复引擎测试"""
    
    @pytest.mark.asyncio
    async def test_remediator_initialization(self):
        """测试修复器初始化"""
        remediator = AutoRemediator()
        assert remediator is not None
    
    @pytest.mark.asyncio
    async def test_execute_restart(self):
        """测试执行重启"""
        remediator = AutoRemediator()
        
        action = RemediationAction(
            action_id="test_restart",
            action_type=RemediationType.RESTART,
            target="test_service"
        )
        
        result = await remediator.execute(action)
        
        assert result is not None
        assert result.status.name in ['SUCCESS', 'FAILED']
    
    def test_success_rate(self):
        """测试成功率统计"""
        remediator = AutoRemediator()
        rate = remediator.get_success_rate()
        assert 0.0 <= rate <= 1.0


class TestRuleEngine:
    """规则引擎测试"""
    
    def test_rule_engine_initialization(self):
        """测试规则引擎初始化"""
        engine = RuleEngine()
        assert engine is not None
    
    def test_add_rule(self):
        """测试添加规则"""
        from app.closed_loop.rules.rule_engine import Rule
        
        engine = RuleEngine()
        rule = Rule(
            rule_id="test_rule",
            name="测试规则",
            description="测试规则描述",
            condition_type=RuleCondition.METRIC_THRESHOLD,
            condition_config={'metric': 'cpu', 'threshold': 80, 'operator': '>'},
            action_type=RuleAction.ALERT,
            action_config={'channel': 'slack'}
        )
        
        rule_id = engine.add_rule(rule)
        assert rule_id in engine.rules
    
    def test_evaluate_metric_threshold(self):
        """测试指标阈值评估"""
        from app.closed_loop.rules.rule_engine import Rule
        
        engine = RuleEngine()
        rule = Rule(
            rule_id="cpu_rule",
            name="CPU规则",
            description="CPU阈值规则",
            condition_type=RuleCondition.METRIC_THRESHOLD,
            condition_config={'metric': 'cpu_usage', 'threshold': 80, 'operator': '>'},
            action_type=RuleAction.ALERT
        )
        engine.add_rule(rule)
        
        # 测试满足条件
        context = {'metrics': {'cpu_usage': 90}}
        triggered = engine.evaluate_rule("cpu_rule", context)
        assert triggered is True
        
        # 测试不满足条件
        context = {'metrics': {'cpu_usage': 70}}
        triggered = engine.evaluate_rule("cpu_rule", context)
        assert triggered is False


class TestClosedLoopMetrics:
    """闭环指标测试"""
    
    def test_metrics_initialization(self):
        """测试指标收集器初始化"""
        metrics = ClosedLoopMetrics()
        assert metrics is not None
    
    def test_counter_increment(self):
        """测试计数器增加"""
        metrics = ClosedLoopMetrics()
        metrics.increment_counter("test_counter", 5)
        assert metrics.get_counter("test_counter") == 5
    
    def test_gauge_set(self):
        """测试仪表盘设置"""
        metrics = ClosedLoopMetrics()
        metrics.set_gauge("test_gauge", 42.0)
        assert metrics.get_gauge("test_gauge") == 42.0


# ==================== Phase 2 测试 ====================

class TestEventCollector:
    """事件收集器测试"""
    
    @pytest.mark.asyncio
    async def test_collector_initialization(self):
        """测试收集器初始化"""
        collector = EventCollector()
        assert collector is not None
    
    @pytest.mark.asyncio
    async def test_collect_event(self):
        """测试收集事件"""
        from app.closed_loop.rca.event_collector import Event
        
        collector = EventCollector()
        event = Event(
            event_id="test_event",
            event_type=EventType.ALERT,
            source="test_service",
            timestamp=datetime.now(),
            severity=EventSeverity.HIGH,
            title="测试事件",
            description="测试描述"
        )
        
        event_id = await collector.collect(event)
        assert event_id == "test_event"
        assert event_id in collector.events
    
    def test_query_events(self):
        """测试查询事件"""
        collector = EventCollector()
        events = collector.query_events(limit=10)
        assert isinstance(events, list)


class TestCorrelationAnalyzer:
    """关联分析器测试"""
    
    def test_analyzer_initialization(self):
        """测试分析器初始化"""
        collector = EventCollector()
        analyzer = CorrelationAnalyzer(collector)
        assert analyzer is not None


class TestForecastEngine:
    """预测引擎测试"""
    
    @pytest.mark.asyncio
    async def test_forecast_initialization(self):
        """测试预测引擎初始化"""
        engine = ForecastEngine()
        assert engine is not None
    
    def test_add_metric_data(self):
        """测试添加指标数据"""
        engine = ForecastEngine()
        
        for i in range(50):
            engine.add_metric_point(
                "test_metric",
                datetime.now() + timedelta(hours=i),
                50.0 + (i % 10)
            )
        
        stats = engine.get_metric_stats("test_metric")
        assert stats['data_points'] == 50
    
    @pytest.mark.asyncio
    async def test_forecast(self):
        """测试预测"""
        engine = ForecastEngine()
        
        # 添加历史数据
        for i in range(48):
            engine.add_metric_point(
                "cpu_usage",
                datetime.now() + timedelta(hours=i),
                50.0 + (i % 20)
            )
        
        # 执行预测
        result = await engine.forecast("cpu_usage", horizon_hours=12)
        
        assert result is not None
        assert len(result.forecast_values) == 12
        assert result.model_used is not None


class TestWorkflowEngine:
    """工作流引擎测试"""
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self):
        """测试工作流引擎初始化"""
        engine = WorkflowEngine()
        assert engine is not None
    
    def test_create_workflow(self):
        """测试创建工作流"""
        engine = WorkflowEngine()
        workflow = engine.create_workflow(
            name="测试工作流",
            description="测试工作流描述"
        )
        
        assert workflow is not None
        assert workflow.workflow_id in engine.workflows
    
    def test_add_workflow_step(self):
        """测试添加工作流步骤"""
        from app.closed_loop.workflow.engine import WorkflowStep
        
        engine = WorkflowEngine()
        workflow = engine.create_workflow("测试工作流", "描述")
        
        step = WorkflowStep(
            step_id="step1",
            name="第一步",
            action_type=WorkflowActionType.NOTIFY,
            params={'message': '测试消息'}
        )
        
        engine.add_workflow_step(workflow.workflow_id, step)
        assert "step1" in workflow.steps


# ==================== 集成测试 ====================

class TestClosedLoopIntegration:
    """闭环系统集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_loop(self):
        """测试完整闭环流程"""
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
        
        # 添加规则
        from app.closed_loop.rules.rule_engine import Rule
        rule = Rule(
            rule_id="test_anomaly_rule",
            name="测试异常规则",
            description="测试",
            condition_type=RuleCondition.ANOMALY_DETECTED,
            condition_config={'min_severity': 0.5},
            action_type=RuleAction.REMEDIATE
        )
        rule_engine.add_rule(rule)
        
        # 预填充正常数据
        for i in range(20):
            await controller.process_metric(
                "cpu_usage",
                50.0,
                datetime.now() + timedelta(minutes=i)
            )
        
        # 发送异常数据
        context = await controller.process_metric(
            "cpu_usage",
            150.0,  # 异常值
            datetime.now() + timedelta(minutes=20)
        )
        
        # 验证
        assert context is not None
        assert context.anomaly is not None


# ==================== 运行测试 ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
