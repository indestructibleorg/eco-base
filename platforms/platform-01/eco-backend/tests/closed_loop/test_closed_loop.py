#!/usr/bin/env python3
# =============================================================================
# Closed Loop System Tests
# =============================================================================
# 閉環系統測試 - Phase 1 基礎閉環測試
# =============================================================================

import pytest
import asyncio
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.closed_loop import (
    # 異常檢測
    AnomalyDetector, DetectionRule, AnomalyEvent, AnomalyType, Severity,
    StatisticalDetector, TrendDetector,
    
    # 自動修復
    Remediator, RemediationAction, RemediationType, RemediationStatus,
    ClearCacheHandler, SwitchProviderHandler,
    
    # 規則引擎
    RuleEngine, Rule, Action, Condition, ConditionOperator,
    CompoundCondition, LogicalOperator, RuleParser,
    
    # 指標
    ClosedLoopMetrics, Counter, Gauge, Histogram,
    
    # 控制器
    ClosedLoopController, ClosedLoopConfig, ClosedLoopMode, ClosedLoopState,
)


# =============================================================================
# 異常檢測測試
# =============================================================================

class TestStatisticalDetector:
    """統計異常檢測測試"""
    
    def test_detect_sigma_normal_data(self):
        """測試 3-sigma 檢測 - 正常數據"""
        # 正態分布數據
        data = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.0]
        
        results = StatisticalDetector.detect_sigma(data)
        
        # 正常數據應該沒有異常
        assert not any(results)
    
    def test_detect_sigma_with_anomaly(self):
        """測試 3-sigma 檢測 - 包含異常"""
        data = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 100.0]  # 最後一個是異常
        
        results = StatisticalDetector.detect_sigma(data)
        
        # 最後一個應該被檢測為異常
        assert results[-1] == True
    
    def test_detect_iqr(self):
        """測試 IQR 檢測"""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 是異常
        
        results = StatisticalDetector.detect_iqr(data)
        
        assert results[-1] == True
    
    def test_detect_mad(self):
        """測試 MAD 檢測"""
        data = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 50.0]  # 50 是異常
        
        results = StatisticalDetector.detect_mad(data)
        
        assert results[-1] == True


class TestTrendDetector:
    """趨勢異常檢測測試"""
    
    def test_detect_trend_change(self):
        """測試趨勢變化檢測"""
        # 前 10 個穩定，後 10 個急劇上升
        data = [10.0] * 10 + [10.0 + i * 2 for i in range(10)]
        
        results = TrendDetector.detect_trend_change(data, window_size=5)
        
        # 趨勢變化點應該被檢測
        assert any(results)


class TestAnomalyDetector:
    """異常檢測引擎測試"""
    
    @pytest.fixture
    def detector(self):
        return AnomalyDetector()
    
    @pytest.fixture
    def sample_rule(self):
        return DetectionRule(
            name="test_rule",
            metric_name="test_metric",
            anomaly_type=AnomalyType.THRESHOLD,
            severity=Severity.WARNING,
            threshold_max=100.0
        )
    
    @pytest.mark.asyncio
    async def test_register_rule(self, detector, sample_rule):
        """測試註冊規則"""
        detector.register_rule(sample_rule)
        
        assert "test_rule" in detector.rules
        assert detector.rules["test_rule"].name == "test_rule"
    
    @pytest.mark.asyncio
    async def test_unregister_rule(self, detector, sample_rule):
        """測試註銷規則"""
        detector.register_rule(sample_rule)
        result = detector.unregister_rule("test_rule")
        
        assert result == True
        assert "test_rule" not in detector.rules
    
    @pytest.mark.asyncio
    async def test_detect_threshold_anomaly(self, detector, sample_rule):
        """測試閾值異常檢測"""
        detector.register_rule(sample_rule)
        
        # 超過閾值應該觸發異常
        anomaly = await detector.detect("test_rule", 150.0)
        
        assert anomaly is not None
        assert anomaly.metric_name == "test_metric"
        assert anomaly.severity == Severity.WARNING
    
    @pytest.mark.asyncio
    async def test_detect_no_anomaly(self, detector, sample_rule):
        """測試無異常情況"""
        detector.register_rule(sample_rule)
        
        # 正常值不應該觸發異常
        anomaly = await detector.detect("test_rule", 50.0)
        
        assert anomaly is None
    
    @pytest.mark.asyncio
    async def test_cooldown(self, detector, sample_rule):
        """測試冷卻機制"""
        sample_rule.cooldown_minutes = 10
        detector.register_rule(sample_rule)
        
        # 第一次檢測
        anomaly1 = await detector.detect("test_rule", 150.0)
        assert anomaly1 is not None
        
        # 冷卻期內再次檢測
        anomaly2 = await detector.detect("test_rule", 150.0)
        assert anomaly2 is None  # 應該被冷卻
    
    def test_list_rules(self, detector, sample_rule):
        """測試列出規則"""
        detector.register_rule(sample_rule)
        
        rules = detector.list_rules()
        
        assert len(rules) == 1
        assert rules[0]["name"] == "test_rule"


# =============================================================================
# 自動修復測試
# =============================================================================

class TestRemediator:
    """自動修復引擎測試"""
    
    @pytest.fixture
    def remediator(self):
        return Remediator()
    
    @pytest.mark.asyncio
    async def test_clear_cache_action(self, remediator):
        """測試清理緩存動作"""
        action = RemediationAction(
            action_type=RemediationType.CLEAR_CACHE,
            target="test",
            parameters={"pattern": "test_*"}
        )
        
        result = await remediator.execute(action)
        
        assert result.status == RemediationStatus.SUCCESS
        assert "Cleared" in result.message
    
    @pytest.mark.asyncio
    async def test_switch_provider_action(self, remediator):
        """測試切換提供者動作"""
        action = RemediationAction(
            action_type=RemediationType.SWITCH_PROVIDER,
            target="test",
            parameters={
                "domain": "cognitive",
                "new_provider": "delta-cognitive"
            }
        )
        
        result = await remediator.execute(action)
        
        assert result.status == RemediationStatus.SUCCESS
        assert "Switched" in result.message
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, remediator):
        """測試重試機制"""
        # 創建一個會失敗的動作（無效的動作類型）
        action = RemediationAction(
            action_type=RemediationType.ENABLE_CIRCUIT_BREAKER,  # 未註冊處理器
            target="test",
            parameters={},
            retry_count=2,
            retry_delay_seconds=0
        )
        
        result = await remediator.execute(action)
        
        # 應該失敗但經過重試
        assert result.status == RemediationStatus.FAILED
    
    def test_get_statistics(self, remediator):
        """測試獲取統計信息"""
        stats = remediator.get_statistics()
        
        assert "total_actions" in stats
        assert "success_rate" in stats


# =============================================================================
# 規則引擎測試
# =============================================================================

class TestCondition:
    """條件測試"""
    
    def test_condition_eq(self):
        """測試等於條件"""
        condition = Condition(
            field="status",
            operator=ConditionOperator.EQ,
            value="error"
        )
        
        assert condition.evaluate({"status": "error"}) == True
        assert condition.evaluate({"status": "ok"}) == False
    
    def test_condition_gt(self):
        """測試大於條件"""
        condition = Condition(
            field="latency",
            operator=ConditionOperator.GT,
            value=100
        )
        
        assert condition.evaluate({"latency": 150}) == True
        assert condition.evaluate({"latency": 50}) == False
    
    def test_condition_contains(self):
        """測試包含條件"""
        condition = Condition(
            field="message",
            operator=ConditionOperator.CONTAINS,
            value="error"
        )
        
        assert condition.evaluate({"message": "an error occurred"}) == True
        assert condition.evaluate({"message": "success"}) == False
    
    def test_nested_field(self):
        """測試嵌套字段"""
        condition = Condition(
            field="data.value",
            operator=ConditionOperator.EQ,
            value=100
        )
        
        assert condition.evaluate({"data": {"value": 100}}) == True
        assert condition.evaluate({"data": {"value": 200}}) == False


class TestCompoundCondition:
    """複合條件測試"""
    
    def test_and_condition(self):
        """測試 AND 條件"""
        condition = CompoundCondition(
            operator=LogicalOperator.AND,
            conditions=[
                Condition("status", ConditionOperator.EQ, "error"),
                Condition("count", ConditionOperator.GT, 10)
            ]
        )
        
        assert condition.evaluate({"status": "error", "count": 15}) == True
        assert condition.evaluate({"status": "error", "count": 5}) == False
        assert condition.evaluate({"status": "ok", "count": 15}) == False
    
    def test_or_condition(self):
        """測試 OR 條件"""
        condition = CompoundCondition(
            operator=LogicalOperator.OR,
            conditions=[
                Condition("status", ConditionOperator.EQ, "error"),
                Condition("status", ConditionOperator.EQ, "warning")
            ]
        )
        
        assert condition.evaluate({"status": "error"}) == True
        assert condition.evaluate({"status": "warning"}) == True
        assert condition.evaluate({"status": "ok"}) == False


class TestRuleParser:
    """規則解析器測試"""
    
    def test_parse_simple_condition(self):
        """測試解析簡單條件"""
        data = {
            "field": "status",
            "operator": "==",
            "value": "error"
        }
        
        condition = RuleParser.parse_condition(data)
        
        assert isinstance(condition, Condition)
        assert condition.field == "status"
        assert condition.operator == ConditionOperator.EQ
    
    def test_parse_compound_condition(self):
        """測試解析複合條件"""
        data = {
            "and": [
                {"field": "status", "operator": "==", "value": "error"},
                {"field": "count", "operator": ">", "value": 10}
            ]
        }
        
        condition = RuleParser.parse_condition(data)
        
        assert isinstance(condition, CompoundCondition)
        assert condition.operator == LogicalOperator.AND
        assert len(condition.conditions) == 2
    
    def test_load_from_yaml(self):
        """測試從 YAML 加載規則"""
        yaml_content = """
rules:
  - name: test_rule
    description: Test rule
    enabled: true
    priority: 100
    cooldown_minutes: 5
    condition:
      field: status
      operator: "=="
      value: error
    actions:
      - type: clear_cache
        parameters:
          pattern: "*"
        delay_seconds: 0
        require_approval: false
"""
        
        rules = RuleParser.load_from_yaml(yaml_content)
        
        assert len(rules) == 1
        assert rules[0].name == "test_rule"
        assert len(rules[0].actions) == 1


class TestRuleEngine:
    """規則引擎測試"""
    
    @pytest.fixture
    def engine(self):
        return RuleEngine()
    
    @pytest.fixture
    def sample_rule(self):
        return Rule(
            name="test_rule",
            description="Test rule",
            condition=Condition("status", ConditionOperator.EQ, "error"),
            actions=[Action("clear_cache", {"pattern": "*"})]
        )
    
    def test_add_rule(self, engine, sample_rule):
        """測試添加規則"""
        engine.add_rule(sample_rule)
        
        assert "test_rule" in engine.rules
    
    def test_evaluate_rule_triggered(self, engine, sample_rule):
        """測試規則觸發"""
        engine.add_rule(sample_rule)
        
        facts = {"status": "error"}
        triggered = engine.evaluate_rule("test_rule", facts)
        
        assert triggered == True
    
    def test_evaluate_rule_not_triggered(self, engine, sample_rule):
        """測試規則未觸發"""
        engine.add_rule(sample_rule)
        
        facts = {"status": "ok"}
        triggered = engine.evaluate_rule("test_rule", facts)
        
        assert triggered == False
    
    def test_evaluate_all(self, engine, sample_rule):
        """測試評估所有規則"""
        engine.add_rule(sample_rule)
        
        facts = {"status": "error"}
        triggered = engine.evaluate_all(facts)
        
        assert "test_rule" in triggered
    
    def test_rule_cooldown(self, engine, sample_rule):
        """測試規則冷卻"""
        sample_rule.cooldown_minutes = 10
        sample_rule.last_triggered = datetime.utcnow()
        engine.add_rule(sample_rule)
        
        facts = {"status": "error"}
        triggered = engine.evaluate_rule("test_rule", facts)
        
        assert triggered == False


# =============================================================================
# 指標測試
# =============================================================================

class TestCounter:
    """計數器測試"""
    
    def test_inc(self):
        """測試增加"""
        counter = Counter("test", "Test counter", ["label"])
        
        counter.inc(label="a")
        counter.inc(2, label="a")
        counter.inc(label="b")
        
        assert counter.get(label="a") == 3
        assert counter.get(label="b") == 1
    
    def test_reset(self):
        """測試重置"""
        counter = Counter("test", "Test counter")
        
        counter.inc()
        counter.reset()
        
        assert counter.get() == 0


class TestGauge:
    """儀表測試"""
    
    def test_set(self):
        """測試設置"""
        gauge = Gauge("test", "Test gauge")
        
        gauge.set(100)
        
        assert gauge.get() == 100
    
    def test_inc_dec(self):
        """測試增減"""
        gauge = Gauge("test", "Test gauge")
        
        gauge.set(100)
        gauge.inc(10)
        gauge.dec(5)
        
        assert gauge.get() == 105


class TestHistogram:
    """直方圖測試"""
    
    def test_observe(self):
        """測試觀察"""
        hist = Histogram("test", "Test histogram")
        
        hist.observe(0.1)
        hist.observe(0.5)
        hist.observe(1.0)
        
        stats = hist.get_stats()
        
        assert stats["count"] == 3
        assert stats["min"] == 0.1
        assert stats["max"] == 1.0


class TestClosedLoopMetrics:
    """閉環指標測試"""
    
    @pytest.fixture
    def metrics(self):
        return ClosedLoopMetrics()
    
    def test_record_anomaly_detection(self, metrics):
        """測試記錄異常檢測"""
        metrics.record_anomaly_detection(
            metric_name="test",
            anomaly_type="threshold",
            severity="warning",
            latency_seconds=0.5
        )
        
        summary = metrics.get_summary()
        
        assert summary["anomaly_detections"]["total"] == 1
    
    def test_record_remediation(self, metrics):
        """測試記錄自動修復"""
        metrics.record_remediation(
            action_type="restart_pod",
            status="success",
            latency_seconds=5.0
        )
        
        summary = metrics.get_summary()
        
        assert summary["remediations"]["total"] == 1
    
    def test_calculate_mttd(self, metrics):
        """測試計算 MTTD"""
        # 記錄一些檢測
        for _ in range(5):
            metrics.record_anomaly_detection(
                metric_name="test",
                anomaly_type="threshold",
                severity="warning",
                latency_seconds=1.0
            )
        
        mttd = metrics.calculate_mttd()
        
        assert mttd == 1.0
    
    def test_export_prometheus_format(self, metrics):
        """測試導出 Prometheus 格式"""
        metrics.record_anomaly_detection(
            metric_name="test",
            anomaly_type="threshold",
            severity="warning",
            latency_seconds=0.5
        )
        
        output = metrics.export_prometheus_format()
        
        assert "closed_loop_anomaly_detections_total" in output
        assert "# HELP" in output
        assert "# TYPE" in output


# =============================================================================
# 控制器測試
# =============================================================================

class TestClosedLoopController:
    """閉環控制器測試"""
    
    @pytest.fixture
    def controller(self):
        config = ClosedLoopConfig(
            mode=ClosedLoopMode.MANUAL,
            detection_interval_seconds=1
        )
        return ClosedLoopController(config)
    
    @pytest.mark.asyncio
    async def test_start_stop(self, controller):
        """測試啟動和停止"""
        await controller.start()
        
        assert controller._running == True
        assert controller.state == ClosedLoopState.IDLE
        
        await controller.stop()
        
        assert controller._running == False
    
    def test_add_detection_rule(self, controller):
        """測試添加檢測規則"""
        rule = DetectionRule(
            name="test",
            metric_name="test_metric",
            anomaly_type=AnomalyType.THRESHOLD,
            severity=Severity.WARNING,
            threshold_max=100
        )
        
        controller.add_detection_rule(rule)
        
        assert "test" in controller.detector.rules
    
    def test_add_remediation_rule(self, controller):
        """測試添加修復規則"""
        rule = Rule(
            name="test",
            condition=Condition("status", ConditionOperator.EQ, "error"),
            actions=[Action("clear_cache", {})]
        )
        
        controller.add_remediation_rule(rule)
        
        assert "test" in controller.rule_engine.rules
    
    def test_get_status(self, controller):
        """測試獲取狀態"""
        status = controller.get_status()
        
        assert "state" in status
        assert "mode" in status
        assert "running" in status
    
    def test_setup_default_rules(self, controller):
        """測試設置默認規則"""
        controller.setup_default_rules()
        
        assert len(controller.detector.rules) > 0
        assert len(controller.rule_engine.rules) > 0


# =============================================================================
# 整合測試
# =============================================================================

class TestClosedLoopIntegration:
    """閉環系統整合測試"""
    
    @pytest.mark.asyncio
    async def test_full_closed_loop_flow(self):
        """測試完整閉環流程"""
        # 創建控制器
        config = ClosedLoopConfig(mode=ClosedLoopMode.FULL_AUTO)
        controller = ClosedLoopController(config)
        
        # 設置檢測規則
        detection_rule = DetectionRule(
            name="high_latency",
            metric_name="latency",
            anomaly_type=AnomalyType.THRESHOLD,
            severity=Severity.WARNING,
            threshold_max=100
        )
        controller.add_detection_rule(detection_rule)
        
        # 設置修復規則
        remediation_rule = Rule(
            name="scale_on_high_latency",
            condition=Condition(
                field="anomaly.rule_name",
                operator=ConditionOperator.EQ,
                value="high_latency"
            ),
            actions=[
                Action(
                    action_type="scale_up",
                    parameters={"namespace": "default", "deployment": "test", "replicas_delta": 1}
                )
            ]
        )
        controller.add_remediation_rule(remediation_rule)
        
        # 手動觸發檢測
        anomaly = await controller.detector.detect("high_latency", 150.0)
        
        assert anomaly is not None
        assert anomaly.metric_name == "latency"
        
        # 驗證規則評估
        facts = {"anomaly": {"rule_name": "high_latency"}}
        triggered = controller.rule_engine.evaluate_all(facts)
        
        assert "scale_on_high_latency" in triggered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
