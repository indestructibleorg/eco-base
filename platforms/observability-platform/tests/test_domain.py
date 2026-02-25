"""Tests for observability domain entities, value objects, events, and exceptions."""

import pytest

from observability_platform.domain.value_objects import (
    MetricType,
    AlertSeverity,
    AlertState,
    LogLevel,
)
from observability_platform.domain.entities import (
    Metric,
    Alert,
    AlertRule,
    HealthCheck,
    LogEntry,
)
from observability_platform.domain.events import (
    MetricRecorded,
    AlertFired,
    AlertResolved,
    HealthChanged,
)
from observability_platform.domain.exceptions import (
    ObservabilityError,
    MetricNotFoundError,
    AlertRuleError,
    HealthCheckError,
)


# -- Value Objects --


class TestMetricType:
    def test_values(self):
        assert MetricType.COUNTER == "counter"
        assert MetricType.GAUGE == "gauge"
        assert MetricType.HISTOGRAM == "histogram"

    def test_from_string(self):
        assert MetricType("counter") == MetricType.COUNTER


class TestAlertSeverity:
    def test_values(self):
        assert AlertSeverity.CRITICAL == "critical"
        assert AlertSeverity.WARNING == "warning"
        assert AlertSeverity.INFO == "info"


class TestAlertState:
    def test_values(self):
        assert AlertState.PENDING == "pending"
        assert AlertState.FIRING == "firing"
        assert AlertState.RESOLVED == "resolved"


class TestLogLevel:
    def test_values(self):
        assert LogLevel.DEBUG == "debug"
        assert LogLevel.INFO == "info"
        assert LogLevel.WARNING == "warning"
        assert LogLevel.ERROR == "error"
        assert LogLevel.CRITICAL == "critical"


# -- Entities --


class TestMetric:
    def test_creation(self):
        metric = Metric(name="cpu.usage", value=75.5, metric_type=MetricType.GAUGE)
        assert metric.name == "cpu.usage"
        assert metric.value == 75.5
        assert metric.metric_type == MetricType.GAUGE
        assert metric.metric_id

    def test_with_tags(self):
        metric = Metric(
            name="requests.total",
            value=100,
            metric_type=MetricType.COUNTER,
            tags={"host": "web-1"},
        )
        assert metric.tags == {"host": "web-1"}


class TestAlertRule:
    def test_evaluate_gt(self):
        rule = AlertRule(name="high_cpu", metric_name="cpu", condition="gt", threshold=90.0)
        assert rule.evaluate(95.0) is True
        assert rule.evaluate(90.0) is False
        assert rule.evaluate(85.0) is False

    def test_evaluate_lt(self):
        rule = AlertRule(name="low_disk", metric_name="disk", condition="lt", threshold=10.0)
        assert rule.evaluate(5.0) is True
        assert rule.evaluate(10.0) is False
        assert rule.evaluate(15.0) is False

    def test_evaluate_eq(self):
        rule = AlertRule(name="exact", metric_name="count", condition="eq", threshold=42.0)
        assert rule.evaluate(42.0) is True
        assert rule.evaluate(41.0) is False

    def test_invalid_condition(self):
        rule = AlertRule(name="bad", metric_name="x", condition="unknown", threshold=1.0)
        assert rule.evaluate(1.0) is False


class TestAlert:
    def test_default_state(self):
        rule = AlertRule(name="test", metric_name="cpu", condition="gt", threshold=90)
        alert = Alert(rule=rule)
        assert alert.state == AlertState.PENDING
        assert alert.fired_at is None
        assert alert.resolved_at is None


class TestHealthCheck:
    def test_creation(self):
        async def check():
            return True

        hc = HealthCheck(name="db", check_fn=check)
        assert hc.name == "db"
        assert hc.timeout_seconds == 5.0


class TestLogEntry:
    def test_creation(self):
        entry = LogEntry(message="test message", level=LogLevel.ERROR, source="app")
        assert entry.message == "test message"
        assert entry.level == LogLevel.ERROR
        assert entry.source == "app"
        assert entry.entry_id


# -- Events --


class TestMetricRecorded:
    def test_creation(self):
        event = MetricRecorded(metric_name="cpu", value=75.0, tags={"host": "a"})
        assert event.metric_name == "cpu"
        assert event.value == 75.0
        assert event.timestamp > 0


class TestAlertFired:
    def test_creation(self):
        event = AlertFired(
            rule_name="high_cpu",
            metric_name="cpu",
            value=95.0,
            threshold=90.0,
            severity=AlertSeverity.CRITICAL,
        )
        assert event.rule_name == "high_cpu"
        assert event.severity == AlertSeverity.CRITICAL


class TestAlertResolved:
    def test_creation(self):
        event = AlertResolved(rule_name="high_cpu", metric_name="cpu")
        assert event.rule_name == "high_cpu"


class TestHealthChanged:
    def test_creation(self):
        event = HealthChanged(component="db", status="unhealthy", previous_status="healthy")
        assert event.component == "db"


# -- Exceptions --


class TestExceptions:
    def test_observability_error(self):
        err = ObservabilityError("test")
        assert str(err) == "test"
        assert err.code == "OBSERVABILITY_ERROR"

    def test_metric_not_found(self):
        err = MetricNotFoundError("cpu.usage")
        assert err.metric_name == "cpu.usage"
        assert "cpu.usage" in str(err)

    def test_alert_rule_error(self):
        err = AlertRuleError("bad rule", rule_name="rule1")
        assert err.rule_name == "rule1"

    def test_health_check_error(self):
        err = HealthCheckError("fail", component="db")
        assert err.component == "db"
