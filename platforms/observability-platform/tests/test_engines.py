"""Tests for observability engines — MetricsEngine, AlertEngine, HealthEngine."""

import asyncio
import time

import pytest

from observability_platform.domain.entities import AlertRule
from observability_platform.domain.value_objects import (
    MetricType,
    AlertSeverity,
    AlertState,
)
from observability_platform.domain.exceptions import (
    MetricNotFoundError,
    AlertRuleError,
)
from observability_platform.engines.metrics_engine import MetricsEngine
from observability_platform.engines.alert_engine import AlertEngine
from observability_platform.engines.health_engine import HealthEngine
from platform_shared.protocols.health import HealthStatus


# ============================================================================
# MetricsEngine
# ============================================================================


class TestMetricsEngine:
    def test_record_and_query(self, metrics_engine: MetricsEngine):
        metrics_engine.record("cpu.usage", 75.0, MetricType.GAUGE)
        metrics_engine.record("cpu.usage", 80.0, MetricType.GAUGE)

        points = metrics_engine.query("cpu.usage")
        assert len(points) == 2
        assert points[0].value == 75.0
        assert points[1].value == 80.0

    def test_record_counter_accumulates(self, metrics_engine: MetricsEngine):
        metrics_engine.record("requests", 10, MetricType.COUNTER)
        metrics_engine.record("requests", 5, MetricType.COUNTER)
        metrics_engine.record("requests", 3, MetricType.COUNTER)

        points = metrics_engine.query("requests")
        assert len(points) == 3
        assert points[0].value == 10
        assert points[1].value == 15
        assert points[2].value == 18

    def test_query_nonexistent_raises(self, metrics_engine: MetricsEngine):
        with pytest.raises(MetricNotFoundError):
            metrics_engine.query("nonexistent")

    def test_query_with_time_range(self, metrics_engine: MetricsEngine):
        now = time.time()
        metrics_engine.record("mem", 100.0, timestamp=now - 10)
        metrics_engine.record("mem", 200.0, timestamp=now - 5)
        metrics_engine.record("mem", 300.0, timestamp=now)

        points = metrics_engine.query("mem", time_range=(now - 7, now - 1))
        assert len(points) == 1
        assert points[0].value == 200.0

    def test_record_with_tags(self, metrics_engine: MetricsEngine):
        metrics_engine.record("http.status", 200, MetricType.GAUGE, tags={"path": "/api"})
        points = metrics_engine.query("http.status")
        assert points[0].tags == {"path": "/api"}

    def test_get_latest(self, metrics_engine: MetricsEngine):
        metrics_engine.record("cpu", 50.0)
        metrics_engine.record("cpu", 60.0)
        latest = metrics_engine.get_latest("cpu")
        assert latest is not None
        assert latest.value == 60.0

    def test_get_latest_nonexistent(self, metrics_engine: MetricsEngine):
        assert metrics_engine.get_latest("missing") is None

    def test_aggregate_avg(self, metrics_engine: MetricsEngine):
        for v in [10.0, 20.0, 30.0]:
            metrics_engine.record("metric", v)
        assert metrics_engine.aggregate("metric", "avg") == 20.0

    def test_aggregate_max(self, metrics_engine: MetricsEngine):
        for v in [10.0, 50.0, 30.0]:
            metrics_engine.record("metric", v)
        assert metrics_engine.aggregate("metric", "max") == 50.0

    def test_aggregate_min(self, metrics_engine: MetricsEngine):
        for v in [10.0, 50.0, 30.0]:
            metrics_engine.record("metric", v)
        assert metrics_engine.aggregate("metric", "min") == 10.0

    def test_aggregate_sum(self, metrics_engine: MetricsEngine):
        for v in [10.0, 20.0, 30.0]:
            metrics_engine.record("metric", v)
        assert metrics_engine.aggregate("metric", "sum") == 60.0

    def test_aggregate_count(self, metrics_engine: MetricsEngine):
        for v in [1.0, 2.0, 3.0]:
            metrics_engine.record("metric", v)
        assert metrics_engine.aggregate("metric", "count") == 3.0

    def test_aggregate_p99(self, metrics_engine: MetricsEngine):
        for v in range(1, 101):
            metrics_engine.record("latency", float(v))
        p99 = metrics_engine.aggregate("latency", "p99")
        assert p99 >= 99.0

    def test_aggregate_unknown_function(self, metrics_engine: MetricsEngine):
        metrics_engine.record("metric", 1.0)
        with pytest.raises(ValueError, match="Unknown aggregation"):
            metrics_engine.aggregate("metric", "median")

    def test_metric_names(self, metrics_engine: MetricsEngine):
        metrics_engine.record("a", 1.0)
        metrics_engine.record("b", 2.0)
        assert sorted(metrics_engine.metric_names) == ["a", "b"]

    def test_clear_specific(self, metrics_engine: MetricsEngine):
        metrics_engine.record("a", 1.0)
        metrics_engine.record("b", 2.0)
        metrics_engine.clear("a")
        assert "a" not in metrics_engine.metric_names
        assert "b" in metrics_engine.metric_names

    def test_clear_all(self, metrics_engine: MetricsEngine):
        metrics_engine.record("a", 1.0)
        metrics_engine.record("b", 2.0)
        metrics_engine.clear()
        assert metrics_engine.metric_names == []


# ============================================================================
# AlertEngine
# ============================================================================


class TestAlertEngine:
    def test_register_rule(self, alert_engine: AlertEngine, metrics_engine: MetricsEngine):
        rule = AlertRule(
            name="high_cpu",
            metric_name="cpu",
            condition="gt",
            threshold=90.0,
            severity=AlertSeverity.CRITICAL,
        )
        alert_engine.register_rule(rule)
        assert "high_cpu" in alert_engine.rules

    def test_register_duplicate_raises(self, alert_engine: AlertEngine):
        rule = AlertRule(name="r1", metric_name="m", condition="gt", threshold=1.0)
        alert_engine.register_rule(rule)
        with pytest.raises(AlertRuleError, match="already registered"):
            alert_engine.register_rule(rule)

    def test_register_invalid_condition_raises(self, alert_engine: AlertEngine):
        rule = AlertRule(name="bad", metric_name="m", condition="invalid", threshold=1.0)
        with pytest.raises(AlertRuleError, match="Invalid condition"):
            alert_engine.register_rule(rule)

    def test_evaluate_fires_alert(
        self, alert_engine: AlertEngine, metrics_engine: MetricsEngine
    ):
        rule = AlertRule(
            name="high_cpu",
            metric_name="cpu",
            condition="gt",
            threshold=90.0,
            severity=AlertSeverity.CRITICAL,
        )
        alert_engine.register_rule(rule)

        # Record a metric above threshold
        metrics_engine.record("cpu", 95.0, MetricType.GAUGE)

        results = alert_engine.evaluate()
        assert len(results) == 1
        assert results[0].state == AlertState.FIRING
        assert results[0].value == 95.0

    def test_evaluate_resolves_alert(
        self, alert_engine: AlertEngine, metrics_engine: MetricsEngine
    ):
        rule = AlertRule(name="high_cpu", metric_name="cpu", condition="gt", threshold=90.0)
        alert_engine.register_rule(rule)

        # Fire it
        metrics_engine.record("cpu", 95.0, MetricType.GAUGE)
        alert_engine.evaluate()

        # Resolve it
        metrics_engine.record("cpu", 50.0, MetricType.GAUGE)
        results = alert_engine.evaluate()
        assert results[0].state == AlertState.RESOLVED

    def test_evaluate_with_for_seconds(
        self, alert_engine: AlertEngine, metrics_engine: MetricsEngine
    ):
        rule = AlertRule(
            name="sustained_cpu",
            metric_name="cpu",
            condition="gt",
            threshold=90.0,
            for_seconds=60.0,  # Must sustain for 60 seconds
        )
        alert_engine.register_rule(rule)

        metrics_engine.record("cpu", 95.0, MetricType.GAUGE)

        # First evaluation: transitions to PENDING
        results = alert_engine.evaluate()
        assert results[0].state == AlertState.PENDING

    def test_evaluate_no_data(self, alert_engine: AlertEngine):
        rule = AlertRule(name="no_data", metric_name="missing", condition="gt", threshold=1.0)
        alert_engine.register_rule(rule)
        results = alert_engine.evaluate()
        assert results[0].state == AlertState.RESOLVED

    def test_firing_alerts_property(
        self, alert_engine: AlertEngine, metrics_engine: MetricsEngine
    ):
        rule = AlertRule(name="test", metric_name="cpu", condition="gt", threshold=50.0)
        alert_engine.register_rule(rule)
        metrics_engine.record("cpu", 100.0, MetricType.GAUGE)
        alert_engine.evaluate()
        assert len(alert_engine.firing_alerts) == 1

    def test_events_emitted(
        self, alert_engine: AlertEngine, metrics_engine: MetricsEngine
    ):
        rule = AlertRule(name="test", metric_name="cpu", condition="gt", threshold=50.0)
        alert_engine.register_rule(rule)

        # Fire
        metrics_engine.record("cpu", 100.0, MetricType.GAUGE)
        alert_engine.evaluate()

        # Resolve
        metrics_engine.record("cpu", 10.0, MetricType.GAUGE)
        alert_engine.evaluate()

        events = alert_engine.events
        assert len(events) == 2
        from observability_platform.domain.events import AlertFired, AlertResolved

        assert isinstance(events[0], AlertFired)
        assert isinstance(events[1], AlertResolved)

    def test_unregister_rule(self, alert_engine: AlertEngine):
        rule = AlertRule(name="temp", metric_name="x", condition="gt", threshold=1.0)
        alert_engine.register_rule(rule)
        alert_engine.unregister_rule("temp")
        assert "temp" not in alert_engine.rules


# ============================================================================
# HealthEngine
# ============================================================================


class TestHealthEngine:
    async def test_register_and_run_healthy(self, health_engine: HealthEngine):
        async def check():
            return True

        health_engine.register_check("db", check)
        reports = await health_engine.run_checks()
        assert len(reports) == 1
        assert reports[0].status == HealthStatus.HEALTHY
        assert reports[0].component == "db"

    async def test_register_and_run_unhealthy(self, health_engine: HealthEngine):
        async def check():
            return False

        health_engine.register_check("db", check)
        reports = await health_engine.run_checks()
        assert reports[0].status == HealthStatus.UNHEALTHY

    async def test_check_timeout(self, health_engine: HealthEngine):
        async def slow_check():
            await asyncio.sleep(10)
            return True

        health_engine.register_check("slow", slow_check, timeout_seconds=0.1)
        reports = await health_engine.run_checks()
        assert reports[0].status == HealthStatus.UNHEALTHY
        assert "timed out" in reports[0].message

    async def test_check_exception(self, health_engine: HealthEngine):
        async def broken_check():
            raise RuntimeError("connection refused")

        health_engine.register_check("broken", broken_check)
        reports = await health_engine.run_checks()
        assert reports[0].status == HealthStatus.UNHEALTHY
        assert "connection refused" in reports[0].message

    async def test_aggregate_all_healthy(self, health_engine: HealthEngine):
        async def ok():
            return True

        health_engine.register_check("a", ok)
        health_engine.register_check("b", ok)
        report = await health_engine.get_aggregate_status()
        assert report.status == HealthStatus.HEALTHY

    async def test_aggregate_one_unhealthy(self, health_engine: HealthEngine):
        async def ok():
            return True

        async def bad():
            return False

        health_engine.register_check("good", ok)
        health_engine.register_check("bad", bad)
        report = await health_engine.get_aggregate_status()
        assert report.status == HealthStatus.UNHEALTHY
        assert "bad" in report.message

    async def test_aggregate_no_checks(self, health_engine: HealthEngine):
        report = await health_engine.get_aggregate_status()
        assert report.status == HealthStatus.HEALTHY
        assert "No checks registered" in report.message

    async def test_registered_checks(self, health_engine: HealthEngine):
        async def ok():
            return True

        health_engine.register_check("a", ok)
        health_engine.register_check("b", ok)
        assert sorted(health_engine.registered_checks) == ["a", "b"]

    async def test_unregister_check(self, health_engine: HealthEngine):
        async def ok():
            return True

        health_engine.register_check("temp", ok)
        health_engine.unregister_check("temp")
        assert "temp" not in health_engine.registered_checks

    async def test_health_changed_events(self, health_engine: HealthEngine):
        async def ok():
            return True

        async def bad():
            return False

        health_engine.register_check("flaky", ok)

        # First run: UNKNOWN -> HEALTHY (no event since previous was UNKNOWN)
        await health_engine.run_checks()

        # Now swap to bad
        health_engine._checks["flaky"].check_fn = bad
        await health_engine.run_checks()

        events = health_engine.events
        assert len(events) == 1
        assert events[0].component == "flaky"
        assert events[0].status == "unhealthy"
