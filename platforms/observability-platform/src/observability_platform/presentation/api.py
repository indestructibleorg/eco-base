"""Observability API — programmatic interface for metrics, alerts, and health.

Provides:
  POST /metrics          — record a metric
  GET  /metrics/{name}   — query metric data
  GET  /alerts           — list current alerts
  POST /alerts/evaluate  — evaluate all alert rules
  GET  /health           — aggregate health check
"""

from __future__ import annotations

from typing import Any

from observability_platform.engines.metrics_engine import MetricsEngine
from observability_platform.engines.alert_engine import AlertEngine
from observability_platform.engines.health_engine import HealthEngine
from observability_platform.domain.value_objects import MetricType, AlertState


class ObservabilityAPI:
    """In-process API facade for the observability platform.

    Wraps the three engines and provides a request/response style interface.
    """

    def __init__(
        self,
        metrics_engine: MetricsEngine | None = None,
        alert_engine: AlertEngine | None = None,
        health_engine: HealthEngine | None = None,
    ) -> None:
        self._metrics = metrics_engine or MetricsEngine()
        self._alerts = alert_engine or AlertEngine(self._metrics)
        self._health = health_engine or HealthEngine()

    async def post_metric(self, data: dict[str, Any]) -> dict[str, Any]:
        """POST /metrics — record a metric data point.

        Expected data:
            name: str
            value: float
            type: str (counter/gauge/histogram)
            tags: dict (optional)
        """
        name = data["name"]
        value = float(data["value"])
        metric_type = MetricType(data.get("type", "gauge"))
        tags = data.get("tags", {})

        metric = self._metrics.record(name, value, metric_type, tags)
        return {
            "metric_id": metric.metric_id,
            "name": metric.name,
            "value": metric.value,
            "type": metric.metric_type.value,
        }

    async def get_metric(self, name: str, time_range: tuple[float, float] | None = None) -> dict[str, Any]:
        """GET /metrics/{name} — query metric data points."""
        points = self._metrics.query(name, time_range)
        return {
            "name": name,
            "points": [
                {"timestamp": p.timestamp, "value": p.value, "tags": p.tags}
                for p in points
            ],
            "count": len(points),
        }

    async def get_alerts(self) -> dict[str, Any]:
        """GET /alerts — list all alert states."""
        alerts = self._alerts.alerts
        return {
            "alerts": [
                {
                    "rule": a.rule.name,
                    "state": a.state.value,
                    "value": a.value,
                    "severity": a.rule.severity.value,
                }
                for a in alerts.values()
            ],
            "firing_count": len(self._alerts.firing_alerts),
        }

    async def evaluate_alerts(self) -> dict[str, Any]:
        """POST /alerts/evaluate — evaluate all alert rules."""
        results = self._alerts.evaluate()
        return {
            "evaluated": len(results),
            "alerts": [
                {
                    "rule": a.rule.name,
                    "state": a.state.value,
                    "value": a.value,
                }
                for a in results
            ],
        }

    async def get_health(self) -> dict[str, Any]:
        """GET /health — aggregate health status."""
        report = await self._health.get_aggregate_status()
        return {
            "status": report.status.value,
            "message": report.message,
            "details": report.details,
        }
