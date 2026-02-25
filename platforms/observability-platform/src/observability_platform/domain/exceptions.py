"""Custom exceptions for the observability platform."""

from __future__ import annotations

from platform_shared.domain.errors import PlatformError


class ObservabilityError(PlatformError):
    """Base error for all observability platform operations."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, code="OBSERVABILITY_ERROR", **kwargs)


class MetricNotFoundError(ObservabilityError):
    """Raised when a queried metric does not exist."""

    def __init__(self, metric_name: str):
        super().__init__(f"Metric not found: {metric_name}")
        self.metric_name = metric_name


class AlertRuleError(ObservabilityError):
    """Raised when an alert rule configuration is invalid."""

    def __init__(self, message: str, rule_name: str = ""):
        super().__init__(message)
        self.rule_name = rule_name


class HealthCheckError(ObservabilityError):
    """Raised when a health check fails unexpectedly."""

    def __init__(self, message: str, app.kubernetes.io/component: str = ""):
        super().__init__(message)
        self.component = component
