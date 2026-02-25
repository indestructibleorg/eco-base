"""Observability domain — entities, value objects, events, exceptions."""

from observability_platform.domain.entities import (
    Metric,
    Alert,
    AlertRule,
    HealthCheck,
    LogEntry,
)
from observability_platform.domain.value_objects import (
    MetricType,
    AlertSeverity,
    AlertState,
    LogLevel,
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

__all__ = [
    "Metric",
    "Alert",
    "AlertRule",
    "HealthCheck",
    "LogEntry",
    "MetricType",
    "AlertSeverity",
    "AlertState",
    "LogLevel",
    "MetricRecorded",
    "AlertFired",
    "AlertResolved",
    "HealthChanged",
    "ObservabilityError",
    "MetricNotFoundError",
    "AlertRuleError",
    "HealthCheckError",
]
