"""Core domain entities for observability."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from observability_platform.domain.value_objects import (
    MetricType,
    AlertSeverity,
    AlertState,
    LogLevel,
)


@dataclass(slots=True)
class Metric:
    """A recorded metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)
    metric_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])


@dataclass(slots=True)
class AlertRule:
    """Rule that defines when an alert should fire."""
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq"
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    for_seconds: float = 0.0
    rule_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def evaluate(self, value: float) -> bool:
        """Check whether the given value triggers this rule."""
        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        elif self.condition == "eq":
            return value == self.threshold
        return False


@dataclass(slots=True)
class Alert:
    """A fired or resolved alert."""
    rule: AlertRule
    state: AlertState = AlertState.PENDING
    value: float = 0.0
    fired_at: float | None = None
    resolved_at: float | None = None
    alert_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    pending_since: float | None = None


@dataclass(slots=True)
class HealthCheck:
    """A health check definition with an async callable."""
    name: str
    check_fn: Callable[[], Awaitable[bool]]
    timeout_seconds: float = 5.0
    check_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])


@dataclass(slots=True)
class LogEntry:
    """A structured log entry."""
    message: str
    level: LogLevel = LogLevel.INFO
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
