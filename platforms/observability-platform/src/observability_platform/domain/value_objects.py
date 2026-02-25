"""Value objects for the observability domain."""

from __future__ import annotations

from enum import Enum


class MetricType(str, Enum):
    """Type of metric being recorded."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class AlertSeverity(str, Enum):
    """Severity level for alerts."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertState(str, Enum):
    """Current state of an alert."""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"


class LogLevel(str, Enum):
    """Log entry severity."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
