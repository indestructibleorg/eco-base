"""Domain events for the observability platform."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from observability_platform.domain.value_objects import AlertSeverity, AlertState


@dataclass(frozen=True, slots=True)
class MetricRecorded:
    """Emitted when a new metric data point is recorded."""
    metric_name: str
    value: float
    tags: dict[str, str] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class AlertFired:
    """Emitted when an alert transitions to FIRING state."""
    rule_name: str
    metric_name: str
    value: float
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class AlertResolved:
    """Emitted when a previously firing alert is resolved."""
    rule_name: str
    metric_name: str
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class HealthChanged:
    """Emitted when a component's health status changes."""
    app.kubernetes.io/component: str
    status: str
    previous_status: str = ""
    message: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)
