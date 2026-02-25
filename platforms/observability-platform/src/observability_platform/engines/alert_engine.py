"""AlertEngine — rule-based alerting against metric values."""

from __future__ import annotations

import time
from typing import Any

from observability_platform.domain.entities import Alert, AlertRule
from observability_platform.domain.value_objects import AlertSeverity, AlertState
from observability_platform.domain.events import AlertFired, AlertResolved
from observability_platform.domain.exceptions import AlertRuleError
from observability_platform.engines.metrics_engine import MetricsEngine


class AlertEngine:
    """Manages alert rules and evaluates them against current metric values.

    Supports alert state transitions: PENDING -> FIRING -> RESOLVED.
    In evaluation, rules check the latest metric value against thresholds.
    The for_seconds parameter controls how long a condition must persist
    before transitioning from PENDING to FIRING.
    """

    def __init__(self, metrics_engine: MetricsEngine) -> None:
        self._metrics = metrics_engine
        self._rules: dict[str, AlertRule] = {}
        self._alerts: dict[str, Alert] = {}
        self._events: list[AlertFired | AlertResolved] = []

    def register_rule(self, rule: AlertRule) -> None:
        """Register a new alert rule.

        Raises:
            AlertRuleError: If a rule with the same name already exists or condition is invalid.
        """
        if rule.condition not in ("gt", "lt", "eq"):
            raise AlertRuleError(
                f"Invalid condition '{rule.condition}': must be gt, lt, or eq",
                rule_name=rule.name,
            )
        if rule.name in self._rules:
            raise AlertRuleError(
                f"Rule '{rule.name}' already registered",
                rule_name=rule.name,
            )
        self._rules[rule.name] = rule
        self._alerts[rule.name] = Alert(
            rule=rule,
            state=AlertState.RESOLVED,
        )

    def unregister_rule(self, name: str) -> None:
        """Remove a rule and its associated alert state."""
        self._rules.pop(name, None)
        self._alerts.pop(name, None)

    def evaluate(self) -> list[Alert]:
        """Evaluate all registered rules against current metric values.

        For each rule:
        - Get the latest value from the metrics engine.
        - If the condition triggers:
          - If RESOLVED -> transition to PENDING (with timestamp).
          - If PENDING and for_seconds elapsed -> transition to FIRING, emit AlertFired.
          - If PENDING and for_seconds not elapsed -> stay PENDING.
          - If already FIRING -> remain FIRING.
        - If the condition does NOT trigger:
          - If FIRING or PENDING -> transition to RESOLVED, emit AlertResolved.

        Returns:
            List of all current Alert objects.
        """
        now = time.time()
        results: list[Alert] = []

        for name, rule in self._rules.items():
            alert = self._alerts[name]
            latest = self._metrics.get_latest(rule.metric_name)

            if latest is None:
                # No data yet — stay as is
                results.append(alert)
                continue

            triggered = rule.evaluate(latest.value)

            if triggered:
                if alert.state == AlertState.RESOLVED:
                    # Start pending
                    alert.state = AlertState.PENDING
                    alert.pending_since = now
                    alert.value = latest.value

                if alert.state == AlertState.PENDING:
                    elapsed = now - (alert.pending_since or now)
                    if elapsed >= rule.for_seconds:
                        # Transition to firing
                        alert.state = AlertState.FIRING
                        alert.fired_at = now
                        alert.value = latest.value
                        event = AlertFired(
                            rule_name=rule.name,
                            metric_name=rule.metric_name,
                            value=latest.value,
                            threshold=rule.threshold,
                            severity=rule.severity,
                        )
                        self._events.append(event)
                # If FIRING, just update value
                elif alert.state == AlertState.FIRING:
                    alert.value = latest.value
            else:
                if alert.state in (AlertState.FIRING, AlertState.PENDING):
                    # Resolve
                    previous_state = alert.state
                    alert.state = AlertState.RESOLVED
                    alert.resolved_at = now
                    alert.pending_since = None
                    if previous_state == AlertState.FIRING:
                        event = AlertResolved(
                            rule_name=rule.name,
                            metric_name=rule.metric_name,
                        )
                        self._events.append(event)

            results.append(alert)

        return results

    @property
    def rules(self) -> dict[str, AlertRule]:
        """All registered rules."""
        return dict(self._rules)

    @property
    def alerts(self) -> dict[str, Alert]:
        """Current alert states."""
        return dict(self._alerts)

    @property
    def firing_alerts(self) -> list[Alert]:
        """All currently firing alerts."""
        return [a for a in self._alerts.values() if a.state == AlertState.FIRING]

    @property
    def events(self) -> list[AlertFired | AlertResolved]:
        """History of alert events."""
        return list(self._events)
