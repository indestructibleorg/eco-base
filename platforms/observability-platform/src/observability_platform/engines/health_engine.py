"""HealthEngine — async health check execution and aggregate reporting."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from platform_shared.protocols.health import HealthStatus, HealthReport

from observability_platform.domain.entities import HealthCheck
from observability_platform.domain.events import HealthChanged
from observability_platform.domain.exceptions import HealthCheckError


@dataclass(slots=True)
class ComponentHealth:
    """Health state for a single registered component."""
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    last_checked: float = 0.0
    latency_ms: float = 0.0


class HealthEngine:
    """Registers health check functions and runs them to produce HealthReports.

    Each registered check is an async callable that returns True (healthy) or False.
    run_checks() executes all checks concurrently and returns a combined report.
    """

    def __init__(self) -> None:
        self._checks: dict[str, HealthCheck] = {}
        self._component_health: dict[str, ComponentHealth] = {}
        self._events: list[HealthChanged] = []

    def register_check(
        self,
        name: str,
        fn: Callable[[], Awaitable[bool]],
        timeout_seconds: float = 5.0,
    ) -> None:
        """Register a health check function for a named component."""
        check = HealthCheck(name=name, check_fn=fn, timeout_seconds=timeout_seconds)
        self._checks[name] = check
        self._component_health[name] = ComponentHealth(name=name)

    def unregister_check(self, name: str) -> None:
        """Remove a registered health check."""
        self._checks.pop(name, None)
        self._component_health.pop(name, None)

    async def run_checks(self) -> list[HealthReport]:
        """Run all registered health checks and return a HealthReport per component.

        Each check is run with its configured timeout. If a check raises or times out,
        the component is marked UNHEALTHY.

        Returns:
            List of HealthReport, one per registered component.
        """
        reports: list[HealthReport] = []

        for name, check in self._checks.items():
            start = time.monotonic()
            previous_status = self._component_health[name].status

            try:
                result = await asyncio.wait_for(
                    check.check_fn(),
                    timeout=check.timeout_seconds,
                )
                elapsed_ms = (time.monotonic() - start) * 1000.0

                if result:
                    status = HealthStatus.HEALTHY
                    message = "OK"
                else:
                    status = HealthStatus.UNHEALTHY
                    message = "Check returned False"

            except asyncio.TimeoutError:
                elapsed_ms = (time.monotonic() - start) * 1000.0
                status = HealthStatus.UNHEALTHY
                message = f"Check timed out after {check.timeout_seconds}s"

            except Exception as exc:
                elapsed_ms = (time.monotonic() - start) * 1000.0
                status = HealthStatus.UNHEALTHY
                message = f"Check failed: {exc}"

            # Update component state
            comp = self._component_health[name]
            comp.status = status
            comp.message = message
            comp.last_checked = time.time()
            comp.latency_ms = elapsed_ms

            # Emit event on status change
            if previous_status != status and previous_status != HealthStatus.UNKNOWN:
                event = HealthChanged(
                    component=name,
                    status=status.value,
                    previous_status=previous_status.value,
                    message=message,
                )
                self._events.append(event)

            report = HealthReport(
                component=name,
                status=status,
                message=message,
                latency_ms=elapsed_ms,
            )
            reports.append(report)

        return reports

    async def get_aggregate_status(self) -> HealthReport:
        """Run all checks and return an aggregate health report.

        - If all components are HEALTHY -> overall HEALTHY.
        - If any component is UNHEALTHY -> overall UNHEALTHY.
        - If some are DEGRADED but none UNHEALTHY -> overall DEGRADED.
        """
        reports = await self.run_checks()

        if not reports:
            return HealthReport(
                component="aggregate",
                status=HealthStatus.HEALTHY,
                message="No checks registered",
            )

        statuses = [r.status for r in reports]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
            msg = f"All {len(reports)} components healthy"
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            unhealthy_names = [r.component for r in reports if r.status == HealthStatus.UNHEALTHY]
            overall = HealthStatus.UNHEALTHY
            msg = f"Unhealthy components: {', '.join(unhealthy_names)}"
        else:
            overall = HealthStatus.DEGRADED
            msg = "Some components degraded"

        return HealthReport(
            component="aggregate",
            status=overall,
            message=msg,
            details={"components": {r.app.kubernetes.io/component: r.status.value for r in reports}},
        )

    @property
    def registered_checks(self) -> list[str]:
        """Names of all registered health checks."""
        return list(self._checks.keys())

    @property
    def component_health(self) -> dict[str, ComponentHealth]:
        """Current health state per component."""
        return dict(self._component_health)

    @property
    def events(self) -> list[HealthChanged]:
        """History of health change events."""
        return list(self._events)
