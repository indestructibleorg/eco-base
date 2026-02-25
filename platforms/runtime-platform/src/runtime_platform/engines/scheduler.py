"""Scheduler — cron-like job scheduling with simple cron expression matching.

Supports standard 5-field cron expressions (minute, hour, day-of-month, month, day-of-week)
and the predefined frequency shortcuts (once, hourly, daily, weekly).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Awaitable

from runtime_platform.domain.value_objects import ScheduleFrequency
from runtime_platform.domain.exceptions import SchedulerError


@dataclass(slots=True)
class ScheduledJob:
    """A registered job in the scheduler."""

    name: str
    cron_expr: str
    callable: Callable[..., Awaitable[Any]]
    frequency: ScheduleFrequency = ScheduleFrequency.CRON
    enabled: bool = True
    last_run: float | None = None
    run_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class Scheduler:
    """Evaluates which registered jobs are due and executes them.

    Usage:
        scheduler = Scheduler()
        scheduler.register_job("cleanup", "0 * * * *", cleanup_fn)  # hourly
        due = scheduler.tick()      # returns list of due job names
        results = await scheduler.run_pending()  # executes them
    """

    def __init__(self) -> None:
        self._jobs: dict[str, ScheduledJob] = {}
        self._pending: list[str] = []
        self._history: list[dict[str, Any]] = []

    @property
    def jobs(self) -> dict[str, ScheduledJob]:
        return dict(self._jobs)

    @property
    def pending(self) -> list[str]:
        return list(self._pending)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def register_job(
        self,
        name: str,
        cron_expr: str,
        callable_: Callable[..., Awaitable[Any]],
        *,
        frequency: ScheduleFrequency = ScheduleFrequency.CRON,
        metadata: dict[str, Any] | None = None,
    ) -> ScheduledJob:
        """Register a new job with the scheduler."""
        if name in self._jobs:
            raise SchedulerError(f"Job '{name}' is already registered")

        # Validate the cron expression
        effective_cron = _resolve_frequency(cron_expr, frequency)

        job = ScheduledJob(
            name=name,
            cron_expr=effective_cron,
            callable=callable_,
            frequency=frequency,
            metadata=metadata or {},
        )
        self._jobs[name] = job
        return job

    def unregister_job(self, name: str) -> None:
        """Remove a job from the scheduler."""
        if name not in self._jobs:
            raise SchedulerError(f"Job '{name}' is not registered")
        del self._jobs[name]
        if name in self._pending:
            self._pending.remove(name)

    def tick(self, now: datetime | None = None) -> list[str]:
        """Evaluate which jobs are due right now. Returns list of due job names.

        Call this periodically (e.g. every minute) to check schedules.
        """
        now = now or datetime.now()
        due: list[str] = []

        for name, job in self._jobs.items():
            if not job.enabled:
                continue

            # ONCE jobs: only run if never run before
            if job.frequency == ScheduleFrequency.ONCE:
                if job.run_count == 0 and _cron_matches(job.cron_expr, now):
                    due.append(name)
                continue

            if _cron_matches(job.cron_expr, now):
                # Avoid double-firing within the same minute
                if job.last_run is not None:
                    last_dt = datetime.fromtimestamp(job.last_run)
                    if (
                        last_dt.year == now.year
                        and last_dt.month == now.month
                        and last_dt.day == now.day
                        and last_dt.hour == now.hour
                        and last_dt.minute == now.minute
                    ):
                        continue
                due.append(name)

        self._pending = due
        return due

    async def run_pending(self) -> list[dict[str, Any]]:
        """Execute all pending (due) jobs. Returns results."""
        results: list[dict[str, Any]] = []

        for name in list(self._pending):
            job = self._jobs.get(name)
            if job is None:
                continue

            start = time.monotonic()
            try:
                result = await job.callable()
                duration = time.monotonic() - start
                job.last_run = time.time()
                job.run_count += 1
                entry = {
                    "job": name,
                    "success": True,
                    "duration_seconds": duration,
                    "result": result,
                }
            except Exception as exc:
                duration = time.monotonic() - start
                entry = {
                    "job": name,
                    "success": False,
                    "duration_seconds": duration,
                    "error": str(exc),
                }

            results.append(entry)
            self._history.append(entry)

        self._pending.clear()
        return results

    async def run_job(self, name: str) -> dict[str, Any]:
        """Force-run a specific job regardless of schedule."""
        job = self._jobs.get(name)
        if job is None:
            raise SchedulerError(f"Job '{name}' is not registered")

        start = time.monotonic()
        try:
            result = await job.callable()
            duration = time.monotonic() - start
            job.last_run = time.time()
            job.run_count += 1
            entry = {
                "job": name,
                "success": True,
                "duration_seconds": duration,
                "result": result,
            }
        except Exception as exc:
            duration = time.monotonic() - start
            entry = {
                "job": name,
                "success": False,
                "duration_seconds": duration,
                "error": str(exc),
            }

        self._history.append(entry)
        return entry


def _resolve_frequency(cron_expr: str, frequency: ScheduleFrequency) -> str:
    """Convert a ScheduleFrequency into a concrete cron expression if needed."""
    if frequency == ScheduleFrequency.CRON:
        _validate_cron(cron_expr)
        return cron_expr

    presets = {
        ScheduleFrequency.ONCE: cron_expr if cron_expr.strip() else "* * * * *",
        ScheduleFrequency.HOURLY: "0 * * * *",
        ScheduleFrequency.DAILY: "0 0 * * *",
        ScheduleFrequency.WEEKLY: "0 0 * * 0",
    }
    resolved = presets.get(frequency, cron_expr)
    _validate_cron(resolved)
    return resolved


def _validate_cron(expr: str) -> None:
    """Basic cron expression validation (5 fields)."""
    parts = expr.strip().split()
    if len(parts) != 5:
        raise SchedulerError(
            f"Invalid cron expression '{expr}': expected 5 fields, got {len(parts)}"
        )


def _cron_matches(expr: str, dt: datetime) -> bool:
    """Check if a cron expression matches a specific datetime.

    Supports:
      - Literal numbers: 5
      - Wildcards: *
      - Ranges: 1-5
      - Lists: 1,3,5
      - Steps: */5 or 1-10/2
    """
    parts = expr.strip().split()
    if len(parts) != 5:
        return False

    values = [dt.minute, dt.hour, dt.day, dt.month, dt.isoweekday() % 7]
    # isoweekday: Mon=1..Sun=7, % 7 gives Mon=1..Sat=6,Sun=0

    for field_expr, current_value in zip(parts, values):
        if not _field_matches(field_expr, current_value):
            return False
    return True


def _field_matches(field_expr: str, value: int) -> bool:
    """Check if a single cron field matches a value."""
    # Handle comma-separated list
    for part in field_expr.split(","):
        if _part_matches(part.strip(), value):
            return True
    return False


def _part_matches(part: str, value: int) -> bool:
    """Match a single cron field part (may contain range, step, or wildcard)."""
    # Step notation: */5 or 1-10/2
    if "/" in part:
        range_part, step_str = part.split("/", 1)
        try:
            step = int(step_str)
        except ValueError:
            return False

        if range_part == "*":
            return value % step == 0
        elif "-" in range_part:
            lo, hi = range_part.split("-", 1)
            try:
                return int(lo) <= value <= int(hi) and (value - int(lo)) % step == 0
            except ValueError:
                return False
        return False

    # Wildcard
    if part == "*":
        return True

    # Range: 1-5
    if "-" in part:
        try:
            lo, hi = part.split("-", 1)
            return int(lo) <= value <= int(hi)
        except ValueError:
            return False

    # Literal number
    try:
        return int(part) == value
    except ValueError:
        return False
