"""Value objects — immutable enumerations for the runtime domain."""

from __future__ import annotations

from enum import Enum


class JobStatus(str, Enum):
    """Lifecycle state of a job within a workflow."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.SKIPPED)

    @property
    def is_success(self) -> bool:
        return self == JobStatus.COMPLETED


class StepType(str, Enum):
    """Kind of operation a step performs."""

    SHELL = "shell"
    PYTHON = "python"
    HTTP = "http"


class ScheduleFrequency(str, Enum):
    """Predefined schedule frequencies for the scheduler."""

    ONCE = "once"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    CRON = "cron"


class ETLPhase(str, Enum):
    """Phases of an ETL pipeline."""

    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"
