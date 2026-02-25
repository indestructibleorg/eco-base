"""Domain entities — Workflow, Job, Step, ETLPipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from runtime_platform.domain.value_objects import JobStatus, StepType, ETLPhase


@dataclass(slots=True)
class Step:
    """A single unit of work within a job."""

    name: str
    type: StepType
    command: str
    timeout: float = 60.0
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Step name must not be empty")
        if not self.command:
            raise ValueError("Step command must not be empty")
        if self.timeout <= 0:
            raise ValueError("Step timeout must be positive")


@dataclass(slots=True)
class Job:
    """A named collection of steps with dependency tracking."""

    id: str = field(default_factory=lambda: f"job-{uuid.uuid4().hex[:8]}")
    name: str = ""
    steps: list[Step] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    status: JobStatus = JobStatus.PENDING
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.id

    def mark_running(self) -> None:
        self.status = JobStatus.RUNNING

    def mark_completed(self) -> None:
        self.status = JobStatus.COMPLETED

    def mark_failed(self, error: str) -> None:
        self.status = JobStatus.FAILED
        self.error = error

    def mark_skipped(self) -> None:
        self.status = JobStatus.SKIPPED


@dataclass(slots=True)
class Workflow:
    """A DAG of jobs — the top-level execution unit."""

    id: str = field(default_factory=lambda: f"wf-{uuid.uuid4().hex[:8]}")
    name: str = ""
    jobs: list[Job] = field(default_factory=list)
    status: JobStatus = JobStatus.PENDING
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.id

    def get_job(self, job_id: str) -> Job | None:
        """Look up a job by id."""
        for job in self.jobs:
            return job if job.id == job_id else None
        return None

    def get_job_by_name(self, name: str) -> Job | None:
        """Look up a job by name."""
        for job in self.jobs:
            if job.name == name:
                return job
        return None

    @property
    def all_completed(self) -> bool:
        return all(j.status.is_terminal for j in self.jobs)

    @property
    def has_failures(self) -> bool:
        return any(j.status == JobStatus.FAILED for j in self.jobs)


@dataclass(slots=True)
class ETLPipeline:
    """An extract-transform-load pipeline specification."""

    id: str = field(default_factory=lambda: f"etl-{uuid.uuid4().hex[:8]}")
    source: str = ""
    transforms: list[str] = field(default_factory=list)
    target: str = ""
    status: ETLPhase | JobStatus = JobStatus.PENDING
    extract_fn: Callable[..., Awaitable[Any]] | None = None
    transform_fns: list[Callable[..., Awaitable[Any]]] = field(default_factory=list)
    load_fn: Callable[..., Awaitable[Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
