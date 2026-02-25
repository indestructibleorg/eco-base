"""Domain events emitted during workflow and ETL execution."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkflowStarted:
    """Emitted when a workflow begins execution."""

    workflow_id: str
    workflow_name: str
    job_count: int
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class JobCompleted:
    """Emitted when a job finishes (successfully or with failure)."""

    workflow_id: str
    job_id: str
    job_name: str
    success: bool
    error: str = ""
    duration_seconds: float = 0.0
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class StepFailed:
    """Emitted when a specific step within a job fails."""

    workflow_id: str
    job_id: str
    step_name: str
    error: str
    exit_code: int = -1
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class ETLCompleted:
    """Emitted when an ETL pipeline finishes."""

    pipeline_id: str
    source: str
    target: str
    rows_extracted: int = 0
    rows_transformed: int = 0
    rows_loaded: int = 0
    success: bool = True
    error: str = ""
    duration_seconds: float = 0.0
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)
