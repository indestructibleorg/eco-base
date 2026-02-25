"""Runtime platform exception hierarchy."""

from __future__ import annotations

from platform_shared.domain.errors import PlatformError


class RuntimePlatformError(PlatformError):
    """Base error for all runtime platform failures."""

    def __init__(self, message: str, **kw):
        super().__init__(message, code="RUNTIME_ERROR", **kw)


class WorkflowError(RuntimePlatformError):
    """Workflow execution failure."""

    def __init__(self, message: str, *, workflow_id: str = "", **kw):
        super().__init__(message, **kw)
        self.code = "WORKFLOW_ERROR"
        self.workflow_id = workflow_id


class JobExecutionError(RuntimePlatformError):
    """Job execution failure."""

    def __init__(self, message: str, *, job_id: str = "", **kw):
        super().__init__(message, **kw)
        self.code = "JOB_EXECUTION_ERROR"
        self.job_id = job_id


class ETLPipelineError(RuntimePlatformError):
    """ETL pipeline failure."""

    def __init__(self, message: str, *, pipeline_id: str = "", phase: str = "", **kw):
        super().__init__(message, **kw)
        self.code = "ETL_PIPELINE_ERROR"
        self.pipeline_id = pipeline_id
        self.phase = phase


class SchedulerError(RuntimePlatformError):
    """Scheduler failure."""

    def __init__(self, message: str, **kw):
        super().__init__(message, **kw)
        self.code = "SCHEDULER_ERROR"


class CyclicDependencyError(WorkflowError):
    """Raised when a workflow DAG contains a cycle."""

    def __init__(self, cycle: list[str], **kw):
        super().__init__(f"Cyclic dependency detected: {' -> '.join(cycle)}", **kw)
        self.code = "CYCLIC_DEPENDENCY"
        self.cycle = cycle
