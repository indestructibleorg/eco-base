"""Tests for the runtime platform domain layer."""

import pytest

from runtime_platform.domain.value_objects import (
    ETLPhase,
    JobStatus,
    ScheduleFrequency,
    StepType,
)
from runtime_platform.domain.entities import ETLPipeline, Job, Step, Workflow
from runtime_platform.domain.events import (
    ETLCompleted,
    JobCompleted,
    StepFailed,
    WorkflowStarted,
)
from runtime_platform.domain.exceptions import (
    CyclicDependencyError,
    ETLPipelineError,
    JobExecutionError,
    RuntimePlatformError,
    SchedulerError,
    WorkflowError,
)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class TestJobStatus:
    def test_pending_is_not_terminal(self):
        assert not JobStatus.PENDING.is_terminal

    def test_completed_is_terminal(self):
        assert JobStatus.COMPLETED.is_terminal

    def test_failed_is_terminal(self):
        assert JobStatus.FAILED.is_terminal

    def test_skipped_is_terminal(self):
        assert JobStatus.SKIPPED.is_terminal

    def test_running_is_not_terminal(self):
        assert not JobStatus.RUNNING.is_terminal

    def test_completed_is_success(self):
        assert JobStatus.COMPLETED.is_success

    def test_failed_is_not_success(self):
        assert not JobStatus.FAILED.is_success

    def test_all_statuses_exist(self):
        expected = {"pending", "queued", "running", "completed", "failed", "skipped"}
        actual = {s.value for s in JobStatus}
        assert actual == expected


class TestStepType:
    def test_shell(self):
        assert StepType.SHELL.value == "shell"

    def test_python(self):
        assert StepType.PYTHON.value == "python"

    def test_http(self):
        assert StepType.HTTP.value == "http"


class TestScheduleFrequency:
    def test_all_frequencies(self):
        expected = {"once", "hourly", "daily", "weekly", "cron"}
        actual = {f.value for f in ScheduleFrequency}
        assert actual == expected


class TestETLPhase:
    def test_all_phases(self):
        expected = {"extract", "transform", "load"}
        actual = {p.value for p in ETLPhase}
        assert actual == expected

    def test_phase_ordering(self):
        phases = list(ETLPhase)
        assert phases == [ETLPhase.EXTRACT, ETLPhase.TRANSFORM, ETLPhase.LOAD]


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


class TestStep:
    def test_create_shell_step(self):
        step = Step(name="build", type=StepType.SHELL, command="make build")
        assert step.name == "build"
        assert step.type == StepType.SHELL
        assert step.command == "make build"
        assert step.timeout == 60.0

    def test_custom_timeout(self):
        step = Step(name="slow", type=StepType.SHELL, command="sleep 100", timeout=120.0)
        assert step.timeout == 120.0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name must not be empty"):
            Step(name="", type=StepType.SHELL, command="echo hi")

    def test_empty_command_raises(self):
        with pytest.raises(ValueError, match="command must not be empty"):
            Step(name="step", type=StepType.SHELL, command="")

    def test_negative_timeout_raises(self):
        with pytest.raises(ValueError, match="timeout must be positive"):
            Step(name="step", type=StepType.SHELL, command="echo hi", timeout=-1)

    def test_step_env(self):
        step = Step(name="s", type=StepType.SHELL, command="env", env={"FOO": "bar"})
        assert step.env == {"FOO": "bar"}


class TestJob:
    def test_default_status_is_pending(self):
        job = Job(name="test")
        assert job.status == JobStatus.PENDING

    def test_mark_running(self):
        job = Job(name="test")
        job.mark_running()
        assert job.status == JobStatus.RUNNING

    def test_mark_completed(self):
        job = Job(name="test")
        job.mark_completed()
        assert job.status == JobStatus.COMPLETED

    def test_mark_failed(self):
        job = Job(name="test")
        job.mark_failed("boom")
        assert job.status == JobStatus.FAILED
        assert job.error == "boom"

    def test_mark_skipped(self):
        job = Job(name="test")
        job.mark_skipped()
        assert job.status == JobStatus.SKIPPED

    def test_depends_on(self):
        job = Job(name="deploy", depends_on=["build", "test"])
        assert job.depends_on == ["build", "test"]

    def test_auto_name_from_id(self):
        job = Job(id="job-abc")
        assert job.name == "job-abc"


class TestWorkflow:
    def test_all_completed_when_all_terminal(self):
        wf = Workflow(name="wf", jobs=[
            Job(name="a", status=JobStatus.COMPLETED),
            Job(name="b", status=JobStatus.FAILED),
        ])
        assert wf.all_completed is True

    def test_all_completed_false_when_pending(self):
        wf = Workflow(name="wf", jobs=[
            Job(name="a", status=JobStatus.COMPLETED),
            Job(name="b", status=JobStatus.PENDING),
        ])
        assert wf.all_completed is False

    def test_has_failures(self):
        wf = Workflow(name="wf", jobs=[
            Job(name="a", status=JobStatus.COMPLETED),
            Job(name="b", status=JobStatus.FAILED),
        ])
        assert wf.has_failures is True

    def test_no_failures(self):
        wf = Workflow(name="wf", jobs=[
            Job(name="a", status=JobStatus.COMPLETED),
        ])
        assert wf.has_failures is False

    def test_get_job_by_name(self):
        job_a = Job(name="alpha")
        wf = Workflow(name="wf", jobs=[job_a])
        assert wf.get_job_by_name("alpha") is job_a
        assert wf.get_job_by_name("missing") is None

    def test_default_status_pending(self):
        wf = Workflow(name="wf")
        assert wf.status == JobStatus.PENDING


class TestETLPipeline:
    def test_create_pipeline(self):
        pipe = ETLPipeline(source="db", target="warehouse", transforms=["clean", "enrich"])
        assert pipe.source == "db"
        assert pipe.target == "warehouse"
        assert pipe.transforms == ["clean", "enrich"]
        assert pipe.status == JobStatus.PENDING

    def test_auto_id(self):
        pipe = ETLPipeline(source="a", target="b")
        assert pipe.id.startswith("etl-")


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class TestEvents:
    def test_workflow_started(self):
        evt = WorkflowStarted(workflow_id="wf-1", workflow_name="test", job_count=3)
        assert evt.workflow_id == "wf-1"
        assert evt.job_count == 3
        assert evt.event_id  # auto-generated
        assert evt.timestamp > 0

    def test_job_completed_success(self):
        evt = JobCompleted(
            workflow_id="wf-1", job_id="j-1", job_name="build",
            success=True, duration_seconds=1.5,
        )
        assert evt.success is True
        assert evt.error == ""

    def test_job_completed_failure(self):
        evt = JobCompleted(
            workflow_id="wf-1", job_id="j-1", job_name="build",
            success=False, error="timeout",
        )
        assert evt.success is False
        assert evt.error == "timeout"

    def test_step_failed(self):
        evt = StepFailed(
            workflow_id="wf-1", job_id="j-1", step_name="compile",
            error="exit code 1", exit_code=1,
        )
        assert evt.exit_code == 1

    def test_etl_completed(self):
        evt = ETLCompleted(
            pipeline_id="etl-1", source="db", target="wh",
            rows_extracted=100, rows_transformed=95, rows_loaded=95,
        )
        assert evt.rows_extracted == 100
        assert evt.success is True


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_runtime_platform_error(self):
        err = RuntimePlatformError("boom")
        assert str(err) == "boom"
        assert err.code == "RUNTIME_ERROR"

    def test_workflow_error(self):
        err = WorkflowError("bad dag", workflow_id="wf-1")
        assert err.workflow_id == "wf-1"
        assert err.code == "WORKFLOW_ERROR"

    def test_job_execution_error(self):
        err = JobExecutionError("step failed", job_id="j-1")
        assert err.job_id == "j-1"

    def test_etl_pipeline_error(self):
        err = ETLPipelineError("extract failed", pipeline_id="p-1", phase="extract")
        assert err.pipeline_id == "p-1"
        assert err.phase == "extract"

    def test_scheduler_error(self):
        err = SchedulerError("duplicate job")
        assert err.code == "SCHEDULER_ERROR"

    def test_cyclic_dependency_error(self):
        err = CyclicDependencyError(["A", "B", "A"])
        assert "Cyclic dependency" in str(err)
        assert err.cycle == ["A", "B", "A"]

    def test_inheritance_chain(self):
        assert issubclass(WorkflowError, RuntimePlatformError)
        assert issubclass(CyclicDependencyError, WorkflowError)
        assert issubclass(ETLPipelineError, RuntimePlatformError)
        assert issubclass(SchedulerError, RuntimePlatformError)
