"""Tests for the runtime platform engines: workflow, ETL, and scheduler."""

import asyncio
from datetime import datetime

import pytest

from runtime_platform.domain.entities import ETLPipeline, Job, Step, Workflow
from runtime_platform.domain.value_objects import JobStatus, StepType, ScheduleFrequency, ETLPhase
from runtime_platform.domain.events import WorkflowStarted, JobCompleted, ETLCompleted
from runtime_platform.domain.exceptions import (
    CyclicDependencyError,
    ETLPipelineError,
    SchedulerError,
    WorkflowError,
)
from runtime_platform.engines.workflow_engine import WorkflowEngine
from runtime_platform.engines.etl_engine import ETLEngine, ETLResult
from runtime_platform.engines.scheduler import Scheduler


# ---------------------------------------------------------------------------
# WorkflowEngine
# ---------------------------------------------------------------------------


class TestWorkflowEngine:
    """Test DAG-based workflow execution."""

    @pytest.fixture
    def engine(self):
        # Use a simple step executor that always succeeds
        async def mock_executor(step):
            return {"step": step.name, "success": True}
        return WorkflowEngine(step_executor=mock_executor)

    @pytest.fixture
    def failing_engine(self):
        # Step executor that fails on steps named "fail-*"
        async def mock_executor(step):
            if step.name.startswith("fail-"):
                raise Exception(f"Step {step.name} failed deliberately")
            return {"step": step.name, "success": True}
        return WorkflowEngine(step_executor=mock_executor)

    async def test_single_job_workflow(self, engine):
        wf = Workflow(
            id="wf-1", name="simple",
            jobs=[Job(
                id="j-1", name="build",
                steps=[Step(name="compile", type=StepType.SHELL, command="echo build")],
            )],
        )
        result = await engine.run_workflow(wf)
        assert result["status"] == "completed"
        assert wf.status == JobStatus.COMPLETED
        assert wf.jobs[0].status == JobStatus.COMPLETED

    async def test_diamond_dag_execution(self, engine, diamond_workflow):
        result = await engine.run_workflow(diamond_workflow)
        assert result["status"] == "completed"
        for job in diamond_workflow.jobs:
            assert job.status == JobStatus.COMPLETED

    async def test_parallel_independent_jobs(self, engine):
        """Jobs with no dependencies should run in the same level."""
        wf = Workflow(
            id="wf-parallel", name="parallel",
            jobs=[
                Job(id="j-a", name="A",
                    steps=[Step(name="a", type=StepType.SHELL, command="echo A")]),
                Job(id="j-b", name="B",
                    steps=[Step(name="b", type=StepType.SHELL, command="echo B")]),
                Job(id="j-c", name="C",
                    steps=[Step(name="c", type=StepType.SHELL, command="echo C")]),
            ],
        )
        result = await engine.run_workflow(wf)
        assert result["status"] == "completed"
        # All three should be completed
        assert all(j.status == JobStatus.COMPLETED for j in wf.jobs)

    async def test_sequential_dependencies(self, engine):
        """A linear chain: A -> B -> C."""
        wf = Workflow(
            id="wf-seq", name="sequential",
            jobs=[
                Job(id="j-a", name="A",
                    steps=[Step(name="a", type=StepType.SHELL, command="echo A")]),
                Job(id="j-b", name="B",
                    steps=[Step(name="b", type=StepType.SHELL, command="echo B")],
                    depends_on=["A"]),
                Job(id="j-c", name="C",
                    steps=[Step(name="c", type=StepType.SHELL, command="echo C")],
                    depends_on=["B"]),
            ],
        )
        result = await engine.run_workflow(wf)
        assert result["status"] == "completed"

    async def test_failed_job_skips_dependents(self, failing_engine):
        """If A fails, B (depends on A) should be skipped."""
        wf = Workflow(
            id="wf-fail", name="fail-skip",
            jobs=[
                Job(id="j-a", name="A",
                    steps=[Step(name="fail-step", type=StepType.SHELL, command="boom")]),
                Job(id="j-b", name="B",
                    steps=[Step(name="ok", type=StepType.SHELL, command="echo ok")],
                    depends_on=["A"]),
            ],
        )
        result = await failing_engine.run_workflow(wf)
        assert result["status"] == "failed"
        assert wf.jobs[0].status == JobStatus.FAILED
        assert wf.jobs[1].status == JobStatus.SKIPPED

    async def test_cyclic_dependency_detected(self, engine):
        """A -> B -> A should raise CyclicDependencyError."""
        wf = Workflow(
            id="wf-cycle", name="cycle",
            jobs=[
                Job(id="j-a", name="A",
                    steps=[Step(name="a", type=StepType.SHELL, command="echo A")],
                    depends_on=["B"]),
                Job(id="j-b", name="B",
                    steps=[Step(name="b", type=StepType.SHELL, command="echo B")],
                    depends_on=["A"]),
            ],
        )
        with pytest.raises(CyclicDependencyError):
            await engine.run_workflow(wf)

    async def test_unknown_dependency_raises(self, engine):
        """Referencing a non-existent dependency should raise WorkflowError."""
        wf = Workflow(
            id="wf-unknown", name="unknown-dep",
            jobs=[
                Job(id="j-a", name="A",
                    steps=[Step(name="a", type=StepType.SHELL, command="echo A")],
                    depends_on=["NONEXISTENT"]),
            ],
        )
        with pytest.raises(WorkflowError, match="unknown job"):
            await engine.run_workflow(wf)

    async def test_events_emitted(self, engine):
        wf = Workflow(
            id="wf-evt", name="events",
            jobs=[Job(
                id="j-1", name="build",
                steps=[Step(name="compile", type=StepType.SHELL, command="echo ok")],
            )],
        )
        await engine.run_workflow(wf)
        events = engine.events

        # Should have WorkflowStarted and JobCompleted
        started = [e for e in events if isinstance(e, WorkflowStarted)]
        completed = [e for e in events if isinstance(e, JobCompleted)]
        assert len(started) == 1
        assert started[0].workflow_id == "wf-evt"
        assert len(completed) == 1
        assert completed[0].success is True

    async def test_empty_workflow(self, engine):
        """A workflow with no jobs should complete immediately."""
        wf = Workflow(id="wf-empty", name="empty", jobs=[])
        result = await engine.run_workflow(wf)
        assert result["status"] == "completed"

    async def test_engine_name_and_status(self, engine):
        assert engine.name == "workflow-engine"
        # Initially idle
        from platform_shared.protocols.engine import EngineStatus
        assert engine.status == EngineStatus.IDLE


# ---------------------------------------------------------------------------
# ETLEngine
# ---------------------------------------------------------------------------


class TestETLEngine:
    """Test ETL pipeline execution."""

    @pytest.fixture
    def engine(self):
        return ETLEngine()

    async def test_successful_pipeline(self, engine, etl_pipeline):
        result = await engine.run_pipeline(etl_pipeline)
        assert result.success is True
        assert result.rows_extracted == 5
        assert result.rows_transformed == 5
        assert result.rows_loaded == 5
        assert result.total_duration > 0
        assert result.errors == []

    async def test_pipeline_phases_timing(self, engine, etl_pipeline):
        result = await engine.run_pipeline(etl_pipeline)
        assert result.extract_duration >= 0
        assert result.transform_duration >= 0
        assert result.load_duration >= 0
        # Total should be >= sum of phases (approximately)
        assert result.total_duration >= 0

    async def test_pipeline_result_stored(self, engine, etl_pipeline):
        result = await engine.run_pipeline(etl_pipeline)
        assert etl_pipeline.id in engine.results
        assert engine.results[etl_pipeline.id] is result

    async def test_extract_failure(self, engine):
        async def bad_extract(source):
            raise ValueError("connection refused")

        async def load(target, data):
            return len(data)

        pipe = ETLPipeline(
            id="etl-fail",
            source="bad-db",
            target="warehouse",
            extract_fn=bad_extract,
            load_fn=load,
        )

        with pytest.raises(ETLPipelineError, match="Extract phase failed"):
            await engine.run_pipeline(pipe)

    async def test_transform_failure(self, engine):
        async def extract(source):
            return [1, 2, 3]

        async def bad_transform(data):
            raise TypeError("invalid data format")

        async def load(target, data):
            return len(data)

        pipe = ETLPipeline(
            id="etl-t-fail",
            source="ok",
            target="ok",
            extract_fn=extract,
            transform_fns=[bad_transform],
            load_fn=load,
        )

        with pytest.raises(ETLPipelineError, match="Transform phase failed"):
            await engine.run_pipeline(pipe)

    async def test_load_failure(self, engine):
        async def extract(source):
            return [1, 2, 3]

        async def load(target, data):
            raise IOError("disk full")

        pipe = ETLPipeline(
            id="etl-l-fail",
            source="ok",
            target="broken",
            extract_fn=extract,
            load_fn=load,
        )

        with pytest.raises(ETLPipelineError, match="Load phase failed"):
            await engine.run_pipeline(pipe)

    async def test_no_extract_fn_raises(self, engine):
        pipe = ETLPipeline(id="etl-no-ext", source="src", target="tgt")
        with pytest.raises(ETLPipelineError, match="No extract function"):
            await engine.run_pipeline(pipe)

    async def test_no_load_fn_raises(self, engine):
        async def extract(source):
            return [1, 2, 3]

        pipe = ETLPipeline(
            id="etl-no-load",
            source="src",
            target="tgt",
            extract_fn=extract,
        )
        with pytest.raises(ETLPipelineError, match="No load function"):
            await engine.run_pipeline(pipe)

    async def test_multiple_transforms(self, engine):
        async def extract(source):
            return [1, 2, 3, 4, 5]

        async def double(data):
            return [x * 2 for x in data]

        async def add_ten(data):
            return [x + 10 for x in data]

        async def load(target, data):
            return len(data)

        pipe = ETLPipeline(
            id="etl-multi",
            source="numbers",
            target="results",
            extract_fn=extract,
            transform_fns=[double, add_ten],
            load_fn=load,
        )

        result = await engine.run_pipeline(pipe)
        assert result.success is True
        assert result.rows_loaded == 5

    async def test_events_emitted(self, engine, etl_pipeline):
        await engine.run_pipeline(etl_pipeline)
        events = engine.events
        completed = [e for e in events if isinstance(e, ETLCompleted)]
        assert len(completed) == 1
        assert completed[0].success is True
        assert completed[0].rows_loaded == 5

    async def test_pipeline_status_tracking(self, engine, etl_pipeline):
        await engine.run_pipeline(etl_pipeline)
        assert etl_pipeline.status == JobStatus.COMPLETED

    async def test_failed_pipeline_status(self, engine):
        async def bad_extract(source):
            raise RuntimeError("fail")

        async def load(target, data):
            return 0

        pipe = ETLPipeline(
            id="etl-fail-status",
            source="x",
            target="y",
            extract_fn=bad_extract,
            load_fn=load,
        )
        with pytest.raises(ETLPipelineError):
            await engine.run_pipeline(pipe)
        assert pipe.status == JobStatus.FAILED

    async def test_engine_name(self, engine):
        assert engine.name == "etl-engine"


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class TestScheduler:
    """Test cron-like job scheduling."""

    @pytest.fixture
    def scheduler(self):
        return Scheduler()

    @pytest.fixture
    def noop_job(self):
        async def noop():
            return "done"
        return noop

    def test_register_job(self, scheduler, noop_job):
        job = scheduler.register_job("cleanup", "0 * * * *", noop_job)
        assert job.name == "cleanup"
        assert job.cron_expr == "0 * * * *"
        assert job.enabled is True
        assert job.run_count == 0

    def test_register_duplicate_raises(self, scheduler, noop_job):
        scheduler.register_job("cleanup", "0 * * * *", noop_job)
        with pytest.raises(SchedulerError, match="already registered"):
            scheduler.register_job("cleanup", "0 * * * *", noop_job)

    def test_unregister_job(self, scheduler, noop_job):
        scheduler.register_job("cleanup", "0 * * * *", noop_job)
        scheduler.unregister_job("cleanup")
        assert "cleanup" not in scheduler.jobs

    def test_unregister_unknown_raises(self, scheduler):
        with pytest.raises(SchedulerError, match="not registered"):
            scheduler.unregister_job("ghost")

    def test_tick_matches_wildcard(self, scheduler, noop_job):
        scheduler.register_job("every-minute", "* * * * *", noop_job)
        now = datetime(2026, 1, 15, 10, 30)
        due = scheduler.tick(now)
        assert "every-minute" in due

    def test_tick_matches_specific_minute(self, scheduler, noop_job):
        scheduler.register_job("at-30", "30 * * * *", noop_job)
        due_yes = scheduler.tick(datetime(2026, 1, 15, 10, 30))
        assert "at-30" in due_yes
        due_no = scheduler.tick(datetime(2026, 1, 15, 10, 15))
        assert "at-30" not in due_no

    def test_tick_matches_hourly(self, scheduler, noop_job):
        scheduler.register_job(
            "hourly", "0 * * * *", noop_job, frequency=ScheduleFrequency.HOURLY
        )
        due = scheduler.tick(datetime(2026, 1, 15, 14, 0))
        assert "hourly" in due

    def test_tick_matches_daily(self, scheduler, noop_job):
        scheduler.register_job(
            "daily", "0 0 * * *", noop_job, frequency=ScheduleFrequency.DAILY
        )
        due = scheduler.tick(datetime(2026, 1, 15, 0, 0))
        assert "daily" in due

    def test_tick_no_match(self, scheduler, noop_job):
        scheduler.register_job("at-midnight", "0 0 * * *", noop_job)
        due = scheduler.tick(datetime(2026, 1, 15, 10, 30))
        assert due == []

    def test_disabled_job_not_triggered(self, scheduler, noop_job):
        job = scheduler.register_job("disabled", "* * * * *", noop_job)
        job.enabled = False
        due = scheduler.tick(datetime(2026, 1, 15, 10, 30))
        assert "disabled" not in due

    async def test_run_pending(self, scheduler, noop_job):
        scheduler.register_job("task", "* * * * *", noop_job)
        scheduler.tick(datetime(2026, 1, 15, 10, 30))
        results = await scheduler.run_pending()
        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["result"] == "done"

    async def test_run_pending_tracks_history(self, scheduler, noop_job):
        scheduler.register_job("task", "* * * * *", noop_job)
        scheduler.tick(datetime(2026, 1, 15, 10, 30))
        await scheduler.run_pending()
        assert len(scheduler.history) == 1

    async def test_run_pending_increments_run_count(self, scheduler, noop_job):
        scheduler.register_job("task", "* * * * *", noop_job)
        scheduler.tick(datetime(2026, 1, 15, 10, 30))
        await scheduler.run_pending()
        assert scheduler.jobs["task"].run_count == 1

    async def test_run_pending_with_failure(self, scheduler):
        async def failing_job():
            raise RuntimeError("boom")

        scheduler.register_job("bad", "* * * * *", failing_job)
        scheduler.tick(datetime(2026, 1, 15, 10, 30))
        results = await scheduler.run_pending()
        assert len(results) == 1
        assert results[0]["success"] is False
        assert "boom" in results[0]["error"]

    async def test_force_run_job(self, scheduler, noop_job):
        scheduler.register_job("manual", "0 0 1 1 *", noop_job)
        result = await scheduler.run_job("manual")
        assert result["success"] is True

    async def test_force_run_unknown_raises(self, scheduler):
        with pytest.raises(SchedulerError, match="not registered"):
            await scheduler.run_job("ghost")

    def test_once_frequency_only_fires_once(self, scheduler, noop_job):
        scheduler.register_job(
            "one-shot", "* * * * *", noop_job, frequency=ScheduleFrequency.ONCE
        )
        due1 = scheduler.tick(datetime(2026, 1, 15, 10, 30))
        assert "one-shot" in due1

    async def test_once_frequency_not_fired_after_run(self, scheduler, noop_job):
        scheduler.register_job(
            "one-shot", "* * * * *", noop_job, frequency=ScheduleFrequency.ONCE
        )
        scheduler.tick(datetime(2026, 1, 15, 10, 30))
        await scheduler.run_pending()
        # After running, run_count=1 so ONCE should not fire again
        due2 = scheduler.tick(datetime(2026, 1, 15, 10, 31))
        assert "one-shot" not in due2

    def test_invalid_cron_expression(self, scheduler, noop_job):
        with pytest.raises(SchedulerError, match="Invalid cron expression"):
            scheduler.register_job("bad-cron", "not a cron", noop_job)

    def test_cron_step_notation(self, scheduler, noop_job):
        """*/15 should match minutes 0, 15, 30, 45."""
        scheduler.register_job("every-15", "*/15 * * * *", noop_job)
        assert "every-15" in scheduler.tick(datetime(2026, 1, 15, 10, 0))
        assert "every-15" in scheduler.tick(datetime(2026, 1, 15, 10, 15))
        assert "every-15" in scheduler.tick(datetime(2026, 1, 15, 10, 30))
        assert "every-15" not in scheduler.tick(datetime(2026, 1, 15, 10, 7))

    def test_cron_range(self, scheduler, noop_job):
        """1-5 in the hour field should match hours 1 through 5."""
        scheduler.register_job("early", "0 1-5 * * *", noop_job)
        assert "early" in scheduler.tick(datetime(2026, 1, 15, 3, 0))
        assert "early" not in scheduler.tick(datetime(2026, 1, 15, 10, 0))

    def test_cron_list(self, scheduler, noop_job):
        """0,30 in the minute field should match minutes 0 and 30."""
        scheduler.register_job("half-hour", "0,30 * * * *", noop_job)
        assert "half-hour" in scheduler.tick(datetime(2026, 1, 15, 10, 0))
        assert "half-hour" in scheduler.tick(datetime(2026, 1, 15, 10, 30))
        assert "half-hour" not in scheduler.tick(datetime(2026, 1, 15, 10, 15))

    def test_no_double_fire_same_minute(self, scheduler, noop_job):
        """A job should not fire twice in the same minute."""
        scheduler.register_job("once-per-min", "* * * * *", noop_job)
        # First tick — should fire
        due1 = scheduler.tick(datetime(2026, 1, 15, 10, 30))
        assert "once-per-min" in due1
        # Simulate that the job ran
        scheduler.jobs["once-per-min"].last_run = datetime(2026, 1, 15, 10, 30).timestamp()
        # Same minute — should NOT fire again
        due2 = scheduler.tick(datetime(2026, 1, 15, 10, 30))
        assert "once-per-min" not in due2
