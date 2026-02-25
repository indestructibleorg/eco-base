"""WorkflowEngine — DAG-based job orchestration with topological execution.

Resolves job dependencies via topological sort, then executes independent jobs
in parallel using asyncio.gather while respecting the dependency order.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Any, Callable, Awaitable

from platform_shared.protocols.engine import EngineStatus

from runtime_platform.domain.entities import Job, Step, Workflow
from runtime_platform.domain.value_objects import JobStatus, StepType
from runtime_platform.domain.events import JobCompleted, StepFailed, WorkflowStarted
from runtime_platform.domain.exceptions import (
    CyclicDependencyError,
    JobExecutionError,
    WorkflowError,
)


# Type alias for the step executor callback
StepExecutor = Callable[[Step], Awaitable[dict[str, Any]]]


class WorkflowEngine:
    """Orchestrates a DAG of jobs with dependency-aware parallel execution.

    The engine topologically sorts jobs, groups them into levels of independent
    jobs, and runs each level concurrently via asyncio.gather.
    """

    def __init__(self, step_executor: StepExecutor | None = None) -> None:
        self._status = EngineStatus.IDLE
        self._events: list[Any] = []
        self._step_executor = step_executor or _default_step_executor

    @property
    def name(self) -> str:
        return "workflow-engine"

    @property
    def status(self) -> EngineStatus:
        return self._status

    @property
    def events(self) -> list[Any]:
        return list(self._events)

    async def start(self) -> None:
        self._status = EngineStatus.RUNNING

    async def stop(self) -> None:
        self._status = EngineStatus.STOPPED

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Protocol-compatible execute — wraps run_workflow."""
        workflow = payload.get("workflow")
        if not isinstance(workflow, Workflow):
            raise WorkflowError("Payload must contain a 'workflow' key with a Workflow instance")
        result = await self.run_workflow(workflow)
        return {"workflow_id": workflow.id, "status": workflow.status.value, "result": result}

    async def run_workflow(self, workflow: Workflow) -> dict[str, Any]:
        """Execute a workflow: topological sort, then parallel-by-level execution."""
        self._status = EngineStatus.RUNNING
        workflow.status = JobStatus.RUNNING
        start_time = time.monotonic()

        self._events.append(WorkflowStarted(
            workflow_id=workflow.id,
            workflow_name=workflow.name,
            job_count=len(workflow.jobs),
        ))

        try:
            # Build job index by name (dependencies reference names)
            job_map = {job.name: job for job in workflow.jobs}
            execution_levels = self._topological_levels(workflow)

            for level in execution_levels:
                # Run all jobs in this level concurrently
                tasks = [self._run_job(workflow, job_map[name]) for name in level]
                await asyncio.gather(*tasks)

                # If any job in this level failed, skip downstream dependents
                failed_names = {
                    name for name in level if job_map[name].status == JobStatus.FAILED
                }
                if failed_names:
                    self._skip_dependents(workflow, failed_names, job_map)

            # Determine final workflow status
            if workflow.has_failures:
                workflow.status = JobStatus.FAILED
                workflow.error = "One or more jobs failed"
            else:
                workflow.status = JobStatus.COMPLETED

        except CyclicDependencyError:
            workflow.status = JobStatus.FAILED
            workflow.error = "Cyclic dependency in job graph"
            raise
        except Exception as exc:
            workflow.status = JobStatus.FAILED
            workflow.error = str(exc)
            self._status = EngineStatus.ERROR
            raise WorkflowError(str(exc), workflow_id=workflow.id) from exc
        finally:
            self._status = EngineStatus.IDLE

        duration = time.monotonic() - start_time
        return {
            "workflow_id": workflow.id,
            "status": workflow.status.value,
            "duration_seconds": duration,
            "jobs": {j.name: j.status.value for j in workflow.jobs},
        }

    def _topological_levels(self, workflow: Workflow) -> list[list[str]]:
        """Kahn's algorithm producing execution levels (groups of independent jobs).

        Each level contains jobs whose dependencies are all in earlier levels.
        """
        job_names = {job.name for job in workflow.jobs}
        in_degree: dict[str, int] = {name: 0 for name in job_names}
        dependents: dict[str, list[str]] = defaultdict(list)

        for job in workflow.jobs:
            for dep in job.depends_on:
                if dep not in job_names:
                    raise WorkflowError(
                        f"Job '{job.name}' depends on unknown job '{dep}'",
                        workflow_id=workflow.id,
                    )
                in_degree[job.name] += 1
                dependents[dep].append(job.name)

        # Seed queue with zero-degree nodes
        queue: deque[str] = deque(name for name, deg in in_degree.items() if deg == 0)
        levels: list[list[str]] = []
        visited = 0

        while queue:
            level = list(queue)
            queue.clear()
            levels.append(level)
            visited += len(level)

            for name in level:
                for dependent in dependents[name]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if visited != len(job_names):
            # Some nodes were never resolved => cycle
            remaining = [n for n in job_names if in_degree[n] > 0]
            raise CyclicDependencyError(remaining, workflow_id=workflow.id)

        return levels

    async def _run_job(self, workflow: Workflow, job: Job) -> None:
        """Execute all steps in a job sequentially."""
        if job.status.is_terminal:
            return

        job.mark_running()
        start_time = time.monotonic()

        for step in job.steps:
            try:
                await self._step_executor(step)
            except Exception as exc:
                job.mark_failed(str(exc))
                self._events.append(StepFailed(
                    workflow_id=workflow.id,
                    job_id=job.id,
                    step_name=step.name,
                    error=str(exc),
                ))
                self._events.append(JobCompleted(
                    workflow_id=workflow.id,
                    job_id=job.id,
                    job_name=job.name,
                    success=False,
                    error=str(exc),
                    duration_seconds=time.monotonic() - start_time,
                ))
                return

        job.mark_completed()
        self._events.append(JobCompleted(
            workflow_id=workflow.id,
            job_id=job.id,
            job_name=job.name,
            success=True,
            duration_seconds=time.monotonic() - start_time,
        ))

    def _skip_dependents(
        self,
        workflow: Workflow,
        failed_names: set[str],
        job_map: dict[str, Job],
    ) -> None:
        """Recursively skip all jobs that depend (directly or transitively) on failed jobs."""
        to_skip: set[str] = set()
        queue = deque(failed_names)

        while queue:
            failed = queue.popleft()
            for job in workflow.jobs:
                if failed in job.depends_on and job.name not in to_skip:
                    to_skip.add(job.name)
                    queue.append(job.name)

        for name in to_skip:
            if name in job_map and not job_map[name].status.is_terminal:
                job_map[name].mark_skipped()


async def _default_step_executor(step: Step) -> dict[str, Any]:
    """Default step executor — runs shell commands, evaluates python, or stubs HTTP."""
    if step.type == StepType.SHELL:
        proc = await asyncio.create_subprocess_shell(
            step.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=step.timeout)
        if proc.returncode != 0:
            raise JobExecutionError(
                f"Shell step '{step.name}' exited with code {proc.returncode}: "
                f"{stderr_b.decode(errors='replace')}",
                job_id="",
            )
        return {
            "exit_code": proc.returncode,
            "stdout": stdout_b.decode(errors="replace"),
            "stderr": stderr_b.decode(errors="replace"),
        }

    elif step.type == StepType.PYTHON:
        # Execute python expression/code and capture result
        result = eval(step.command)  # noqa: S307
        return {"result": result}

    elif step.type == StepType.HTTP:
        # Stub: in a real implementation this would make an HTTP request
        return {"status": "http_stub", "command": step.command}

    raise JobExecutionError(f"Unknown step type: {step.type}")
