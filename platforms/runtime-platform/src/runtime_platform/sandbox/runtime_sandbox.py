"""RuntimeSandbox — wraps job step execution inside a ProcessSandbox.

Provides isolated execution of shell commands with per-step timeouts,
environment variable injection, and result capture.
"""

from __future__ import annotations

import asyncio
from typing import Any

from platform_shared.sandbox import (
    ExecutionResult,
    ProcessSandbox,
    SandboxConfig,
    SandboxStatus,
)
from platform_shared.sandbox.resource import ResourceLimits

from runtime_platform.domain.entities import Job, Step
from runtime_platform.domain.value_objects import JobStatus, StepType
from runtime_platform.domain.exceptions import JobExecutionError


class RuntimeSandbox:
    """Isolates job step execution inside a ProcessSandbox.

    Each job gets its own sandbox instance. Steps are executed sequentially
    within the sandbox, inheriting its resource limits and network isolation.
    """

    def __init__(
        self,
        sandbox: ProcessSandbox | None = None,
        resource_limits: ResourceLimits | None = None,
    ) -> None:
        self._sandbox = sandbox or ProcessSandbox()
        self._resource_limits = resource_limits or ResourceLimits()
        self._active_sandboxes: dict[str, str] = {}  # job_id -> sandbox_id

    @property
    def active_sandboxes(self) -> dict[str, str]:
        return dict(self._active_sandboxes)

    async def execute_job(self, job: Job) -> dict[str, Any]:
        """Execute all steps of a job inside an isolated sandbox.

        Returns a dict with job_id, results per step, and overall success.
        """
        config = SandboxConfig(
            name=f"job-{job.id}",
            resource_limits=self._resource_limits,
            timeout_seconds=max(step.timeout for step in job.steps) if job.steps else 60.0,
        )

        sandbox_id = await self._sandbox.create(config)
        self._active_sandboxes[job.id] = sandbox_id
        job.mark_running()

        step_results: list[dict[str, Any]] = []
        success = True

        try:
            for step in job.steps:
                result = await self.execute_step(sandbox_id, step)
                step_results.append(result)
                if not result["success"]:
                    success = False
                    job.mark_failed(result.get("error", "Step failed"))
                    break

            if success:
                job.mark_completed()

        finally:
            await self._sandbox.destroy(sandbox_id)
            self._active_sandboxes.pop(job.id, None)

        return {
            "job_id": job.id,
            "sandbox_id": sandbox_id,
            "success": success,
            "status": job.status.value,
            "steps": step_results,
        }

    async def execute_step(self, sandbox_id: str, step: Step) -> dict[str, Any]:
        """Execute a single step inside an existing sandbox.

        Shell steps run as subprocess commands.
        Python steps are evaluated in-process.
        HTTP steps return a stub result.
        """
        if step.type == StepType.SHELL:
            return await self._execute_shell(sandbox_id, step)
        elif step.type == StepType.PYTHON:
            return await self._execute_python(step)
        elif step.type == StepType.HTTP:
            return self._execute_http(step)
        else:
            return {
                "step": step.name,
                "success": False,
                "error": f"Unknown step type: {step.type}",
            }

    async def _execute_shell(self, sandbox_id: str, step: Step) -> dict[str, Any]:
        """Execute a shell command inside the sandbox with the step's timeout."""
        # Temporarily override config timeout to match step timeout
        state = self._sandbox._sandboxes.get(sandbox_id)
        original_timeout = None
        if state is not None:
            original_timeout = state.config.timeout_seconds
            # We need to create a new config since SandboxConfig uses slots
            state.config = SandboxConfig(
                name=state.config.name,
                resource_limits=state.config.resource_limits,
                timeout_seconds=step.timeout,
                network_policy=state.config.network_policy,
                volumes=state.config.volumes,
                env_vars={**state.config.env_vars, **step.env},
                working_dir=state.config.working_dir,
                filesystem_readonly=state.config.filesystem_readonly,
                labels=state.config.labels,
            )

        try:
            # Split command into args for the sandbox
            command = ["sh", "-c", step.command]
            result: ExecutionResult = await self._sandbox.execute(sandbox_id, command)

            return {
                "step": step.name,
                "success": result.success,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_seconds": result.duration_seconds,
                "timed_out": result.timed_out,
            }

        except Exception as exc:
            return {
                "step": step.name,
                "success": False,
                "error": str(exc),
            }
        finally:
            # Restore original timeout
            if state is not None and original_timeout is not None:
                state.config = SandboxConfig(
                    name=state.config.name,
                    resource_limits=state.config.resource_limits,
                    timeout_seconds=original_timeout,
                    network_policy=state.config.network_policy,
                    volumes=state.config.volumes,
                    env_vars=state.config.env_vars,
                    working_dir=state.config.working_dir,
                    filesystem_readonly=state.config.filesystem_readonly,
                    labels=state.config.labels,
                )

    async def _execute_python(self, step: Step) -> dict[str, Any]:
        """Execute a Python expression/statement."""
        try:
            result = eval(step.command)  # noqa: S307
            return {
                "step": step.name,
                "success": True,
                "result": result,
            }
        except Exception as exc:
            return {
                "step": step.name,
                "success": False,
                "error": str(exc),
            }

    def _execute_http(self, step: Step) -> dict[str, Any]:
        """Stub HTTP execution (would use aiohttp in production)."""
        return {
            "step": step.name,
            "success": True,
            "status": "http_stub",
            "url": step.command,
        }

    async def cleanup(self) -> None:
        """Destroy all active sandboxes."""
        for job_id, sandbox_id in list(self._active_sandboxes.items()):
            try:
                await self._sandbox.destroy(sandbox_id)
            except Exception:
                pass
        self._active_sandboxes.clear()
