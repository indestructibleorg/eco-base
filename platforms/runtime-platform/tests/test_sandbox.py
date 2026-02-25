"""Tests for the RuntimeSandbox — isolated job step execution."""

import pytest

from platform_shared.sandbox import ProcessSandbox, SandboxConfig, SandboxStatus
from platform_shared.sandbox.resource import ResourceLimits

from runtime_platform.domain.entities import Job, Step
from runtime_platform.domain.value_objects import JobStatus, StepType
from runtime_platform.sandbox.runtime_sandbox import RuntimeSandbox


class TestRuntimeSandbox:
    """Test sandbox-isolated job execution."""

    @pytest.fixture
    def sandbox(self):
        return RuntimeSandbox()

    @pytest.fixture
    def echo_job(self):
        return Job(
            id="job-echo",
            name="echo-job",
            steps=[
                Step(name="say-hello", type=StepType.SHELL, command="echo hello", timeout=10.0),
            ],
        )

    @pytest.fixture
    def multi_step_job(self):
        return Job(
            id="job-multi",
            name="multi-step",
            steps=[
                Step(name="step-1", type=StepType.SHELL, command="echo step1", timeout=10.0),
                Step(name="step-2", type=StepType.SHELL, command="echo step2", timeout=10.0),
            ],
        )

    @pytest.fixture
    def python_job(self):
        return Job(
            id="job-py",
            name="python-job",
            steps=[
                Step(name="calc", type=StepType.PYTHON, command="2 + 2", timeout=5.0),
            ],
        )

    @pytest.fixture
    def failing_job(self):
        return Job(
            id="job-fail",
            name="failing-job",
            steps=[
                Step(name="bad-cmd", type=StepType.SHELL, command="exit 1", timeout=10.0),
            ],
        )

    async def test_execute_shell_step(self, sandbox, echo_job):
        result = await sandbox.execute_job(echo_job)
        assert result["success"] is True
        assert result["job_id"] == "job-echo"
        assert echo_job.status == JobStatus.COMPLETED
        assert len(result["steps"]) == 1
        assert "hello" in result["steps"][0]["stdout"]

    async def test_execute_multi_step(self, sandbox, multi_step_job):
        result = await sandbox.execute_job(multi_step_job)
        assert result["success"] is True
        assert len(result["steps"]) == 2
        assert "step1" in result["steps"][0]["stdout"]
        assert "step2" in result["steps"][1]["stdout"]

    async def test_execute_python_step(self, sandbox, python_job):
        result = await sandbox.execute_job(python_job)
        assert result["success"] is True
        assert result["steps"][0]["result"] == 4

    async def test_failing_step_marks_job_failed(self, sandbox, failing_job):
        result = await sandbox.execute_job(failing_job)
        assert result["success"] is False
        assert failing_job.status == JobStatus.FAILED

    async def test_sandbox_cleanup_after_execution(self, sandbox, echo_job):
        await sandbox.execute_job(echo_job)
        # After execution, the sandbox should be destroyed
        assert echo_job.id not in sandbox.active_sandboxes

    async def test_http_step(self, sandbox):
        job = Job(
            id="job-http",
            name="http-job",
            steps=[
                Step(name="api-call", type=StepType.HTTP, command="https://example.com", timeout=5.0),
            ],
        )
        result = await sandbox.execute_job(job)
        assert result["success"] is True
        assert result["steps"][0]["status"] == "http_stub"

    async def test_custom_resource_limits(self):
        limits = ResourceLimits(cpu_cores=0.5, memory_mb=256)
        sandbox = RuntimeSandbox(resource_limits=limits)
        job = Job(
            id="job-limited",
            name="limited",
            steps=[Step(name="s", type=StepType.SHELL, command="echo ok", timeout=5.0)],
        )
        result = await sandbox.execute_job(job)
        assert result["success"] is True

    async def test_multi_step_stops_on_failure(self, sandbox):
        """When a step fails, subsequent steps should not execute."""
        job = Job(
            id="job-stop",
            name="stop-on-fail",
            steps=[
                Step(name="ok-step", type=StepType.SHELL, command="echo ok", timeout=10.0),
                Step(name="fail-step", type=StepType.SHELL, command="exit 1", timeout=10.0),
                Step(name="never-run", type=StepType.SHELL, command="echo nope", timeout=10.0),
            ],
        )
        result = await sandbox.execute_job(job)
        assert result["success"] is False
        # Only 2 steps should have been executed (ok + fail), third was skipped
        assert len(result["steps"]) == 2
        assert result["steps"][0]["success"] is True
        assert result["steps"][1]["success"] is False

    async def test_cleanup(self, sandbox, echo_job):
        """Test the explicit cleanup method."""
        await sandbox.cleanup()
        assert sandbox.active_sandboxes == {}
