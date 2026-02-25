"""Tests for the ObservabilitySandbox."""

import asyncio

import pytest

from observability_platform.sandbox.observability_sandbox import (
    ObservabilitySandbox,
    ProbeResult,
)
from platform_shared.sandbox import SandboxConfig, SandboxStatus
from platform_shared.sandbox.runtime import InProcessSandbox


class TestObservabilitySandbox:
    @pytest.fixture
    def sandbox(self):
        runtime = InProcessSandbox()
        return ObservabilitySandbox(runtime=runtime, timeout_seconds=5.0)

    async def test_initialize_creates_sandbox(self, sandbox: ObservabilitySandbox):
        sid = await sandbox.initialize()
        assert sid is not None
        assert sandbox.sandbox_id == sid

    async def test_register_and_run_probes(self, sandbox: ObservabilitySandbox):
        await sandbox.initialize()

        async def healthy_probe():
            return True

        async def unhealthy_probe():
            return False

        sandbox.register_probe("healthy", healthy_probe)
        sandbox.register_probe("unhealthy", unhealthy_probe)

        results = await sandbox.run_probes()
        assert len(results) == 2

        healthy_result = next(r for r in results if r.component == "healthy")
        assert healthy_result.healthy is True

        unhealthy_result = next(r for r in results if r.component == "unhealthy")
        assert unhealthy_result.healthy is False

    async def test_probe_timeout(self, sandbox: ObservabilitySandbox):
        sandbox._timeout = 0.1
        await sandbox.initialize()

        async def slow_probe():
            await asyncio.sleep(10)
            return True

        sandbox.register_probe("slow", slow_probe)
        results = await sandbox.run_probes()
        assert len(results) == 1
        assert results[0].healthy is False
        assert "timed out" in results[0].message

    async def test_probe_exception(self, sandbox: ObservabilitySandbox):
        await sandbox.initialize()

        async def failing_probe():
            raise RuntimeError("connection error")

        sandbox.register_probe("failing", failing_probe)
        results = await sandbox.run_probes()
        assert len(results) == 1
        assert results[0].healthy is False
        assert "connection error" in results[0].message

    async def test_destroy(self, sandbox: ObservabilitySandbox):
        await sandbox.initialize()
        assert sandbox.sandbox_id is not None
        await sandbox.destroy()
        assert sandbox.sandbox_id is None

    async def test_registered_probes(self, sandbox: ObservabilitySandbox):
        async def probe():
            return True

        sandbox.register_probe("a", probe)
        sandbox.register_probe("b", probe)
        assert sorted(sandbox.registered_probes) == ["a", "b"]

    async def test_probe_result_fields(self, sandbox: ObservabilitySandbox):
        await sandbox.initialize()

        async def ok():
            return True

        sandbox.register_probe("test", ok)
        results = await sandbox.run_probes()
        result = results[0]
        assert isinstance(result, ProbeResult)
        assert result.component == "test"
        assert result.healthy is True
        assert result.duration_ms >= 0
