"""Tests for the SecuritySandbox."""

import pytest

from security_platform.sandbox.security_sandbox import (
    SecuritySandbox,
    SecuritySandboxConfig,
)
from platform_shared.sandbox import SandboxStatus
from platform_shared.sandbox.runtime import InProcessSandbox


class TestSecuritySandbox:
    @pytest.fixture
    def sandbox(self):
        runtime = InProcessSandbox()
        config = SecuritySandboxConfig(max_memory_mb=128, timeout_seconds=30.0)
        return SecuritySandbox(runtime=runtime, config=config)

    async def test_create_sandbox(self, sandbox: SecuritySandbox):
        sid = await sandbox.create()
        assert sid is not None
        assert sandbox.sandbox_id == sid

    async def test_verify_isolation(self, sandbox: SecuritySandbox):
        await sandbox.create()
        status = sandbox.verify_isolation()
        assert status.isolated is True
        assert status.no_network is True
        assert status.readonly_fs is True
        assert status.memory_limited is True
        assert status.violations == []

    async def test_get_status(self, sandbox: SecuritySandbox):
        await sandbox.create()
        status = await sandbox.get_status()
        assert status == SandboxStatus.READY

    async def test_destroy(self, sandbox: SecuritySandbox):
        await sandbox.create()
        await sandbox.destroy()
        assert sandbox.sandbox_id is None

    async def test_isolation_rule_created(self, sandbox: SecuritySandbox):
        await sandbox.create()
        rule = sandbox.isolation_rule
        assert rule is not None
        assert rule.no_network is True
        assert rule.readonly_fs is True
        assert rule.max_memory_mb == 128

    async def test_verify_before_create_raises(self, sandbox: SecuritySandbox):
        with pytest.raises(RuntimeError, match="not created"):
            sandbox.verify_isolation()

    async def test_status_before_create_raises(self, sandbox: SecuritySandbox):
        with pytest.raises(RuntimeError, match="not created"):
            await sandbox.get_status()

    async def test_custom_config(self):
        runtime = InProcessSandbox()
        config = SecuritySandboxConfig(
            max_memory_mb=256,
            timeout_seconds=60.0,
            allowed_paths=["/tmp/safe"],
        )
        sandbox = SecuritySandbox(runtime=runtime, config=config)
        await sandbox.create()
        rule = sandbox.isolation_rule
        assert rule is not None
        assert rule.max_memory_mb == 256
        assert rule.allowed_paths == ["/tmp/safe"]
