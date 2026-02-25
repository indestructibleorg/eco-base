"""SecuritySandbox — hardened sandbox with all caps dropped, read-only fs, no network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from platform_shared.sandbox import (
    SandboxConfig,
    SandboxStatus,
    ExecutionResult,
    SandboxRuntime,
    ProcessSandbox,
)
from platform_shared.sandbox.network import POLICY_NO_NETWORK
from platform_shared.sandbox.resource import ResourceLimits

from security_platform.domain.entities import IsolationRule
from security_platform.engines.isolation_engine import IsolationEngine, IsolationStatus


@dataclass(slots=True)
class SecuritySandboxConfig:
    """Hardened sandbox configuration preset."""
    max_memory_mb: int = 128
    timeout_seconds: float = 30.0
    allowed_paths: list[str] = field(default_factory=list)


class SecuritySandbox:
    """A hardened sandbox environment with maximum isolation.

    All capabilities are dropped, filesystem is read-only,
    network access is blocked, and resource limits are strict.
    """

    def __init__(
        self,
        runtime: SandboxRuntime | None = None,
        config: SecuritySandboxConfig | None = None,
    ) -> None:
        self._runtime = runtime or ProcessSandbox()
        self._config = config or SecuritySandboxConfig()
        self._isolation_engine = IsolationEngine()
        self._sandbox_id: str | None = None
        self._isolation_rule: IsolationRule | None = None

    async def create(self) -> str:
        """Create a hardened sandbox with maximum isolation constraints."""
        sandbox_config = SandboxConfig(
            name="security-sandbox",
            resource_limits=ResourceLimits(
                cpu_cores=0.5,
                memory_mb=self._config.max_memory_mb,
                max_processes=8,
                max_open_fds=64,
            ),
            timeout_seconds=self._config.timeout_seconds,
            network_policy=POLICY_NO_NETWORK,
            filesystem_readonly=True,
        )

        self._sandbox_id = await self._runtime.create(sandbox_config)

        # Register isolation rule
        self._isolation_rule = IsolationRule(
            sandbox_id=self._sandbox_id,
            no_network=True,
            readonly_fs=True,
            max_memory_mb=self._config.max_memory_mb,
            allowed_paths=self._config.allowed_paths,
        )
        self._isolation_engine.apply_isolation(self._isolation_rule)

        return self._sandbox_id

    def verify_isolation(self) -> IsolationStatus:
        """Verify that the sandbox meets isolation requirements."""
        if self._sandbox_id is None:
            raise RuntimeError("Sandbox not created")
        return self._isolation_engine.verify_isolation(self._sandbox_id)

    async def get_status(self) -> SandboxStatus:
        """Get current sandbox status."""
        if self._sandbox_id is None:
            raise RuntimeError("Sandbox not created")
        return await self._runtime.get_status(self._sandbox_id)

    async def destroy(self) -> None:
        """Tear down the hardened sandbox."""
        if self._sandbox_id:
            await self._runtime.destroy(self._sandbox_id)
            self._isolation_engine.remove_rule(self._sandbox_id)
            self._sandbox_id = None

    @property
    def sandbox_id(self) -> str | None:
        return self._sandbox_id

    @property
    def isolation_rule(self) -> IsolationRule | None:
        return self._isolation_rule
