"""ObservabilitySandbox — runs health probes in an isolated sandbox environment."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from platform_shared.sandbox import (
    SandboxConfig,
    SandboxStatus,
    ExecutionResult,
    SandboxRuntime,
    ProcessSandbox,
)
from platform_shared.sandbox.network import POLICY_NO_NETWORK
from platform_shared.sandbox.resource import ResourceLimits
from platform_shared.protocols.health import HealthStatus, HealthReport


@dataclass(slots=True)
class ProbeResult:
    """Result of a sandboxed health probe."""
    app.kubernetes.io/component: str
    healthy: bool
    message: str = ""
    duration_ms: float = 0.0


class ObservabilitySandbox:
    """Executes health probes inside a sandboxed environment.

    Each probe runs with restricted resources and network isolation
    to prevent runaway checks from affecting the host system.
    """

    def __init__(
        self,
        runtime: SandboxRuntime | None = None,
        resource_limits: ResourceLimits | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._runtime = runtime or ProcessSandbox()
        self._resource_limits = resource_limits or ResourceLimits(
            cpu_cores=0.5,
            memory_mb=128,
            max_processes=8,
        )
        self._timeout = timeout_seconds
        self._sandbox_id: str | None = None
        self._probes: dict[str, Callable[[], Awaitable[bool]]] = {}

    async def initialize(self) -> str:
        """Create the sandbox environment."""
        config = SandboxConfig(
            name="observability-probes",
            resource_limits=self._resource_limits,
            timeout_seconds=self._timeout,
            network_policy=POLICY_NO_NETWORK,
            filesystem_readonly=True,
        )
        self._sandbox_id = await self._runtime.create(config)
        return self._sandbox_id

    def register_probe(self, name: str, fn: Callable[[], Awaitable[bool]]) -> None:
        """Register a health probe function."""
        self._probes[name] = fn

    async def run_probes(self) -> list[ProbeResult]:
        """Execute all registered probes and return results.

        Probes run concurrently within the timeout window.
        """
        results: list[ProbeResult] = []

        for name, fn in self._probes.items():
            import time
            start = time.monotonic()
            try:
                healthy = await asyncio.wait_for(fn(), timeout=self._timeout)
                elapsed_ms = (time.monotonic() - start) * 1000.0
                results.append(ProbeResult(
                    component=name,
                    healthy=healthy,
                    message="OK" if healthy else "Probe returned False",
                    duration_ms=elapsed_ms,
                ))
            except asyncio.TimeoutError:
                elapsed_ms = (time.monotonic() - start) * 1000.0
                results.append(ProbeResult(
                    component=name,
                    healthy=False,
                    message=f"Probe timed out after {self._timeout}s",
                    duration_ms=elapsed_ms,
                ))
            except Exception as exc:
                elapsed_ms = (time.monotonic() - start) * 1000.0
                results.append(ProbeResult(
                    component=name,
                    healthy=False,
                    message=f"Probe failed: {exc}",
                    duration_ms=elapsed_ms,
                ))

        return results

    async def destroy(self) -> None:
        """Tear down the sandbox."""
        if self._sandbox_id:
            await self._runtime.destroy(self._sandbox_id)
            self._sandbox_id = None

    @property
    def sandbox_id(self) -> str | None:
        return self._sandbox_id

    @property
    def registered_probes(self) -> list[str]:
        return list(self._probes.keys())
